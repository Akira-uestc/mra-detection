import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 数据预处理与加载模块
# ==========================================

class TEPDatasetBuilder:
    def __init__(self, seq_len=60, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        """
        读取数据并进行预处理
        注意：根据你的描述，前1500为正常，后1500为异常。
        通常训练只使用正常数据（Unsupervised AD setting）。
        """
        try:
            # 读取数据
            df = pd.read_csv(file_path)
            data_df = df.filter(like='xmeas_').iloc[:, :41]
            raw_data = data_df.to_numpy() # shape: (3000, 41)
            
            # 1. 处理 NaN (mask生成)
            # 代码逻辑：np.isnan(data) -> 1 (缺失), 0 (观测)
            # 文本逻辑：Mask=1 (观测), 0 (待插补)
            # 我们在此处统一转换为文本逻辑：mask_obs = 1 (有值), 0 (缺失)
            isnan_mask = np.isnan(raw_data)
            mask_obs = (~isnan_mask).astype(float) # 1=Observed
            
            # 填补 NaN 为 0 (仅为了Scaler能运行，后续会被Mask盖住)
            data_filled = np.nan_to_num(raw_data, nan=0.0)
            
            # 2. 数据标准化 (仅在Observed数据上fit，避免泄露)
            # 这里简单起见对填补后的数据fit，更严谨的做法是只fit mask=1的部分
            self.scaler.fit(data_filled[:1500]) # 仅用正常数据fit
            data_scaled = self.scaler.transform(data_filled)
            
            # 再次将缺失值置为0 (标准化后0不再是0，需要mask作用)
            data_scaled = data_scaled * mask_obs 
            
            return data_scaled, mask_obs
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None, None

    def create_windows(self, data, mask):
        """生成滑动窗口序列 (Batch, Seq_Len, Nodes)"""
        n_samples, n_nodes = data.shape
        windows_data = []
        windows_mask = []
        
        for i in range(0, n_samples - self.seq_len + 1, self.stride):
            windows_data.append(data[i:i+self.seq_len])
            windows_mask.append(mask[i:i+self.seq_len])
            
        return np.array(windows_data), np.array(windows_mask)

# ==========================================
# 2. 核心网络模块
# ==========================================

class GraphLearner(nn.Module):
    """
    3.3 自适应变量关联学习模块
    生成邻接矩阵 A_adp
    """
    def __init__(self, num_nodes, embed_dim, alpha=3.0):
        super(GraphLearner, self).__init__()
        self.num_nodes = num_nodes
        # E1, E2: (N, d)
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.alpha = alpha
        
    def forward(self):
        # M1 = tanh(alpha * E1)
        M1 = torch.tanh(self.alpha * self.E1)
        M2 = torch.tanh(self.alpha * self.E2)
        
        # A_adp = Softmax(ReLU(M1 @ M2.T))
        adj = torch.matmul(M1, M2.transpose(0, 1))
        adj = F.relu(adj)
        
        # Row-wise Softmax implies incoming information normalization
        adj = F.softmax(adj, dim=-1) 
        return adj

class GCNLayer(nn.Module):
    """图卷积层: X' = A * X * W"""
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        # x: (Batch, Seq, Nodes, Features) or (Batch, Nodes, Features)
        # 这里为了简化，假设输入是 (Batch, Seq, Nodes) -> (Batch, Seq, Nodes, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)
            
        # 1. Feature transform: XW
        x_trans = self.linear(x) # (Batch, Seq, Nodes, Out_dim)
        
        # 2. Graph propagation: AX
        # adj: (Nodes, Nodes)
        # x_trans: (Batch, Seq, Nodes, Out_dim)
        # Einsum: bn (nodes, nodes), bsnd (batch, seq, nodes, dim) -> bsnd
        out = torch.einsum('nm, bsmd -> bsnd', adj, x_trans)
        
        return out.squeeze(-1) # Return (Batch, Seq, Nodes)

class TCNLayer(nn.Module):
    """时序卷积层 (Dilated Conv)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
                              
    def forward(self, x):
        # x: (Batch, Nodes, Seq) for Conv1d
        # Causal padding: remove the last padding
        out = self.conv(x)
        return out[:, :, :-self.padding]

class FrequencyImputer(nn.Module):
    """
    3.4.3 频域增强插补分支
    FFT -> Attention -> IFFT
    """
    def __init__(self, seq_len, num_nodes):
        super(FrequencyImputer, self).__init__()
        # 频域长度: rfft 后长度为 seq_len//2 + 1
        self.freq_len = seq_len // 2 + 1
        
        # 简单的频域注意力机制: 学习每个频率分量的权重
        self.freq_att = nn.Sequential(
            nn.Linear(self.freq_len * 2, 64), # *2 因为实部和虚部
            nn.ReLU(),
            nn.Linear(64, self.freq_len * 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Nodes)
        # 1. FFT
        x_freq = torch.fft.rfft(x, dim=1) # (Batch, Freq, Nodes) Complex
        
        # 2. Attention
        # 将实部和虚部拼接用于计算 Attention
        real = x_freq.real
        imag = x_freq.imag
        feat = torch.cat([real, imag], dim=1) # (Batch, 2*Freq, Nodes)
        
        # 调整维度以通过 Linear: (Batch, Nodes, 2*Freq)
        feat = feat.permute(0, 2, 1)
        att_weights = self.freq_att(feat) # (Batch, Nodes, 2*Freq)
        att_weights = att_weights.permute(0, 2, 1) # (Batch, 2*Freq, Nodes)
        
        # 应用权重
        weight_real = att_weights[:, :self.freq_len, :]
        weight_imag = att_weights[:, self.freq_len:, :]
        
        x_freq_enhanced = torch.complex(
            real * weight_real,
            imag * weight_imag
        )
        
        # 3. IFFT
        x_restored = torch.fft.irfft(x_freq_enhanced, n=x.shape[1], dim=1)
        return x_restored

# ==========================================
# 3. AGF-ADNet 整体模型
# ==========================================

class AGF_ADNet(nn.Module):
    def __init__(self, num_nodes=41, seq_len=60, d_model=64):
        super(AGF_ADNet, self).__init__()
        
        # 3.3 Graph Learning
        self.graph_learner = GraphLearner(num_nodes=num_nodes, embed_dim=16)
        
        # 3.4.1 Time Domain Branch
        self.gcn = GCNLayer(in_dim=1, out_dim=1) # Mapping scalar to scalar with graph prop
        self.tcn = TCNLayer(in_channels=num_nodes, out_channels=num_nodes)
        
        # 3.4.3 Frequency Domain Branch
        self.freq_imputer = FrequencyImputer(seq_len, num_nodes)
        
        # 3.4.4 Fusion
        # Concat (Time, Freq) -> Conv1x1 -> 1
        self.fusion_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        
        # 3.4.5 Transformer Encoder for AD
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_nodes, nhead=1, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final Reconstruction
        self.output_proj = nn.Linear(num_nodes, num_nodes)

    def forward(self, x, mask):
        """
        x: (Batch, Seq, Nodes) - Zero filled at missing parts
        mask: (Batch, Seq, Nodes) - 1 for Observed, 0 for Missing
        """
        batch, seq, nodes = x.shape
        
        # Get Adjacency Matrix
        adj = self.graph_learner()
        
        # --- Imputation Phase ---
        
        # 1. Time Branch
        # GCN: captures variable correlation
        h_gcn = self.gcn(x, adj) # (Batch, Seq, Nodes)
        
        # TCN: captures temporal dynamics (permute for Conv1d: B, N, S)
        h_tcn = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        h_time = h_gcn + h_tcn
        
        # 2. Freq Branch
        h_freq = self.freq_imputer(x)
        
        # 3. Fusion
        # Stack for Conv2d: (Batch, 2, Seq, Nodes)
        h_stacked = torch.stack([h_time, h_freq], dim=1)
        x_imputed_latent = self.fusion_conv(h_stacked).squeeze(1) # (Batch, Seq, Nodes)
        
        # 4. Mix with Observed Data
        # Replace predicted values with observed values where mask=1
        # This acts as the input to the AD module
        x_filled = x * mask + x_imputed_latent * (1 - mask)
        
        # --- Anomaly Detection Phase (Reconstruction) ---
        
        # Transformer Input: x_filled
        h_trans = self.transformer(x_filled)
        
        # Output Reconstruction
        x_recon = self.output_proj(h_trans)
        
        return x_recon, adj, x_filled

# ==========================================
# 4. 训练与运行逻辑
# ==========================================

def train_agf_adnet():
    # 参数设置
    SEQ_LEN = 1 # 窗口长度
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 50
    LAMBDA_SPARSITY = 0.5
    
    # 1. 加载数据
    builder = TEPDatasetBuilder(seq_len=SEQ_LEN, stride=5) # stride设大一点避免数据过多
    raw_data, mask_obs = builder.load_data("./TEP_3000_Block_Split.csv")
    
    if raw_data is None: return

    # 划分训练集（正常数据）和测试集（正常+异常）
    # 前1500是正常，后1500是异常
    train_data = raw_data[:1500]
    train_mask = mask_obs[:1500]
    
    test_data = raw_data[1500:] # 包含部分正常和全部异常
    test_mask = mask_obs[1500:]
    
    # 创建窗口
    X_train, M_train = builder.create_windows(train_data, train_mask)
    X_test, M_test = builder.create_windows(test_data, test_mask)
    
    # 转Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(M_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AGF_ADNet(num_nodes=41, seq_len=SEQ_LEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Start Training on {device}...")
    
    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, m_batch in train_loader:
            x_batch, m_batch = x_batch.to(device), m_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            x_recon, adj, _ = model(x_batch, m_batch)
            
            # Loss 1: Reconstruction Error (仅计算观测部分 M=1)
            # 损失函数定义：L_recon = || (X_recon - X_in) * M ||^2
            recon_loss = torch.mean(((x_recon - x_batch) * m_batch) ** 2)
            
            # Loss 2: Graph Sparsity (L1 norm of Adj)
            sparsity_loss = torch.mean(torch.abs(adj))
            
            loss = recon_loss + LAMBDA_SPARSITY * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}")

    # ==========================
    # 5. 异常检测评估
    # ==========================
    print("Evaluating...")
    model.eval()
    
    test_scores = []
    
    # 将测试集放入Loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(M_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for x_batch, m_batch in test_loader:
            x_batch, m_batch = x_batch.to(device), m_batch.to(device)
            
            x_recon, _, x_filled = model(x_batch, m_batch)
            
            # 异常得分定义：
            # 通常基于重构误差。对于缺失部分，使用Imputed值计算也可以，
            # 但最保守的方法是看观测值的重构误差。
            # 这里我们计算整个序列的平均误差作为该样本的得分。
            
            error = (x_recon - x_filled) ** 2 # (B, S, N)
            
            # 对每个时间步和变量求和，得到每个样本的得分
            score = torch.mean(error, dim=[1, 2]) # (B, )
            test_scores.extend(score.cpu().numpy())
            
    test_scores = np.array(test_scores)
    
    # 简单的可视化或阈值判定
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(test_scores, label='Anomaly Score')
    # 假设后半部分是异常，我们可以画一条竖线
    # 注意：滑动窗口会导致索引偏移，大致划分一下
    split_point = len(X_test) - (1500 // 5) # 粗略估算异常数据起始点
    plt.axvline(x=len(X_test) - (1500//5), color='r', linestyle='--', label='Fault Start (Approx)')
    plt.title("AGF-ADNet Anomaly Scores")
    plt.legend()
    plt.show()
    
    print("Done. Check the plot for anomaly scores.")

if __name__ == "__main__":
    train_agf_adnet()