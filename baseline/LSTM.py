import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据读取函数
# ==========================================
def generate_mask_matrix():
    """读取之前生成的块状分布 CSV 数据"""
    try:
        # 假设 csv 在当前路径
        df = pd.read_csv("./TEP_3000_Block_Split.csv")
        data_df = df.filter(like='xmeas_').iloc[:, :41]
        data = data_df.astype(float).to_numpy()
        mask = np.isnan(data).astype(int)
        return data, mask
    except FileNotFoundError:
        print("找不到数据文件，请确保 TEP_3000_Block_Split.csv 在当前目录下")
        return None, None

# ==========================================
# 2. 数据预处理
# ==========================================
def create_sequences(data, seq_length):
    """将2D数据转换为3D序列数据 (Samples, Seq_Len, Features)"""
    xs = []
    for i in range(len(data) - seq_length + 1):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

# 读取数据
raw_data, mask = generate_mask_matrix()

if raw_data is not None:
    print(f"数据加载成功喵！原始形状: {raw_data.shape}")

    # --- 关键步骤：处理 NaN ---
    # LSTM 无法处理 NaN，我们将 NaN 填充为 0 (假设数据已经中心化，或者0代表无信号)
    # 可以根据业务逻辑改成填均值
    raw_data = np.nan_to_num(raw_data, nan=0.0)

    # --- 划分训练集和测试集 ---
    # 前 1500 个是正常数据 (用于训练)
    train_data_raw = raw_data[:1500] 
    # 全部数据用于测试 (观察后 1500 个是否报警)
    test_data_raw = raw_data 

    # --- 归一化---
    scaler = MinMaxScaler()
    # 只在训练集(正常数据)上 fit，防止通过测试集泄露未来信息
    train_data_norm = scaler.fit_transform(train_data_raw)
    test_data_norm = scaler.transform(test_data_raw)

    # --- 创建时间序列窗口 ---
    SEQ_LENGTH = 10
    
    X_train = create_sequences(train_data_norm, SEQ_LENGTH)
    X_test = create_sequences(test_data_norm, SEQ_LENGTH)

    # 转为 PyTorch Tensor
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)
    
    print(f"训练集形状: {train_tensor.shape}") # (Samples, 10, 41)
    print(f"测试集形状: {test_tensor.shape}")

    # 创建 DataLoader
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=32, shuffle=True)

# ==========================================
# 3. 定义 LSTM 自编码器模型
# ==========================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder: 压缩信息
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Decoder: 还原信息
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        
        # Encoder
        # enc_out: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        _, (hidden_n, _) = self.encoder(x)
        
        # 将 Encoder 最后的隐状态复制 seq_len 次，作为 Decoder 的输入
        # 这一步是为了让 Decoder 根据压缩的特征重构整个序列
        # hidden_n 形状: (1, batch, hidden) -> squeeze -> (batch, hidden)
        repeated_hidden = hidden_n.squeeze(0).unsqueeze(1).repeat(1, x.shape[1], 1)
        
        # Decoder
        dec_out, _ = self.decoder(repeated_hidden)
        
        # 映射回原始特征维度
        reconstructed = self.output_layer(dec_out)
        
        return reconstructed

# ==========================================
# 4. 训练模型
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAutoencoder(seq_len=SEQ_LENGTH, n_features=41, hidden_dim=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if raw_data is not None:
    print("开始训练模型...)")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.6f}")

# ==========================================
# 5. 异常检测与评估
# ==========================================
if raw_data is not None:
    model.eval()
    with torch.no_grad():
        test_tensor = test_tensor.to(device)
        predictions = model(test_tensor)
        
        # 计算每个样本的重构误差 (MSE)
        # shape: (Samples, Seq_Len, Features) -> mean over axis 1 and 2
        loss_dist = torch.mean((predictions - test_tensor) ** 2, dim=[1, 2]).cpu().numpy()

    # --- 设定阈值 ---
    # 我们看前 1500 个样本（也就是正常部分）的误差分布
    # 注意：由于滑动窗口，数据点会少 seq_length 个
    split_point = 1500 - SEQ_LENGTH + 1
    
    normal_losses = loss_dist[:split_point]
    abnormal_losses = loss_dist[split_point:]
    
    # 简单的阈值策略：均值 + 3倍标准差
    threshold = np.mean(normal_losses) + 3 * np.std(normal_losses)
    
    print(f"\n计算出的异常阈值: {threshold:.6f}")
    
    # --- 判定 ---
    anomalies = loss_dist > threshold
    print(f"检测到的异常数量: {np.sum(anomalies)}")
    
    # --- 可视化结果 ---
    plt.figure(figsize=(12, 6))
    plt.plot(loss_dist, label='Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.axvline(x=split_point, color='g', linestyle='-', label='True Anomaly Start')
    plt.title('Anomaly Detection using LSTM Autoencoder (TEP Data)')
    plt.xlabel('Sample Index')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()
    
    print("图表已生成")