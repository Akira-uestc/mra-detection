import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# ==========================================
# 1. 数据读取函数
# ==========================================
def generate_mask_matrix():
    """读取之前生成的块状分布 CSV 数据"""
    try:
        # 假设 csv 在当前路径
        df = pd.read_csv("../TEP_3000_Block_Split.csv")
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
def create_sequences(data, seq_length, stride=1):
    """
    将2D数据转换为3D序列数据 (Samples, Seq_Len, Features)
    参考 mra.py 的滑窗逻辑：
    - 每个时间点都有一个窗口
    - 前期样本不足 seq_len 时用首条样本前置填充
    """
    xs = []
    n = len(data)
    if n == 0:
        return np.zeros((0, seq_length, data.shape[1]), dtype=data.dtype)

    for i in range(0, n, stride):
        if i < seq_length:
            pad_len = seq_length - i - 1
            if pad_len > 0:
                pad = np.tile(data[0:1], (pad_len, 1))
                x = np.concatenate([pad, data[0:i + 1]], axis=0)
            else:
                x = data[0:i + 1]
        else:
            x = data[i - seq_length + 1:i + 1]
        xs.append(x)

    return np.stack(xs)

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
    SPLIT_INDEX = 1500
    train_data_raw = raw_data[:SPLIT_INDEX]
    # 后 1500 个作为测试集 (仅用于测试/可视化)
    test_data_raw = raw_data[SPLIT_INDEX:]

    # --- 归一化---
    scaler = MinMaxScaler()
    # 只在训练集(正常数据)上 fit，防止通过测试集泄露未来信息
    train_data_norm = scaler.fit_transform(train_data_raw)
    test_data_norm = scaler.transform(test_data_raw)

    # --- 创建时间序列窗口 ---
    SEQ_LENGTH = 10
    
    X_train = create_sequences(train_data_norm, SEQ_LENGTH, stride=1)
    X_test = create_sequences(test_data_norm, SEQ_LENGTH, stride=1)

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
        train_tensor_eval = train_tensor.to(device)
        train_predictions = model(train_tensor_eval)
        # 计算训练集每个样本的重构误差 (MSE)
        train_loss_dist = torch.mean(
            (train_predictions - train_tensor_eval) ** 2, dim=[1, 2]
        ).cpu().numpy()

        test_tensor_eval = test_tensor.to(device)
        test_predictions = model(test_tensor_eval)
        # 计算测试集每个样本的重构误差 (MSE)
        test_loss_dist = torch.mean(
            (test_predictions - test_tensor_eval) ** 2, dim=[1, 2]
        ).cpu().numpy()

    # --- 设定阈值 ---
    # 使用训练集（正常数据）的误差分布设定阈值
    # 简单的阈值策略：均值 + 3倍标准差
    threshold = np.mean(train_loss_dist) + 3 * np.std(train_loss_dist)
    
    print(f"\n计算出的异常阈值: {threshold:.6f}")
    
    # --- 判定 ---
    anomalies = test_loss_dist > threshold
    print(f"检测到的异常数量: {np.sum(anomalies)}")

    # Classification Metrics
    # Ground truth: all test sequences are anomalous (label=1)
    y_true = np.ones(len(test_loss_dist), dtype=int)
    y_pred = anomalies.astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # --- 可视化结果 (match mra.py style exactly) ---
    scores = test_loss_dist
    plt.figure(figsize=(6, 5))
    plt.plot(scores, label='异常分数', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    plt.xlabel('样本索引')
    plt.ylabel('重构误差')
    plt.title('LSTM异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/akira/codespace/mra-detection/anomaly_detection_results.png', dpi=150)
    print("\nPlot saved to: /home/akira/codespace/mra-detection/anomaly_detection_results.png")
    plt.show()
