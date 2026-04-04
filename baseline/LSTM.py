import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import os
import glob
from pathlib import Path

from window_utils import build_prompt_test_windows

plt.rcParams['font.sans-serif'] = ['SimHei']

WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000

# ==========================================
# 1. 数据读取函数
# ==========================================
def load_csv_dir(dir_path, file_pattern="*.csv"):
    """Load all CSV files matching file_pattern from a directory and concatenate.
    CSVs are headerless with numeric columns.
    Returns (data, num_features).
    """
    csv_files = sorted(glob.glob(os.path.join(dir_path, file_pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {dir_path}")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, header=None)
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} rows, {df.shape[1]} cols")
    data = pd.concat(dfs, ignore_index=True).to_numpy(dtype=np.float32)
    return data, data.shape[1]

# ==========================================
# 2. 数据预处理
# ==========================================
def create_sequences(data, seq_length, stride=1):
    """
    将 2D 数据转换为 3D 序列数据 (Samples, Seq_Len, Features)
    参考 mra.py 的滑窗逻辑
    """
    xs = []
    n = len(data)
    if n == 0:
        return np.zeros((0, seq_length, data.shape[1]), dtype=data.dtype)

    stop_idx = min(n, WINDOW_START_INDEX + WINDOW_SAMPLE_COUNT * stride)
    if stop_idx <= WINDOW_START_INDEX:
        return np.zeros((0, seq_length, data.shape[1]), dtype=data.dtype)

    for i in range(WINDOW_START_INDEX, stop_idx, stride):
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

# 加载数据
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATTERN = "train_*.csv"
TEST_PATTERN  = "test_*.csv"

print("Loading training data...")
train_data_raw, num_features = load_csv_dir(str(DATA_DIR / "train"), TRAIN_PATTERN)
print(f"Training data: {train_data_raw.shape}, num_features={num_features}")

print("\nLoading test data...")
test_data_raw, _ = load_csv_dir(str(DATA_DIR / "test"), TEST_PATTERN)
print(f"Test data: {test_data_raw.shape}")

# 处理缺失值
train_data_raw = np.nan_to_num(train_data_raw, nan=0.0)
test_data_raw = np.nan_to_num(test_data_raw, nan=0.0)

# 归一化 (fit on training data only)
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data_raw)
test_data_norm = scaler.transform(test_data_raw)

# 创建时间序列窗口
SEQ_LENGTH = 50

X_train = create_sequences(train_data_norm, SEQ_LENGTH, stride=1)
X_test, test_labels = build_prompt_test_windows(test_data_norm, SEQ_LENGTH, stride=1)

# 转为 PyTorch Tensor
train_tensor = torch.FloatTensor(X_train)
test_tensor = torch.FloatTensor(X_test)

print(f"训练集形状: {train_tensor.shape}")
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
model = LSTMAutoencoder(seq_len=SEQ_LENGTH, n_features=num_features, hidden_dim=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("开始训练模型...)")
num_epochs = 10

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
model.eval()
with torch.no_grad():
    train_tensor_eval = train_tensor.to(device)
    train_predictions = model(train_tensor_eval)
    train_loss_dist = torch.mean(
        (train_predictions - train_tensor_eval) ** 2, dim=[1, 2]
    ).cpu().numpy()

    test_tensor_eval = test_tensor.to(device)
    test_predictions = model(test_tensor_eval)
    test_loss_dist = torch.mean(
        (test_predictions - test_tensor_eval) ** 2, dim=[1, 2]
    ).cpu().numpy()

test_split_idx = int(np.sum(test_labels == 0))

# 设定阈值
threshold = float(np.mean(train_loss_dist))

print(f"\n计算出的异常阈值: {threshold:.6f}")

y_true = test_labels
y_pred = (test_loss_dist > threshold).astype(int)

print(f"测试集分界: [0:{test_split_idx}) normal, [{test_split_idx}:{len(test_loss_dist)}) anomaly")
print(f"检测到的异常数量: {(y_pred == 1).sum()} / {len(y_pred)}")

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
normal_mask = y_true == 0
fault_mask = y_true == 1
fra = float(np.mean(y_pred[normal_mask] == 1)) if np.any(normal_mask) else 0.0
fdr = float(np.mean(y_pred[fault_mask] == 1)) if np.any(fault_mask) else 0.0

print(f"\nClassification Metrics:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  FDR:       {fdr:.4f}")
print(f"  FRA:       {fra:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# 可视化结果
plt.figure(figsize=(6, 5))
plt.plot(test_loss_dist, label='测试异常分数', alpha=0.7)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
plt.axvline(x=test_split_idx, color='g', linestyle=':', label='测试集分界')
plt.xlabel('测试样本索引')
plt.ylabel('重构误差')
plt.title('LSTM异常检测')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/akira/codespace/mra-detection/anomaly_detection_results.png', dpi=150)
print("\nPlot saved to: /home/akira/codespace/mra-detection/anomaly_detection_results.png")
plt.show()
