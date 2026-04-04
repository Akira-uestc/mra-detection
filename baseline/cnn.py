import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import os
import glob

from window_utils import build_prompt_test_windows

plt.rcParams['font.sans-serif'] = ['SimHei']

WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000

def plot_results(scores, threshold, split_idx, save_path='/home/akira/codespace/mra-detection/anomaly_detection_results.png'):
    """绘制异常检测可视化图（与 mra.py 完全一致）"""
    plt.figure(figsize=(6, 5))
    plt.plot(scores, label='测试异常分数', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    plt.axvline(x=split_idx, color='g', linestyle=':', label='测试集分界')
    plt.xlabel('测试样本索引')
    plt.ylabel('重构误差')
    plt.title('CNN异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

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

def create_windows(data, seq_len=60, stride=1):
    """参考 mra.py 的滑窗方式，返回形状 (num_windows, seq_len, num_features)"""
    n = len(data)
    if n == 0:
        return np.zeros((0, seq_len, data.shape[1]), dtype=data.dtype)

    stop_idx = min(n, WINDOW_START_INDEX + WINDOW_SAMPLE_COUNT * stride)
    if stop_idx <= WINDOW_START_INDEX:
        return np.zeros((0, seq_len, data.shape[1]), dtype=data.dtype)

    windows = []
    for i in range(WINDOW_START_INDEX, stop_idx, stride):
        if i < seq_len:
            # 头部不足 seq_len 时，用首样本做前向填充
            pad_len = seq_len - i - 1
            window_data = np.concatenate(
                [np.tile(data[0:1], (pad_len, 1)), data[0 : i + 1]],
                axis=0,
            )
        else:
            window_data = data[i - seq_len + 1 : i + 1]
        windows.append(window_data)

    return np.stack(windows)

def prepare_data(seq_len=60, stride=1):
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    TRAIN_PATTERN = "train_*.csv"
    TEST_PATTERN  = "test_*.csv"

    # 1. 加载训练和测试数据
    print("Loading training data...")
    train_data, num_features = load_csv_dir(str(DATA_DIR / "train"), TRAIN_PATTERN)
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, _ = load_csv_dir(str(DATA_DIR / "test"), TEST_PATTERN)
    print(f"Test data: {test_data.shape}")

    # 2. 处理缺失值
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    # 3. 标准化 (fit on training data only)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data).astype(np.float32)
    test_data_scaled = scaler.transform(test_data).astype(np.float32)

    # 4. 生成滑窗
    X_train = create_windows(train_data_scaled, seq_len=seq_len, stride=stride)
    X_test, test_labels = build_prompt_test_windows(
        test_data_scaled,
        seq_len=seq_len,
        stride=stride,
    )

    # 5. 重塑数据以适应 1D-CNN: (Batch_Size, Channels, Length)
    X_train = np.transpose(X_train, (0, 2, 1))  # (N, num_features, seq_len)
    X_test = np.transpose(X_test, (0, 2, 1))

    # 6. 转为 PyTorch Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    return X_train_tensor, X_test_tensor, test_labels, num_features

# ---------------------------------------------------------
# 2. 定义 1D-CNN 模型
# ---------------------------------------------------------

class AnomalyDetectorCNN(nn.Module):
    def __init__(self, num_features=18):
        super(AnomalyDetectorCNN, self).__init__()

        # 1D-CNN Autoencoder: train on normal, detect anomalies by reconstruction error
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_len -> seq_len/2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_len/2 -> seq_len/4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=2),  # seq_len/4 -> seq_len/2
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=16,
                out_channels=num_features,
                kernel_size=2,
                stride=2,
                output_padding=0,  # seq_len/2 -> seq_len
            ),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        # 若长度不一致，进行裁剪或右侧补零
        if x_rec.size(-1) != x.size(-1):
            if x_rec.size(-1) > x.size(-1):
                x_rec = x_rec[..., : x.size(-1)]
            else:
                x_rec = F.pad(x_rec, (0, x.size(-1) - x_rec.size(-1)))
        return x_rec

# ---------------------------------------------------------
# 3. 训练与评估流程
# ---------------------------------------------------------

def train_model():
    # 准备数据
    SEQ_LEN = 60
    STRIDE = 1
    X_train, X_test, test_labels, num_features = prepare_data(
        seq_len=SEQ_LEN,
        stride=STRIDE,
    )
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyDetectorCNN(num_features=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 1
    print(f"开始训练，共 {epochs} 个 Epoch...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            recon = model(inputs)
            loss = criterion(recon, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 评估模型
    print("\n--- 异常检测评估结果 ---")
    model.eval()
    with torch.no_grad():
        X_train_dev = X_train.to(device)
        X_test_dev = X_test.to(device)

        recon_train = model(X_train_dev)
        train_scores = (recon_train - X_train_dev).pow(2).mean(dim=[1, 2]).detach().cpu().numpy()

        train_mean = float(np.mean(train_scores))
        train_std = float(np.std(train_scores))
        threshold = train_mean

        recon_test = model(X_test_dev)
        test_scores = (recon_test - X_test_dev).pow(2).mean(dim=[1, 2]).detach().cpu().numpy()
        y_true = test_labels
        split_idx = int(np.sum(y_true == 0))
        y_pred = (test_scores > threshold).astype(int)

        print(f"Device: {device}")
        print(f"Train recon error: mean={train_mean:.6f}, std={train_std:.6f}")
        print(f"Threshold (mean train score): {threshold:.6f}")
        print(f"Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores)}) anomaly")
        print(f"Anomalies detected: {(y_pred == 1).sum()} / {len(y_pred)}")

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

        plot_results(test_scores, threshold, split_idx)
        
if __name__ == "__main__":
    train_model()
