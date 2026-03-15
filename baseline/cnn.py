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

plt.rcParams['font.sans-serif'] = ['SimHei']

def plot_results(scores, threshold, save_path='/home/akira/codespace/mra-detection/anomaly_detection_results.png'):
    """绘制异常检测可视化图（与 mra.py 完全一致）"""
    plt.figure(figsize=(6, 5))
    plt.plot(scores, label='异常分数', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    plt.xlabel('样本索引')
    plt.ylabel('重构误差')
    plt.title('CNN异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

"""
读取原始数据并生成掩码矩阵
取前41个状态变量，返回原始数据，并替换NaN为1、正常数据为0生成掩码
"""
def generate_mask_matrix():
    try:
        data_path = Path(__file__).resolve().parent.parent / "TEP_3000_Block_Split.csv"
        df = pd.read_csv(data_path)
        # 提取 xmeas_1 到 xmeas_41 (丢弃后面的 xmv)
        data_df = df.filter(like='xmeas_').iloc[:, :41]
        data = data_df.astype(float).to_numpy()
        # 1 代表缺失 (NaN), 0 代表观测值
        mask = np.isnan(data).astype(int)
        return data, mask
    except FileNotFoundError:
        print("警告: 找不到数据文件。正在生成模拟数据用于演示代码运行...")
        # 生成模拟数据以确保代码可运行
        mock_data = np.random.randn(3000, 41)
        # 随机设置一些 NaN
        mock_data[np.random.rand(*mock_data.shape) < 0.1] = np.nan
        mock_mask = np.isnan(mock_data).astype(int)
        return mock_data, mock_mask

def create_windows(data, seq_len=60, stride=1):
    """参考 mra.py 的滑窗方式，返回形状 (num_windows, seq_len, num_features)"""
    n = len(data)
    if n == 0:
        return np.zeros((0, seq_len, data.shape[1]))

    windows = []
    for i in range(0, n, stride):
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
    # 1. 加载数据
    data, mask = generate_mask_matrix()
    
    if data is None:
        return None, None, None, None

    # 2. 处理缺失值 (NaN)
    # 神经网络不能输入 NaN。这里我们将 NaN 填充为 0。
    # 由于后续会做标准化 (StandardScaler)，0 通常接近均值，是一个安全的填充值。
    # 如果希望模型感知"缺失"这个特征，可以将 mask 也作为输入拼接到 data 后面，
    # 但为了保持 CNN 简单，这里仅做填充。
    data = np.nan_to_num(data, nan=0.0)

    # 3. 生成标签 (0: 正常, 1: 异常)
    # 前1500正常，后1500异常
    y_normal = np.zeros(1500)
    y_faulty = np.ones(1500)
    y = np.concatenate([y_normal, y_faulty])

    # 4. 数据分割 (训练集/测试集)
    # 前 50% 用于训练，后 50% 用于测试（保持时间顺序，不打乱）
    split_idx = len(data) // 2

    # 5. 标准化 (Standardization)
    scaler = StandardScaler()
    scaler.fit(data[:split_idx])
    data_scaled = scaler.transform(data)  # 使用训练集的参数转换全量数据

    # 6. 生成滑窗 (参考 mra.py)
    X_train = create_windows(data_scaled[:split_idx], seq_len=seq_len, stride=stride)
    X_test = create_windows(data_scaled[split_idx:], seq_len=seq_len, stride=stride)
    y_train = y[:split_idx:stride]
    y_test = y[split_idx::stride]

    # 7. 重塑数据以适应 1D-CNN
    # PyTorch Conv1d 输入形状: (Batch_Size, Channels, Length)
    # 将 41 个变量作为通道数，滑窗长度 seq_len 作为序列长度
    X_train = np.transpose(X_train, (0, 2, 1))  # (N, 41, seq_len)
    X_test = np.transpose(X_test, (0, 2, 1))

    # 8. 转为 PyTorch Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # shape变为 (N, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# ---------------------------------------------------------
# 2. 定义 1D-CNN 模型
# ---------------------------------------------------------

class AnomalyDetectorCNN(nn.Module):
    def __init__(self, num_features=41):
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
    X_train, y_train, X_test, y_test = prepare_data(seq_len=SEQ_LEN, stride=STRIDE)
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyDetectorCNN(num_features=41).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 30
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
        threshold = train_mean + 3.0 * train_std

        recon_test = model(X_test_dev)
        test_scores = (recon_test - X_test_dev).pow(2).mean(dim=[1, 2]).detach().cpu().numpy()

        # Splice train scores to front of test scores for evaluation
        all_scores = np.concatenate([train_scores, test_scores])
        # Labels: 0 for train (normal), 1 for test (anomaly)
        y_true = np.concatenate([np.zeros(len(train_scores), dtype=int), np.ones(len(test_scores), dtype=int)])
        y_pred = (all_scores > threshold).astype(int)

        print(f"Device: {device}")
        print(f"Train recon error: mean={train_mean:.6f}, std={train_std:.6f}")
        print(f"Threshold (mean + 3*std): {threshold:.6f}")
        print(f"Anomalies detected: {(y_pred == 1).sum()} / {len(y_pred)}")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        plot_results(all_scores, threshold)
        
if __name__ == "__main__":
    train_model()
