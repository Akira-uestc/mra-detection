import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from _project_root import PROJECT_ROOT

from utils.methods.data_loading import load_csv_dir_values
from utils.methods.display import (
    compute_binary_classification_metrics,
    plot_detection_scores,
    print_metrics,
)
from utils.methods.postprocess import (
    apply_ewaf_by_segments,
    choose_threshold,
    infer_segment_lengths,
    split_index_from_labels,
)
from utils.methods.windowing import (
    build_front_padded_windows,
    build_prompt_test_windows_values,
)

plt.rcParams['font.sans-serif'] = ['SimHei']

WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000
USE_EWAF = True
EWAF_ALPHA = 0.15

def prepare_data(seq_len=60, stride=1):
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_PATTERN = "train_*.csv"
    TEST_PATTERN  = "test_*.csv"

    # 1. 加载训练和测试数据
    print("Loading training data...")
    train_data, num_features = load_csv_dir_values(str(DATA_DIR / "train"), TRAIN_PATTERN)
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, _ = load_csv_dir_values(str(DATA_DIR / "test"), TEST_PATTERN)
    print(f"Test data: {test_data.shape}")

    # 2. 处理缺失值
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    # 3. 标准化 (fit on training data only)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data).astype(np.float32)
    test_data_scaled = scaler.transform(test_data).astype(np.float32)

    # 4. 生成滑窗
    X_train = build_front_padded_windows(
        train_data_scaled,
        seq_len=seq_len,
        stride=stride,
        start_index=WINDOW_START_INDEX,
        max_window_count=WINDOW_SAMPLE_COUNT,
    )
    X_test, test_labels = build_prompt_test_windows_values(
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
        if USE_EWAF:
            train_scores = apply_ewaf_by_segments(train_scores, EWAF_ALPHA)

        train_mean = float(np.mean(train_scores))
        train_std = float(np.std(train_scores))
        threshold = choose_threshold(train_scores, method="mean")

        recon_test = model(X_test_dev)
        test_scores = (recon_test - X_test_dev).pow(2).mean(dim=[1, 2]).detach().cpu().numpy()
        y_true = test_labels
        split_idx = split_index_from_labels(y_true)
        if USE_EWAF:
            test_scores = apply_ewaf_by_segments(
                test_scores,
                EWAF_ALPHA,
                infer_segment_lengths(y_true),
            )
        y_pred = (test_scores > threshold).astype(int)

        print(f"Device: {device}")
        print(f"Train recon error: mean={train_mean:.6f}, std={train_std:.6f}")
        print(f"Threshold (mean train score): {threshold:.6f}")
        print(f"Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores)}) anomaly")
        print(f"Anomalies detected: {(y_pred == 1).sum()} / {len(y_pred)}")

        metrics = compute_binary_classification_metrics(y_true, y_pred)
        print_metrics(
            "\nClassification Metrics:",
            metrics,
            order=["accuracy", "precision", "recall", "fdr", "fra", "f1"],
        )

        plot_detection_scores(
            test_scores,
            threshold,
            split_idx,
            "/home/akira/codespace/mra-detection/outputs/cnn/anomaly_detection_results.png",
            title="CNN异常检测",
            ylabel="重构误差",
            show=True,
        )
        
if __name__ == "__main__":
    train_model()
