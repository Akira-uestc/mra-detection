import glob
import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]

from window_utils import build_prompt_test_windows


WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000


def seed_everything(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_results(scores, threshold, split_idx, save_path):
    plt.figure(figsize=(6, 5))
    plt.plot(scores, label="测试异常分数", alpha=0.7)
    plt.axhline(y=threshold, color="r", linestyle="--", label=f"阈值 ({threshold:.4f})")
    plt.axvline(x=split_idx, color="g", linestyle=":", label="测试集分界")
    plt.xlabel("测试样本索引")
    plt.ylabel("重构误差")
    plt.title("Transformer异常检测")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def load_csv_dir(dir_path, file_pattern="*.csv"):
    csv_files = sorted(glob.glob(os.path.join(dir_path, file_pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {dir_path}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)
        dfs.append(df)
        print(f"  Loaded {csv_file}: {len(df)} rows, {df.shape[1]} cols")

    data = pd.concat(dfs, ignore_index=True).to_numpy(dtype=np.float32)
    return data, data.shape[1]


def create_windows(data, seq_len=60, stride=1):
    n = len(data)
    if n == 0:
        return np.zeros((0, seq_len, data.shape[1]), dtype=data.dtype)

    stop_idx = min(n, WINDOW_START_INDEX + WINDOW_SAMPLE_COUNT * stride)
    if stop_idx <= WINDOW_START_INDEX:
        return np.zeros((0, seq_len, data.shape[1]), dtype=data.dtype)

    windows = []
    for i in range(WINDOW_START_INDEX, stop_idx, stride):
        if i < seq_len:
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
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_pattern = "train_*.csv"
    test_pattern = "test_*.csv"

    print("Loading training data...")
    train_data, num_features = load_csv_dir(str(data_dir / "train"), train_pattern)
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, _ = load_csv_dir(str(data_dir / "test"), test_pattern)
    print(f"Test data: {test_data.shape}")

    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data).astype(np.float32)
    test_data_scaled = scaler.transform(test_data).astype(np.float32)

    x_train = create_windows(train_data_scaled, seq_len=seq_len, stride=stride)
    x_test, test_labels = build_prompt_test_windows(
        test_data_scaled,
        seq_len=seq_len,
        stride=stride,
    )

    return torch.FloatTensor(x_train), torch.FloatTensor(x_test), test_labels, num_features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AnomalyDetectorTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_features),
        )

    def forward(self, x):
        hidden = self.input_proj(x)
        hidden = self.positional_encoding(hidden)
        hidden = self.encoder(hidden)
        return self.output_proj(hidden)


def train_model():
    seed_everything(40)

    seq_len = 60
    stride = 1
    batch_size = 64
    epochs = 10
    output_path = Path(__file__).resolve().parent.parent / "outputs" / "transformer_detection.png"

    x_train, x_test, test_labels, num_features = prepare_data(
        seq_len=seq_len,
        stride=stride,
    )
    train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyDetectorTransformer(num_features=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("\n--- 异常检测评估结果 ---")
    model.eval()
    with torch.no_grad():
        x_train_dev = x_train.to(device)
        x_test_dev = x_test.to(device)

        recon_train = model(x_train_dev)
        train_scores = (recon_train - x_train_dev).pow(2).mean(dim=[1, 2]).cpu().numpy()

        train_mean = float(np.mean(train_scores))
        train_std = float(np.std(train_scores))
        threshold = train_mean

        recon_test = model(x_test_dev)
        test_scores = (recon_test - x_test_dev).pow(2).mean(dim=[1, 2]).cpu().numpy()
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

        print("\nClassification Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  FDR:       {fdr:.4f}")
        print(f"  FRA:       {fra:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        plot_results(test_scores, threshold, split_idx, output_path)


if __name__ == "__main__":
    train_model()
