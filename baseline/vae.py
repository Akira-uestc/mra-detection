import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimSun"]
plt.rcParams["font.sans-serif"] = ["SimSun", "SimSun-ExtB", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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

USE_EWAF = True
EWAF_ALPHA = 0.15


def seed_everything(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(seq_len=60, stride=1):
    data_dir = PROJECT_ROOT / "data"
    train_pattern = "train_*.csv"
    test_pattern = "test_*.csv"

    print("Loading training data...")
    train_data, num_features = load_csv_dir_values(str(data_dir / "train"), train_pattern)
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, _ = load_csv_dir_values(str(data_dir / "test"), test_pattern)
    print(f"Test data: {test_data.shape}")

    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data).astype(np.float32)
    test_data_scaled = scaler.transform(test_data).astype(np.float32)

    x_train = build_front_padded_windows(
        train_data_scaled,
        seq_len=seq_len,
        stride=stride,
    )
    x_test, test_labels = build_prompt_test_windows_values(
        test_data_scaled,
        seq_len=seq_len,
        stride=stride,
    )

    return torch.FloatTensor(x_train), torch.FloatTensor(x_test), test_labels, num_features


class WindowVAE(nn.Module):
    def __init__(self, seq_len, num_features, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        input_dim = seq_len * num_features
        bottleneck_dim = hidden_dim // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(bottleneck_dim, latent_dim)
        self.logvar_layer = nn.Linear(bottleneck_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        flattened = x.reshape(batch_size, -1)
        encoded = self.encoder(flattened)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).reshape(batch_size, self.seq_len, self.num_features)
        return recon, mu, logvar


def compute_vae_loss(recon, target, mu, logvar, beta=1e-3):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_model():
    seed_everything(40)

    seq_len = 10
    stride = 1
    batch_size = 32
    epochs = 10
    beta = 1e-3
    output_path = Path(__file__).resolve().parent.parent / "outputs" / "vae_detection_results.png"

    x_train, x_test, test_labels, num_features = prepare_data(
        seq_len=seq_len,
        stride=stride,
    )
    train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WindowVAE(seq_len=seq_len, num_features=num_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"开始训练，共 {epochs} 个 Epoch...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss, recon_loss, kl_loss = compute_vae_loss(recon, inputs, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()

        if (epoch + 1) % 5 == 0:
            denom = len(train_loader)
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Loss: {running_loss / denom:.4f}, "
                f"Recon: {running_recon / denom:.4f}, "
                f"KL: {running_kl / denom:.4f}"
            )

    print("\n--- 异常检测评估结果 ---")
    model.eval()
    with torch.no_grad():
        x_train_dev = x_train.to(device)
        x_test_dev = x_test.to(device)

        recon_train, _, _ = model(x_train_dev)
        train_scores = (recon_train - x_train_dev).pow(2).mean(dim=[1, 2]).cpu().numpy()
        if USE_EWAF:
            train_scores = apply_ewaf_by_segments(train_scores, EWAF_ALPHA)

        train_mean = float(np.mean(train_scores))
        train_std = float(np.std(train_scores))
        threshold = choose_threshold(train_scores, method="gaussian_quantile_max")

        recon_test, _, _ = model(x_test_dev)
        test_scores = (recon_test - x_test_dev).pow(2).mean(dim=[1, 2]).cpu().numpy()
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
            output_path,
            show=True,
            ylabel="重构误差",
            color_scheme="mra",
        )


if __name__ == "__main__":
    train_model()
