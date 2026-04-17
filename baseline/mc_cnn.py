"""
MC-CNN baseline adapted from:

Song et al. (2025), "A Soft Sensor for Multirate Quality Variables Based on
MC-CNN".

Paper-faithful pieces kept here:
- multitemporal channels (M = 5)
- shared network: FC -> Conv1d -> Conv1d -> flatten -> latent feature
- parallel prediction heads, one head per quality variable
- masked loss so unsampled targets do not participate in backpropagation

Repo-specific adaptation:
- infer fully observed columns as process variables
- infer sparsely observed columns as multirate quality variables
- use train split as normal data and prediction error as anomaly score
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from _project_root import PROJECT_ROOT
from utils.methods.data_loading import load_csv_dir_with_mask
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
    build_front_padded_windows_with_mask,
    build_prompt_test_windows,
)


plt.rcParams["font.family"] = ["SimSun"]
plt.rcParams["font.sans-serif"] = ["SimSun", "SimSun-ExtB", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EPS = 1e-6
USE_EWAF = True
EWAF_ALPHA = 0.15


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_observed_standardizer(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(train_data, axis=0)
    stds = np.nanstd(train_data, axis=0)
    means[~np.isfinite(means)] = 0.0
    stds[~np.isfinite(stds) | (stds < EPS)] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def transform_observed(
    data: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    scaled = (data - means) / stds
    scaled[np.isnan(data)] = np.nan
    return scaled.astype(np.float32)




def infer_process_and_quality_columns(
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_counts = (1.0 - train_mask).sum(axis=0)
    total_rows = train_mask.shape[0]

    process_cols = np.flatnonzero(observed_counts == total_rows)
    quality_cols = np.flatnonzero(observed_counts < total_rows)

    if len(process_cols) == 0:
        raise ValueError("MC-CNN requires at least one fully observed process column.")
    if len(quality_cols) == 0:
        raise ValueError("MC-CNN requires at least one sparse quality column.")

    return process_cols.astype(int), quality_cols.astype(int)


def infer_interval(mask_column: np.ndarray) -> int:
    observed_idx = np.flatnonzero(mask_column == 0)
    if len(observed_idx) <= 1:
        return 1
    return int(np.gcd.reduce(np.diff(observed_idx)))


@dataclass
class PreparedData:
    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_loader: DataLoader
    test_labels: np.ndarray
    process_cols: np.ndarray
    quality_cols: np.ndarray
    num_process_features: int
    num_quality_features: int
    num_channels: int


def prepare_data(
    num_channels: int = 50,
    stride: int = 1,
    batch_size: int = 128,
) -> PreparedData:
    data_dir = PROJECT_ROOT / "data"
    train_pattern = "train_*.csv"
    test_pattern = "test_*.csv"

    print("Loading training data...")
    train_raw, train_mask, num_features = load_csv_dir_with_mask(
        data_dir / "train",
        train_pattern,
    )
    print(f"Training data: {train_raw.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_raw, test_mask, _ = load_csv_dir_with_mask(
        data_dir / "test",
        test_pattern,
    )
    print(f"Test data: {test_raw.shape}")

    means, stds = fit_observed_standardizer(train_raw)
    train_scaled = transform_observed(train_raw, means, stds)
    test_scaled = transform_observed(test_raw, means, stds)

    process_cols, quality_cols = infer_process_and_quality_columns(train_mask)

    print("\nInferred process columns:", process_cols.tolist())
    print("Inferred quality columns:", quality_cols.tolist())
    print("Quality sampling intervals:")
    for col_idx in quality_cols.tolist():
        interval = infer_interval(train_mask[:, col_idx])
        observed = int((1.0 - train_mask[:, col_idx]).sum())
        print(f"  column {col_idx}: interval={interval}, observed_samples={observed}")

    x_train = np.nan_to_num(train_scaled[:, process_cols], nan=0.0).astype(np.float32)
    x_test = np.nan_to_num(test_scaled[:, process_cols], nan=0.0).astype(np.float32)
    x_train_mask = train_mask[:, process_cols].astype(np.float32)
    x_test_mask = test_mask[:, process_cols].astype(np.float32)

    y_train = np.nan_to_num(train_scaled[:, quality_cols], nan=0.0).astype(np.float32)
    y_test = np.nan_to_num(test_scaled[:, quality_cols], nan=0.0).astype(np.float32)
    y_train_mask = train_mask[:, quality_cols].astype(np.float32)
    y_test_mask = test_mask[:, quality_cols].astype(np.float32)

    dx_train, _ = build_front_padded_windows_with_mask(
        x_train,
        x_train_mask,
        seq_len=num_channels,
        stride=stride,
    )
    dx_test, _, test_labels = build_prompt_test_windows(
        x_test,
        x_test_mask,
        seq_len=num_channels,
        stride=stride,
    )
    dy_train, dy_train_mask = build_front_padded_windows_with_mask(
        y_train,
        y_train_mask,
        seq_len=num_channels,
        stride=stride,
    )
    dy_test, dy_test_mask, _ = build_prompt_test_windows(
        y_test,
        y_test_mask,
        seq_len=num_channels,
        stride=stride,
    )

    y_train_latest = dy_train[:, -1, :]
    y_test_latest = dy_test[:, -1, :]
    y_train_latest_mask = dy_train_mask[:, -1, :]
    y_test_latest_mask = dy_test_mask[:, -1, :]

    train_dataset = TensorDataset(
        torch.tensor(dx_train),
        torch.tensor(dy_train),
        torch.tensor(dy_train_mask),
        torch.tensor(y_train_latest),
        torch.tensor(y_train_latest_mask),
    )
    test_dataset = TensorDataset(
        torch.tensor(dx_test),
        torch.tensor(dy_test),
        torch.tensor(dy_test_mask),
        torch.tensor(y_test_latest),
        torch.tensor(y_test_latest_mask),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return PreparedData(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        test_labels=test_labels,
        process_cols=process_cols,
        quality_cols=quality_cols,
        num_process_features=len(process_cols),
        num_quality_features=len(quality_cols),
        num_channels=num_channels,
    )


class SharedNetwork(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_process_features: int,
        num_quality_features: int,
        fc_hidden_dim: int = 40,
        latent_dim: int = 25,
    ) -> None:
        super().__init__()

        conv_out_length = fc_hidden_dim - 4
        if conv_out_length <= 0:
            raise ValueError(
                "fc_hidden_dim must be at least 5 for the two Conv1d layers."
            )

        self.num_channels = num_channels
        self.num_quality_features = num_quality_features
        self.fc_hidden = nn.Linear(num_process_features, fc_hidden_dim)
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=3, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3)
        self.feature_projection = nn.Linear(conv_out_length, latent_dim)
        self.window_head = nn.Linear(latent_dim, num_channels * num_quality_features)
        self.activation = nn.SELU()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc_hidden(x))
        hidden = self.activation(self.conv1(hidden))
        hidden = self.activation(self.conv2(hidden))
        hidden = hidden.flatten(start_dim=1)
        return self.activation(self.feature_projection(hidden))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        window_pred = self.window_head(features)
        window_pred = window_pred.view(-1, self.num_channels, self.num_quality_features)
        return window_pred, features


class QualityHead(nn.Module):
    def __init__(self, latent_dim: int = 25, hidden_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MCCNN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_process_features: int,
        num_quality_features: int,
        fc_hidden_dim: int = 40,
        latent_dim: int = 25,
        head_hidden_dim: int = 5,
    ) -> None:
        super().__init__()
        self.shared = SharedNetwork(
            num_channels=num_channels,
            num_process_features=num_process_features,
            num_quality_features=num_quality_features,
            fc_hidden_dim=fc_hidden_dim,
            latent_dim=latent_dim,
        )
        self.heads = nn.ModuleList(
            [
                QualityHead(latent_dim=latent_dim, hidden_dim=head_hidden_dim)
                for _ in range(num_quality_features)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_pred, features = self.shared(x)
        latest_pred = torch.cat([head(features) for head in self.heads], dim=1)
        return window_pred, latest_pred, features


def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor
) -> torch.Tensor:
    observed = 1.0 - missing_mask
    squared_error = (pred - target).pow(2) * observed
    denom = observed.sum().clamp_min(1.0)
    return squared_error.sum() / denom


def masked_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = 1.0 - missing_mask
    squared_error = (pred - target).pow(2) * observed
    error_sum = squared_error.sum(dim=dims)
    observed_count = observed.sum(dim=dims)
    scores = error_sum / observed_count.clamp_min(1.0)
    return scores, observed_count


def train_shared_network(
    model: MCCNN,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    learning_rate: float = 1e-3,
) -> None:
    optimizer = optim.Adam(model.shared.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.shared.state_dict())

    print(f"\nStage 1: pretraining shared network for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for dx, dy, dy_mask, _, _ in train_loader:
            dx = dx.to(device)
            dy = dy.to(device)
            dy_mask = dy_mask.to(device)

            optimizer.zero_grad()
            window_pred, _ = model.shared(dx)
            loss = masked_mse_loss(window_pred, dy, dy_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.shared.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1:03d}/{epochs}] shared_loss={avg_loss:.6f}")

    model.shared.load_state_dict(best_state)


def train_full_model(
    model: MCCNN,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    learning_rate: float = 5e-4,
    shared_aux_weight: float = 0.2,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    print(f"\nStage 2: fine-tuning full MC-CNN for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for dx, dy, dy_mask, y_latest, y_latest_mask in train_loader:
            dx = dx.to(device)
            dy = dy.to(device)
            dy_mask = dy_mask.to(device)
            y_latest = y_latest.to(device)
            y_latest_mask = y_latest_mask.to(device)

            optimizer.zero_grad()
            window_pred, latest_pred, _ = model(dx)
            shared_loss = masked_mse_loss(window_pred, dy, dy_mask)
            latest_loss = masked_mse_loss(latest_pred, y_latest, y_latest_mask)
            loss = latest_loss + shared_aux_weight * shared_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1:03d}/{epochs}] total_loss={avg_loss:.6f}")

    model.load_state_dict(best_state)


def score_dataset(
    model: MCCNN, data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    scores = []
    model.eval()

    with torch.no_grad():
        for dx, dy, dy_mask, y_latest, y_latest_mask in data_loader:
            dx = dx.to(device)
            dy = dy.to(device)
            dy_mask = dy_mask.to(device)
            y_latest = y_latest.to(device)
            y_latest_mask = y_latest_mask.to(device)

            window_pred, latest_pred, _ = model(dx)
            window_score, _ = masked_mse_per_sample(
                window_pred, dy, dy_mask, dims=(1, 2)
            )
            latest_score, latest_count = masked_mse_per_sample(
                latest_pred,
                y_latest,
                y_latest_mask,
                dims=(1,),
            )

            batch_score = window_score + torch.where(
                latest_count > 0,
                latest_score,
                torch.zeros_like(latest_score),
            )
            scores.extend(batch_score.cpu().numpy())

    return np.asarray(scores, dtype=np.float32)

def train_model() -> None:
    seed_everything(42)

    num_channels = 5
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare_data(
        num_channels=num_channels,
        stride=1,
        batch_size=batch_size,
    )

    model = MCCNN(
        num_channels=data.num_channels,
        num_process_features=data.num_process_features,
        num_quality_features=data.num_quality_features,
        fc_hidden_dim=40,
        latent_dim=25,
        head_hidden_dim=5,
    ).to(device)

    print("\nModel configuration:")
    print(f"  Device: {device}")
    print(f"  Multitemporal channels (M): {data.num_channels}")
    print(f"  Process features: {data.num_process_features}")
    print(f"  Quality features: {data.num_quality_features}")

    train_shared_network(
        model=model,
        train_loader=data.train_loader,
        device=device,
        epochs=30,
        learning_rate=1e-3,
    )
    train_full_model(
        model=model,
        train_loader=data.train_loader,
        device=device,
        epochs=25,
        learning_rate=5e-4,
        shared_aux_weight=0.2,
    )

    train_scores = score_dataset(model, data.train_eval_loader, device)
    test_scores = score_dataset(model, data.test_loader, device)
    if USE_EWAF:
        train_scores = apply_ewaf_by_segments(train_scores, EWAF_ALPHA)
    # threshold = choose_threshold(train_scores, method="mean")
    threshold = choose_threshold(train_scores, method="gaussian_quantile_max")

    y_true = data.test_labels
    split_idx = split_index_from_labels(y_true)
    if USE_EWAF:
        test_scores = apply_ewaf_by_segments(
            test_scores,
            EWAF_ALPHA,
            infer_segment_lengths(y_true),
        )
    y_pred = (test_scores > threshold).astype(int)
    metrics = compute_binary_classification_metrics(
        y_true,
        y_pred,
        threshold=threshold,
    )

    print("\n--- Anomaly detection evaluation ---")
    print(
        f"Train score: mean={np.mean(train_scores):.6f}, std={np.std(train_scores):.6f}"
    )
    print(f"Threshold (mean train score): {threshold:.6f}")
    print(
        f"Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores)}) anomaly"
    )
    print(f"Anomalies detected: {(y_pred == 1).sum()} / {len(y_pred)}")

    print_metrics(
        "\nClassification Metrics:",
        metrics,
        order=["threshold", "accuracy", "precision", "recall", "fdr", "fra", "f1"],
    )

    plot_detection_scores(
        test_scores,
        threshold,
        split_idx,
        PROJECT_ROOT / "outputs" / "mc_cnn_detection.png",
        ylabel="异常分数",
        color_scheme="mra",
    )


if __name__ == "__main__":
    train_model()
