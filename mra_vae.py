#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from graph_gcn_reconstruction import (
    ObservedStandardScaler,
    configure_chinese_font,
    seed_everything,
)
from utils.methods.data_loading import load_csv_glob_with_mask
from utils.methods.display import (
    compute_binary_classification_metrics,
    plot_detection_scores,
)
from utils.methods.postprocess import (
    apply_ewaf_by_segments,
    choose_threshold,
    infer_segment_lengths,
    split_index_from_labels,
)
from utils.methods.windowing import (
    build_prompt_test_windows,
    build_standard_windows,
    build_windows,
)


@dataclass
class LossWeights:
    imputation: float = 1.0
    vae_reconstruction: float = 0.4
    kl: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAE 插补，并将补全结果送入 Transformer 做无监督异常检测。"
    )
    parser.add_argument("--train-glob", default="data/train/train_1.csv")
    parser.add_argument("--test-glob", default="data/test/test_C5_1.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/vae_imputer_transformer_detection",
    )
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.15,
        help="训练时从观测点额外随机遮挡的比例。",
    )
    parser.add_argument("--vae-hidden-dim", type=int, default=256)
    parser.add_argument("--vae-latent-dim", type=int, default=64)
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=1e-3,
        help="VAE KL 项权重。",
    )
    parser.add_argument("--detector-d-model", type=int, default=128)
    parser.add_argument("--detector-heads", type=int, default=4)
    parser.add_argument("--detector-layers", type=int, default=3)
    parser.add_argument(
        "--detector-epochs",
        type=int,
        default=None,
        help="检测器单独训练轮数。默认与 --epochs 相同；设为 0 则跳过检测器训练。",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--score-vae-weight",
        type=float,
        default=0.25,
        help="将 VAE 观测点重构误差并入最终异常分数时的权重。",
    )
    parser.add_argument(
        "--score-shift-weight",
        type=float,
        default=0,
        help="将补全窗口的分布偏移分数并入最终异常分数时的权重。",
    )
    parser.add_argument(
        "--threshold-std-factor",
        type=float,
        default=2.0,
        help="阈值 = max(mean + k * std, quantile)。",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="基于训练分数的分位数阈值。",
    )
    parser.add_argument(
        "--ewaf-alpha",
        type=float,
        default=0.3,
        help="异常分数 EWAF 平滑系数，取值范围 (0, 1]。",
    )
    parser.add_argument(
        "--min-anomaly-duration",
        type=int,
        default=50,
        help="最短异常持续长度，单位为窗口点数；小于该长度的连续异常片段会被抑制。",
    )
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_csv(data: np.ndarray, feature_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=feature_names).to_csv(output_path, index=False)


def infer_rate_metadata(missing_mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    missing_ratio = missing_mask.mean(axis=0)
    observed_ratio = np.clip(1.0 - missing_ratio, 1e-6, 1.0)
    inferred_stride = np.rint(1.0 / observed_ratio).astype(np.int64)
    unique_stride = sorted(np.unique(inferred_stride).tolist())
    rate_lookup = {stride_value: idx for idx, stride_value in enumerate(unique_stride)}
    rate_id = np.asarray(
        [rate_lookup[item] for item in inferred_stride], dtype=np.int64
    )
    return torch.tensor(rate_id, dtype=torch.long), torch.tensor(
        inferred_stride, dtype=torch.long
    )


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = ((prediction - target) * target_mask).pow(2)
    return squared_error.sum() / target_mask.sum().clamp_min(1.0)


def sample_holdout_mask(missing_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    observed = ~missing_mask.bool()
    holdout = ((torch.rand_like(missing_mask) < ratio) & observed).view(
        missing_mask.size(0), -1
    )
    observed_flat = observed.view(missing_mask.size(0), -1)

    for batch_idx in range(holdout.size(0)):
        if observed_flat[batch_idx].sum() == 0:
            continue
        if holdout[batch_idx].any():
            continue
        candidate_idx = torch.nonzero(observed_flat[batch_idx], as_tuple=False).squeeze(
            -1
        )
        picked = candidate_idx[
            torch.randint(0, len(candidate_idx), (1,), device=missing_mask.device)
        ]
        holdout[batch_idx, picked] = True

    return holdout.view_as(missing_mask).float()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.pe[:length].to(device=device, dtype=dtype)


class MaskAwareWindowVAEImputer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        input_dim = seq_len * num_nodes * 2
        output_dim = seq_len * num_nodes
        bottleneck_dim = max(hidden_dim // 2, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.mu_layer = nn.Linear(bottleneck_dim, latent_dim)
        self.logvar_layer = nn.Linear(bottleneck_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(
            torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

    def forward(
        self,
        x_input: torch.Tensor,
        model_missing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        observed_mask = 1.0 - model_missing_mask
        x_observed = torch.where(
            observed_mask.bool(), x_input, torch.zeros_like(x_input)
        )

        batch_size, time_steps, num_nodes = x_input.shape
        if time_steps != self.seq_len or num_nodes != self.num_nodes:
            raise ValueError(
                f"期望窗口形状 (*, {self.seq_len}, {self.num_nodes})，"
                f"收到 {tuple(x_input.shape)}"
            )

        encoder_input = torch.cat([x_observed, observed_mask], dim=-1).reshape(
            batch_size, -1
        )
        encoded = self.encoder(encoder_input)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded).clamp(min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        x_imputed = self.decoder(z).reshape(batch_size, self.seq_len, self.num_nodes)
        x_complete = torch.where(model_missing_mask.bool(), x_imputed, x_input)

        return {
            "x_observed": x_observed,
            "x_imputed": x_imputed,
            "x_complete": x_complete,
            "mu": mu,
            "logvar": logvar,
            "kl_loss": self.kl_divergence(mu, logvar),
        }


class SamplingAwareTransformerAD(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_rates: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        per_variable_dim: int = 16,
        phase_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.variable_embedding = nn.Embedding(num_nodes, 8)
        self.rate_embedding = nn.Embedding(num_rates, 8)
        self.phase_mlp = nn.Sequential(
            nn.Linear(1, phase_dim),
            nn.GELU(),
            nn.Linear(phase_dim, phase_dim),
        )
        self.per_variable_proj = nn.Sequential(
            nn.Linear(1 + 1 + 8 + 8 + phase_dim, per_variable_dim),
            nn.GELU(),
            nn.Linear(per_variable_dim, per_variable_dim),
        )
        token_input_dim = num_nodes * 2 + num_nodes * per_variable_dim
        self.token_proj = nn.Linear(token_input_dim, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_nodes),
        )

    def forward(
        self,
        x_complete: torch.Tensor,
        observed_mask: torch.Tensor,
        rate_id: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, num_nodes = x_complete.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"期望 {self.num_nodes} 个变量，收到 {num_nodes}")

        variable_embed = self.variable_embedding(
            torch.arange(num_nodes, device=x_complete.device)
        )
        rate_embed = self.rate_embedding(rate_id)

        time_index = torch.arange(
            time_steps, device=x_complete.device, dtype=x_complete.dtype
        ).view(time_steps, 1)
        stride_value = (
            stride.to(device=x_complete.device, dtype=x_complete.dtype)
            .view(1, num_nodes)
            .clamp_min(1.0)
        )
        phase = torch.remainder(time_index, stride_value) / stride_value
        phase = phase.unsqueeze(-1)
        phase_embed = self.phase_mlp(phase)

        variable_embed = (
            variable_embed.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, time_steps, -1, -1)
        )
        rate_embed = (
            rate_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1, -1)
        )
        phase_embed = phase_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)

        per_variable_input = torch.cat(
            [
                x_complete.unsqueeze(-1),
                observed_mask.unsqueeze(-1),
                variable_embed,
                rate_embed,
                phase_embed,
            ],
            dim=-1,
        )
        rate_aware_features = self.per_variable_proj(per_variable_input)

        token_input = torch.cat(
            [
                x_complete,
                observed_mask,
                rate_aware_features.reshape(batch_size, time_steps, -1),
            ],
            dim=-1,
        )
        tokens = self.token_proj(token_input)
        tokens = tokens + self.position_encoding(
            time_steps, x_complete.device, x_complete.dtype
        ).unsqueeze(0)

        encoded = self.encoder(tokens)
        delta = self.reconstruction_head(encoded)
        return x_complete + delta


def compute_imputer_losses(
    outputs: dict[str, torch.Tensor],
    x_true: torch.Tensor,
    structural_missing_mask: torch.Tensor,
    holdout_mask: torch.Tensor,
    loss_weights: LossWeights,
    vae_beta: float,
) -> dict[str, torch.Tensor]:
    observed_mask = 1.0 - structural_missing_mask
    imputation_loss = masked_mse(outputs["x_imputed"], x_true, holdout_mask)
    vae_reconstruction_loss = masked_mse(outputs["x_imputed"], x_true, observed_mask)
    kl_loss = outputs["kl_loss"]

    total_loss = (
        loss_weights.imputation * imputation_loss
        + loss_weights.vae_reconstruction * vae_reconstruction_loss
        + loss_weights.kl * vae_beta * kl_loss
    )
    return {
        "imputation_loss": imputation_loss,
        "vae_reconstruction_loss": vae_reconstruction_loss,
        "kl_loss": kl_loss,
        "total_loss": total_loss,
    }


def train_imputer(
    imputer: MaskAwareWindowVAEImputer,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    holdout_ratio: float,
    vae_beta: float,
    loss_weights: LossWeights,
) -> list[dict[str, float]]:
    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(imputer.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        imputer.train()
        meter = {
            "imputation_loss": 0.0,
            "vae_reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "total_loss": 0.0,
        }

        for x_true, structural_missing_mask in loader:
            x_true = x_true.to(device)
            structural_missing_mask = structural_missing_mask.to(device)
            holdout_mask = sample_holdout_mask(structural_missing_mask, holdout_ratio)
            model_missing_mask = torch.maximum(structural_missing_mask, holdout_mask)
            x_input = x_true.masked_fill(model_missing_mask.bool(), 0.0)

            outputs = imputer(x_input, model_missing_mask)
            losses = compute_imputer_losses(
                outputs=outputs,
                x_true=x_true,
                structural_missing_mask=structural_missing_mask,
                holdout_mask=holdout_mask,
                loss_weights=loss_weights,
                vae_beta=vae_beta,
            )

            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(imputer.parameters(), max_norm=1.0)
            optimizer.step()

            for key in meter:
                meter[key] += float(losses[key].detach().cpu())

        epoch_stats = {key: value / max(len(loader), 1) for key, value in meter.items()}
        history.append(epoch_stats)
        print(
            f"Imputer Epoch {epoch + 1:02d}/{epochs} "
            f"total={epoch_stats['total_loss']:.5f} "
            f"impute={epoch_stats['imputation_loss']:.5f} "
            f"vae_rec={epoch_stats['vae_reconstruction_loss']:.5f} "
            f"kl={epoch_stats['kl_loss']:.5f}"
        )

    return history


def train_detector(
    detector: SamplingAwareTransformerAD,
    imputer: MaskAwareWindowVAEImputer,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    rate_id: torch.Tensor,
    stride: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    if epochs <= 0:
        return []

    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(detector.parameters(), lr=lr, weight_decay=weight_decay)
    rate_id = rate_id.to(device)
    stride = stride.to(device)
    history: list[dict[str, float]] = []
    imputer.eval()

    for epoch in range(epochs):
        detector.train()
        total = 0.0

        for x_true, structural_missing_mask in loader:
            x_true = x_true.to(device)
            structural_missing_mask = structural_missing_mask.to(device)
            observed_mask = 1.0 - structural_missing_mask
            x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)

            with torch.no_grad():
                imputer_outputs = imputer(x_input, structural_missing_mask)
                x_complete = imputer_outputs["x_complete"].detach()

            reconstruction = detector(x_complete, observed_mask, rate_id, stride)
            loss = masked_mse(reconstruction, x_complete, observed_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
            optimizer.step()

            total += float(loss.detach().cpu())

        epoch_stats = {"total_loss": total / max(len(loader), 1)}
        history.append(epoch_stats)
        print(
            f"Detector Epoch {epoch + 1:02d}/{epochs} "
            f"loss={epoch_stats['total_loss']:.5f}"
        )

    return history


@torch.no_grad()
def reconstruct_full_sequence(
    imputer: MaskAwareWindowVAEImputer,
    windows: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )

    completed = []
    imputer.eval()

    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = imputer(x_input, structural_missing_mask)
        completed.append(outputs["x_complete"][:, -1, :].cpu().numpy())

    return np.concatenate(completed, axis=0).astype(np.float32)


@torch.no_grad()
def collect_window_statistics(
    imputer: MaskAwareWindowVAEImputer,
    detector: SamplingAwareTransformerAD | None,
    windows: np.ndarray,
    masks: np.ndarray,
    rate_id: torch.Tensor,
    stride: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )
    rate_id = rate_id.to(device)
    stride = stride.to(device)

    collected = {
        "detector_score": [],
        "vae_reconstruction_score": [],
        "vae_kl_score": [],
    }

    imputer.eval()
    if detector is not None:
        detector.eval()
    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        observed_mask = 1.0 - structural_missing_mask
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = imputer(x_input, structural_missing_mask)
        if detector is None:
            detector_score = torch.zeros(x_true.size(0), device=device)
        else:
            detector_reconstruction = detector(
                outputs["x_complete"], observed_mask, rate_id, stride
            )
            detector_sq = (
                (detector_reconstruction - outputs["x_complete"]) * observed_mask
            ).pow(2)
            detector_score = detector_sq.sum(dim=(1, 2)) / observed_mask.sum(
                dim=(1, 2)
            ).clamp_min(1.0)
        vae_sq = ((outputs["x_imputed"] - x_true) * observed_mask).pow(2)
        vae_reconstruction_score = vae_sq.sum(dim=(1, 2)) / observed_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        vae_kl_score = -0.5 * torch.sum(
            1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp(),
            dim=1,
        )

        collected["detector_score"].append(
            detector_score.cpu().numpy().astype(np.float32)
        )
        collected["vae_reconstruction_score"].append(
            vae_reconstruction_score.cpu().numpy().astype(np.float32)
        )
        collected["vae_kl_score"].append(vae_kl_score.cpu().numpy().astype(np.float32))

    return {
        key: np.concatenate(value, axis=0).astype(np.float32)
        for key, value in collected.items()
    }


def normalize_scores(
    train_values: np.ndarray, eval_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    std = max(std, 1e-6)
    return (
        ((train_values - mean) / std).astype(np.float32),
        ((eval_values - mean) / std).astype(np.float32),
    )


def combine_scores(
    train_stats: dict[str, np.ndarray],
    eval_stats: dict[str, np.ndarray],
    vae_weight: float,
    shift_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_det, eval_det = normalize_scores(
        train_stats["detector_score"], eval_stats["detector_score"]
    )
    train_vae, eval_vae = normalize_scores(
        train_stats["vae_reconstruction_score"],
        eval_stats["vae_reconstruction_score"],
    )
    train_shift, eval_shift = normalize_scores(
        train_stats["shift_score"], eval_stats["shift_score"]
    )
    train_score = (
        np.abs(train_det) + vae_weight * train_vae + shift_weight * train_shift
    )
    eval_score = np.abs(eval_det) + vae_weight * eval_vae + shift_weight * eval_shift
    return train_score.astype(np.float32), eval_score.astype(np.float32)


def apply_min_anomaly_duration(
    prediction: np.ndarray,
    min_duration: int,
) -> np.ndarray:
    if min_duration <= 1:
        return prediction.astype(np.int64)

    filtered = np.zeros_like(prediction, dtype=np.int64)
    run_start: int | None = None

    for idx in range(len(prediction) + 1):
        current = int(prediction[idx]) if idx < len(prediction) else 0
        if current == 1 and run_start is None:
            run_start = idx
            continue
        if current == 0 and run_start is not None:
            if idx - run_start >= min_duration:
                filtered[run_start:idx] = 1
            run_start = None

    return filtered


def save_loss_curve(
    history: list[dict[str, float]],
    output_path: Path,
    curves: list[tuple[str, str, str]],
    title: str,
) -> None:
    if not history:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, color in curves:
        if key not in history[0]:
            continue
        ax.plot(epochs, [item[key] for item in history], label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_window_feature_matrix(windows: np.ndarray) -> np.ndarray:
    features = []
    for column_slice in (slice(0, 6), slice(6, 12), slice(12, 18), slice(0, 18)):
        block = windows[:, :, column_slice]
        features.append(block.mean(axis=1))
        features.append(block.std(axis=1))

    features.append(np.abs(np.diff(windows, axis=1)).mean(axis=1))
    fft_feature = (
        np.abs(np.fft.rfft(windows, axis=1))[:, 1:6, :].mean(axis=1).astype(np.float32)
    )
    features.append(fft_feature)
    return np.concatenate(features, axis=1).astype(np.float32)


def compute_distribution_shift_scores(
    train_windows: np.ndarray,
    eval_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = build_window_feature_matrix(train_windows)
    eval_features = build_window_feature_matrix(eval_windows)
    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1e-6)

    train_z = (train_features - feature_mean) / feature_std
    eval_z = (eval_features - feature_mean) / feature_std
    train_score = np.sqrt((train_z**2).mean(axis=1))
    eval_score = np.sqrt((eval_z**2).mean(axis=1))
    return train_score.astype(np.float32), eval_score.astype(np.float32)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ewaf_alpha <= 1.0:
        raise ValueError(f"--ewaf-alpha 必须在 (0, 1] 内，收到 {args.ewaf_alpha}")
    if args.min_anomaly_duration < 1:
        raise ValueError(
            f"--min-anomaly-duration 必须 >= 1，收到 {args.min_anomaly_duration}"
        )
    if args.vae_hidden_dim < 1 or args.vae_latent_dim < 1:
        raise ValueError("--vae-hidden-dim 和 --vae-latent-dim 必须为正整数")
    if args.vae_beta < 0.0:
        raise ValueError(f"--vae-beta 必须 >= 0，收到 {args.vae_beta}")
    detector_epochs = (
        args.epochs if args.detector_epochs is None else args.detector_epochs
    )
    if detector_epochs < 0:
        raise ValueError(f"--detector-epochs 必须 >= 0，收到 {detector_epochs}")
    seed_everything(args.seed)
    configure_chinese_font()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    loss_weights = LossWeights()

    train_raw, train_mask, feature_names = load_csv_glob_with_mask(args.train_glob)
    test_raw, test_mask, _ = load_csv_glob_with_mask(args.test_glob)

    scaler = ObservedStandardScaler().fit(train_raw, train_mask)
    train_scaled = scaler.transform(train_raw, train_mask)
    test_scaled = scaler.transform(test_raw, test_mask)

    train_impute_windows, train_impute_masks = build_windows(
        train_scaled,
        train_mask.astype(np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )
    test_impute_windows, test_impute_masks = build_windows(
        test_scaled,
        test_mask.astype(np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )
    train_eval_windows, train_eval_masks = build_standard_windows(
        train_scaled,
        train_mask.astype(np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )
    test_eval_windows, test_eval_masks, test_labels = build_prompt_test_windows(
        test_scaled,
        test_mask.astype(np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )

    rate_id, stride = infer_rate_metadata(train_mask.astype(np.float32))
    print(f"使用设备: {device}")
    print(f"训练插补窗口: {train_impute_windows.shape}")
    print(f"训练评分窗口: {train_eval_windows.shape}")
    print(f"测试评分窗口: {test_eval_windows.shape}")
    print(f"采样率分组 rate_id: {rate_id.tolist()}")
    print(f"推断步长 stride: {stride.tolist()}")

    imputer = MaskAwareWindowVAEImputer(
        num_nodes=train_scaled.shape[1],
        seq_len=args.seq_len,
        hidden_dim=args.vae_hidden_dim,
        latent_dim=args.vae_latent_dim,
        dropout=args.dropout,
    ).to(device)
    detector = SamplingAwareTransformerAD(
        num_nodes=train_scaled.shape[1],
        num_rates=int(rate_id.max().item()) + 1,
        d_model=args.detector_d_model,
        num_heads=args.detector_heads,
        num_layers=args.detector_layers,
        dropout=args.dropout,
        per_variable_dim=max(args.detector_d_model // 8, 8),
    ).to(device)

    imputer_history = train_imputer(
        imputer=imputer,
        train_windows=train_impute_windows,
        train_masks=train_impute_masks,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        holdout_ratio=args.holdout_ratio,
        vae_beta=args.vae_beta,
        loss_weights=loss_weights,
    )
    detector_history = train_detector(
        detector=detector,
        imputer=imputer,
        train_windows=train_eval_windows,
        train_masks=train_eval_masks,
        rate_id=rate_id,
        stride=stride,
        device=device,
        epochs=detector_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    torch.save(
        {
            "imputer_state_dict": imputer.state_dict(),
            "detector_state_dict": detector.state_dict(),
            "args": vars(args),
            "detector_epochs": detector_epochs,
            "rate_id": rate_id,
            "stride": stride,
            "loss_weights": asdict(loss_weights),
        },
        output_dir / "model.pt",
    )

    train_complete_scaled = reconstruct_full_sequence(
        imputer=imputer,
        windows=train_impute_windows,
        masks=train_impute_masks,
        device=device,
        batch_size=args.batch_size,
    )
    test_complete_scaled = reconstruct_full_sequence(
        imputer=imputer,
        windows=test_impute_windows,
        masks=test_impute_masks,
        device=device,
        batch_size=args.batch_size,
    )
    train_complete = scaler.inverse_transform(train_complete_scaled)
    test_complete = scaler.inverse_transform(test_complete_scaled)

    save_csv(train_complete, feature_names, output_dir / "train_completed.csv")
    save_csv(test_complete, feature_names, output_dir / "test_completed.csv")

    trained_detector = detector if detector_history else None
    train_stats = collect_window_statistics(
        imputer=imputer,
        detector=trained_detector,
        windows=train_eval_windows,
        masks=train_eval_masks,
        rate_id=rate_id,
        stride=stride,
        device=device,
        batch_size=args.batch_size,
    )
    test_stats = collect_window_statistics(
        imputer=imputer,
        detector=trained_detector,
        windows=test_eval_windows,
        masks=test_eval_masks,
        rate_id=rate_id,
        stride=stride,
        device=device,
        batch_size=args.batch_size,
    )

    train_complete_eval_windows, _ = build_standard_windows(
        train_complete_scaled,
        np.zeros_like(train_mask, dtype=np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )
    test_complete_eval_windows, _, _ = build_prompt_test_windows(
        test_complete_scaled,
        np.zeros_like(test_mask, dtype=np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )
    train_shift_scores, test_shift_scores = compute_distribution_shift_scores(
        train_windows=train_complete_eval_windows,
        eval_windows=test_complete_eval_windows,
    )
    train_stats["shift_score"] = train_shift_scores
    test_stats["shift_score"] = test_shift_scores

    train_raw_scores, test_raw_scores = combine_scores(
        train_stats=train_stats,
        eval_stats=test_stats,
        vae_weight=args.score_vae_weight,
        shift_weight=args.score_shift_weight,
    )
    test_segment_lengths = infer_segment_lengths(test_labels)
    test_split_idx = split_index_from_labels(test_labels)
    train_scores = apply_ewaf_by_segments(train_raw_scores, args.ewaf_alpha)
    test_scores = apply_ewaf_by_segments(
        test_raw_scores,
        args.ewaf_alpha,
        segment_lengths=test_segment_lengths,
    )
    threshold = choose_threshold(
        train_scores=train_scores,
        # method="gaussian_quantile_max",
        method="mean",
        std_factor=args.threshold_std_factor,
        quantile=args.threshold_quantile,
    )
    threshold_prediction = (test_scores >= threshold).astype(np.int64)
    final_prediction = apply_min_anomaly_duration(
        threshold_prediction,
        args.min_anomaly_duration,
    )
    metrics = compute_binary_classification_metrics(
        test_labels,
        final_prediction,
        threshold=threshold,
    )

    prediction_df = pd.DataFrame(
        {
            "sample_index": np.arange(1, len(test_scores) + 1),
            "label": test_labels,
            "detector_score": test_stats["detector_score"],
            "vae_reconstruction_score": test_stats["vae_reconstruction_score"],
            "vae_kl_score": test_stats["vae_kl_score"],
            "shift_score": test_stats["shift_score"],
            "raw_final_score": test_raw_scores,
            "final_score": test_scores,
            "threshold_prediction": threshold_prediction,
            "prediction": final_prediction,
        }
    )
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "device": str(device),
        "config": vars(args),
        "loss_weights": asdict(loss_weights),
        "metrics": metrics,
        "ewaf_alpha": args.ewaf_alpha,
        "min_anomaly_duration": args.min_anomaly_duration,
        "detector_epochs": detector_epochs,
        "train_raw_score_mean": float(np.mean(train_raw_scores)),
        "train_raw_score_std": float(np.std(train_raw_scores)),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "train_threshold_quantile": float(
            np.quantile(train_scores, args.threshold_quantile)
        ),
        "rate_id": rate_id.tolist(),
        "stride": stride.tolist(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    configure_chinese_font()
    save_loss_curve(
        imputer_history,
        output_dir / "training_curve.png",
        curves=[
            ("total_loss", "总损失", "#1D3557"),
            ("imputation_loss", "插补损失", "#457B9D"),
            ("vae_reconstruction_loss", "VAE重构损失", "#2A9D8F"),
            ("kl_loss", "KL损失", "#E76F51"),
        ],
        title="VAE 插补器训练曲线",
    )
    save_loss_curve(
        detector_history,
        output_dir / "detector_training_curve.png",
        curves=[("total_loss", "检测器重构损失", "#E63946")],
        title="检测器训练曲线",
    )
    plot_detection_scores(
        scores=test_scores,
        threshold=threshold,
        split_idx=test_split_idx,
        save_path=output_dir / "anomaly_scores.png",
        color_scheme="mra",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("输出文件:")
    print(f"  模型: {output_dir / 'model.pt'}")
    print(f"  Train 补全: {output_dir / 'train_completed.csv'}")
    print(f"  Test 补全 : {output_dir / 'test_completed.csv'}")
    print(f"  预测结果 : {output_dir / 'test_predictions.csv'}")
    print(f"  指标汇总 : {output_dir / 'metrics.json'}")
    print(f"  插补训练曲线: {output_dir / 'training_curve.png'}")
    if detector_history:
        print(f"  检测器训练曲线: {output_dir / 'detector_training_curve.png'}")
    print(f"  曲线图   : {output_dir / 'anomaly_scores.png'}")


if __name__ == "__main__":
    main()
