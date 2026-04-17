#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from freq_reconstruction import FrequencyOnlyReconstructor
from graph_gcn_reconstruction import (
    GraphGCNReconstructor,
    ObservedStandardScaler,
    configure_chinese_font,
    save_adjacency_heatmap,
    seed_everything,
    summarize_adjacency,
)
from utils.methods.data_loading import load_csv_glob_with_mask
from utils.methods.display import (
    compute_binary_classification_metrics,
    plot_detection_scores,
)
from utils.methods.postprocess import choose_threshold, split_index_from_labels
from utils.methods.windowing import (
    build_prompt_test_windows,
    build_standard_windows,
    build_windows,
)


def build_parser(
    *,
    description: str,
    default_output_dir: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-glob", default="data/train/train_1.csv")
    parser.add_argument("--test-glob", default="data/test/test_C5_1.csv")
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--graph-hidden-dim", type=int, default=32)
    parser.add_argument("--gcn-hidden-dim", type=int, default=32)
    parser.add_argument("--detector-d-model", type=int, default=128)
    parser.add_argument("--detector-heads", type=int, default=4)
    parser.add_argument("--detector-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold-std-factor", type=float, default=2.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default=None)
    return parser


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_csv(data: np.ndarray, feature_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=feature_names).to_csv(output_path, index=False)


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
        missing_mask.size(0),
        -1,
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
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.pe[:length].to(device=device, dtype=dtype)


class SimpleTransformerAD(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_proj = nn.Linear(num_nodes * 2, d_model)
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
    ) -> torch.Tensor:
        batch_size, time_steps, num_nodes = x_complete.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"期望 {self.num_nodes} 个变量，收到 {num_nodes}")

        token_input = torch.cat([x_complete, observed_mask], dim=-1)
        tokens = self.input_proj(token_input)
        tokens = tokens + self.position_encoding(
            time_steps,
            x_complete.device,
            x_complete.dtype,
        ).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.reconstruction_head(encoded)


class SingleExpertTransformerModel(nn.Module):
    def __init__(
        self,
        *,
        expert_type: str,
        num_nodes: int,
        seq_len: int,
        graph_hidden_dim: int,
        gcn_hidden_dim: int,
        detector_d_model: int,
        detector_heads: int,
        detector_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.expert_type = expert_type
        if expert_type == "freq":
            self.expert = FrequencyOnlyReconstructor(
                num_features=num_nodes,
                seq_len=seq_len,
            )
        elif expert_type == "graph":
            self.expert = GraphGCNReconstructor(
                num_nodes=num_nodes,
                graph_hidden_dim=graph_hidden_dim,
                gcn_hidden_dim=gcn_hidden_dim,
            )
        else:
            raise ValueError(f"不支持的 expert_type: {expert_type}")

        self.detector = SimpleTransformerAD(
            num_nodes=num_nodes,
            d_model=detector_d_model,
            num_heads=detector_heads,
            num_layers=detector_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x_input: torch.Tensor,
        model_missing_mask: torch.Tensor,
        structural_missing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if self.expert_type == "graph":
            x_expert, adjacency = self.expert(x_input, model_missing_mask)
        else:
            x_expert = self.expert(x_input)
            adjacency = None

        x_complete = torch.where(model_missing_mask.bool(), x_expert, x_input)
        observed_mask = 1.0 - structural_missing_mask
        detector_reconstruction = self.detector(x_complete, observed_mask)

        return {
            "x_expert": x_expert,
            "x_complete": x_complete,
            "detector_reconstruction": detector_reconstruction,
            "adjacency": adjacency,
        }


def compute_losses(
    outputs: dict[str, torch.Tensor | None],
    x_true: torch.Tensor,
    structural_missing_mask: torch.Tensor,
    holdout_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    x_expert = outputs["x_expert"]
    x_complete = outputs["x_complete"]
    detector_reconstruction = outputs["detector_reconstruction"]
    assert isinstance(x_expert, torch.Tensor)
    assert isinstance(x_complete, torch.Tensor)
    assert isinstance(detector_reconstruction, torch.Tensor)

    expert_loss = masked_mse(x_expert, x_true, holdout_mask)
    observed_mask = 1.0 - structural_missing_mask
    detector_loss = masked_mse(
        detector_reconstruction,
        x_complete.detach(),
        observed_mask,
    )
    total_loss = expert_loss + detector_loss
    return {
        "expert_loss": expert_loss,
        "detector_loss": detector_loss,
        "total_loss": total_loss,
    }


def train_model(
    model: SingleExpertTransformerModel,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    holdout_ratio: float,
) -> list[dict[str, float]]:
    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        meter = {
            "expert_loss": 0.0,
            "detector_loss": 0.0,
            "total_loss": 0.0,
        }

        for x_true, structural_missing_mask in loader:
            x_true = x_true.to(device)
            structural_missing_mask = structural_missing_mask.to(device)
            holdout_mask = sample_holdout_mask(structural_missing_mask, holdout_ratio)
            model_missing_mask = torch.maximum(structural_missing_mask, holdout_mask)
            x_input = x_true.masked_fill(model_missing_mask.bool(), 0.0)

            outputs = model(x_input, model_missing_mask, structural_missing_mask)
            losses = compute_losses(
                outputs=outputs,
                x_true=x_true,
                structural_missing_mask=structural_missing_mask,
                holdout_mask=holdout_mask,
            )

            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for key in meter:
                meter[key] += float(losses[key].detach().cpu())

        epoch_stats = {key: value / max(len(loader), 1) for key, value in meter.items()}
        history.append(epoch_stats)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"total={epoch_stats['total_loss']:.5f} "
            f"expert={epoch_stats['expert_loss']:.5f} "
            f"det={epoch_stats['detector_loss']:.5f}"
        )

    return history


@torch.no_grad()
def reconstruct_full_sequence(
    model: SingleExpertTransformerModel,
    windows: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )

    completed = []
    adjacencies = []
    has_adjacency = False
    model.eval()

    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = model(x_input, structural_missing_mask, structural_missing_mask)

        x_complete = outputs["x_complete"]
        adjacency = outputs["adjacency"]
        assert isinstance(x_complete, torch.Tensor)
        completed.append(x_complete[:, -1, :].cpu().numpy())
        if adjacency is not None:
            has_adjacency = True
            adjacencies.append(adjacency.cpu().numpy())

    completed_array = np.concatenate(completed, axis=0).astype(np.float32)
    adjacency_array = None
    if has_adjacency:
        adjacency_array = np.concatenate(adjacencies, axis=0).astype(np.float32)
    return completed_array, adjacency_array


@torch.no_grad()
def collect_detection_scores(
    model: SingleExpertTransformerModel,
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
    scores = []
    model.eval()

    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        observed_mask = 1.0 - structural_missing_mask
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = model(x_input, structural_missing_mask, structural_missing_mask)

        x_complete = outputs["x_complete"]
        detector_reconstruction = outputs["detector_reconstruction"]
        assert isinstance(x_complete, torch.Tensor)
        assert isinstance(detector_reconstruction, torch.Tensor)

        detector_sq = ((detector_reconstruction - x_complete) * observed_mask).pow(2)
        detector_score = detector_sq.sum(dim=(1, 2)) / observed_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        scores.append(detector_score.cpu().numpy().astype(np.float32))

    return np.concatenate(scores, axis=0).astype(np.float32)


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history) + 1)
    total = [item["total_loss"] for item in history]
    expert = [item["expert_loss"] for item in history]
    detector = [item["detector_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, total, label="总损失", color="#1D3557")
    ax.plot(epochs, expert, label="专家插补损失", color="#457B9D")
    ax.plot(epochs, detector, label="检测损失", color="#E63946")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练曲线")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_experiment(
    *,
    args: argparse.Namespace,
    expert_type: str,
    experiment_name: str,
    description: str,
) -> None:
    seed_everything(args.seed)
    configure_chinese_font()

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

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

    print(description)
    print(f"使用设备: {device}")
    print(f"训练插补窗口: {train_impute_windows.shape}")
    print(f"训练评分窗口: {train_eval_windows.shape}")
    print(f"测试评分窗口: {test_eval_windows.shape}")

    model = SingleExpertTransformerModel(
        expert_type=expert_type,
        num_nodes=train_scaled.shape[1],
        seq_len=args.seq_len,
        graph_hidden_dim=args.graph_hidden_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
        detector_d_model=args.detector_d_model,
        detector_heads=args.detector_heads,
        detector_layers=args.detector_layers,
        dropout=args.dropout,
    ).to(device)

    history = train_model(
        model=model,
        train_windows=train_impute_windows,
        train_masks=train_impute_masks,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        holdout_ratio=args.holdout_ratio,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "expert_type": expert_type,
            "experiment_name": experiment_name,
        },
        output_dir / "model.pt",
    )

    train_complete_scaled, train_adjacency = reconstruct_full_sequence(
        model=model,
        windows=train_impute_windows,
        masks=train_impute_masks,
        device=device,
        batch_size=args.batch_size,
    )
    test_complete_scaled, _ = reconstruct_full_sequence(
        model=model,
        windows=test_impute_windows,
        masks=test_impute_masks,
        device=device,
        batch_size=args.batch_size,
    )

    save_csv(
        scaler.inverse_transform(train_complete_scaled),
        feature_names,
        output_dir / "train_completed.csv",
    )
    save_csv(
        scaler.inverse_transform(test_complete_scaled),
        feature_names,
        output_dir / "test_completed.csv",
    )

    if train_adjacency is not None:
        summarize_adjacency(train_adjacency)
        save_adjacency_heatmap(train_adjacency, output_dir / "train_adjacency_heatmap.png")
        save_adjacency_heatmap(
            train_adjacency,
            output_dir / "train_adjacency_heatmap_offdiag.png",
            suppress_diagonal=True,
        )

    train_scores = collect_detection_scores(
        model=model,
        windows=train_eval_windows,
        masks=train_eval_masks,
        device=device,
        batch_size=args.batch_size,
    )
    test_scores = collect_detection_scores(
        model=model,
        windows=test_eval_windows,
        masks=test_eval_masks,
        device=device,
        batch_size=args.batch_size,
    )

    threshold = choose_threshold(
        train_scores=train_scores,
        method="gaussian_quantile_max",
        std_factor=args.threshold_std_factor,
        quantile=args.threshold_quantile,
    )
    prediction = (test_scores >= threshold).astype(np.int64)
    metrics = compute_binary_classification_metrics(
        test_labels,
        prediction,
        threshold=threshold,
    )

    prediction_df = pd.DataFrame(
        {
            "sample_index": np.arange(1, len(test_scores) + 1),
            "label": test_labels,
            "score": test_scores,
            "prediction": prediction,
        }
    )
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "experiment": experiment_name,
        "description": description,
        "expert_type": expert_type,
        "device": str(device),
        "config": vars(args),
        "metrics": metrics,
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "train_threshold_quantile": float(np.quantile(train_scores, args.threshold_quantile)),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    save_training_curve(history, output_dir / "training_curve.png")
    plot_detection_scores(
        scores=test_scores,
        threshold=threshold,
        split_idx=split_index_from_labels(test_labels),
        save_path=output_dir / "anomaly_scores.png",
        color_scheme="mra",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
