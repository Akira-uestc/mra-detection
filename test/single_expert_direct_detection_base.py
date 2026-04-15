#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    adjacency_balance_loss,
    configure_chinese_font,
    resolve_distance_prior_matrix,
    save_adjacency_heatmap,
    seed_everything,
    summarize_adjacency,
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
    parser.add_argument(
        "--physical-coords-path",
        default=None,
        help="可选：节点物理坐标文件（CSV/NPY），按变量顺序逐行排列。",
    )
    parser.add_argument(
        "--physical-distance-path",
        default=None,
        help="可选：节点物理距离矩阵文件（CSV/NPY），按变量顺序排列。",
    )
    parser.add_argument("--diag-target", type=float, default=0.25)
    parser.add_argument("--graph-loss-weight", type=float, default=0.1)
    parser.add_argument("--threshold-std-factor", type=float, default=2.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--ewaf-alpha", type=float, default=0.3)
    parser.add_argument("--min-anomaly-duration", type=int, default=50)
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


class SingleExpertDirectDetectionModel(nn.Module):
    def __init__(
        self,
        *,
        expert_type: str,
        num_nodes: int,
        seq_len: int,
        graph_hidden_dim: int,
        gcn_hidden_dim: int,
        physical_distance_matrix: torch.Tensor | np.ndarray | None = None,
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
                physical_distance_matrix=physical_distance_matrix,
            )
        else:
            raise ValueError(f"不支持的 expert_type: {expert_type}")

    def forward(
        self,
        x_input: torch.Tensor,
        model_missing_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if self.expert_type == "graph":
            x_expert, adjacency = self.expert(x_input, model_missing_mask)
        else:
            x_expert = self.expert(x_input)
            adjacency = None

        x_complete = torch.where(model_missing_mask.bool(), x_expert, x_input)
        return {
            "x_expert": x_expert,
            "x_complete": x_complete,
            "adjacency": adjacency,
        }


def compute_losses(
    outputs: dict[str, torch.Tensor | None],
    x_true: torch.Tensor,
    holdout_mask: torch.Tensor,
    *,
    expert_type: str,
    diag_target: float,
    graph_loss_weight: float,
) -> dict[str, torch.Tensor]:
    x_expert = outputs["x_expert"]
    assert isinstance(x_expert, torch.Tensor)

    expert_loss = masked_mse(x_expert, x_true, holdout_mask)
    if expert_type == "graph":
        adjacency = outputs["adjacency"]
        assert isinstance(adjacency, torch.Tensor)
        graph_loss = adjacency_balance_loss(adjacency, target_diag=diag_target)
    else:
        graph_loss = expert_loss.new_zeros(())

    total_loss = expert_loss + graph_loss_weight * graph_loss
    return {
        "expert_loss": expert_loss,
        "graph_loss": graph_loss,
        "total_loss": total_loss,
    }


def train_model(
    model: SingleExpertDirectDetectionModel,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    *,
    expert_type: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    holdout_ratio: float,
    diag_target: float,
    graph_loss_weight: float,
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
            "graph_loss": 0.0,
            "total_loss": 0.0,
        }

        for x_true, structural_missing_mask in loader:
            x_true = x_true.to(device)
            structural_missing_mask = structural_missing_mask.to(device)
            holdout_mask = sample_holdout_mask(structural_missing_mask, holdout_ratio)
            model_missing_mask = torch.maximum(structural_missing_mask, holdout_mask)
            x_input = x_true.masked_fill(model_missing_mask.bool(), 0.0)

            outputs = model(x_input, model_missing_mask)
            losses = compute_losses(
                outputs=outputs,
                x_true=x_true,
                holdout_mask=holdout_mask,
                expert_type=expert_type,
                diag_target=diag_target,
                graph_loss_weight=graph_loss_weight,
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
            f"graph={epoch_stats['graph_loss']:.5f}"
        )

    return history


@torch.no_grad()
def reconstruct_full_sequence(
    model: SingleExpertDirectDetectionModel,
    windows: np.ndarray,
    masks: np.ndarray,
    *,
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
        outputs = model(x_input, structural_missing_mask)

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
    model: SingleExpertDirectDetectionModel,
    windows: np.ndarray,
    masks: np.ndarray,
    *,
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
        outputs = model(x_input, structural_missing_mask)

        x_expert = outputs["x_expert"]
        assert isinstance(x_expert, torch.Tensor)
        expert_sq = ((x_expert - x_true) * observed_mask).pow(2)
        expert_score = expert_sq.sum(dim=(1, 2)) / observed_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        scores.append(expert_score.cpu().numpy().astype(np.float32))

    return np.concatenate(scores, axis=0).astype(np.float32)


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history) + 1)
    total = [item["total_loss"] for item in history]
    expert = [item["expert_loss"] for item in history]
    graph = [item["graph_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, total, label="总损失", color="#1D3557")
    ax.plot(epochs, expert, label="重构损失", color="#457B9D")
    ax.plot(epochs, graph, label="图正则损失", color="#E76F51")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练曲线")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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


def run_experiment(
    *,
    args: argparse.Namespace,
    expert_type: str,
    experiment_name: str,
    description: str,
) -> None:
    if not 0.0 < args.ewaf_alpha <= 1.0:
        raise ValueError(f"--ewaf-alpha 必须在 (0, 1] 内，收到 {args.ewaf_alpha}")
    if args.min_anomaly_duration < 1:
        raise ValueError(
            f"--min-anomaly-duration 必须 >= 1，收到 {args.min_anomaly_duration}"
        )

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

    physical_distance_matrix = None
    if expert_type == "graph":
        physical_distance_matrix, distance_prior_source = resolve_distance_prior_matrix(
            data=train_scaled,
            missing_mask=train_mask,
            coords_path=args.physical_coords_path,
            distance_path=args.physical_distance_path,
        )
    else:
        distance_prior_source = None

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
    if distance_prior_source == "external_physical":
        print("图先验: 使用物理距离构造的欧氏距离相似度先验")
    elif distance_prior_source == "train_data_auto":
        print("图先验: 根据训练数据自动计算欧氏距离相似度先验")

    model = SingleExpertDirectDetectionModel(
        expert_type=expert_type,
        num_nodes=train_scaled.shape[1],
        seq_len=args.seq_len,
        graph_hidden_dim=args.graph_hidden_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
        physical_distance_matrix=physical_distance_matrix,
    ).to(device)

    history = train_model(
        model=model,
        train_windows=train_impute_windows,
        train_masks=train_impute_masks,
        expert_type=expert_type,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        holdout_ratio=args.holdout_ratio,
        diag_target=args.diag_target,
        graph_loss_weight=args.graph_loss_weight,
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
        save_adjacency_heatmap(
            train_adjacency,
            output_dir / "train_adjacency_heatmap.png",
        )
        save_adjacency_heatmap(
            train_adjacency,
            output_dir / "train_adjacency_heatmap_offdiag.png",
            suppress_diagonal=True,
        )

    train_raw_scores = collect_detection_scores(
        model=model,
        windows=train_eval_windows,
        masks=train_eval_masks,
        device=device,
        batch_size=args.batch_size,
    )
    test_raw_scores = collect_detection_scores(
        model=model,
        windows=test_eval_windows,
        masks=test_eval_masks,
        device=device,
        batch_size=args.batch_size,
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
        method="gaussian_quantile_max",
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
            "raw_score": test_raw_scores,
            "final_score": test_scores,
            "threshold_prediction": threshold_prediction,
            "prediction": final_prediction,
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
        "train_raw_score_mean": float(np.mean(train_raw_scores)),
        "train_raw_score_std": float(np.std(train_raw_scores)),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "train_threshold_quantile": float(
            np.quantile(train_scores, args.threshold_quantile)
        ),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    save_training_curve(history, output_dir / "training_curve.png")
    plot_detection_scores(
        scores=test_scores,
        threshold=threshold,
        split_idx=test_split_idx,
        save_path=output_dir / "anomaly_scores.png",
        title=f"{experiment_name} 异常检测结果",
        style="mra",
        figsize=(16, 5),
        dpi=180,
        threshold_label_fmt="阈值 = {threshold:.4f}",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
