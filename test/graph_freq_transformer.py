#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

from graph_gcn_reconstruction import (
    ObservedStandardScaler,
    adjacency_balance_loss,
    configure_chinese_font,
    resolve_distance_prior_matrix,
    save_adjacency_heatmap,
    seed_everything,
    summarize_adjacency,
)
from mra import (
    LossWeights,
    TwoExpertGatedImputer,
    apply_min_anomaly_duration,
    choose_device,
    infer_rate_metadata,
    masked_mse,
    normalize_scores,
    sample_holdout_mask,
    save_csv,
    save_gate_statistics,
)
from single_expert_transformer_base import SimpleTransformerAD
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="双专家门控插补 + 普通 Transformer 异常检测。"
    )
    parser.add_argument("--train-glob", default="data/train/train_1.csv")
    parser.add_argument("--test-glob", default="data/test/test_C5_1.csv")
    parser.add_argument("--output-dir", default="test/outputs/graph_freq_transformer")
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
    parser.add_argument("--graph-hidden-dim", type=int, default=32)
    parser.add_argument("--gcn-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--rate-embed-dim", type=int, default=8)
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
    parser.add_argument("--detector-d-model", type=int, default=128)
    parser.add_argument("--detector-heads", type=int, default=4)
    parser.add_argument("--detector-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diag-target", type=float, default=0.25)
    parser.add_argument(
        "--score-disagreement-weight",
        type=float,
        default=0.25,
        help="将专家分歧并入最终异常分数时的权重。",
    )
    parser.add_argument("--threshold-std-factor", type=float, default=2.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
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


class GraphFreqTransformerAnomalyModel(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        seq_len: int,
        num_rates: int,
        graph_hidden_dim: int,
        gcn_hidden_dim: int,
        gate_hidden_dim: int,
        rate_embed_dim: int,
        detector_d_model: int,
        detector_heads: int,
        detector_layers: int,
        dropout: float,
        physical_distance_matrix: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.imputer = TwoExpertGatedImputer(
            num_nodes=num_nodes,
            seq_len=seq_len,
            num_rates=num_rates,
            graph_hidden_dim=graph_hidden_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gate_hidden_dim=gate_hidden_dim,
            rate_embed_dim=rate_embed_dim,
            physical_distance_matrix=physical_distance_matrix,
        )
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
        rate_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.imputer(x_input, model_missing_mask, rate_id)
        observed_mask = 1.0 - structural_missing_mask
        detector_reconstruction = self.detector(outputs["x_complete"], observed_mask)

        outputs["detector_reconstruction"] = detector_reconstruction
        outputs["detector_error"] = torch.abs(
            detector_reconstruction - outputs["x_complete"]
        )
        return outputs


def compute_losses(
    outputs: dict[str, torch.Tensor],
    x_true: torch.Tensor,
    structural_missing_mask: torch.Tensor,
    holdout_mask: torch.Tensor,
    loss_weights: LossWeights,
    diag_target: float,
) -> dict[str, torch.Tensor]:
    fusion_loss = masked_mse(outputs["x_imputed"], x_true, holdout_mask)
    gcn_loss = masked_mse(outputs["x_gcn"], x_true, holdout_mask)
    freq_loss = masked_mse(outputs["x_freq"], x_true, holdout_mask)
    graph_loss = adjacency_balance_loss(outputs["adjacency"], target_diag=diag_target)

    observed_mask = 1.0 - structural_missing_mask
    detector_loss = masked_mse(
        outputs["detector_reconstruction"],
        outputs["x_complete"].detach(),
        observed_mask,
    )

    total_loss = (
        loss_weights.fusion * fusion_loss
        + loss_weights.gcn * gcn_loss
        + loss_weights.freq * freq_loss
        + loss_weights.detector * detector_loss
        + loss_weights.graph * graph_loss
    )
    return {
        "fusion_loss": fusion_loss,
        "gcn_loss": gcn_loss,
        "freq_loss": freq_loss,
        "detector_loss": detector_loss,
        "graph_loss": graph_loss,
        "total_loss": total_loss,
    }


def train_model(
    model: GraphFreqTransformerAnomalyModel,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    rate_id: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    holdout_ratio: float,
    diag_target: float,
    loss_weights: LossWeights,
) -> list[dict[str, float]]:
    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rate_id = rate_id.to(device)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        meter = {
            "fusion_loss": 0.0,
            "gcn_loss": 0.0,
            "freq_loss": 0.0,
            "detector_loss": 0.0,
            "graph_loss": 0.0,
            "total_loss": 0.0,
        }

        for x_true, structural_missing_mask in loader:
            x_true = x_true.to(device)
            structural_missing_mask = structural_missing_mask.to(device)
            holdout_mask = sample_holdout_mask(structural_missing_mask, holdout_ratio)
            model_missing_mask = torch.maximum(structural_missing_mask, holdout_mask)
            x_input = x_true.masked_fill(model_missing_mask.bool(), 0.0)

            outputs = model(
                x_input=x_input,
                model_missing_mask=model_missing_mask,
                structural_missing_mask=structural_missing_mask,
                rate_id=rate_id,
            )
            losses = compute_losses(
                outputs=outputs,
                x_true=x_true,
                structural_missing_mask=structural_missing_mask,
                holdout_mask=holdout_mask,
                loss_weights=loss_weights,
                diag_target=diag_target,
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
            f"fusion={epoch_stats['fusion_loss']:.5f} "
            f"det={epoch_stats['detector_loss']:.5f} "
            f"graph={epoch_stats['graph_loss']:.5f}"
        )

    return history


@torch.no_grad()
def reconstruct_full_sequence(
    model: GraphFreqTransformerAnomalyModel,
    windows: np.ndarray,
    masks: np.ndarray,
    rate_id: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )
    rate_id = rate_id.to(device)

    completed = []
    gate_weights = []
    adj_samples = []
    model.eval()

    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = model(
            x_input=x_input,
            model_missing_mask=structural_missing_mask,
            structural_missing_mask=structural_missing_mask,
            rate_id=rate_id,
        )
        completed.append(outputs["x_complete"][:, -1, :].cpu().numpy())
        gate_weights.append(outputs["gate_weights"][:, -1, :, :].cpu().numpy())
        adj_samples.append(outputs["adjacency"].cpu().numpy())

    return (
        np.concatenate(completed, axis=0).astype(np.float32),
        np.concatenate(gate_weights, axis=0).astype(np.float32),
        np.concatenate(adj_samples, axis=0).astype(np.float32),
    )


@torch.no_grad()
def collect_window_statistics(
    model: GraphFreqTransformerAnomalyModel,
    windows: np.ndarray,
    masks: np.ndarray,
    rate_id: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )
    rate_id = rate_id.to(device)

    collected = {
        "detector_score": [],
        "disagreement_score": [],
        "gate_entropy": [],
    }
    model.eval()

    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        observed_mask = 1.0 - structural_missing_mask
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = model(
            x_input=x_input,
            model_missing_mask=structural_missing_mask,
            structural_missing_mask=structural_missing_mask,
            rate_id=rate_id,
        )

        detector_sq = (
            (outputs["detector_reconstruction"] - outputs["x_complete"]) * observed_mask
        ).pow(2)
        detector_score = detector_sq.sum(dim=(1, 2)) / observed_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        disagreement_score = torch.abs(outputs["x_gcn"] - outputs["x_freq"]).mean(
            dim=(1, 2)
        )
        gate = outputs["gate_weights"].clamp_min(1e-8)
        gate_entropy = -(gate * gate.log()).sum(dim=-1).mean(dim=(1, 2))

        collected["detector_score"].append(
            detector_score.cpu().numpy().astype(np.float32)
        )
        collected["disagreement_score"].append(
            disagreement_score.cpu().numpy().astype(np.float32)
        )
        collected["gate_entropy"].append(gate_entropy.cpu().numpy().astype(np.float32))

    return {
        key: np.concatenate(value, axis=0).astype(np.float32)
        for key, value in collected.items()
    }


def combine_scores(
    train_stats: dict[str, np.ndarray],
    eval_stats: dict[str, np.ndarray],
    disagreement_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_det, eval_det = normalize_scores(
        train_stats["detector_score"],
        eval_stats["detector_score"],
    )
    train_score = train_det
    eval_score = eval_det

    if disagreement_weight != 0.0:
        train_gap, eval_gap = normalize_scores(
            train_stats["disagreement_score"],
            eval_stats["disagreement_score"],
        )
        train_score = train_score + disagreement_weight * train_gap
        eval_score = eval_score + disagreement_weight * eval_gap

    return train_score.astype(np.float32), eval_score.astype(np.float32)


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history) + 1)
    total = [item["total_loss"] for item in history]
    fusion = [item["fusion_loss"] for item in history]
    detector = [item["detector_loss"] for item in history]
    gcn = [item["gcn_loss"] for item in history]
    freq = [item["freq_loss"] for item in history]
    graph = [item["graph_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, total, label="总损失", color="#1D3557")
    ax.plot(epochs, fusion, label="融合插补损失", color="#457B9D")
    ax.plot(epochs, detector, label="Transformer 检测损失", color="#E63946")
    ax.plot(epochs, gcn, label="GCN 损失", color="#2A9D8F")
    ax.plot(epochs, freq, label="频域损失", color="#E9C46A")
    ax.plot(epochs, graph, label="图正则损失", color="#E76F51")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练曲线")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
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
    loss_weights = LossWeights()

    train_raw, train_mask, feature_names = load_csv_glob_with_mask(args.train_glob)
    test_raw, test_mask, _ = load_csv_glob_with_mask(args.test_glob)

    scaler = ObservedStandardScaler().fit(train_raw, train_mask)
    train_scaled = scaler.transform(train_raw, train_mask)
    test_scaled = scaler.transform(test_raw, test_mask)
    distance_prior_matrix, distance_prior_source = resolve_distance_prior_matrix(
        data=train_scaled,
        missing_mask=train_mask,
        coords_path=args.physical_coords_path,
        distance_path=args.physical_distance_path,
    )

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
    print("双专家门控插补 + 普通 Transformer 异常检测")
    print(f"使用设备: {device}")
    print(f"训练插补窗口: {train_impute_windows.shape}")
    print(f"训练评分窗口: {train_eval_windows.shape}")
    print(f"测试评分窗口: {test_eval_windows.shape}")
    print(f"采样率分组 rate_id: {rate_id.tolist()}")
    print(f"推断步长 stride: {stride.tolist()}")
    if distance_prior_source == "external_physical":
        print("图先验: 使用物理距离构造的欧氏距离相似度先验")
    else:
        print("图先验: 根据训练数据自动计算欧氏距离相似度先验")

    model = GraphFreqTransformerAnomalyModel(
        num_nodes=train_scaled.shape[1],
        seq_len=args.seq_len,
        num_rates=int(rate_id.max().item()) + 1,
        graph_hidden_dim=args.graph_hidden_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        rate_embed_dim=args.rate_embed_dim,
        detector_d_model=args.detector_d_model,
        detector_heads=args.detector_heads,
        detector_layers=args.detector_layers,
        dropout=args.dropout,
        physical_distance_matrix=distance_prior_matrix,
    ).to(device)

    history = train_model(
        model=model,
        train_windows=train_impute_windows,
        train_masks=train_impute_masks,
        rate_id=rate_id,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        holdout_ratio=args.holdout_ratio,
        diag_target=args.diag_target,
        loss_weights=loss_weights,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "rate_id": rate_id,
            "stride": stride,
            "loss_weights": asdict(loss_weights),
            "detector_type": "simple_transformer",
        },
        output_dir / "model.pt",
    )

    train_complete_scaled, train_gate_weights, train_adjacency = (
        reconstruct_full_sequence(
            model=model,
            windows=train_impute_windows,
            masks=train_impute_masks,
            rate_id=rate_id,
            device=device,
            batch_size=args.batch_size,
        )
    )
    test_complete_scaled, test_gate_weights, _ = reconstruct_full_sequence(
        model=model,
        windows=test_impute_windows,
        masks=test_impute_masks,
        rate_id=rate_id,
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
    save_gate_statistics(
        train_gate_weights,
        feature_names,
        output_dir / "train_gate_weights.csv",
    )
    save_gate_statistics(
        test_gate_weights,
        feature_names,
        output_dir / "test_gate_weights.csv",
    )
    summarize_adjacency(train_adjacency)
    save_adjacency_heatmap(train_adjacency, output_dir / "train_adjacency_heatmap.png")
    save_adjacency_heatmap(
        train_adjacency,
        output_dir / "train_adjacency_heatmap_offdiag.png",
        suppress_diagonal=True,
    )

    train_stats = collect_window_statistics(
        model=model,
        windows=train_eval_windows,
        masks=train_eval_masks,
        rate_id=rate_id,
        device=device,
        batch_size=args.batch_size,
    )
    test_stats = collect_window_statistics(
        model=model,
        windows=test_eval_windows,
        masks=test_eval_masks,
        rate_id=rate_id,
        device=device,
        batch_size=args.batch_size,
    )

    train_raw_scores, test_raw_scores = combine_scores(
        train_stats=train_stats,
        eval_stats=test_stats,
        disagreement_weight=args.score_disagreement_weight,
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
            "detector_score": test_stats["detector_score"],
            "disagreement_score": test_stats["disagreement_score"],
            "gate_entropy": test_stats["gate_entropy"],
            "raw_final_score": test_raw_scores,
            "final_score": test_scores,
            "threshold_prediction": threshold_prediction,
            "prediction": final_prediction,
        }
    )
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "experiment": "graph_freq_transformer",
        "description": "双专家门控插补 + 普通 Transformer 异常检测",
        "detector_type": "simple_transformer",
        "device": str(device),
        "config": vars(args),
        "loss_weights": asdict(loss_weights),
        "metrics": metrics,
        "ewaf_alpha": args.ewaf_alpha,
        "min_anomaly_duration": args.min_anomaly_duration,
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

    save_training_curve(history, output_dir / "training_curve.png")
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
    print(f"  门控权重 : {output_dir / 'train_gate_weights.csv'}")
    print(f"  预测结果 : {output_dir / 'test_predictions.csv'}")
    print(f"  指标汇总 : {output_dir / 'metrics.json'}")
    print(f"  曲线图   : {output_dir / 'anomaly_scores.png'}")


if __name__ == "__main__":
    main()
