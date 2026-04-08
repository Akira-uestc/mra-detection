#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ablation.ablation_model import (
    AblationConfig,
    FusionAnomalyModel,
    LossWeights,
    apply_min_anomaly_duration,
    choose_device,
    collect_window_statistics,
    combine_scores,
    compute_distribution_shift_scores,
    infer_rate_metadata,
    reconstruct_full_sequence,
    save_csv,
    save_gate_statistics,
    train_model,
)
from graph_gcn_reconstruction import (
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
    save_training_curve,
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


def build_experiment_registry() -> dict[str, AblationConfig]:
    experiments = [
        AblationConfig(
            name="A00_full",
            description="完整 MRA 基线：双专家门控 + 采样率感知检测 + EWAF",
        ),
        AblationConfig(
            name="A01_no_gcn_expert",
            description="去掉 GCN 专家，仅保留频域专家插补",
            fusion_mode="single_freq",
        ),
        AblationConfig(
            name="A02_no_freq_expert",
            description="去掉频域专家，仅保留 GCN 专家插补",
            fusion_mode="single_gcn",
        ),
        AblationConfig(
            name="A03_mean_fusion",
            description="保留双专家，但使用均值融合代替门控融合",
            fusion_mode="mean",
        ),
        AblationConfig(
            name="A04_no_gate_rate",
            description="门控层不使用采样率嵌入",
            gate_use_rate=False,
        ),
        AblationConfig(
            name="A05_no_detector_rate",
            description="检测器不使用采样率嵌入",
            detector_use_rate=False,
        ),
        AblationConfig(
            name="A06_no_phase",
            description="检测器不使用 phase 特征",
            detector_use_phase=False,
        ),
        AblationConfig(
            name="A07_no_rate_aware",
            description="门控与检测器均不使用采样率感知信息",
            gate_use_rate=False,
            detector_use_rate=False,
            detector_use_phase=False,
        ),
        AblationConfig(
            name="A08_no_ewaf",
            description="异常检测方式消融 D4：去掉 EWAF 平滑",
            use_ewaf=False,
        ),
    ]
    return {item.name: item for item in experiments}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 ablation/ 下独立运行 MRA 模型消融实验，不修改原始 mra.py。",
    )
    parser.add_argument("--train-glob", default="data/train/train_1.csv")
    parser.add_argument("--test-glob", default="data/test/test_C5_1.csv")
    parser.add_argument("--output-dir", default="ablation/outputs")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="实验名列表，默认 all。",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="仅列出实验名并退出。",
    )
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--graph-hidden-dim", type=int, default=32)
    parser.add_argument("--gcn-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--rate-embed-dim", type=int, default=8)
    parser.add_argument("--detector-d-model", type=int, default=128)
    parser.add_argument("--detector-heads", type=int, default=4)
    parser.add_argument("--detector-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diag-target", type=float, default=0.25)
    parser.add_argument("--score-disagreement-weight", type=float, default=0.25)
    parser.add_argument("--score-shift-weight", type=float, default=1.0)
    parser.add_argument("--threshold-std-factor", type=float, default=2.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--ewaf-alpha", type=float, default=0.3)
    parser.add_argument("--min-anomaly-duration", type=int, default=50)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_experiments(
    requested: list[str],
    registry: dict[str, AblationConfig],
) -> list[AblationConfig]:
    if not requested or requested == ["all"]:
        return list(registry.values())

    resolved = []
    for item in requested:
        if item not in registry:
            raise ValueError(f"未知实验名: {item}")
        resolved.append(registry[item])
    return resolved


def print_experiment_list(registry: dict[str, AblationConfig]) -> None:
    print("可用消融实验:")
    for key, value in registry.items():
        print(f"  {key}: {value.description}")


def maybe_apply_ewaf(
    scores: np.ndarray,
    alpha: float,
    segment_lengths: list[int] | None,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return scores.astype(np.float32)
    return apply_ewaf_by_segments(scores, alpha, segment_lengths=segment_lengths)


def run_single_experiment(
    args: argparse.Namespace,
    experiment: AblationConfig,
    output_root: Path,
) -> dict[str, object]:
    print(f"\n========== {experiment.name} ==========")
    print(experiment.description)

    seed_everything(args.seed)
    configure_chinese_font()

    output_dir = output_root / experiment.name
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

    model = FusionAnomalyModel(
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
        ablation=experiment,
    ).to(device)

    history = train_model(
        model=model,
        train_windows=train_impute_windows,
        train_masks=train_impute_masks,
        rate_id=rate_id,
        stride=stride,
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
            "ablation": asdict(experiment),
            "rate_id": rate_id,
            "stride": stride,
            "loss_weights": asdict(loss_weights),
        },
        output_dir / "model.pt",
    )

    train_complete_scaled, train_gate_weights, train_adjacency = reconstruct_full_sequence(
        model=model,
        windows=train_impute_windows,
        masks=train_impute_masks,
        rate_id=rate_id,
        stride=stride,
        device=device,
        batch_size=args.batch_size,
    )
    test_complete_scaled, test_gate_weights, _ = reconstruct_full_sequence(
        model=model,
        windows=test_impute_windows,
        masks=test_impute_masks,
        rate_id=rate_id,
        stride=stride,
        device=device,
        batch_size=args.batch_size,
    )
    train_complete = scaler.inverse_transform(train_complete_scaled)
    test_complete = scaler.inverse_transform(test_complete_scaled)

    save_csv(train_complete, feature_names, output_dir / "train_completed.csv")
    save_csv(test_complete, feature_names, output_dir / "test_completed.csv")
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

    if train_adjacency is not None:
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
        stride=stride,
        device=device,
        batch_size=args.batch_size,
    )
    test_stats = collect_window_statistics(
        model=model,
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
        disagreement_weight=args.score_disagreement_weight,
        shift_weight=args.score_shift_weight,
    )
    test_segment_lengths = infer_segment_lengths(test_labels)
    test_split_idx = split_index_from_labels(test_labels)
    train_scores = maybe_apply_ewaf(
        train_raw_scores,
        args.ewaf_alpha,
        None,
        enabled=experiment.use_ewaf,
    )
    test_scores = maybe_apply_ewaf(
        test_raw_scores,
        args.ewaf_alpha,
        test_segment_lengths,
        enabled=experiment.use_ewaf,
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
            "shift_score": test_stats["shift_score"],
            "raw_final_score": test_raw_scores,
            "final_score": test_scores,
            "threshold_prediction": threshold_prediction,
            "prediction": final_prediction,
        }
    )
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "experiment": experiment.name,
        "description": experiment.description,
        "device": str(device),
        "config": vars(args),
        "ablation": asdict(experiment),
        "loss_weights": asdict(loss_weights),
        "metrics": metrics,
        "ewaf_enabled": experiment.use_ewaf,
        "ewaf_alpha": args.ewaf_alpha,
        "min_anomaly_duration": args.min_anomaly_duration,
        "train_raw_score_mean": float(np.mean(train_raw_scores)),
        "train_raw_score_std": float(np.std(train_raw_scores)),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "train_threshold_quantile": float(np.quantile(train_scores, args.threshold_quantile)),
        "rate_id": rate_id.tolist(),
        "stride": stride.tolist(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    configure_chinese_font()
    save_training_curve(history, output_dir / "training_curve.png")
    plot_detection_scores(
        scores=test_scores,
        threshold=threshold,
        split_idx=test_split_idx,
        save_path=output_dir / "anomaly_scores.png",
        title=f"{experiment.name} 异常检测结果",
        style="mra",
        figsize=(16, 5),
        dpi=180,
        threshold_label_fmt="阈值 = {threshold:.4f}",
    )

    row = {
        "experiment": experiment.name,
        "description": experiment.description,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "fdr": float(metrics["fdr"]),
        "fra": float(metrics["fra"]),
        "f1": float(metrics["f1"]),
        "threshold": float(metrics["threshold"]),
        "ewaf_enabled": experiment.use_ewaf,
        "output_dir": str(output_dir),
    }
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ewaf_alpha <= 1.0:
        raise ValueError(f"--ewaf-alpha 必须在 (0, 1] 内，收到 {args.ewaf_alpha}")
    if args.min_anomaly_duration < 1:
        raise ValueError(
            f"--min-anomaly-duration 必须 >= 1，收到 {args.min_anomaly_duration}"
        )

    registry = build_experiment_registry()
    if args.list_experiments:
        print_experiment_list(registry)
        return

    experiments = resolve_experiments(args.experiments, registry)
    output_root = ROOT_DIR / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for experiment in experiments:
        summary_rows.append(run_single_experiment(args, experiment, output_root))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n已汇总至: {summary_path}")


if __name__ == "__main__":
    main()
