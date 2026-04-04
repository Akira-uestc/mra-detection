#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ablation_detection_utils import choose_device, run_mra_style_detection
from graph_gcn_reconstruction import (
    GraphGCNReconstructor,
    ObservedStandardScaler,
    build_windows as build_graph_windows,
    configure_chinese_font,
    load_csv_series,
    reconstruct_sequence,
    seed_everything,
    summarize_adjacency,
    train_model as train_graph_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单 GCN 模块消融：使用 GCN 完成重构，并接入 mra.py 的异常检测流程。"
    )
    parser.add_argument(
        "--train-glob",
        default="data/train/train_*.csv",
        help="训练集 CSV 模式，默认: data/train/train_*.csv",
    )
    parser.add_argument(
        "--test-glob",
        default="data/test/test_*.csv",
        help="测试集 CSV 模式，默认: data/test/test_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gcn_transformer_detection",
        help="输出目录，默认: outputs/gcn_transformer_detection",
    )
    parser.add_argument(
        "--graph-seq-len",
        type=int,
        default=50,
        help="GCN 重构窗口长度，默认: 50",
    )
    parser.add_argument(
        "--detector-seq-len",
        "--transformer-seq-len",
        dest="detector_seq_len",
        type=int,
        default=50,
        help="MRA 风格检测器窗口长度，默认: 50",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="滑动窗口步长，默认: 1",
    )
    parser.add_argument(
        "--graph-epochs",
        type=int,
        default=10,
        help="GCN 重构训练轮数，默认: 10",
    )
    parser.add_argument(
        "--detector-epochs",
        "--transformer-epochs",
        dest="detector_epochs",
        type=int,
        default=12,
        help="MRA 风格检测器训练轮数，默认: 12",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小，默认: 64",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率，默认: 1e-3",
    )
    parser.add_argument(
        "--graph-hidden-dim",
        type=int,
        default=32,
        help="图学习隐藏维度，默认: 32",
    )
    parser.add_argument(
        "--gcn-hidden-dim",
        type=int,
        default=32,
        help="GCN 隐藏维度，默认: 32",
    )
    parser.add_argument(
        "--adj-reg-weight",
        type=float,
        default=0.1,
        help="邻接矩阵正则权重，默认: 0.1",
    )
    parser.add_argument(
        "--diag-target",
        type=float,
        default=0.25,
        help="邻接矩阵目标对角占比，默认: 0.25",
    )
    parser.add_argument("--detector-d-model", type=int, default=128)
    parser.add_argument("--detector-heads", type=int, default=4)
    parser.add_argument("--detector-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--score-shift-weight",
        type=float,
        default=1.0,
        help="分布偏移分数在最终异常分数中的权重。",
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
        help="最短异常持续长度，小于该长度的连续异常片段会被抑制。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40,
        help="随机种子，默认: 40",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="运行设备，默认自动选择 cuda 或 cpu",
    )
    return parser.parse_args()


def save_csv(data: np.ndarray, feature_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=feature_names).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_chinese_font()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw, train_mask, feature_names = load_csv_series(args.train_glob)
    test_raw, test_mask, _ = load_csv_series(args.test_glob)

    scaler = ObservedStandardScaler().fit(train_raw, train_mask)
    train_scaled = scaler.transform(train_raw, train_mask)
    test_scaled = scaler.transform(test_raw, test_mask)

    train_windows, train_window_masks = build_graph_windows(
        train_scaled,
        train_mask.astype(np.float32),
        seq_len=args.graph_seq_len,
        stride=args.stride,
    )
    test_windows, test_window_masks = build_graph_windows(
        test_scaled,
        test_mask.astype(np.float32),
        seq_len=args.graph_seq_len,
        stride=args.stride,
    )

    device = choose_device(args.device)
    print(f"使用设备: {device}")

    graph_model = GraphGCNReconstructor(
        num_nodes=train_scaled.shape[1],
        graph_hidden_dim=args.graph_hidden_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
    ).to(device)

    print("\n开始训练 GCN 重构模块...")
    train_graph_model(
        model=graph_model,
        train_windows=train_windows,
        train_masks=train_window_masks,
        device=device,
        epochs=args.graph_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adj_reg_weight=args.adj_reg_weight,
        diag_target=args.diag_target,
    )

    print("\n重构 train/test 序列...")
    train_reconstructed_scaled, train_adjacency = reconstruct_sequence(
        model=graph_model,
        windows=train_windows,
        masks=train_window_masks,
        device=device,
        batch_size=args.batch_size,
    )
    test_reconstructed_scaled, _ = reconstruct_sequence(
        model=graph_model,
        windows=test_windows,
        masks=test_window_masks,
        device=device,
        batch_size=args.batch_size,
    )
    summarize_adjacency(train_adjacency)

    train_completed_scaled = np.where(
        train_mask > 0.5,
        train_reconstructed_scaled,
        train_scaled,
    ).astype(np.float32)
    test_completed_scaled = np.where(
        test_mask > 0.5,
        test_reconstructed_scaled,
        test_scaled,
    ).astype(np.float32)

    train_reconstructed = scaler.inverse_transform(train_reconstructed_scaled)
    test_reconstructed = scaler.inverse_transform(test_reconstructed_scaled)
    train_completed = scaler.inverse_transform(train_completed_scaled)
    test_completed = scaler.inverse_transform(test_completed_scaled)

    save_csv(
        train_reconstructed,
        feature_names,
        output_dir / "train_reconstructed_final.csv",
    )
    save_csv(
        test_reconstructed,
        feature_names,
        output_dir / "test_reconstructed_final.csv",
    )
    save_csv(
        train_completed,
        feature_names,
        output_dir / "train_completed_final.csv",
    )
    save_csv(
        test_completed,
        feature_names,
        output_dir / "test_completed_final.csv",
    )

    run_mra_style_detection(
        train_completed_scaled=train_completed_scaled,
        test_completed_scaled=test_completed_scaled,
        train_mask=train_mask,
        test_mask=test_mask,
        seq_len=args.detector_seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        detector_epochs=args.detector_epochs,
        lr=args.lr,
        detector_d_model=args.detector_d_model,
        detector_heads=args.detector_heads,
        detector_layers=args.detector_layers,
        dropout=args.dropout,
        score_shift_weight=args.score_shift_weight,
        threshold_std_factor=args.threshold_std_factor,
        threshold_quantile=args.threshold_quantile,
        ewaf_alpha=args.ewaf_alpha,
        min_anomaly_duration=args.min_anomaly_duration,
        device=device,
        output_dir=output_dir,
        plot_title="单 GCN 模块 MRA 风格异常检测",
    )

    print("\n输出文件:")
    print(f"  Train 重构序列: {output_dir / 'train_reconstructed_final.csv'}")
    print(f"  Test 重构序列 : {output_dir / 'test_reconstructed_final.csv'}")
    print(f"  Train 补全序列: {output_dir / 'train_completed_final.csv'}")
    print(f"  Test 补全序列 : {output_dir / 'test_completed_final.csv'}")
    print(f"  检测模型     : {output_dir / 'detector.pt'}")
    print(f"  检测结果     : {output_dir / 'test_predictions.csv'}")
    print(f"  指标汇总     : {output_dir / 'metrics.json'}")
    print(f"  检测曲线图   : {output_dir / 'anomaly_scores.png'}")


if __name__ == "__main__":
    main()
