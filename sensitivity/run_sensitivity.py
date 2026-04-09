#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graph_gcn_reconstruction import (
    ObservedStandardScaler,
    configure_chinese_font,
    seed_everything,
)
from mra import (
    FusionAnomalyModel,
    LossWeights,
    apply_min_anomaly_duration,
    choose_device,
    collect_window_statistics,
    combine_scores,
    compute_distribution_shift_scores,
    infer_rate_metadata,
    reconstruct_full_sequence,
    train_model,
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


RUN_CONFIG_FIELDS = [
    "train_glob",
    "test_glob",
    "seq_len",
    "stride",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "holdout_ratio",
    "graph_hidden_dim",
    "gcn_hidden_dim",
    "gate_hidden_dim",
    "rate_embed_dim",
    "detector_d_model",
    "detector_heads",
    "detector_layers",
    "dropout",
    "diag_target",
    "score_disagreement_weight",
    "score_shift_weight",
    "threshold_std_factor",
    "threshold_quantile",
    "ewaf_alpha",
    "min_anomaly_duration",
    "seed",
    "device",
]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    group: str
    description: str
    kind: str
    candidates: tuple[Any, ...]

    def parse_value(self, raw: str) -> int | float:
        if self.kind == "int":
            return int(float(raw))
        return float(raw)


@dataclass
class LoadedData:
    feature_names: list[str]
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_scaled: np.ndarray
    test_scaled: np.ndarray
    rate_id: torch.Tensor
    sampling_stride: torch.Tensor


@dataclass
class WindowBundle:
    train_impute_windows: np.ndarray
    train_impute_masks: np.ndarray
    test_impute_windows: np.ndarray
    test_impute_masks: np.ndarray
    train_eval_windows: np.ndarray
    train_eval_masks: np.ndarray
    test_eval_windows: np.ndarray
    test_eval_masks: np.ndarray
    test_labels: np.ndarray


@dataclass
class ScoreContext:
    train_detector_score: np.ndarray
    train_disagreement_score: np.ndarray
    train_gate_entropy: np.ndarray
    train_shift_score: np.ndarray
    test_detector_score: np.ndarray
    test_disagreement_score: np.ndarray
    test_gate_entropy: np.ndarray
    test_shift_score: np.ndarray
    test_labels: np.ndarray

    @property
    def train_stats(self) -> dict[str, np.ndarray]:
        return {
            "detector_score": self.train_detector_score,
            "disagreement_score": self.train_disagreement_score,
            "gate_entropy": self.train_gate_entropy,
            "shift_score": self.train_shift_score,
        }

    @property
    def test_stats(self) -> dict[str, np.ndarray]:
        return {
            "detector_score": self.test_detector_score,
            "disagreement_score": self.test_disagreement_score,
            "gate_entropy": self.test_gate_entropy,
            "shift_score": self.test_shift_score,
        }

    @property
    def test_segment_lengths(self) -> list[int]:
        return infer_segment_lengths(self.test_labels)

    @property
    def test_split_idx(self) -> int:
        return split_index_from_labels(self.test_labels)

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            train_detector_score=self.train_detector_score.astype(np.float32),
            train_disagreement_score=self.train_disagreement_score.astype(np.float32),
            train_gate_entropy=self.train_gate_entropy.astype(np.float32),
            train_shift_score=self.train_shift_score.astype(np.float32),
            test_detector_score=self.test_detector_score.astype(np.float32),
            test_disagreement_score=self.test_disagreement_score.astype(np.float32),
            test_gate_entropy=self.test_gate_entropy.astype(np.float32),
            test_shift_score=self.test_shift_score.astype(np.float32),
            test_labels=self.test_labels.astype(np.int64),
        )

    @classmethod
    def load(cls, input_path: Path) -> "ScoreContext":
        saved = np.load(input_path)
        return cls(
            train_detector_score=saved["train_detector_score"].astype(np.float32),
            train_disagreement_score=saved["train_disagreement_score"].astype(
                np.float32
            ),
            train_gate_entropy=saved["train_gate_entropy"].astype(np.float32),
            train_shift_score=saved["train_shift_score"].astype(np.float32),
            test_detector_score=saved["test_detector_score"].astype(np.float32),
            test_disagreement_score=saved["test_disagreement_score"].astype(
                np.float32
            ),
            test_gate_entropy=saved["test_gate_entropy"].astype(np.float32),
            test_shift_score=saved["test_shift_score"].astype(np.float32),
            test_labels=saved["test_labels"].astype(np.int64),
        )


def build_parameter_registry() -> dict[str, ParameterSpec]:
    specs = [
        ParameterSpec(
            name="seq_len",
            group="training",
            description="窗口长度",
            kind="int",
            candidates=(20, 30, 50, 80, 100),
        ),
        ParameterSpec(
            name="stride",
            group="training",
            description="滑窗步长",
            kind="int",
            candidates=(1, 2, 5, 10),
        ),
        ParameterSpec(
            name="holdout_ratio",
            group="training",
            description="训练随机额外遮挡比例",
            kind="float",
            candidates=(0.05, 0.10, 0.15, 0.20, 0.30),
        ),
        ParameterSpec(
            name="lr",
            group="training",
            description="AdamW 学习率",
            kind="float",
            candidates=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        ),
        ParameterSpec(
            name="weight_decay",
            group="training",
            description="AdamW 权重衰减",
            kind="float",
            candidates=(0.0, 1e-5, 1e-4, 1e-3, 1e-2),
        ),
        ParameterSpec(
            name="graph_hidden_dim",
            group="training",
            description="图学习隐藏维度",
            kind="int",
            candidates=(16, 32, 64, 96, 128),
        ),
        ParameterSpec(
            name="gcn_hidden_dim",
            group="training",
            description="GCN 隐藏维度",
            kind="int",
            candidates=(16, 32, 64, 96, 128),
        ),
        ParameterSpec(
            name="gate_hidden_dim",
            group="training",
            description="门控 MLP 隐藏维度",
            kind="int",
            candidates=(32, 64, 128, 192, 256),
        ),
        ParameterSpec(
            name="rate_embed_dim",
            group="training",
            description="采样率嵌入维度",
            kind="int",
            candidates=(4, 8, 16, 32),
        ),
        ParameterSpec(
            name="detector_d_model",
            group="training",
            description="Transformer d_model",
            kind="int",
            candidates=(64, 96, 128, 192, 256),
        ),
        ParameterSpec(
            name="detector_heads",
            group="training",
            description="Transformer 头数",
            kind="int",
            candidates=(2, 4, 8),
        ),
        ParameterSpec(
            name="detector_layers",
            group="training",
            description="Transformer 层数",
            kind="int",
            candidates=(1, 2, 3, 4, 5),
        ),
        ParameterSpec(
            name="dropout",
            group="training",
            description="Transformer dropout",
            kind="float",
            candidates=(0.0, 0.05, 0.10, 0.20, 0.30),
        ),
        ParameterSpec(
            name="diag_target",
            group="training",
            description="邻接矩阵对角约束目标",
            kind="float",
            candidates=(0.10, 0.20, 0.25, 0.30, 0.40),
        ),
        ParameterSpec(
            name="score_disagreement_weight",
            group="postprocess",
            description="专家分歧分数权重",
            kind="float",
            candidates=(0.0, 0.10, 0.25, 0.50, 1.0),
        ),
        ParameterSpec(
            name="score_shift_weight",
            group="postprocess",
            description="分布偏移分数权重",
            kind="float",
            candidates=(0.0, 0.5, 1.0, 1.5, 2.0),
        ),
        ParameterSpec(
            name="threshold_std_factor",
            group="postprocess",
            description="阈值高斯项标准差系数",
            kind="float",
            candidates=(0.5, 1.0, 2.0, 3.0, 4.0),
        ),
        ParameterSpec(
            name="threshold_quantile",
            group="postprocess",
            description="训练分数分位数阈值",
            kind="float",
            candidates=(0.90, 0.95, 0.97, 0.99),
        ),
        ParameterSpec(
            name="ewaf_alpha",
            group="postprocess",
            description="EWAF 平滑系数",
            kind="float",
            candidates=(0.10, 0.20, 0.30, 0.60, 1.0),
        ),
        ParameterSpec(
            name="min_anomaly_duration",
            group="postprocess",
            description="最短异常持续长度",
            kind="int",
            candidates=(1, 10, 20, 50, 100),
        ),
    ]
    return {spec.name: spec for spec in specs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 sensitivity/ 下独立运行 MRA 参数敏感性实验，不修改原始 mra.py。"
    )
    parser.add_argument("--train-glob", default="data/train/train_1.csv")
    parser.add_argument("--test-glob", default="data/test/test_C5_1.csv")
    parser.add_argument("--output-dir", default="sensitivity/outputs")
    parser.add_argument(
        "--group",
        choices=("training", "postprocess", "all"),
        default="all",
        help="按大类筛选敏感性参数。",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=["all"],
        help="参数名列表，默认 all。",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="仅列出可用参数及候选值并退出。",
    )
    parser.add_argument(
        "--num-values",
        type=int,
        default=3,
        help="每个参数默认保留的候选值个数。",
    )
    parser.add_argument(
        "--param-counts",
        nargs="*",
        default=[],
        help="为指定参数覆盖候选值个数，例如 seq_len=5 lr=4。",
    )
    parser.add_argument(
        "--param-values",
        nargs="*",
        default=[],
        help="显式指定参数测试值，例如 seq_len=30,50,80。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在且配置一致的实验。",
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


def validate_args(args: argparse.Namespace) -> None:
    if args.num_values < 1:
        raise ValueError(f"--num-values 必须 >= 1，收到 {args.num_values}")
    if not 0.0 < args.ewaf_alpha <= 1.0:
        raise ValueError(f"--ewaf-alpha 必须在 (0, 1] 内，收到 {args.ewaf_alpha}")
    if args.min_anomaly_duration < 1:
        raise ValueError(
            f"--min-anomaly-duration 必须 >= 1，收到 {args.min_anomaly_duration}"
        )
    if args.detector_d_model % args.detector_heads != 0:
        raise ValueError(
            "--detector-d-model 必须能被 --detector-heads 整除，"
            f"收到 {args.detector_d_model} 和 {args.detector_heads}"
        )


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in RUN_CONFIG_FIELDS}


def format_value(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def value_token(value: Any) -> str:
    return (
        format_value(value)
        .replace("-", "m")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def parse_param_count_overrides(
    items: list[str],
    registry: dict[str, ParameterSpec],
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--param-counts 格式错误: {item}")
        name, raw_count = item.split("=", 1)
        if name not in registry:
            raise ValueError(f"未知参数: {name}")
        count = int(raw_count)
        if count < 1:
            raise ValueError(f"参数 {name} 的候选值个数必须 >= 1，收到 {count}")
        overrides[name] = count
    return overrides


def parse_param_value_overrides(
    items: list[str],
    registry: dict[str, ParameterSpec],
) -> dict[str, list[int | float]]:
    overrides: dict[str, list[int | float]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--param-values 格式错误: {item}")
        name, raw_values = item.split("=", 1)
        if name not in registry:
            raise ValueError(f"未知参数: {name}")
        spec = registry[name]
        parsed = [
            spec.parse_value(part.strip())
            for part in raw_values.split(",")
            if part.strip()
        ]
        if not parsed:
            raise ValueError(f"参数 {name} 没有提供有效取值。")
        overrides[name] = sort_unique_values(parsed)
    return overrides


def sort_unique_values(values: list[Any]) -> list[Any]:
    normalized = []
    seen = set()
    for value in values:
        key = format_value(value)
        if key in seen:
            continue
        normalized.append(value)
        seen.add(key)
    return sorted(normalized, key=float)


def select_spread_values(values: list[Any], count: int, default_value: Any) -> list[Any]:
    if count <= 1:
        return [default_value]

    candidates = sort_unique_values(values + [default_value])
    if count >= len(candidates):
        return candidates

    default_index = candidates.index(default_value)
    selected = {default_index}
    if len(selected) < count:
        selected.add(0)
    if len(selected) < count:
        selected.add(len(candidates) - 1)

    while len(selected) < count:
        best_idx = None
        best_distance = -1
        for idx in range(len(candidates)):
            if idx in selected:
                continue
            distance = min(abs(idx - chosen) for chosen in selected)
            if distance > best_distance:
                best_idx = idx
                best_distance = distance
        if best_idx is None:
            break
        selected.add(best_idx)

    return [candidates[idx] for idx in sorted(selected)]


def resolve_param_values(
    spec: ParameterSpec,
    baseline_value: Any,
    default_count: int,
    count_overrides: dict[str, int],
    explicit_overrides: dict[str, list[int | float]],
) -> list[int | float]:
    if spec.name in explicit_overrides:
        return sort_unique_values(explicit_overrides[spec.name] + [baseline_value])
    count = count_overrides.get(spec.name, default_count)
    return select_spread_values(list(spec.candidates), count, baseline_value)


def resolve_selected_parameters(
    requested: list[str],
    group: str,
    registry: dict[str, ParameterSpec],
) -> list[ParameterSpec]:
    if requested == ["all"] or not requested:
        if group == "all":
            return list(registry.values())
        return [spec for spec in registry.values() if spec.group == group]

    selected = []
    for name in requested:
        if name not in registry:
            raise ValueError(f"未知参数: {name}")
        spec = registry[name]
        if group != "all" and spec.group != group:
            raise ValueError(
                f"参数 {name} 属于 {spec.group}，与 --group={group} 不一致。"
            )
        selected.append(spec)
    return selected


def validate_override_targets(
    selected_specs: list[ParameterSpec],
    count_overrides: dict[str, int],
    explicit_overrides: dict[str, list[int | float]],
) -> None:
    selected_names = {spec.name for spec in selected_specs}
    for name in count_overrides:
        if name not in selected_names:
            raise ValueError(f"--param-counts 中的参数 {name} 未包含在本次 --params 里。")
    for name in explicit_overrides:
        if name not in selected_names:
            raise ValueError(f"--param-values 中的参数 {name} 未包含在本次 --params 里。")


def print_parameter_list(
    registry: dict[str, ParameterSpec],
    args: argparse.Namespace,
) -> None:
    print("可用敏感性参数:")
    for spec in registry.values():
        baseline_value = getattr(args, spec.name)
        candidates = sort_unique_values(list(spec.candidates) + [baseline_value])
        print(
            f"  {spec.name} [{spec.group}] 默认={format_value(baseline_value)} "
            f"候选={','.join(format_value(item) for item in candidates)}"
        )
        print(f"    {spec.description}")


def load_base_data(args: argparse.Namespace) -> LoadedData:
    train_raw, train_mask, feature_names = load_csv_glob_with_mask(args.train_glob)
    test_raw, test_mask, _ = load_csv_glob_with_mask(args.test_glob)

    scaler = ObservedStandardScaler().fit(train_raw, train_mask)
    train_scaled = scaler.transform(train_raw, train_mask)
    test_scaled = scaler.transform(test_raw, test_mask)
    rate_id, sampling_stride = infer_rate_metadata(train_mask.astype(np.float32))

    return LoadedData(
        feature_names=feature_names,
        train_mask=train_mask.astype(np.float32),
        test_mask=test_mask.astype(np.float32),
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        rate_id=rate_id,
        sampling_stride=sampling_stride,
    )


def build_window_bundle(
    data: LoadedData,
    config: argparse.Namespace,
) -> WindowBundle:
    train_impute_windows, train_impute_masks = build_windows(
        data.train_scaled,
        data.train_mask,
        seq_len=config.seq_len,
        stride=config.stride,
    )
    test_impute_windows, test_impute_masks = build_windows(
        data.test_scaled,
        data.test_mask,
        seq_len=config.seq_len,
        stride=config.stride,
    )
    train_eval_windows, train_eval_masks = build_standard_windows(
        data.train_scaled,
        data.train_mask,
        seq_len=config.seq_len,
        stride=config.stride,
    )
    test_eval_windows, test_eval_masks, test_labels = build_prompt_test_windows(
        data.test_scaled,
        data.test_mask,
        seq_len=config.seq_len,
        stride=config.stride,
    )
    return WindowBundle(
        train_impute_windows=train_impute_windows,
        train_impute_masks=train_impute_masks,
        test_impute_windows=test_impute_windows,
        test_impute_masks=test_impute_masks,
        train_eval_windows=train_eval_windows,
        train_eval_masks=train_eval_masks,
        test_eval_windows=test_eval_windows,
        test_eval_masks=test_eval_masks,
        test_labels=test_labels,
    )


def make_namespace(args: argparse.Namespace, overrides: dict[str, Any]) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(overrides)
    return argparse.Namespace(**payload)


def build_model(
    data: LoadedData,
    config: argparse.Namespace,
    device: torch.device,
) -> FusionAnomalyModel:
    if config.detector_d_model % config.detector_heads != 0:
        raise ValueError(
            "--detector-d-model 必须能被 --detector-heads 整除，"
            f"收到 {config.detector_d_model} 和 {config.detector_heads}"
        )
    return FusionAnomalyModel(
        num_nodes=data.train_scaled.shape[1],
        seq_len=config.seq_len,
        num_rates=int(data.rate_id.max().item()) + 1,
        graph_hidden_dim=config.graph_hidden_dim,
        gcn_hidden_dim=config.gcn_hidden_dim,
        gate_hidden_dim=config.gate_hidden_dim,
        rate_embed_dim=config.rate_embed_dim,
        detector_d_model=config.detector_d_model,
        detector_heads=config.detector_heads,
        detector_layers=config.detector_layers,
        dropout=config.dropout,
    ).to(device)


def build_score_context(
    model: FusionAnomalyModel,
    data: LoadedData,
    windows: WindowBundle,
    config: argparse.Namespace,
    device: torch.device,
) -> ScoreContext:
    train_complete_scaled, _, _ = reconstruct_full_sequence(
        model=model,
        windows=windows.train_impute_windows,
        masks=windows.train_impute_masks,
        rate_id=data.rate_id,
        stride=data.sampling_stride,
        device=device,
        batch_size=config.batch_size,
    )
    test_complete_scaled, _, _ = reconstruct_full_sequence(
        model=model,
        windows=windows.test_impute_windows,
        masks=windows.test_impute_masks,
        rate_id=data.rate_id,
        stride=data.sampling_stride,
        device=device,
        batch_size=config.batch_size,
    )

    train_stats = collect_window_statistics(
        model=model,
        windows=windows.train_eval_windows,
        masks=windows.train_eval_masks,
        rate_id=data.rate_id,
        stride=data.sampling_stride,
        device=device,
        batch_size=config.batch_size,
    )
    test_stats = collect_window_statistics(
        model=model,
        windows=windows.test_eval_windows,
        masks=windows.test_eval_masks,
        rate_id=data.rate_id,
        stride=data.sampling_stride,
        device=device,
        batch_size=config.batch_size,
    )

    train_complete_eval_windows, _ = build_standard_windows(
        train_complete_scaled,
        np.zeros_like(data.train_mask, dtype=np.float32),
        seq_len=config.seq_len,
        stride=config.stride,
    )
    test_complete_eval_windows, _, _ = build_prompt_test_windows(
        test_complete_scaled,
        np.zeros_like(data.test_mask, dtype=np.float32),
        seq_len=config.seq_len,
        stride=config.stride,
    )
    train_shift_scores, test_shift_scores = compute_distribution_shift_scores(
        train_windows=train_complete_eval_windows,
        eval_windows=test_complete_eval_windows,
    )

    return ScoreContext(
        train_detector_score=train_stats["detector_score"],
        train_disagreement_score=train_stats["disagreement_score"],
        train_gate_entropy=train_stats["gate_entropy"],
        train_shift_score=train_shift_scores,
        test_detector_score=test_stats["detector_score"],
        test_disagreement_score=test_stats["disagreement_score"],
        test_gate_entropy=test_stats["gate_entropy"],
        test_shift_score=test_shift_scores,
        test_labels=windows.test_labels,
    )


def evaluate_postprocess(
    context: ScoreContext,
    config: argparse.Namespace,
) -> dict[str, Any]:
    train_raw_scores, test_raw_scores = combine_scores(
        train_stats=context.train_stats,
        eval_stats=context.test_stats,
        disagreement_weight=config.score_disagreement_weight,
        shift_weight=config.score_shift_weight,
    )
    train_scores = apply_ewaf_by_segments(train_raw_scores, config.ewaf_alpha)
    test_scores = apply_ewaf_by_segments(
        test_raw_scores,
        config.ewaf_alpha,
        segment_lengths=context.test_segment_lengths,
    )
    threshold = choose_threshold(
        train_scores=train_scores,
        method="gaussian_quantile_max",
        std_factor=config.threshold_std_factor,
        quantile=config.threshold_quantile,
    )
    threshold_prediction = (test_scores >= threshold).astype(np.int64)
    final_prediction = apply_min_anomaly_duration(
        threshold_prediction,
        config.min_anomaly_duration,
    )
    metrics = compute_binary_classification_metrics(
        context.test_labels,
        final_prediction,
        threshold=threshold,
    )
    return {
        "metrics": metrics,
        "train_raw_scores": train_raw_scores.astype(np.float32),
        "test_raw_scores": test_raw_scores.astype(np.float32),
        "train_scores": train_scores.astype(np.float32),
        "test_scores": test_scores.astype(np.float32),
        "threshold_prediction": threshold_prediction.astype(np.int64),
        "final_prediction": final_prediction.astype(np.int64),
        "threshold": float(threshold),
    }


def save_predictions(
    output_dir: Path,
    context: ScoreContext,
    evaluation: dict[str, Any],
) -> None:
    prediction_df = pd.DataFrame(
        {
            "sample_index": np.arange(1, len(evaluation["test_scores"]) + 1),
            "label": context.test_labels,
            "detector_score": context.test_detector_score,
            "disagreement_score": context.test_disagreement_score,
            "gate_entropy": context.test_gate_entropy,
            "shift_score": context.test_shift_score,
            "raw_final_score": evaluation["test_raw_scores"],
            "final_score": evaluation["test_scores"],
            "threshold_prediction": evaluation["threshold_prediction"],
            "prediction": evaluation["final_prediction"],
        }
    )
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)


def save_metrics(
    output_dir: Path,
    config: argparse.Namespace,
    evaluation: dict[str, Any],
    *,
    row_kind: str,
    parameter: str,
    parameter_group: str,
    parameter_value: Any,
    reused_baseline: bool,
) -> dict[str, Any]:
    summary = {
        "row_kind": row_kind,
        "parameter": parameter,
        "parameter_group": parameter_group,
        "parameter_value": parameter_value,
        "reused_baseline": reused_baseline,
        "config": build_run_config(config),
        "metrics": {
            key: float(value) for key, value in evaluation["metrics"].items()
        },
        "train_raw_score_mean": float(np.mean(evaluation["train_raw_scores"])),
        "train_raw_score_std": float(np.std(evaluation["train_raw_scores"])),
        "train_score_mean": float(np.mean(evaluation["train_scores"])),
        "train_score_std": float(np.std(evaluation["train_scores"])),
        "train_threshold_quantile": float(
            np.quantile(evaluation["train_scores"], config.threshold_quantile)
        ),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def config_matches(output_dir: Path, config: argparse.Namespace) -> bool:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return saved.get("config") == build_run_config(config)


def load_metrics_summary(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))


def build_summary_row(
    metrics_summary: dict[str, Any],
    *,
    output_dir: Path,
    row_kind: str,
    parameter: str,
    parameter_group: str,
    parameter_value: Any,
    reused_baseline: bool,
) -> dict[str, Any]:
    metrics = metrics_summary["metrics"]
    numeric_value = (
        float(parameter_value)
        if isinstance(parameter_value, (int, float, np.integer, np.floating))
        else np.nan
    )
    return {
        "row_kind": row_kind,
        "parameter": parameter,
        "parameter_group": parameter_group,
        "parameter_value": format_value(parameter_value),
        "parameter_value_numeric": numeric_value,
        "reused_baseline": bool(reused_baseline),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "fdr": float(metrics["fdr"]),
        "fra": float(metrics["fra"]),
        "f1": float(metrics["f1"]),
        "threshold": float(metrics["threshold"]),
        "output_dir": str(output_dir),
    }


def save_sensitivity_plot(
    rows: pd.DataFrame,
    spec: ParameterSpec,
    output_path: Path,
) -> None:
    if rows.empty:
        return
    rows = rows.sort_values("parameter_value_numeric")
    x_values = rows["parameter_value_numeric"].to_numpy(dtype=float)
    f1_values = rows["f1"].to_numpy(dtype=float)
    fdr_values = rows["fdr"].to_numpy(dtype=float)
    baseline_rows = rows[rows["row_kind"] == "baseline_reference"]
    baseline_x = (
        float(baseline_rows.iloc[0]["parameter_value_numeric"])
        if not baseline_rows.empty
        else None
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_values, f1_values, marker="o", linewidth=1.8, color="#1D3557", label="F1")
    ax.plot(x_values, fdr_values, marker="s", linewidth=1.8, color="#E76F51", label="FDR")
    if baseline_x is not None:
        ax.axvline(
            baseline_x,
            color="#6C757D",
            linestyle="--",
            linewidth=1.2,
            label=f"Baseline = {format_value(baseline_x)}",
        )
    ax.set_title(f"{spec.name} 敏感性")
    ax.set_xlabel(spec.name)
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_selection_plan(output_root: Path, plan: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "selection_plan.json", "w", encoding="utf-8") as file:
        json.dump(plan, file, ensure_ascii=False, indent=2)


def training_run(
    data: LoadedData,
    config: argparse.Namespace,
    output_dir: Path,
    *,
    row_kind: str,
    parameter: str,
    parameter_group: str,
    parameter_value: Any,
    skip_existing: bool,
    require_context: bool,
) -> tuple[dict[str, Any], ScoreContext | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / "score_context.npz"

    if skip_existing and config_matches(output_dir, config):
        metrics_summary = load_metrics_summary(output_dir)
        if require_context and context_path.exists():
            print(f"复用已有实验: {output_dir}")
            return metrics_summary, ScoreContext.load(context_path)
        if not require_context:
            print(f"复用已有实验: {output_dir}")
            return metrics_summary, None

    print(f"运行训练实验: {parameter}={format_value(parameter_value)}")
    seed_everything(config.seed)
    device = choose_device(config.device)
    loss_weights = LossWeights()
    windows = build_window_bundle(data, config)
    model = build_model(data, config, device)

    history = train_model(
        model=model,
        train_windows=windows.train_impute_windows,
        train_masks=windows.train_impute_masks,
        rate_id=data.rate_id,
        stride=data.sampling_stride,
        device=device,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        holdout_ratio=config.holdout_ratio,
        diag_target=config.diag_target,
        loss_weights=loss_weights,
    )

    context = build_score_context(model, data, windows, config, device)
    evaluation = evaluate_postprocess(context, config)
    save_predictions(output_dir, context, evaluation)
    metrics_summary = save_metrics(
        output_dir=output_dir,
        config=config,
        evaluation=evaluation,
        row_kind=row_kind,
        parameter=parameter,
        parameter_group=parameter_group,
        parameter_value=parameter_value,
        reused_baseline=False,
    )
    context.save(context_path)
    save_training_curve(history, output_dir / "training_curve.png")
    plot_detection_scores(
        scores=evaluation["test_scores"],
        threshold=evaluation["threshold"],
        split_idx=context.test_split_idx,
        save_path=output_dir / "anomaly_scores.png",
        title=f"{parameter}={format_value(parameter_value)}",
        style="mra",
        figsize=(16, 5),
        dpi=180,
        threshold_label_fmt="阈值 = {threshold:.4f}",
    )
    return metrics_summary, context if require_context else None


def postprocess_run(
    config: argparse.Namespace,
    context: ScoreContext,
    output_dir: Path,
    *,
    row_kind: str,
    parameter: str,
    parameter_group: str,
    parameter_value: Any,
    skip_existing: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and config_matches(output_dir, config):
        print(f"复用已有实验: {output_dir}")
        return load_metrics_summary(output_dir)

    print(f"运行后处理实验: {parameter}={format_value(parameter_value)}")
    evaluation = evaluate_postprocess(context, config)
    save_predictions(output_dir, context, evaluation)
    metrics_summary = save_metrics(
        output_dir=output_dir,
        config=config,
        evaluation=evaluation,
        row_kind=row_kind,
        parameter=parameter,
        parameter_group=parameter_group,
        parameter_value=parameter_value,
        reused_baseline=False,
    )
    plot_detection_scores(
        scores=evaluation["test_scores"],
        threshold=evaluation["threshold"],
        split_idx=context.test_split_idx,
        save_path=output_dir / "anomaly_scores.png",
        title=f"{parameter}={format_value(parameter_value)}",
        style="mra",
        figsize=(16, 5),
        dpi=180,
        threshold_label_fmt="阈值 = {threshold:.4f}",
    )
    return metrics_summary


def create_baseline_reference_row(
    metrics_summary: dict[str, Any],
    output_dir: Path,
    spec: ParameterSpec,
    baseline_value: Any,
) -> dict[str, Any]:
    return build_summary_row(
        metrics_summary,
        output_dir=output_dir,
        row_kind="baseline_reference",
        parameter=spec.name,
        parameter_group=spec.group,
        parameter_value=baseline_value,
        reused_baseline=True,
    )


def summarize_results(
    output_root: Path,
    rows: list[dict[str, Any]],
    selected_specs: list[ParameterSpec],
) -> None:
    summary_df = pd.DataFrame(rows)
    baseline_row = summary_df[summary_df["row_kind"] == "baseline"].iloc[0]
    for metric in ("accuracy", "precision", "recall", "fdr", "fra", "f1", "threshold"):
        summary_df[f"delta_{metric}"] = summary_df[metric] - float(baseline_row[metric])
    summary_df.to_csv(output_root / "sensitivity_summary.csv", index=False)

    ranking_rows = []
    variant_df = summary_df[summary_df["parameter"] != "baseline"].copy()
    for spec in selected_specs:
        param_df = variant_df[variant_df["parameter"] == spec.name]
        if param_df.empty:
            continue
        ranking_rows.append(
            {
                "parameter": spec.name,
                "parameter_group": spec.group,
                "tested_values": int(param_df["parameter_value"].nunique()),
                "best_f1": float(param_df["f1"].max()),
                "worst_f1": float(param_df["f1"].min()),
                "f1_range": float(param_df["f1"].max() - param_df["f1"].min()),
                "best_fdr": float(param_df["fdr"].max()),
                "worst_fdr": float(param_df["fdr"].min()),
                "fdr_range": float(param_df["fdr"].max() - param_df["fdr"].min()),
                "max_abs_delta_f1": float(param_df["delta_f1"].abs().max()),
                "max_abs_delta_fdr": float(param_df["delta_fdr"].abs().max()),
            }
        )
        save_sensitivity_plot(
            rows=param_df,
            spec=spec,
            output_path=output_root / "plots" / f"{spec.name}_f1_fdr.png",
        )

    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            by=["max_abs_delta_f1", "max_abs_delta_fdr"],
            ascending=[False, False],
        )
    ranking_df.to_csv(output_root / "sensitivity_ranking.csv", index=False)


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_chinese_font()

    registry = build_parameter_registry()
    if args.list_params:
        print_parameter_list(registry, args)
        return

    selected_specs = resolve_selected_parameters(args.params, args.group, registry)
    count_overrides = parse_param_count_overrides(args.param_counts, registry)
    explicit_overrides = parse_param_value_overrides(args.param_values, registry)
    validate_override_targets(selected_specs, count_overrides, explicit_overrides)

    output_root = ROOT_DIR / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    selection_plan = {
        "group": args.group,
        "num_values": args.num_values,
        "parameters": {},
    }
    for spec in selected_specs:
        selected_values = resolve_param_values(
            spec=spec,
            baseline_value=getattr(args, spec.name),
            default_count=args.num_values,
            count_overrides=count_overrides,
            explicit_overrides=explicit_overrides,
        )
        selection_plan["parameters"][spec.name] = {
            "group": spec.group,
            "baseline": getattr(args, spec.name),
            "values": selected_values,
        }
    save_selection_plan(output_root, selection_plan)

    print("本次实验计划:")
    for name, item in selection_plan["parameters"].items():
        print(f"  {name} [{item['group']}]: {item['values']}")

    data = load_base_data(args)

    baseline_dir = output_root / "baseline"
    baseline_summary, baseline_context = training_run(
        data=data,
        config=make_namespace(args, {}),
        output_dir=baseline_dir,
        row_kind="baseline",
        parameter="baseline",
        parameter_group="baseline",
        parameter_value="baseline",
        skip_existing=args.skip_existing,
        require_context=True,
    )
    if baseline_context is None:
        raise RuntimeError("baseline 分数上下文缺失，无法继续后处理敏感性实验。")

    rows = [
        build_summary_row(
            baseline_summary,
            output_dir=baseline_dir,
            row_kind="baseline",
            parameter="baseline",
            parameter_group="baseline",
            parameter_value="baseline",
            reused_baseline=False,
        )
    ]

    for spec in selected_specs:
        baseline_value = getattr(args, spec.name)
        values = resolve_param_values(
            spec=spec,
            baseline_value=baseline_value,
            default_count=args.num_values,
            count_overrides=count_overrides,
            explicit_overrides=explicit_overrides,
        )
        rows.append(
            create_baseline_reference_row(
                baseline_summary,
                baseline_dir,
                spec,
                baseline_value,
            )
        )

        for value in values:
            if value == baseline_value:
                continue

            config = make_namespace(args, {spec.name: value})
            if spec.group == "training":
                output_dir = output_root / "training" / spec.name / value_token(value)
                metrics_summary, _ = training_run(
                    data=data,
                    config=config,
                    output_dir=output_dir,
                    row_kind="variant",
                    parameter=spec.name,
                    parameter_group=spec.group,
                    parameter_value=value,
                    skip_existing=args.skip_existing,
                    require_context=False,
                )
            else:
                output_dir = output_root / "postprocess" / spec.name / value_token(value)
                metrics_summary = postprocess_run(
                    config=config,
                    context=baseline_context,
                    output_dir=output_dir,
                    row_kind="variant",
                    parameter=spec.name,
                    parameter_group=spec.group,
                    parameter_value=value,
                    skip_existing=args.skip_existing,
                )

            rows.append(
                build_summary_row(
                    metrics_summary,
                    output_dir=output_dir,
                    row_kind="variant",
                    parameter=spec.name,
                    parameter_group=spec.group,
                    parameter_value=value,
                    reused_baseline=False,
                )
            )

    summarize_results(output_root, rows, selected_specs)
    print(f"\n已汇总至: {output_root / 'sensitivity_summary.csv'}")
    print(f"敏感性排序: {output_root / 'sensitivity_ranking.csv'}")


if __name__ == "__main__":
    main()
