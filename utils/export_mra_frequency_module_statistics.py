#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mra import PreImpute
from graph_gcn_reconstruction import ObservedStandardScaler, seed_everything
from utils.methods.data_loading import load_csv_glob_with_mask
from utils.methods.windowing import build_windows
from utils.plot_mra_frequency_module_figures import (
    DEFAULT_CHECKPOINT,
    checkpoint_args,
    choose_device,
    load_checkpoint,
    load_frequency_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "导出训练完成 MRA 频域模块在全部训练窗口上的注意力权重、增强频谱，"
            "并筛选低频修正后更强的变量/窗口。"
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"训练完成 checkpoint 路径。默认: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--train-glob",
        default=None,
        help="训练 CSV 匹配模式。默认优先读取 checkpoint 配置，否则 data/train/train_1.csv。",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mra_frequency_module_statistics",
        help="输出目录。默认: outputs/mra_frequency_module_statistics",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="窗口长度。默认优先读取 checkpoint 配置，否则 50。",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="窗口步长。默认优先读取 checkpoint 配置，否则 1。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="批量前向的 batch size。默认: 256",
    )
    parser.add_argument(
        "--low-freq-count",
        type=int,
        default=4,
        help="低频非直流 bin 数量。默认使用频率索引 1..4。",
    )
    parser.add_argument(
        "--include-dc",
        action="store_true",
        help="把频率索引 0 的直流分量也纳入低频统计。默认不纳入。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="保存排名靠前的候选数量。默认: 30",
    )
    parser.add_argument(
        "--candidate-min-start-row",
        type=int,
        default=0,
        help="候选窗口的最小原始起始行。默认 0，即排除前端 padding 窗口。",
    )
    parser.add_argument(
        "--candidate-min-missing-count",
        type=int,
        default=0,
        help="主候选表要求该变量窗口内至少有多少个缺失点。默认 0。",
    )
    parser.add_argument(
        "--min-delta-low-high-ratio",
        type=float,
        default=1.0,
        help="候选要求低频平均修正量 / 高频平均修正量至少达到该值。默认 1.0。",
    )
    parser.add_argument(
        "--save-long-csv",
        action="store_true",
        help="额外保存全部 window-feature-frequency 明细 CSV。文件会比较大。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="运行设备，例如 cpu 或 cuda。默认自动选择。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40,
        help="随机种子。默认: 40",
    )
    return parser.parse_args()


@torch.no_grad()
def collect_frequency_tensors(
    model: torch.nn.Module,
    windows: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(windows), torch.from_numpy(masks)),
        batch_size=batch_size,
        shuffle=False,
    )

    collected: dict[str, list[np.ndarray]] = {
        "attention_weight": [],
        "spectrum_real": [],
        "spectrum_imag": [],
        "spectrum_magnitude": [],
        "delta_real": [],
        "delta_imag": [],
        "delta_magnitude": [],
        "enhanced_real": [],
        "enhanced_imag": [],
        "enhanced_magnitude": [],
    }

    model.eval()
    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        x_seed = PreImpute.fill(x_input, structural_missing_mask)

        x_perm = x_seed.permute(0, 2, 1)
        spectrum = torch.fft.rfft(x_perm, dim=2)
        freq_len = spectrum.size(-1)
        real = spectrum.real
        imag = spectrum.imag
        freq_feature = torch.cat([real, imag], dim=-1)

        attention = model.freq.attention(freq_feature)
        enhancement = model.freq.freq_enhance(freq_feature)
        delta_real = enhancement[..., :freq_len] * attention
        delta_imag = enhancement[..., freq_len:] * attention
        delta = torch.complex(delta_real, delta_imag)
        enhanced_spectrum = spectrum + delta

        collected["attention_weight"].append(attention.cpu().numpy().astype(np.float32))
        collected["spectrum_real"].append(real.cpu().numpy().astype(np.float32))
        collected["spectrum_imag"].append(imag.cpu().numpy().astype(np.float32))
        collected["spectrum_magnitude"].append(
            torch.abs(spectrum).cpu().numpy().astype(np.float32)
        )
        collected["delta_real"].append(delta_real.cpu().numpy().astype(np.float32))
        collected["delta_imag"].append(delta_imag.cpu().numpy().astype(np.float32))
        collected["delta_magnitude"].append(
            torch.abs(delta).cpu().numpy().astype(np.float32)
        )
        collected["enhanced_real"].append(
            enhanced_spectrum.real.cpu().numpy().astype(np.float32)
        )
        collected["enhanced_imag"].append(
            enhanced_spectrum.imag.cpu().numpy().astype(np.float32)
        )
        collected["enhanced_magnitude"].append(
            torch.abs(enhanced_spectrum).cpu().numpy().astype(np.float32)
        )

    return {
        key: np.concatenate(value, axis=0).astype(np.float32)
        for key, value in collected.items()
    }


def resolve_low_frequency_indices(
    freq_len: int,
    low_freq_count: int,
    include_dc: bool,
) -> np.ndarray:
    if low_freq_count < 1:
        raise ValueError(f"--low-freq-count 必须 >= 1，收到 {low_freq_count}")

    start = 0 if include_dc else 1
    stop = min(freq_len, start + low_freq_count)
    if start >= stop:
        raise ValueError(
            f"频谱长度为 {freq_len}，无法构造低频索引；"
            "请减小 --low-freq-count 或启用 --include-dc。"
        )
    return np.arange(start, stop, dtype=np.int64)


def summarize_window_features(
    tensors: dict[str, np.ndarray],
    masks: np.ndarray,
    feature_names: list[str],
    stride: int,
    seq_len: int,
    low_indices: np.ndarray,
) -> pd.DataFrame:
    spectrum_magnitude = tensors["spectrum_magnitude"]
    enhanced_magnitude = tensors["enhanced_magnitude"]
    delta_magnitude = tensors["delta_magnitude"]
    attention_weight = tensors["attention_weight"]

    freq_len = spectrum_magnitude.shape[-1]
    high_indices = np.setdiff1d(np.arange(freq_len), low_indices)
    if high_indices.size == 0:
        high_indices = low_indices

    low_original = spectrum_magnitude[:, :, low_indices].mean(axis=-1)
    low_enhanced = enhanced_magnitude[:, :, low_indices].mean(axis=-1)
    low_gain = (enhanced_magnitude[:, :, low_indices] - spectrum_magnitude[:, :, low_indices]).mean(axis=-1)
    low_gain_ratio = (
        enhanced_magnitude[:, :, low_indices].sum(axis=-1)
        / np.maximum(spectrum_magnitude[:, :, low_indices].sum(axis=-1), 1e-6)
    )
    low_delta = delta_magnitude[:, :, low_indices].mean(axis=-1)
    low_attention = attention_weight[:, :, low_indices].mean(axis=-1)

    high_original = spectrum_magnitude[:, :, high_indices].mean(axis=-1)
    high_enhanced = enhanced_magnitude[:, :, high_indices].mean(axis=-1)
    high_gain = (enhanced_magnitude[:, :, high_indices] - spectrum_magnitude[:, :, high_indices]).mean(axis=-1)
    high_delta = delta_magnitude[:, :, high_indices].mean(axis=-1)
    high_attention = attention_weight[:, :, high_indices].mean(axis=-1)

    delta_low_high_ratio = low_delta / np.maximum(high_delta, 1e-6)
    low_gain_minus_high_gain = low_gain - high_gain
    # 排名偏向“低频修正量更大，且修正比高频更集中”的候选。
    score = (
        low_delta
        * np.maximum(delta_low_high_ratio, 0.0)
        * (0.5 + low_attention)
    )

    rows = []
    window_count, feature_count = low_gain.shape
    observed_count = (1.0 - masks).sum(axis=1)
    missing_count = masks.sum(axis=1)

    for window_index in range(window_count):
        end_row_index = window_index * stride
        start_row_index = end_row_index - seq_len + 1
        for feature_index in range(feature_count):
            rows.append(
                {
                    "window_index": window_index,
                    "raw_window_start_row_index": start_row_index,
                    "raw_window_end_row_index": end_row_index,
                    "feature_index_1based": feature_index + 1,
                    "feature_name": feature_names[feature_index],
                    "observed_count": int(observed_count[window_index, feature_index]),
                    "missing_count": int(missing_count[window_index, feature_index]),
                    "low_original_magnitude_mean": float(low_original[window_index, feature_index]),
                    "low_enhanced_magnitude_mean": float(low_enhanced[window_index, feature_index]),
                    "low_magnitude_gain_mean": float(low_gain[window_index, feature_index]),
                    "low_magnitude_gain_ratio": float(low_gain_ratio[window_index, feature_index]),
                    "low_delta_magnitude_mean": float(low_delta[window_index, feature_index]),
                    "high_original_magnitude_mean": float(high_original[window_index, feature_index]),
                    "high_enhanced_magnitude_mean": float(high_enhanced[window_index, feature_index]),
                    "high_magnitude_gain_mean": float(high_gain[window_index, feature_index]),
                    "high_delta_magnitude_mean": float(high_delta[window_index, feature_index]),
                    "delta_low_high_ratio": float(delta_low_high_ratio[window_index, feature_index]),
                    "low_gain_minus_high_gain": float(low_gain_minus_high_gain[window_index, feature_index]),
                    "low_attention_mean": float(low_attention[window_index, feature_index]),
                    "high_attention_mean": float(high_attention[window_index, feature_index]),
                    "low_correction_score": float(score[window_index, feature_index]),
                    "low_frequency_score": float(score[window_index, feature_index]),
                }
            )

    return pd.DataFrame(rows)


def build_frequency_detail_rows(
    tensors: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    top_df = summary_df.head(top_k)
    rows = []
    freq_count = tensors["attention_weight"].shape[-1]
    for rank, item in enumerate(top_df.itertuples(index=False), start=1):
        window_index = int(item.window_index)
        feature_index = int(item.feature_index_1based) - 1
        for frequency_index in range(freq_count):
            rows.append(
                {
                    "rank": rank,
                    "window_index": window_index,
                    "feature_index_1based": feature_index + 1,
                    "feature_name": item.feature_name,
                    "frequency_index": frequency_index,
                    "attention_weight": float(
                        tensors["attention_weight"][window_index, feature_index, frequency_index]
                    ),
                    "spectrum_real": float(
                        tensors["spectrum_real"][window_index, feature_index, frequency_index]
                    ),
                    "spectrum_imag": float(
                        tensors["spectrum_imag"][window_index, feature_index, frequency_index]
                    ),
                    "spectrum_magnitude": float(
                        tensors["spectrum_magnitude"][window_index, feature_index, frequency_index]
                    ),
                    "delta_real": float(
                        tensors["delta_real"][window_index, feature_index, frequency_index]
                    ),
                    "delta_imag": float(
                        tensors["delta_imag"][window_index, feature_index, frequency_index]
                    ),
                    "delta_magnitude": float(
                        tensors["delta_magnitude"][window_index, feature_index, frequency_index]
                    ),
                    "enhanced_real": float(
                        tensors["enhanced_real"][window_index, feature_index, frequency_index]
                    ),
                    "enhanced_imag": float(
                        tensors["enhanced_imag"][window_index, feature_index, frequency_index]
                    ),
                    "enhanced_magnitude": float(
                        tensors["enhanced_magnitude"][window_index, feature_index, frequency_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_table(
    summary_df: pd.DataFrame,
    top_k: int,
    min_start_row: int,
    min_missing_count: int,
    min_delta_low_high_ratio: float,
) -> pd.DataFrame:
    candidate_df = summary_df[
        (summary_df["raw_window_start_row_index"] >= min_start_row)
        & (summary_df["missing_count"] >= min_missing_count)
        & (summary_df["low_delta_magnitude_mean"] > 0.0)
        & (summary_df["delta_low_high_ratio"] >= min_delta_low_high_ratio)
    ].copy()

    if candidate_df.empty:
        candidate_df = summary_df[
            (summary_df["raw_window_start_row_index"] >= min_start_row)
            & (summary_df["missing_count"] >= min_missing_count)
        ].copy()

    candidate_df = candidate_df.head(top_k).copy()
    candidate_df.insert(0, "rank", np.arange(1, len(candidate_df) + 1))
    return candidate_df


def save_long_frequency_csv(
    tensors: dict[str, np.ndarray],
    feature_names: list[str],
    output_path: Path,
) -> None:
    window_count, feature_count, freq_count = tensors["attention_weight"].shape
    chunks = []
    for feature_index in range(feature_count):
        window_idx, freq_idx = np.meshgrid(
            np.arange(window_count, dtype=np.int64),
            np.arange(freq_count, dtype=np.int64),
            indexing="ij",
        )
        chunks.append(
            pd.DataFrame(
                {
                    "window_index": window_idx.ravel(),
                    "feature_index_1based": feature_index + 1,
                    "feature_name": feature_names[feature_index],
                    "frequency_index": freq_idx.ravel(),
                    "attention_weight": tensors["attention_weight"][:, feature_index, :].ravel(),
                    "spectrum_magnitude": tensors["spectrum_magnitude"][:, feature_index, :].ravel(),
                    "delta_magnitude": tensors["delta_magnitude"][:, feature_index, :].ravel(),
                    "enhanced_magnitude": tensors["enhanced_magnitude"][:, feature_index, :].ravel(),
                }
            )
        )
    pd.concat(chunks, ignore_index=True).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    checkpoint = load_checkpoint(args.checkpoint, allow_random_init=False)
    ckpt_args = checkpoint_args(checkpoint)
    train_glob = args.train_glob or ckpt_args.get("train_glob", "data/train/train_1.csv")
    seq_len = int(args.seq_len or ckpt_args.get("seq_len", 50))
    stride = int(args.stride or ckpt_args.get("stride", 1))
    device = choose_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data, missing_mask, feature_names = load_csv_glob_with_mask(train_glob)
    scaler = ObservedStandardScaler().fit(raw_data, missing_mask)
    scaled_data = scaler.transform(raw_data, missing_mask)
    windows, window_masks = build_windows(
        scaled_data,
        missing_mask.astype(np.float32),
        seq_len=seq_len,
        stride=stride,
    )

    model, model_source = load_frequency_model(
        checkpoint=checkpoint,
        num_features=scaled_data.shape[1],
        seq_len=seq_len,
        device=device,
    )

    tensors = collect_frequency_tensors(
        model=model,
        windows=windows,
        masks=window_masks,
        device=device,
        batch_size=args.batch_size,
    )
    low_indices = resolve_low_frequency_indices(
        freq_len=tensors["attention_weight"].shape[-1],
        low_freq_count=args.low_freq_count,
        include_dc=args.include_dc,
    )

    tensor_path = output_dir / "frequency_module_tensors.npz"
    np.savez_compressed(
        tensor_path,
        **tensors,
        feature_names=np.asarray(feature_names),
        window_indices=np.arange(windows.shape[0], dtype=np.int64),
        frequency_indices=np.arange(tensors["attention_weight"].shape[-1], dtype=np.int64),
        raw_window_end_row_indices=np.arange(windows.shape[0], dtype=np.int64) * stride,
        low_frequency_indices=low_indices,
    )

    summary_df = summarize_window_features(
        tensors=tensors,
        masks=window_masks,
        feature_names=feature_names,
        stride=stride,
        seq_len=seq_len,
        low_indices=low_indices,
    )
    summary_df = summary_df.sort_values(
        by=[
            "low_correction_score",
            "low_delta_magnitude_mean",
            "delta_low_high_ratio",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_path = output_dir / "window_feature_low_frequency_summary.csv"
    top_path = output_dir / "top_low_frequency_candidates.csv"
    detail_path = output_dir / "top_candidate_frequency_values.csv"
    top_missing_path = output_dir / "top_low_frequency_candidates_with_missing.csv"
    missing_detail_path = output_dir / "top_missing_candidate_frequency_values.csv"
    metadata_path = output_dir / "metadata.json"
    long_csv_path = output_dir / "frequency_module_long_values.csv"

    summary_df.to_csv(summary_path, index=False)
    top_df = build_candidate_table(
        summary_df=summary_df,
        top_k=args.top_k,
        min_start_row=args.candidate_min_start_row,
        min_missing_count=args.candidate_min_missing_count,
        min_delta_low_high_ratio=args.min_delta_low_high_ratio,
    )
    top_df.to_csv(top_path, index=False)
    detail_df = build_frequency_detail_rows(
        tensors,
        top_df.drop(columns=["rank"]) if "rank" in top_df else top_df,
        args.top_k,
    )
    detail_df.to_csv(detail_path, index=False)

    top_missing_df = build_candidate_table(
        summary_df=summary_df,
        top_k=args.top_k,
        min_start_row=args.candidate_min_start_row,
        min_missing_count=max(1, args.candidate_min_missing_count),
        min_delta_low_high_ratio=args.min_delta_low_high_ratio,
    )
    top_missing_df.to_csv(top_missing_path, index=False)
    missing_detail_df = build_frequency_detail_rows(
        tensors,
        (
            top_missing_df.drop(columns=["rank"])
            if "rank" in top_missing_df
            else top_missing_df
        ),
        args.top_k,
    )
    missing_detail_df.to_csv(missing_detail_path, index=False)

    if args.save_long_csv:
        save_long_frequency_csv(tensors, feature_names, long_csv_path)

    metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_epochs": ckpt_args.get("epochs"),
        "model_source": model_source,
        "train_glob": train_glob,
        "seq_len": seq_len,
        "stride": stride,
        "device": str(device),
        "window_count": int(windows.shape[0]),
        "feature_count": int(windows.shape[2]),
        "frequency_count": int(tensors["attention_weight"].shape[-1]),
        "low_frequency_indices": low_indices.tolist(),
        "include_dc": bool(args.include_dc),
        "low_freq_count": int(args.low_freq_count),
        "tensor_npz": str(tensor_path),
        "summary_csv": str(summary_path),
        "top_candidates_csv": str(top_path),
        "top_candidate_frequency_values_csv": str(detail_path),
        "top_candidates_with_missing_csv": str(top_missing_path),
        "top_missing_candidate_frequency_values_csv": str(missing_detail_path),
        "candidate_min_start_row": int(args.candidate_min_start_row),
        "candidate_min_missing_count": int(args.candidate_min_missing_count),
        "min_delta_low_high_ratio": float(args.min_delta_low_high_ratio),
        "selection_metric": "low_delta_magnitude_mean prioritized by delta_low_high_ratio",
        "long_csv": str(long_csv_path) if args.save_long_csv else None,
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("频域模块统计已输出:")
    print(f"  输出目录: {output_dir}")
    print(f"  checkpoint epochs: {ckpt_args.get('epochs')}")
    print(f"  低频索引: {low_indices.tolist()}")
    print(f"  全量张量: {tensor_path}")
    print(f"  全窗口汇总: {summary_path}")
    print(f"  Top 候选: {top_path}")
    print(f"  Top 候选频率明细: {detail_path}")
    print(f"  含缺失点 Top 候选: {top_missing_path}")
    print(f"  含缺失点候选频率明细: {missing_detail_path}")
    if args.save_long_csv:
        print(f"  全量长表 CSV: {long_csv_path}")
    if not top_df.empty:
        best = top_df.iloc[0]
        print("最强低频修正候选:")
        print(
            "  "
            f"rank=1, feature={best['feature_name']}({int(best['feature_index_1based'])}), "
            f"window={int(best['window_index'])}, "
            f"rows={int(best['raw_window_start_row_index'])}..{int(best['raw_window_end_row_index'])}, "
            f"low_delta={best['low_delta_magnitude_mean']:.6f}, "
            f"high_delta={best['high_delta_magnitude_mean']:.6f}, "
            f"low_gain={best['low_magnitude_gain_mean']:.6f}, "
            f"delta_low/high={best['delta_low_high_ratio']:.6f}"
        )
    if not top_missing_df.empty:
        best_missing = top_missing_df.iloc[0]
        print("含缺失点的最强低频修正候选:")
        print(
            "  "
            f"rank=1, feature={best_missing['feature_name']}({int(best_missing['feature_index_1based'])}), "
            f"window={int(best_missing['window_index'])}, "
            f"rows={int(best_missing['raw_window_start_row_index'])}..{int(best_missing['raw_window_end_row_index'])}, "
            f"missing={int(best_missing['missing_count'])}, "
            f"low_delta={best_missing['low_delta_magnitude_mean']:.6f}, "
            f"high_delta={best_missing['high_delta_magnitude_mean']:.6f}, "
            f"low_gain={best_missing['low_magnitude_gain_mean']:.6f}, "
            f"delta_low/high={best_missing['delta_low_high_ratio']:.6f}"
        )


if __name__ == "__main__":
    main()
