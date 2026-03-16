#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

EXPECTED_FEATURES = 18
FEATURE_NAMES = [f"特征{i:02d}" for i in range(1, EXPECTED_FEATURES + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 data/train 下的时序 CSV 样本执行 FFT 并生成中文可视化。"
    )
    parser.add_argument(
        "--input-glob",
        default="data/train/*.csv",
        help="输入文件匹配模式，默认: data/train/*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fft_visualizations",
        help="图片输出目录，默认: outputs/fft_visualizations",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="采样时间间隔，默认 1.0",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="图片 DPI，默认 180",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多处理多少个文件，默认处理全部",
    )
    return parser.parse_args()


def configure_chinese_font() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    selected = None
    for name in candidates:
        if name in available_fonts:
            selected = name
            break

    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


def load_sample(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{csv_path} 的列数为 {df.shape[1]}，预期应为 {EXPECTED_FEATURES}。"
        )

    df.columns = FEATURE_NAMES
    return df


def preprocess_series(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(series, errors="coerce")
    filled = numeric.interpolate(method="linear", limit_direction="both")

    if filled.isna().any():
        fallback = 0.0 if filled.dropna().empty else float(filled.dropna().mean())
        filled = filled.fillna(fallback)

    values = filled.to_numpy(dtype=float)
    centered = values - values.mean()
    return values, centered


def compute_fft(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum) / n

    if len(magnitude) > 2:
        magnitude[1:-1] *= 2

    return freqs, magnitude


def create_grid_figure(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(6, 3, figsize=(18, 20), constrained_layout=False)
    fig.suptitle(title, fontsize=20)
    return fig, axes.flatten()


def save_time_domain_figure(
    df: pd.DataFrame, sample_name: str, output_path: Path, dpi: int
) -> None:
    fig, axes = create_grid_figure(f"{sample_name} 时域信号")

    for idx, column in enumerate(df.columns):
        values, _ = preprocess_series(df[column])
        ax = axes[idx]
        ax.plot(values, color="#1f77b4", linewidth=0.9)
        ax.set_title(column, fontsize=11)
        ax.set_xlabel("时间步", fontsize=9)
        ax.set_ylabel("数值", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)

    for ax in axes[len(df.columns) :]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def dominant_frequency_text(freqs: np.ndarray, magnitude: np.ndarray) -> str:
    if len(freqs) <= 1:
        return "主频: 0.000"

    peak_idx = int(np.argmax(magnitude[1:]) + 1)
    return f"主频: {freqs[peak_idx]:.4f}"


def save_fft_figure(
    df: pd.DataFrame, sample_name: str, output_path: Path, dt: float, dpi: int
) -> None:
    fig, axes = create_grid_figure(f"{sample_name} FFT 频谱")

    for idx, column in enumerate(df.columns):
        _, centered = preprocess_series(df[column])
        freqs, magnitude = compute_fft(centered, dt)

        ax = axes[idx]
        ax.plot(freqs, magnitude, color="#d62728", linewidth=0.9)
        ax.set_title(column, fontsize=11)
        ax.set_xlabel("频率", fontsize=9)
        ax.set_ylabel("幅值", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.text(
            0.98,
            0.92,
            dominant_frequency_text(freqs, magnitude),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75},
        )

    for ax in axes[len(df.columns) :]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_file(csv_path: Path, output_dir: Path, dt: float, dpi: int) -> None:
    sample_name = csv_path.stem
    df = load_sample(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    time_output = output_dir / f"{sample_name}_time_domain.png"
    fft_output = output_dir / f"{sample_name}_fft_spectrum.png"

    save_time_domain_figure(df, sample_name, time_output, dpi)
    save_fft_figure(df, sample_name, fft_output, dt, dpi)

    print(f"[完成] {csv_path.name}")
    print(f"  时域图: {time_output}")
    print(f"  FFT图 : {fft_output}")


def main() -> None:
    args = parse_args()
    configure_chinese_font()

    csv_paths = sorted(Path().glob(args.input_glob))
    if args.max_files is not None:
        csv_paths = csv_paths[: args.max_files]

    if not csv_paths:
        raise SystemExit(f"没有找到匹配文件: {args.input_glob}")

    output_dir = Path(args.output_dir)
    print(f"共找到 {len(csv_paths)} 个文件，开始处理。")

    for csv_path in csv_paths:
        process_file(csv_path, output_dir, args.dt, args.dpi)

    print("全部处理完成。")


if __name__ == "__main__":
    main()
