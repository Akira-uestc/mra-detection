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
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paper-ready waveform -> FFT spectrum -> reconstruction figure."
    )
    parser.add_argument(
        "--output-dir",
        default="Docs/describe/figures",
        help="Output directory for PNG/PDF/SVG files.",
    )
    parser.add_argument(
        "--stem",
        default="fft_reconstruction_pipeline",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Number of waveform samples used for the FFT.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for the PNG export.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    candidates = [
        "SimSun",
        "宋体",
        "NSimSun",
        "新宋体",
        "Songti SC",
        "Noto Serif CJK SC",
        "FangSong",
        "I.Ming",
        "Noto Serif",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available_fonts), None)

    if selected:
        plt.rcParams["font.family"] = selected
        plt.rcParams["font.serif"] = [selected, "DejaVu Serif"]
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["DejaVu Serif"]

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 10.5,
        }
    )


def build_waveform(samples: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    signal = (
        0.82 * np.sin(2 * np.pi * 4.0 * t)
        + 0.42 * np.sin(2 * np.pi * 11.0 * t + 0.65)
        + 0.24 * np.cos(2 * np.pi * 23.0 * t - 0.20)
        + 0.34 * np.exp(-0.5 * ((t - 0.62) / 0.045) ** 2)
        - 0.25 * np.exp(-0.5 * ((t - 0.34) / 0.025) ** 2)
        + 0.16 * (t - 0.5)
    )
    signal = signal / np.max(np.abs(signal))
    return t, signal


def build_missing_mask(t: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    intervals = [(0.18, 0.25), (0.51, 0.58), (0.77, 0.83)]
    mask = np.zeros_like(t, dtype=bool)
    for start, end in intervals:
        mask |= (t >= start) & (t <= end)

    isolated = np.array([0.115, 0.305, 0.438, 0.682, 0.902])
    nearest = np.abs(t[:, None] - isolated[None, :]).argmin(axis=0)
    mask[nearest] = True
    return mask, intervals


def fill_missing_linear(t: np.ndarray, values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    observed = ~missing
    if not np.any(observed):
        raise ValueError("At least one observed sample is required for interpolation.")
    return np.interp(t, t[observed], values[observed])


def compute_fft(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    magnitude = np.abs(spectrum) / signal.size
    if magnitude.size > 2:
        magnitude[1:-1] *= 2
    return freqs, magnitude, spectrum


def reconstruct_waveform(spectrum: np.ndarray, samples: int) -> np.ndarray:
    return np.fft.irfft(spectrum, n=samples)


def style_plot_axis(ax: plt.Axes, caption: str) -> None:
    ax.set_title(caption, pad=13, fontsize=14, fontweight="normal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#344054")
    ax.grid(True, color="#e4e7ec", linewidth=0.7, alpha=0.9)
    ax.tick_params(axis="both", labelsize=9.5, colors="#475467", length=3)


def draw_arrow_axis(ax: plt.Axes, top_label: str, bottom_label: str) -> None:
    ax.set_axis_off()
    arrow = FancyArrowPatch(
        (0.08, 0.50),
        (0.92, 0.50),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=2.0,
        color="#2f3a4a",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    ax.text(
        0.50,
        0.65,
        top_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="normal",
        color="#1d2939",
    )
    ax.text(
        0.50,
        0.34,
        bottom_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#667085",
    )


def create_figure(size: int) -> tuple[plt.Figure, float]:
    t, original = build_waveform(size)
    missing, intervals = build_missing_mask(t)
    observed = original.copy()
    observed[missing] = np.nan

    filled = fill_missing_linear(t, original, missing)
    dt = float(t[1] - t[0])
    freqs, magnitude, spectrum = compute_fft(filled, dt)
    reconstructed = reconstruct_waveform(spectrum, size)
    mse = float(np.mean((filled - reconstructed) ** 2))

    y_min = min(np.nanmin(observed), np.min(reconstructed), np.min(filled)) - 0.12
    y_max = max(np.nanmax(observed), np.max(reconstructed), np.max(filled)) + 0.12

    fig = plt.figure(figsize=(11.6, 3.45), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 0.28, 1.0, 0.28, 1.0],
        left=0.035,
        right=0.965,
        top=0.84,
        bottom=0.15,
        wspace=0.08,
    )

    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    plot_axes = [axes[0], axes[2], axes[4]]

    for start, end in intervals:
        plot_axes[0].axvspan(start, end, color="#f04438", alpha=0.12, lw=0)
    plot_axes[0].plot(t, observed, color="#1f5aa6", linewidth=1.8, label="观测波形")
    plot_axes[0].scatter(
        t[missing],
        np.full(np.count_nonzero(missing), y_min + 0.07 * (y_max - y_min)),
        marker="|",
        s=32,
        color="#d92d20",
        linewidths=1.1,
        label="缺失值",
        zorder=4,
    )
    plot_axes[0].set_xlim(0.0, 1.0)
    plot_axes[0].set_ylim(y_min, y_max)
    plot_axes[0].set_xlabel("时间")
    plot_axes[0].set_ylabel("幅值")

    freq_limit = 34.0
    shown = freqs <= freq_limit
    plot_axes[1].plot(freqs[shown], magnitude[shown], color="#a1123d", linewidth=1.7)
    plot_axes[1].fill_between(
        freqs[shown],
        magnitude[shown],
        color="#d92d20",
        alpha=0.16,
        linewidth=0,
    )
    plot_axes[1].set_xlim(0.0, freq_limit)
    plot_axes[1].set_ylim(0.0, float(np.max(magnitude[shown])) * 1.18)
    plot_axes[1].set_xlabel("频率")
    plot_axes[1].set_ylabel("幅值")

    plot_axes[2].plot(t, reconstructed, color="#087443", linewidth=1.8)
    plot_axes[2].set_xlim(0.0, 1.0)
    plot_axes[2].set_ylim(y_min, y_max)
    plot_axes[2].set_xlabel("时间")
    plot_axes[2].set_ylabel("幅值")

    style_plot_axis(plot_axes[0], "原始波形（含缺失值）")
    style_plot_axis(plot_axes[1], "FFT 后的频谱")
    style_plot_axis(plot_axes[2], "重构后的波形")

    draw_arrow_axis(axes[1], "FFT", r"$\mathcal{F}\{x(t)\}$")
    draw_arrow_axis(axes[3], "IFFT", r"$\mathcal{F}^{-1}\{X(f)\}$")

    return fig, mse


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, mse = create_figure(args.size)
    for suffix in ("png", "pdf", "svg"):
        output_path = output_dir / f"{args.stem}.{suffix}"
        save_kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.05}
        if suffix == "png":
            save_kwargs["dpi"] = args.dpi
        fig.savefig(output_path, **save_kwargs)

    plt.close(fig)
    print(f"Saved figure set to: {output_dir}")
    print(f"Reconstruction MSE: {mse:.6e}")


if __name__ == "__main__":
    main()
