from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


_METRIC_LABELS = {
    "threshold": "Threshold",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "fdr": "FDR",
    "fra": "FRA",
    "f1": "F1-Score",
    "specificity": "Specificity",
    "TP": "TP",
    "TN": "TN",
    "FP": "FP",
    "FN": "FN",
}


_NON_INTERACTIVE_BACKENDS = {
    "agg",
    "cairo",
    "pdf",
    "pgf",
    "ps",
    "svg",
    "template",
    "module://matplotlib_inline.backend_inline",
}


_SONGTI_FONT_CANDIDATES = [
    "SimSun",
    "宋体",
    "NSimSun",
    "新宋体",
    "Songti SC",
    "Noto Serif CJK SC",
    "Noto Serif CJK JP",
    "Source Han Serif SC",
    "DejaVu Serif",
]


def configure_songti_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next(
        (name for name in _SONGTI_FONT_CANDIDATES if name in available),
        "DejaVu Serif",
    )
    plt.rcParams["font.family"] = [selected]
    plt.rcParams["font.serif"] = [selected, "DejaVu Serif"]
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _can_show_interactive_figure() -> bool:
    backend = str(matplotlib.get_backend()).lower()
    if backend not in _NON_INTERACTIVE_BACKENDS:
        return True

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False

    for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            plt.switch_backend(candidate)
            return True
        except Exception:
            continue

    return False


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float | None = None,
    include_specificity: bool = False,
    include_counts: bool = False,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    metrics: dict[str, float | int] = {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fdr": float(tp / max(tp + fn, 1)),
        "fra": float(fp / max(fp + tn, 1)),
        "f1": float((2 * tp) / max(2 * tp + fp + fn, 1)),
    }

    if threshold is not None:
        metrics["threshold"] = float(threshold)
    if include_specificity:
        metrics["specificity"] = float(tn / max(tn + fp, 1))
    if include_counts:
        metrics.update({"TP": tp, "TN": tn, "FP": fp, "FN": fn})
    return metrics


def print_metrics(
    title: str,
    metrics: dict[str, float | int],
    *,
    order: list[str] | None = None,
) -> None:
    if title:
        print(title)

    keys = order or [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "fdr",
        "fra",
        "f1",
        "specificity",
        "TP",
        "TN",
        "FP",
        "FN",
    ]
    for key in keys:
        if key not in metrics:
            continue
        label = _METRIC_LABELS.get(key, key)
        value = metrics[key]
        if isinstance(value, (int, np.integer)):
            print(f"  {label}: {int(value)}")
        else:
            print(f"  {label}: {float(value):.4f}")


def plot_detection_scores(
    scores: np.ndarray,
    threshold: float,
    split_idx: int,
    save_path: str | Path,
    *,
    title: str | None = None,
    ylabel: str = "重构误差",
    xlabel: str = "测试样本索引",
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
    style: str = "compact",
    color_scheme: str = "default",
    score_label: str = "异常分数",
    threshold_label_fmt: str = "阈值",
    split_label: str = "测试集分界",
    label_fontsize: float = 14,
    tick_fontsize: float = 12,
    legend_fontsize: float = 15,
    show: bool = False,
) -> None:
    configure_songti_font()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    can_show = _can_show_interactive_figure() if show else False

    fig, ax = plt.subplots(figsize=figsize)
    x_axis = np.arange(1, len(scores) + 1)
    mra_score_color = "#0B6E4F"
    mra_threshold_color = "#D1495B"
    mra_split_color = "#222222"

    if style == "mra":
        ax.plot(x_axis, scores, color=mra_score_color, linewidth=1.6, label=score_label)
        ax.axhline(
            threshold,
            color=mra_threshold_color,
            linestyle="--",
            linewidth=1.4,
            label=threshold_label_fmt.format(threshold=threshold),
        )
        ax.axvline(
            split_idx,
            color=mra_split_color,
            linestyle="--",
            linewidth=1.2,
            label=split_label,
        )
        if split_idx > 0:
            ax.fill_between(
                x_axis[:split_idx],
                scores[:split_idx],
                alpha=0.08,
                color="#2A9D8F",
            )
        if split_idx < len(scores):
            start = max(split_idx - 1, 0)
            ax.fill_between(
                x_axis[start:],
                scores[start:],
                alpha=0.08,
                color="#E76F51",
            )
        ax.set_xlabel("窗口索引", fontsize=label_fontsize)
        ax.set_ylabel("分数", fontsize=label_fontsize)
    else:
        use_mra_colors = color_scheme == "mra"
        ax.plot(
            scores,
            color=mra_score_color if use_mra_colors else None,
            linewidth=1.6 if use_mra_colors else None,
            label=score_label,
            alpha=0.7,
        )
        ax.axhline(
            y=threshold,
            color=mra_threshold_color if use_mra_colors else "r",
            linestyle="--",
            linewidth=1.4 if use_mra_colors else None,
            label=threshold_label_fmt.format(threshold=threshold),
        )
        ax.axvline(
            x=split_idx,
            color=mra_split_color if use_mra_colors else "g",
            linestyle="--" if use_mra_colors else ":",
            linewidth=1.2 if use_mra_colors else None,
            label=split_label,
        )
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if title:
        ax.set_title(title, fontsize=label_fontsize + 1)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3 if style != "mra" else 0.2)
    legend_loc = "upper left" if style == "mra" or color_scheme == "mra" else None
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    print(f"\nPlot saved to: {save_path}")
    if show and can_show:
        plt.show()
    elif show:
        print("Interactive display skipped: no GUI backend/display is available.")
    plt.close(fig)


def save_training_curve(
    history: list[dict[str, float]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history) + 1)
    total = [item["total_loss"] for item in history]
    fusion = [item["fusion_loss"] for item in history]
    detector = [item["detector_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, total, label="总损失", color="#1D3557")
    ax.plot(epochs, fusion, label="插补损失", color="#457B9D")
    ax.plot(epochs, detector, label="检测损失", color="#E63946")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练曲线")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
