#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freq_reconstruction import FrequencyOnlyReconstructor
from graph_gcn_reconstruction import ObservedStandardScaler, seed_everything
from mra import PreImpute
from utils.methods.data_loading import load_csv_glob_with_mask
from utils.methods.windowing import build_windows

DEFAULT_CHECKPOINT = "outputs/gcn_freq_fusion_transformer_detection/model.pt"

BLUE = "#2f6db3"
ORANGE = "#f28e2b"
RED = "#d62728"
BLUE_PURPLE = "#5b5fc7"
DARK_GREEN = "#006b3c"
DASHED_GRAY = "#6b7280"
GRID = "#d9dee8"
SPINE = "#344054"
TEXT = "#1f2937"

LABEL_SIZE = 24
TICK_SIZE = 18
LEGEND_SIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用 mra.py 的频域专家和真实训练数据，导出可放入论文的频域模块直观图。"
        )
    )
    parser.add_argument(
        "--train-glob",
        default=None,
        help="训练 CSV 匹配模式。默认优先读取 checkpoint 中的 train_glob，否则 data/train/train_1.csv。",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"MRA checkpoint 路径。默认: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mra_frequency_module_figures",
        help="图片和中间数据输出目录。默认: outputs/mra_frequency_module_figures",
    )
    parser.add_argument(
        "--feature-index",
        type=int,
        default=10,
        help="要绘制的变量编号，1-based。默认: 10",
    )
    parser.add_argument(
        "--feature-name",
        default=None,
        help="可选：用变量名指定变量，例如 特征10；若设置则优先于 --feature-index。",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=None,
        help="可选：训练窗口编号，0-based。默认自动选择该变量观测点较充足的窗口。",
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
        "--min-observed-count",
        type=int,
        default=None,
        help="自动选窗时该变量在窗口内的最少观测点数。默认 max(3, seq_len//4)。",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        help="图片格式，可选 png pdf svg。默认同时导出三种格式。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG 输出 DPI。默认: 600",
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
        help="随机种子；仅在允许随机初始化时影响模型初始化。默认: 40",
    )
    parser.add_argument(
        "--epoch-label",
        default=None,
        help=(
            "图例中的 epoch 标注。默认从 checkpoint 的 args['epochs'] 推断，"
            "例如 Epoch 12；传入空字符串可不显示。"
        ),
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="checkpoint 缺失时仍用随机初始化频域专家生成图。论文制图不建议开启。",
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
        "SimHei",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), None)

    if selected is None:
        selected = "DejaVu Serif"
        print("警告: 未找到宋体/SimSun，已退回 DejaVu Serif。")

    plt.rcParams.update(
        {
            "font.family": selected,
            "font.serif": [selected, "DejaVu Serif"],
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "font.size": TICK_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_formats(formats: Iterable[str]) -> list[str]:
    valid = {"png", "pdf", "svg"}
    normalized = []
    for item in formats:
        suffix = item.lower().lstrip(".")
        if suffix not in valid:
            raise ValueError(f"不支持的图片格式: {item}，可选 {sorted(valid)}")
        if suffix not in normalized:
            normalized.append(suffix)
    return normalized


def load_checkpoint(path_text: str | None, allow_random_init: bool) -> dict | None:
    if not path_text:
        if allow_random_init:
            return None
        raise FileNotFoundError("未提供 checkpoint；若要随机初始化，请显式添加 --allow-random-init。")

    checkpoint_path = Path(path_text)
    if not checkpoint_path.exists():
        if allow_random_init:
            print(f"警告: checkpoint 不存在，使用随机初始化: {checkpoint_path}")
            return None
        raise FileNotFoundError(
            f"checkpoint 不存在: {checkpoint_path}。请先运行 mra.py 训练，"
            "或指定 --checkpoint，或显式添加 --allow-random-init。"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 格式不受支持: {checkpoint_path}")
    return checkpoint


def checkpoint_args(checkpoint: dict | None) -> dict:
    if checkpoint is None:
        return {}
    raw_args = checkpoint.get("args", {})
    return raw_args if isinstance(raw_args, dict) else {}


def resolve_epoch_label(requested_label: str | None, ckpt_args: dict) -> str | None:
    if requested_label is not None:
        label = requested_label.strip()
        return label or None

    checkpoint_epochs = ckpt_args.get("epochs")
    if checkpoint_epochs is None:
        return None
    return f"Epoch {checkpoint_epochs}"


def resolve_feature_index(
    feature_names: list[str],
    feature_index_1based: int,
    feature_name: str | None,
) -> int:
    if feature_name:
        if feature_name not in feature_names:
            raise ValueError(f"变量名 {feature_name} 不存在，可选: {feature_names}")
        return feature_names.index(feature_name)

    feature_index = feature_index_1based - 1
    if feature_index < 0 or feature_index >= len(feature_names):
        raise ValueError(
            f"--feature-index={feature_index_1based} 越界，"
            f"当前训练数据共有 {len(feature_names)} 个变量。"
        )
    return feature_index


def choose_window_index(
    windows: np.ndarray,
    masks: np.ndarray,
    feature_index: int,
    requested_index: int | None,
    seq_len: int,
    min_observed_count: int | None,
) -> int:
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(windows):
            raise ValueError(
                f"--window-index={requested_index} 越界，窗口总数为 {len(windows)}。"
            )
        return requested_index

    min_count = min_observed_count
    if min_count is None:
        min_count = max(3, seq_len // 4)

    observed_count = (1.0 - masks[:, :, feature_index]).sum(axis=1)
    feature_windows = windows[:, :, feature_index]
    window_std = feature_windows.std(axis=1)
    first_full_window = min(max(seq_len - 1, 0), len(windows) - 1)
    candidate_idx = np.arange(first_full_window, len(windows))

    enough_observed = observed_count[candidate_idx] >= min_count
    non_constant = window_std[candidate_idx] > 1e-6
    usable = candidate_idx[enough_observed & non_constant]
    if usable.size:
        return int(usable[0])

    best_local = int(np.argmax(observed_count[candidate_idx]))
    return int(candidate_idx[best_local])


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def extract_frequency_state(
    checkpoint: dict | None,
    model_keys: set[str],
) -> dict[str, torch.Tensor] | None:
    if checkpoint is None:
        return None

    state = checkpoint
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state = checkpoint["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("checkpoint 中没有可读取的 state_dict。")

    state = strip_module_prefix(state)
    if model_keys.issubset(state.keys()):
        return {key: state[key] for key in model_keys}

    prefixes = [
        "imputer.freq.",
        "model.imputer.freq.",
        "freq_reconstructor.",
    ]
    for prefix in prefixes:
        sub_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if model_keys.issubset(sub_state.keys()):
            return {key: sub_state[key] for key in model_keys}

    raise ValueError(
        "checkpoint 中未找到 FrequencyOnlyReconstructor 权重；"
        "需要包含 imputer.freq.* 或直接包含 freq.* / output_head.* 键。"
    )


def load_frequency_model(
    checkpoint: dict | None,
    num_features: int,
    seq_len: int,
    device: torch.device,
) -> tuple[FrequencyOnlyReconstructor, str]:
    model = FrequencyOnlyReconstructor(num_features=num_features, seq_len=seq_len)
    model_keys = set(model.state_dict().keys())
    frequency_state = extract_frequency_state(checkpoint, model_keys)

    source = "random_initialization"
    if frequency_state is not None:
        missing, unexpected = model.load_state_dict(frequency_state, strict=True)
        if missing or unexpected:
            raise ValueError(f"频域权重加载不完整: missing={missing}, unexpected={unexpected}")
        source = "checkpoint"

    model.to(device)
    model.eval()
    return model, source


@torch.no_grad()
def collect_frequency_module_values(
    model: FrequencyOnlyReconstructor,
    window: np.ndarray,
    mask: np.ndarray,
    feature_index: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x_true = torch.from_numpy(window).unsqueeze(0).to(device)
    structural_missing_mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
    x_seed = PreImpute.fill(x_input, structural_missing_mask)

    x_perm = x_seed.permute(0, 2, 1)
    spectrum = torch.fft.rfft(x_perm, dim=2)
    real = spectrum.real
    imag = spectrum.imag
    freq_feature = torch.cat([real, imag], dim=-1)

    attention = model.freq.attention(freq_feature)
    enhancement = model.freq.freq_enhance(freq_feature)
    freq_len = spectrum.size(-1)
    delta_real = enhancement[..., :freq_len] * attention
    delta_imag = enhancement[..., freq_len:] * attention
    enhanced_spectrum = spectrum + torch.complex(delta_real, delta_imag)

    core_reconstruction = torch.fft.irfft(
        enhanced_spectrum,
        n=x_seed.size(1),
        dim=2,
    ).permute(0, 2, 1)
    reconstruction_output = model(x_seed)

    selected = (0, feature_index)
    return {
        "spectrum_real": real[selected].detach().cpu().numpy(),
        "spectrum_imag": imag[selected].detach().cpu().numpy(),
        "spectrum_magnitude": torch.abs(spectrum[selected]).detach().cpu().numpy(),
        "attention_weight": attention[selected].detach().cpu().numpy(),
        "delta_real": delta_real[selected].detach().cpu().numpy(),
        "delta_imag": delta_imag[selected].detach().cpu().numpy(),
        "enhanced_real": enhanced_spectrum.real[selected].detach().cpu().numpy(),
        "enhanced_imag": enhanced_spectrum.imag[selected].detach().cpu().numpy(),
        "enhanced_magnitude": torch.abs(enhanced_spectrum[selected]).detach().cpu().numpy(),
        "zero_filled_input": x_input[0, :, feature_index].detach().cpu().numpy(),
        "x_seed": x_seed[0, :, feature_index].detach().cpu().numpy(),
        "frequency_core_reconstruction": core_reconstruction[
            0, :, feature_index
        ].detach().cpu().numpy(),
        "frequency_reconstruction_output": reconstruction_output[
            0, :, feature_index
        ].detach().cpu().numpy(),
    }


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=TEXT, labelsize=TICK_SIZE, length=4, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_color(SPINE)


def finish_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    *,
    legend_loc: str = "upper right",
    legend_title: str | None = None,
    show_legend: bool = True,
) -> None:
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=12)
    style_axis(ax)
    if not show_legend:
        return

    legend = ax.legend(
        loc=legend_loc,
        frameon=True,
        fontsize=LEGEND_SIZE,
        title=legend_title,
    )
    if legend_title is not None:
        legend.set_title(legend_title, prop={"size": LEGEND_SIZE})
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cbd5e1")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.72)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    paths = []
    for suffix in formats:
        output_path = output_dir / f"{stem}.{suffix}"
        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if suffix == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        paths.append(output_path)
    plt.close(fig)
    return paths


def plot_bar(
    values: np.ndarray,
    label: str,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    epoch_label: str | None,
    show_legend: bool = True,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(values.size)
    ax.bar(x, values, color=BLUE, width=0.78, label=label)
    finish_axis(
        ax,
        "频率索引",
        "数值",
        legend_title=epoch_label,
        show_legend=show_legend,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, stem, formats, dpi)


def plot_line(
    series: list[tuple[np.ndarray, str, str]],
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    epoch_label: str | None,
    show_legend: bool = True,
    *,
    linewidth: float = 2.6,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for values, label, color in series:
        ax.plot(np.arange(values.size), values, color=color, linewidth=linewidth, label=label)
    finish_axis(
        ax,
        xlabel,
        ylabel,
        legend_title=epoch_label,
        show_legend=show_legend,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, stem, formats, dpi)


def plot_enhanced_spectrum(
    enhanced_magnitude: np.ndarray,
    original_magnitude: np.ndarray,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    epoch_label: str | None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(enhanced_magnitude.size)
    ax.plot(
        x,
        original_magnitude,
        color=BLUE,
        linewidth=2.0,
        linestyle=(0, (5, 3)),
        alpha=0.9,
        label="原始频谱幅值",
    )
    ax.plot(
        x,
        enhanced_magnitude,
        color=BLUE_PURPLE,
        linewidth=2.6,
        label="增强后频谱幅值",
    )
    finish_axis(ax, "频率索引", "幅值", legend_title=epoch_label)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem, formats, dpi)


def plot_time_line_with_original_points(
    values: np.ndarray,
    observed_values: np.ndarray,
    observed_mask: np.ndarray,
    missing_mask: np.ndarray,
    line_label: str,
    dashed_series: list[tuple[np.ndarray, str, str]] | None,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    epoch_label: str | None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(values.size)
    ax.plot(x, values, color=DARK_GREEN, linewidth=2.6, label=line_label)
    if dashed_series is not None:
        for dashed_values, dashed_label, dashed_color in dashed_series:
            ax.plot(
                x,
                dashed_values,
                color=dashed_color,
                linewidth=2.0,
                linestyle=(0, (5, 3)),
                alpha=0.82,
                label=dashed_label,
            )

    valid_observed = observed_mask & np.isfinite(observed_values)
    observed_x = x[valid_observed]
    observed_y = observed_values[valid_observed]
    if observed_x.size:
        ax.scatter(
            observed_x,
            observed_y,
            s=14,
            color="#111111",
            label="原始观测点",
            zorder=4,
        )

    valid_missing = missing_mask & np.isfinite(values)
    missing_x = x[valid_missing]
    missing_y = values[valid_missing]
    if missing_x.size:
        ax.scatter(
            missing_x,
            missing_y,
            s=42,
            facecolors="none",
            edgecolors=RED,
            linewidths=1.25,
            label="原始缺失位置",
            zorder=5,
        )

    finish_axis(ax, "时间索引", "数值", legend_title=epoch_label)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem, formats, dpi)


def save_source_data(
    values: dict[str, np.ndarray],
    raw_window: np.ndarray,
    mask_window: np.ndarray,
    feature_index: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    freq_len = values["spectrum_real"].size
    frequency_df = pd.DataFrame(
        {
            "frequency_index": np.arange(freq_len),
            "spectrum_real": values["spectrum_real"],
            "spectrum_imag": values["spectrum_imag"],
            "spectrum_magnitude": values["spectrum_magnitude"],
            "attention_weight": values["attention_weight"],
            "delta_real": values["delta_real"],
            "delta_imag": values["delta_imag"],
            "enhanced_real": values["enhanced_real"],
            "enhanced_imag": values["enhanced_imag"],
            "enhanced_magnitude": values["enhanced_magnitude"],
        }
    )

    seq_len = values["x_seed"].size
    time_df = pd.DataFrame(
        {
            "time_index": np.arange(seq_len),
            "raw_training_value": raw_window[:, feature_index],
            "structural_missing_mask": mask_window[:, feature_index].astype(np.int64),
            "zero_filled_input": values["zero_filled_input"],
            "x_seed": values["x_seed"],
            "frequency_core_reconstruction": values["frequency_core_reconstruction"],
            "frequency_reconstruction_output": values["frequency_reconstruction_output"],
        }
    )

    frequency_path = output_dir / "frequency_domain_values.csv"
    time_path = output_dir / "time_domain_values.csv"
    frequency_df.to_csv(frequency_path, index=False)
    time_df.to_csv(time_path, index=False)
    return frequency_path, time_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_matplotlib()
    formats = normalize_formats(args.formats)

    checkpoint = load_checkpoint(args.checkpoint, args.allow_random_init)
    ckpt_args = checkpoint_args(checkpoint)
    epoch_label = resolve_epoch_label(args.epoch_label, ckpt_args)
    train_glob = args.train_glob or ckpt_args.get("train_glob", "data/train/train_1.csv")
    seq_len = int(args.seq_len or ckpt_args.get("seq_len", 50))
    stride = int(args.stride or ckpt_args.get("stride", 1))
    if seq_len < 2:
        raise ValueError(f"--seq-len 必须 >= 2，收到 {seq_len}")
    if stride < 1:
        raise ValueError(f"--stride 必须 >= 1，收到 {stride}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    raw_data, missing_mask, feature_names = load_csv_glob_with_mask(train_glob)
    feature_index = resolve_feature_index(feature_names, args.feature_index, args.feature_name)
    scaler = ObservedStandardScaler().fit(raw_data, missing_mask)
    scaled_data = scaler.transform(raw_data, missing_mask)

    windows, window_masks = build_windows(
        scaled_data,
        missing_mask.astype(np.float32),
        seq_len=seq_len,
        stride=stride,
    )
    raw_windows, _ = build_windows(
        raw_data,
        missing_mask.astype(np.float32),
        seq_len=seq_len,
        stride=stride,
    )
    window_index = choose_window_index(
        windows=windows,
        masks=window_masks,
        feature_index=feature_index,
        requested_index=args.window_index,
        seq_len=seq_len,
        min_observed_count=args.min_observed_count,
    )

    model, model_source = load_frequency_model(
        checkpoint=checkpoint,
        num_features=scaled_data.shape[1],
        seq_len=seq_len,
        device=device,
    )
    values = collect_frequency_module_values(
        model=model,
        window=windows[window_index],
        mask=window_masks[window_index],
        feature_index=feature_index,
        device=device,
    )

    saved_paths: list[Path] = []
    saved_paths += plot_bar(
        values["spectrum_real"],
        "频谱实部",
        output_dir,
        "01_spectrum_real",
        formats,
        args.dpi,
        epoch_label,
        show_legend=False,
    )
    saved_paths += plot_bar(
        values["spectrum_imag"],
        "频谱虚部",
        output_dir,
        "02_spectrum_imag",
        formats,
        args.dpi,
        epoch_label,
        show_legend=False,
    )
    saved_paths += plot_line(
        [(values["attention_weight"], "频域注意力权重", ORANGE)],
        "频率索引",
        "权重",
        output_dir,
        "03_frequency_attention_weight",
        formats,
        args.dpi,
        epoch_label,
        show_legend=False,
    )
    saved_paths += plot_line(
        [
            (values["delta_real"], "频谱增量实部", ORANGE),
            (values["delta_imag"], "频谱增量虚部", RED),
        ],
        "频率索引",
        "数值",
        output_dir,
        "04_spectrum_increment",
        formats,
        args.dpi,
        epoch_label,
    )
    saved_paths += plot_enhanced_spectrum(
        values["enhanced_magnitude"],
        values["spectrum_magnitude"],
        output_dir,
        "05_enhanced_spectrum",
        formats,
        args.dpi,
        epoch_label,
    )
    observed_values = windows[window_index, :, feature_index]
    observed_mask = window_masks[window_index, :, feature_index] < 0.5
    missing_position_mask = window_masks[window_index, :, feature_index] >= 0.5
    saved_paths += plot_time_line_with_original_points(
        values["x_seed"],
        observed_values,
        observed_mask,
        missing_position_mask,
        r"输入窗口 $x^{seed}$",
        [(values["zero_filled_input"], "置零输入", DASHED_GRAY)],
        output_dir,
        "06_input_x_seed",
        formats,
        args.dpi,
        epoch_label,
    )
    saved_paths += plot_time_line_with_original_points(
        values["frequency_reconstruction_output"],
        observed_values,
        observed_mask,
        missing_position_mask,
        "频域重构输出",
        [(values["x_seed"], r"模块输入 $x^{seed}$", DASHED_GRAY)],
        output_dir,
        "07_frequency_reconstruction_output",
        formats,
        args.dpi,
        epoch_label,
    )

    frequency_data_path, time_data_path = save_source_data(
        values=values,
        raw_window=raw_windows[window_index],
        mask_window=window_masks[window_index],
        feature_index=feature_index,
        output_dir=output_dir,
    )

    end_row_index = window_index * stride
    start_row_index = end_row_index - seq_len + 1
    metadata = {
        "train_glob": train_glob,
        "checkpoint": args.checkpoint,
        "model_source": model_source,
        "device": str(device),
        "seq_len": seq_len,
        "stride": stride,
        "window_index": window_index,
        "raw_window_start_row_index": start_row_index,
        "raw_window_end_row_index": end_row_index,
        "feature_index_1based": feature_index + 1,
        "feature_name": feature_names[feature_index],
        "epoch_label": epoch_label,
        "checkpoint_epochs": ckpt_args.get("epochs"),
        "value_space": "standardized_by_training_observed_values",
        "figure_count": 7,
        "formats": formats,
        "frequency_data_csv": str(frequency_data_path),
        "time_data_csv": str(time_data_path),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("频域模块论文图已输出:")
    print(f"  输出目录: {output_dir}")
    print(f"  变量: {feature_names[feature_index]} (index={feature_index + 1})")
    print(f"  窗口: {window_index}，原始行范围: {start_row_index}..{end_row_index}")
    print(f"  Epoch 标注: {epoch_label or '未显示'}")
    print(f"  模型来源: {model_source}")
    print(f"  图片数量: {len(saved_paths)} 个文件，7 张图 x {len(formats)} 种格式")
    print(f"  频域数据: {frequency_data_path}")
    print(f"  时域数据: {time_data_path}")
    print(f"  元数据: {metadata_path}")


if __name__ == "__main__":
    main()
