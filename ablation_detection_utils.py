#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from graph_gcn_reconstruction import configure_chinese_font
from mra import (
    SamplingAwareTransformerAD,
    TEST_SEGMENT_LENGTH,
    TEST_WINDOW_COUNT,
    apply_ewaf_by_segments,
    apply_min_anomaly_duration,
    build_standard_windows,
    choose_threshold,
    compute_distribution_shift_scores,
    compute_metrics_from_prediction,
    infer_rate_metadata,
    masked_mse,
    normalize_scores,
)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_segmented_test_windows(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if len(data) == 0:
        empty = np.zeros((0, seq_len, 0), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.int64), []

    windows = []
    window_masks = []
    labels = []
    segment_lengths: list[int] = []

    for segment_start, label in ((0, 0), (TEST_SEGMENT_LENGTH, 1)):
        segment_data = data[segment_start : segment_start + TEST_SEGMENT_LENGTH]
        segment_mask = mask[segment_start : segment_start + TEST_SEGMENT_LENGTH]
        segment_count = 0

        for end_idx in range(seq_len, len(segment_data), stride):
            windows.append(segment_data[end_idx - seq_len : end_idx])
            window_masks.append(segment_mask[end_idx - seq_len : end_idx])
            labels.append(label)
            segment_count += 1
            if segment_count >= TEST_WINDOW_COUNT:
                break

        segment_lengths.append(segment_count)

    if not windows:
        shape = (0, seq_len, data.shape[1])
        empty = np.zeros(shape, dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.int64), segment_lengths

    return (
        np.stack(windows).astype(np.float32),
        np.stack(window_masks).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        segment_lengths,
    )


def train_sampling_aware_detector(
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    rate_id: torch.Tensor,
    stride_meta: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> tuple[SamplingAwareTransformerAD, list[float]]:
    model = SamplingAwareTransformerAD(
        num_nodes=train_windows.shape[2],
        num_rates=int(rate_id.max().item()) + 1,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        per_variable_dim=max(d_model // 8, 8),
    ).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rate_id = rate_id.to(device)
    stride_meta = stride_meta.to(device)
    history: list[float] = []

    print(f"\n开始训练 MRA 风格检测器，共 {epochs} 个 Epoch...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_complete, structural_missing_mask in loader:
            x_complete = x_complete.to(device)
            structural_missing_mask = structural_missing_mask.to(device)
            observed_mask = 1.0 - structural_missing_mask

            reconstruction = model(x_complete, observed_mask, rate_id, stride_meta)
            loss = masked_mse(reconstruction, x_complete, observed_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        avg_loss = total_loss / max(len(loader), 1)
        history.append(avg_loss)
        print(f"Detector Epoch {epoch + 1:02d}/{epochs}  Loss: {avg_loss:.6f}")

    return model, history


@torch.no_grad()
def collect_detector_scores(
    model: SamplingAwareTransformerAD,
    windows: np.ndarray,
    masks: np.ndarray,
    rate_id: torch.Tensor,
    stride_meta: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )
    rate_id = rate_id.to(device)
    stride_meta = stride_meta.to(device)

    model.eval()
    collected = []

    for x_complete, structural_missing_mask in loader:
        x_complete = x_complete.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        observed_mask = 1.0 - structural_missing_mask

        reconstruction = model(x_complete, observed_mask, rate_id, stride_meta)
        detector_sq = ((reconstruction - x_complete) * observed_mask).pow(2)
        detector_score = detector_sq.sum(dim=(1, 2)) / observed_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        collected.append(detector_score.cpu().numpy().astype(np.float32))

    return np.concatenate(collected, axis=0).astype(np.float32)


def plot_anomaly_scores(
    scores: np.ndarray,
    threshold: float,
    split_idx: int,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 5))
    x_axis = np.arange(1, len(scores) + 1)
    ax.plot(x_axis, scores, color="#0B6E4F", linewidth=1.6, label="异常分数")
    ax.axhline(
        threshold,
        color="#D1495B",
        linestyle="--",
        linewidth=1.4,
        label=f"阈值 = {threshold:.4f}",
    )

    if 0 < split_idx < len(scores):
        ax.axvline(
            split_idx,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label="正常/异常分界",
        )
        ax.fill_between(
            x_axis[:split_idx],
            scores[:split_idx],
            alpha=0.08,
            color="#2A9D8F",
        )
        ax.fill_between(
            x_axis[split_idx - 1 :],
            scores[split_idx - 1 :],
            alpha=0.08,
            color="#E76F51",
        )

    ax.set_xlabel("窗口索引")
    ax.set_ylabel("分数")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_detector_training_curve(history: list[float], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    configure_chinese_font()
    epochs = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history, color="#1D3557", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("检测器训练曲线")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_mra_style_detection(
    *,
    train_completed_scaled: np.ndarray,
    test_completed_scaled: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seq_len: int,
    stride: int,
    batch_size: int,
    detector_epochs: int,
    lr: float,
    detector_d_model: int,
    detector_heads: int,
    detector_layers: int,
    dropout: float,
    score_shift_weight: float,
    threshold_std_factor: float,
    threshold_quantile: float,
    ewaf_alpha: float,
    min_anomaly_duration: int,
    device: torch.device,
    output_dir: Path,
    plot_title: str,
) -> dict[str, float]:
    train_eval_windows, train_eval_masks = build_standard_windows(
        train_completed_scaled,
        train_mask.astype(np.float32),
        seq_len=seq_len,
        stride=stride,
    )
    test_eval_windows, test_eval_masks, test_labels, segment_lengths = (
        build_segmented_test_windows(
            test_completed_scaled,
            test_mask.astype(np.float32),
            seq_len=seq_len,
            stride=stride,
        )
    )

    if train_eval_windows.size == 0:
        raise ValueError("训练检测窗口为空，请检查 detector seq_len 与训练数据长度。")
    if test_eval_windows.size == 0:
        raise ValueError("测试检测窗口为空，请检查 detector seq_len 与测试数据长度。")

    rate_id, stride_meta = infer_rate_metadata(train_mask.astype(np.float32))
    print(f"检测训练窗口: {train_eval_windows.shape}")
    print(f"检测测试窗口: {test_eval_windows.shape}")
    print(f"采样率分组 rate_id: {rate_id.tolist()}")
    print(f"推断步长 stride: {stride_meta.tolist()}")

    detector, detector_history = train_sampling_aware_detector(
        train_windows=train_eval_windows,
        train_masks=train_eval_masks,
        rate_id=rate_id,
        stride_meta=stride_meta,
        device=device,
        epochs=detector_epochs,
        batch_size=batch_size,
        lr=lr,
        d_model=detector_d_model,
        num_heads=detector_heads,
        num_layers=detector_layers,
        dropout=dropout,
    )

    train_detector_scores = collect_detector_scores(
        model=detector,
        windows=train_eval_windows,
        masks=train_eval_masks,
        rate_id=rate_id,
        stride_meta=stride_meta,
        device=device,
        batch_size=batch_size,
    )
    test_detector_scores = collect_detector_scores(
        model=detector,
        windows=test_eval_windows,
        masks=test_eval_masks,
        rate_id=rate_id,
        stride_meta=stride_meta,
        device=device,
        batch_size=batch_size,
    )

    train_shift_scores, test_shift_scores = compute_distribution_shift_scores(
        train_windows=train_eval_windows,
        eval_windows=test_eval_windows,
    )
    train_det_norm, test_det_norm = normalize_scores(
        train_detector_scores, test_detector_scores
    )
    train_shift_norm, test_shift_norm = normalize_scores(
        train_shift_scores, test_shift_scores
    )

    train_raw_scores = np.abs(train_det_norm) + score_shift_weight * train_shift_norm
    test_raw_scores = np.abs(test_det_norm) + score_shift_weight * test_shift_norm
    train_scores = apply_ewaf_by_segments(train_raw_scores, ewaf_alpha)
    test_scores = apply_ewaf_by_segments(
        test_raw_scores,
        ewaf_alpha,
        segment_lengths=segment_lengths,
    )

    threshold = choose_threshold(
        train_scores=train_scores,
        std_factor=threshold_std_factor,
        quantile=threshold_quantile,
    )
    threshold_prediction = (test_scores >= threshold).astype(np.int64)
    final_prediction = apply_min_anomaly_duration(
        threshold_prediction,
        min_anomaly_duration,
    )
    metrics = compute_metrics_from_prediction(
        test_labels,
        final_prediction,
        threshold,
    )

    split_idx = int(segment_lengths[0]) if segment_lengths else 0
    prediction_df = pd.DataFrame(
        {
            "sample_index": np.arange(1, len(test_scores) + 1),
            "label": test_labels,
            "detector_score": test_detector_scores,
            "shift_score": test_shift_scores,
            "raw_final_score": test_raw_scores,
            "final_score": test_scores,
            "threshold_prediction": threshold_prediction,
            "prediction": final_prediction,
        }
    )
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    torch.save(
        {
            "model_state_dict": detector.state_dict(),
            "rate_id": rate_id,
            "stride": stride_meta,
        },
        output_dir / "detector.pt",
    )
    save_detector_training_curve(
        detector_history, output_dir / "detector_training_curve.png"
    )
    plot_anomaly_scores(
        scores=test_scores,
        threshold=threshold,
        split_idx=split_idx,
        output_path=output_dir / "anomaly_scores.png",
        title=plot_title,
    )

    summary = {
        "device": str(device),
        "metrics": metrics,
        "segment_lengths": segment_lengths,
        "detector_seq_len": seq_len,
        "score_shift_weight": score_shift_weight,
        "threshold_std_factor": threshold_std_factor,
        "threshold_quantile": threshold_quantile,
        "ewaf_alpha": ewaf_alpha,
        "min_anomaly_duration": min_anomaly_duration,
        "train_raw_score_mean": float(np.mean(train_raw_scores)),
        "train_raw_score_std": float(np.std(train_raw_scores)),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        "rate_id": rate_id.tolist(),
        "stride": stride_meta.tolist(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics