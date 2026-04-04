from __future__ import annotations

import numpy as np


def apply_ewaf(scores: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"ewaf alpha 必须在 (0, 1] 内，收到 {alpha}")
    if scores.size == 0:
        return scores.astype(np.float32)

    smoothed = np.empty_like(scores, dtype=np.float32)
    smoothed[0] = np.float32(scores[0])
    for idx in range(1, len(scores)):
        smoothed[idx] = np.float32(
            alpha * scores[idx] + (1.0 - alpha) * smoothed[idx - 1]
        )
    return smoothed


def apply_ewaf_by_segments(
    scores: np.ndarray,
    alpha: float,
    segment_lengths: list[int] | None = None,
) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    if not segment_lengths:
        return apply_ewaf(scores, alpha)

    parts = []
    cursor = 0
    for length in segment_lengths:
        if length <= 0:
            continue
        next_cursor = min(cursor + length, len(scores))
        parts.append(apply_ewaf(scores[cursor:next_cursor], alpha))
        cursor = next_cursor
        if cursor >= len(scores):
            break

    if cursor < len(scores):
        parts.append(apply_ewaf(scores[cursor:], alpha))

    if not parts:
        return scores.astype(np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def choose_threshold(
    train_scores: np.ndarray,
    *,
    method: str = "mean",
    std_factor: float = 1.0,
    quantile: float = 0.95,
) -> float:
    if train_scores.size == 0:
        raise ValueError("train_scores 为空，无法计算阈值。")

    mean_score = float(np.mean(train_scores))
    std_score = float(np.std(train_scores))

    if method == "mean":
        return mean_score
    if method == "mean_std":
        return mean_score + std_factor * std_score
    if method in {"gaussian_quantile_max", "max_gaussian_quantile"}:
        gaussian_threshold = mean_score + std_factor * std_score
        quantile_threshold = float(np.quantile(train_scores, quantile))
        return max(gaussian_threshold, quantile_threshold)

    raise ValueError(f"不支持的阈值方法: {method}")


def infer_segment_lengths(labels: np.ndarray) -> list[int]:
    if labels.size == 0:
        return []

    lengths = []
    run_start = 0
    current = labels[0]
    for idx in range(1, len(labels) + 1):
        if idx < len(labels) and labels[idx] == current:
            continue
        lengths.append(idx - run_start)
        if idx < len(labels):
            run_start = idx
            current = labels[idx]
    return lengths


def split_index_from_labels(labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0
    first_label = labels[0]
    change_indices = np.flatnonzero(labels != first_label)
    if change_indices.size == 0:
        return len(labels)
    return int(change_indices[0])
