from __future__ import annotations

import numpy as np


TEST_SEGMENT_LENGTH = 2050
TEST_WINDOW_COUNT = 2000


def _empty_window_array(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.zeros((0, seq_len, data.shape[1]), dtype=data.dtype)


def build_windows(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    window_masks = []
    num_steps = data.shape[0]

    for end_idx in range(0, num_steps, stride):
        start_idx = end_idx - seq_len + 1
        if start_idx >= 0:
            window = data[start_idx : end_idx + 1]
            window_mask = mask[start_idx : end_idx + 1]
        else:
            pad_len = -start_idx
            window = np.concatenate(
                [np.repeat(data[0:1], pad_len, axis=0), data[: end_idx + 1]],
                axis=0,
            )
            window_mask = np.concatenate(
                [np.repeat(mask[0:1], pad_len, axis=0), mask[: end_idx + 1]],
                axis=0,
            )

        windows.append(window)
        window_masks.append(window_mask)

    return np.stack(windows).astype(np.float32), np.stack(window_masks).astype(np.float32)


def build_standard_windows(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if len(data) < seq_len:
        shape = (0, seq_len, data.shape[1])
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    windows = []
    window_masks = []
    for end_idx in range(seq_len, len(data) + 1, stride):
        windows.append(data[end_idx - seq_len : end_idx])
        window_masks.append(mask[end_idx - seq_len : end_idx])
    return np.stack(windows).astype(np.float32), np.stack(window_masks).astype(np.float32)


def build_prompt_test_windows(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    segment_length: int = TEST_SEGMENT_LENGTH,
    window_count: int = TEST_WINDOW_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    window_masks = []
    labels = []
    stop_idx = min(segment_length, seq_len + window_count)

    for segment_start, label in ((0, 0), (segment_length, 1)):
        segment_data = data[segment_start : segment_start + segment_length]
        segment_mask = mask[segment_start : segment_start + segment_length]
        if len(segment_data) < seq_len:
            continue

        for end_idx in range(seq_len, stop_idx, stride):
            if end_idx > len(segment_data):
                break
            windows.append(segment_data[end_idx - seq_len : end_idx])
            window_masks.append(segment_mask[end_idx - seq_len : end_idx])
            labels.append(label)

    if not windows:
        return (
            _empty_window_array(data, seq_len).astype(np.float32),
            _empty_window_array(mask, seq_len).astype(np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(windows).astype(np.float32),
        np.stack(window_masks).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def build_prompt_test_windows_values(
    data: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    segment_length: int = TEST_SEGMENT_LENGTH,
    window_count: int = TEST_WINDOW_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    labels = []
    stop_idx = min(segment_length, seq_len + window_count)

    for segment_start, label in ((0, 0), (segment_length, 1)):
        segment_data = data[segment_start : segment_start + segment_length]
        if len(segment_data) < seq_len:
            continue

        for end_idx in range(seq_len, stop_idx, stride):
            if end_idx > len(segment_data):
                break
            windows.append(segment_data[end_idx - seq_len : end_idx])
            labels.append(label)

    if not windows:
        return _empty_window_array(data, seq_len), np.zeros((0,), dtype=np.int64)

    return np.stack(windows).astype(data.dtype, copy=False), np.asarray(
        labels, dtype=np.int64
    )


def build_front_padded_windows(
    data: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    start_index: int = 49,
    max_window_count: int = 4000,
) -> np.ndarray:
    n = len(data)
    if n == 0:
        return _empty_window_array(data, seq_len)

    stop_idx = min(n, start_index + max_window_count * stride)
    if stop_idx <= start_index:
        return _empty_window_array(data, seq_len)

    windows = []
    for idx in range(start_index, stop_idx, stride):
        if idx < seq_len:
            pad_len = seq_len - idx - 1
            window = np.concatenate(
                [np.tile(data[0:1], (pad_len, 1)), data[0 : idx + 1]],
                axis=0,
            )
        else:
            window = data[idx - seq_len + 1 : idx + 1]
        windows.append(window)

    return np.stack(windows)


def build_front_padded_windows_with_mask(
    data: np.ndarray,
    mask: np.ndarray,
    seq_len: int,
    stride: int = 1,
    *,
    start_index: int = 49,
    max_window_count: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    if n == 0:
        shape = (0, seq_len, data.shape[1])
        return np.zeros(shape, dtype=data.dtype), np.zeros(shape, dtype=mask.dtype)

    stop_idx = min(n, start_index + max_window_count * stride)
    if stop_idx <= start_index:
        shape = (0, seq_len, data.shape[1])
        return np.zeros(shape, dtype=data.dtype), np.zeros(shape, dtype=mask.dtype)

    windows = []
    window_masks = []
    for idx in range(start_index, stop_idx, stride):
        if idx < seq_len:
            pad_len = seq_len - idx - 1
            window = np.concatenate(
                [np.tile(data[0:1], (pad_len, 1)), data[0 : idx + 1]],
                axis=0,
            )
            window_mask = np.concatenate(
                [np.tile(mask[0:1], (pad_len, 1)), mask[0 : idx + 1]],
                axis=0,
            )
        else:
            window = data[idx - seq_len + 1 : idx + 1]
            window_mask = mask[idx - seq_len + 1 : idx + 1]
        windows.append(window)
        window_masks.append(window_mask)

    return np.stack(windows).astype(data.dtype, copy=False), np.stack(window_masks).astype(
        mask.dtype, copy=False
    )


def _build_forecast_target(
    values: np.ndarray,
    start_idx: int,
    prediction_horizon: int,
) -> np.ndarray:
    end_idx = start_idx + prediction_horizon
    if end_idx <= len(values):
        return values[start_idx:end_idx]

    available = values[start_idx:] if start_idx < len(values) else values[-1:]
    pad_needed = prediction_horizon - len(available)
    return np.concatenate([available, np.tile(values[-1:], (pad_needed, 1))], axis=0)


def build_forecasting_windows(
    data: np.ndarray,
    sequence_length: int,
    prediction_horizon: int = 1,
    stride: int = 1,
    *,
    training: bool = True,
    start_index: int = 49,
    max_window_count: int = 4000,
    test_segment_length: int = TEST_SEGMENT_LENGTH,
    test_window_count: int = TEST_WINDOW_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = data.astype(np.float32)
    x_windows = []
    y_windows = []
    labels = []

    if training:
        stop_idx = min(len(values), start_index + max_window_count * stride)
        for idx in range(start_index, stop_idx, stride):
            if idx < sequence_length:
                pad_len = sequence_length - idx - 1
                window = np.concatenate(
                    [np.tile(values[0:1], (pad_len, 1)), values[0 : idx + 1]],
                    axis=0,
                )
            else:
                window = values[idx - sequence_length + 1 : idx + 1]
            target = _build_forecast_target(values, idx + 1, prediction_horizon)
            x_windows.append(window)
            y_windows.append(target)
            labels.append(0)
    else:
        stop_idx = min(test_segment_length, sequence_length + test_window_count)
        for segment_start, label in ((0, 0), (test_segment_length, 1)):
            segment_values = values[segment_start : segment_start + test_segment_length]
            if len(segment_values) < sequence_length:
                continue

            for end_idx in range(sequence_length, stop_idx, stride):
                if end_idx > len(segment_values):
                    break
                window = segment_values[end_idx - sequence_length : end_idx]
                global_end_idx = segment_start + end_idx - 1
                target = _build_forecast_target(
                    values, global_end_idx + 1, prediction_horizon
                )
                x_windows.append(window)
                y_windows.append(target)
                labels.append(label)

    if not x_windows:
        return (
            np.zeros((0, sequence_length, values.shape[1]), dtype=np.float32),
            np.zeros((0, prediction_horizon, values.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(x_windows).astype(np.float32),
        np.stack(y_windows).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )
