from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_csv_paths(input_glob: str) -> list[str]:
    csv_paths = sorted(glob.glob(input_glob))
    if not csv_paths:
        raise FileNotFoundError(f"没有找到匹配文件: {input_glob}")
    return csv_paths


def _read_csv_arrays(
    csv_paths: list[str],
    *,
    dtype: np.dtype,
    include_mask: bool,
    log_template: str,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    num_features: int | None = None

    for path in csv_paths:
        df = pd.read_csv(path, header=None)
        arr = df.to_numpy(dtype=dtype)
        if num_features is None:
            num_features = arr.shape[1]
        elif arr.shape[1] != num_features:
            raise ValueError("输入 CSV 的列数不一致，无法拼接。")

        arrays.append(arr)
        if include_mask:
            masks.append(np.isnan(arr).astype(np.float32))
        print(log_template.format(path=path, rows=arr.shape[0], cols=arr.shape[1]))

    if num_features is None:
        raise RuntimeError("没有读取到任何 CSV 内容。")
    return arrays, masks, num_features


def load_csv_dir_values(
    dir_path: str | Path,
    file_pattern: str = "*.csv",
    *,
    dtype: np.dtype = np.float32,
    log_template: str = "  Loaded {path}: {rows} rows, {cols} cols",
) -> tuple[np.ndarray, int]:
    csv_paths = _resolve_csv_paths(str(Path(dir_path) / file_pattern))
    arrays, _, num_features = _read_csv_arrays(
        csv_paths,
        dtype=dtype,
        include_mask=False,
        log_template=log_template,
    )
    return np.concatenate(arrays, axis=0), num_features


def load_csv_dir_with_mask(
    dir_path: str | Path,
    file_pattern: str = "*.csv",
    *,
    dtype: np.dtype = np.float32,
    log_template: str = "  Loaded {path}: {rows} rows, {cols} cols",
) -> tuple[np.ndarray, np.ndarray, int]:
    csv_paths = _resolve_csv_paths(str(Path(dir_path) / file_pattern))
    arrays, masks, num_features = _read_csv_arrays(
        csv_paths,
        dtype=dtype,
        include_mask=True,
        log_template=log_template,
    )
    return np.concatenate(arrays, axis=0), np.concatenate(masks, axis=0), num_features


def load_csv_glob_with_mask(
    input_glob: str,
    *,
    dtype: np.dtype = np.float32,
    feature_name_prefix: str = "特征",
    log_template: str = "加载 {path}: {rows} 行, {cols} 列",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    csv_paths = _resolve_csv_paths(input_glob)
    arrays, masks, num_features = _read_csv_arrays(
        csv_paths,
        dtype=dtype,
        include_mask=True,
        log_template=log_template,
    )
    feature_names = [
        f"{feature_name_prefix}{feature_idx:02d}"
        for feature_idx in range(1, num_features + 1)
    ]
    return np.concatenate(arrays, axis=0), np.concatenate(masks, axis=0), feature_names
