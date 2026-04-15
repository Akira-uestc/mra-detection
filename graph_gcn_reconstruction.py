#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import font_manager
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 mra.py 中的图学习思路，使用 GCN 对时序数据执行重构并输出可视化。"
    )
    parser.add_argument(
        "--input-glob",
        default="data/train/train_*.csv",
        help="输入 CSV 匹配模式，默认: data/train/train_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gcn_reconstruction",
        help="输出目录，默认: outputs/gcn_reconstruction",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="滑动窗口长度，默认: 50",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="窗口步长，默认: 1",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="训练轮数，默认: 30",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小，默认: 64",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率，默认: 1e-3",
    )
    parser.add_argument(
        "--graph-hidden-dim",
        type=int,
        default=32,
        help="图学习隐藏维度，默认: 32",
    )
    parser.add_argument(
        "--gcn-hidden-dim",
        type=int,
        default=32,
        help="GCN 隐藏维度，默认: 32",
    )
    parser.add_argument(
        "--plot-points",
        type=int,
        default=300,
        help="绘图时展示的时间步数，默认: 300",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=40,
        help="随机种子，默认: 40",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="运行设备，默认自动选择 cuda 或 cpu",
    )
    parser.add_argument(
        "--adj-reg-weight",
        type=float,
        default=0.1,
        help="邻接矩阵正则权重，默认: 0.1",
    )
    parser.add_argument(
        "--diag-target",
        type=float,
        default=0.25,
        help="邻接矩阵期望的对角占比，默认: 0.25",
    )
    parser.add_argument(
        "--physical-coords-path",
        default=None,
        help="可选：节点物理坐标文件（CSV/NPY），按变量顺序逐行排列。",
    )
    parser.add_argument(
        "--physical-distance-path",
        default=None,
        help="可选：节点物理距离矩阵文件（CSV/NPY），按变量顺序排列。",
    )
    return parser.parse_args()


def seed_everything(seed: int = 40) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), None)
    if selected is not None:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_csv_series(input_glob: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    csv_paths = sorted(glob.glob(input_glob))
    if not csv_paths:
        raise FileNotFoundError(f"没有找到匹配文件: {input_glob}")

    arrays = []
    masks = []
    for path in csv_paths:
        df = pd.read_csv(path, header=None)
        arr = df.to_numpy(dtype=np.float32)
        arrays.append(arr)
        masks.append(np.isnan(arr).astype(np.float32))
        print(f"加载 {path}: {arr.shape[0]} 行, {arr.shape[1]} 列")

    first_cols = arrays[0].shape[1]
    if any(arr.shape[1] != first_cols for arr in arrays):
        raise ValueError("输入 CSV 的列数不一致，无法拼接。")

    data = np.concatenate(arrays, axis=0)
    mask = np.concatenate(masks, axis=0)
    feature_names = [f"特征{i:02d}" for i in range(1, first_cols + 1)]
    return data, mask, feature_names


def _load_numeric_table(path: str | Path) -> np.ndarray:
    table_path = Path(path)
    if table_path.suffix.lower() == ".npy":
        array = np.load(table_path)
        if array.ndim != 2:
            raise ValueError(f"{table_path} 应为二维数组，收到形状 {array.shape}")
        return np.asarray(array, dtype=np.float32)

    raw_df = pd.read_csv(table_path, header=None)
    numeric_df = raw_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric_df.empty:
        raise ValueError(f"{table_path} 中未解析出有效数值表。")
    return numeric_df.to_numpy(dtype=np.float32)


def load_physical_distance_matrix(
    *,
    num_nodes: int,
    coords_path: str | None = None,
    distance_path: str | None = None,
) -> np.ndarray | None:
    if coords_path is not None and distance_path is not None:
        raise ValueError("--physical-coords-path 和 --physical-distance-path 只能二选一。")
    if coords_path is None and distance_path is None:
        return None

    if coords_path is not None:
        coords = _load_numeric_table(coords_path)
        if coords.shape[0] != num_nodes:
            raise ValueError(
                f"{coords_path} 的节点数为 {coords.shape[0]}，与特征数 {num_nodes} 不一致。"
            )
        if coords.shape[1] < 1:
            raise ValueError(f"{coords_path} 至少需要 1 个坐标维度。")
        diff = coords[:, None, :] - coords[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)
    else:
        distance = _load_numeric_table(distance_path)
        if distance.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"{distance_path} 的形状为 {distance.shape}，期望 {(num_nodes, num_nodes)}。"
            )
        distance = 0.5 * (distance + distance.T)

    distance = np.asarray(distance, dtype=np.float32)
    distance = np.maximum(distance, 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def build_data_driven_distance_matrix(
    data: np.ndarray,
    missing_mask: np.ndarray,
    *,
    mask_weight: float = 0.5,
) -> np.ndarray:
    if data.shape != missing_mask.shape:
        raise ValueError(
            f"data 和 missing_mask 形状不一致: {data.shape} vs {missing_mask.shape}"
        )

    observed_mask = 1.0 - missing_mask.astype(np.float32)
    finite_data = np.where(np.isfinite(data), data, 0.0).astype(np.float32)
    # 用训练期标准化轨迹和观测模式共同描述每个节点，
    # 自动得到一个数据驱动的欧氏距离先验。
    node_descriptors = np.concatenate(
        [finite_data.T, mask_weight * observed_mask.T],
        axis=1,
    )
    distance = np.linalg.norm(
        node_descriptors[:, None, :] - node_descriptors[None, :, :],
        axis=-1,
    )
    distance = np.asarray(distance, dtype=np.float32)
    distance = np.maximum(distance, 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def resolve_distance_prior_matrix(
    *,
    data: np.ndarray,
    missing_mask: np.ndarray,
    coords_path: str | None = None,
    distance_path: str | None = None,
) -> tuple[np.ndarray, str]:
    external_distance = load_physical_distance_matrix(
        num_nodes=data.shape[1],
        coords_path=coords_path,
        distance_path=distance_path,
    )
    if external_distance is not None:
        return external_distance, "external_physical"

    auto_distance = build_data_driven_distance_matrix(data, missing_mask)
    return auto_distance, "train_data_auto"


class ObservedStandardScaler:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, data: np.ndarray, missing_mask: np.ndarray) -> "ObservedStandardScaler":
        means = []
        scales = []
        for feature_idx in range(data.shape[1]):
            observed = data[:, feature_idx][missing_mask[:, feature_idx] < 0.5]
            observed = observed[np.isfinite(observed)]
            if observed.size == 0:
                means.append(0.0)
                scales.append(1.0)
                continue

            mean = float(np.mean(observed))
            std = float(np.std(observed))
            means.append(mean)
            scales.append(max(std, self.eps))

        self.mean_ = np.asarray(means, dtype=np.float32)
        self.scale_ = np.asarray(scales, dtype=np.float32)
        return self

    def transform(self, data: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ObservedStandardScaler 尚未 fit。")

        missing = missing_mask.astype(bool) | ~np.isfinite(data)
        filled = np.where(missing, self.mean_[None, :], data)
        transformed = (filled - self.mean_[None, :]) / self.scale_[None, :]
        return transformed.astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ObservedStandardScaler 尚未 fit。")
        restored = data * self.scale_[None, :] + self.mean_[None, :]
        return restored.astype(np.float32)


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


class GraphLearner(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 32,
        static_dim: int = 8,
        coord_dim: int = 8,
        self_loop_logit: float = 0.5,
        physical_distance_matrix: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.self_loop_logit = nn.Parameter(torch.tensor(self_loop_logit, dtype=torch.float32))
        self.prior_strength = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.similarity_strength = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))

        self.static_context = nn.Parameter(torch.randn(num_nodes, static_dim))
        self.node_coords = nn.Parameter(torch.randn(num_nodes, coord_dim))
        self.register_buffer("physical_distance_matrix", None)
        self.register_buffer("physical_distance_scale", None)
        if physical_distance_matrix is not None:
            distance_tensor = torch.as_tensor(physical_distance_matrix, dtype=torch.float32)
            if distance_tensor.shape != (num_nodes, num_nodes):
                raise ValueError(
                    "physical_distance_matrix 形状错误，"
                    f"期望 {(num_nodes, num_nodes)}，收到 {tuple(distance_tensor.shape)}。"
                )
            distance_tensor = 0.5 * (distance_tensor + distance_tensor.transpose(0, 1))
            distance_tensor = distance_tensor.clamp_min(0.0)
            distance_tensor.fill_diagonal_(0.0)
            positive_distance = distance_tensor[distance_tensor > 0]
            if positive_distance.numel() == 0:
                distance_scale = torch.tensor(1.0, dtype=torch.float32)
            else:
                distance_scale = positive_distance.median().clamp_min(1e-6)
            self.physical_distance_matrix = distance_tensor
            self.physical_distance_scale = distance_scale

        dynamic_dim = 4
        self.node_encoder = nn.Sequential(
            nn.Linear(dynamic_dim + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_prior_adjacency(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eye = torch.eye(self.num_nodes, device=device, dtype=dtype)
        if self.physical_distance_matrix is not None:
            dist = self.physical_distance_matrix.to(device=device, dtype=dtype)
            scale = self.physical_distance_scale.to(device=device, dtype=dtype)
            prior = 1.0 / (1.0 + dist / scale.clamp_min(1e-6))
        else:
            dist = torch.cdist(self.node_coords, self.node_coords, p=2)
            prior = 1.0 / (1.0 + dist)
        return prior * (1.0 - eye)

    def _last_observed(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        time_index = torch.arange(x.size(1), device=x.device, dtype=x.dtype).view(1, -1, 1)
        latest_index = (time_index * observed).argmax(dim=1).long()
        return x.gather(1, latest_index.unsqueeze(1)).squeeze(1)

    def _dynamic_context(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = 1.0 - mask
        count = observed.sum(dim=1).clamp_min(1.0)
        mean = (x * observed).sum(dim=1) / count
        centered = (x - mean.unsqueeze(1)) * observed
        std = torch.sqrt(centered.pow(2).sum(dim=1) / count + 1e-6)
        last = self._last_observed(x, observed)
        missing_ratio = mask.mean(dim=1)
        return torch.stack([mean, std, last, missing_ratio], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        node_dynamic = self._dynamic_context(x, mask)
        node_static = self.static_context.unsqueeze(0).expand(batch_size, -1, -1)
        node_features = torch.cat([node_dynamic, node_static], dim=-1)
        node_repr = self.node_encoder(node_features)
        node_repr_norm = F.normalize(node_repr, p=2, dim=-1)

        h_i = node_repr.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        h_j = node_repr.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        pair_features = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j], dim=-1)

        edge_logits = self.edge_mlp(pair_features).squeeze(-1)
        cosine_similarity = torch.einsum("bnh,bmh->bnm", node_repr_norm, node_repr_norm)
        prior = self._build_prior_adjacency(x.device, x.dtype).unsqueeze(0)
        eye = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype).unsqueeze(0)
        offdiag_mask = 1.0 - eye

        edge_logits = edge_logits * offdiag_mask
        cosine_similarity = cosine_similarity * offdiag_mask
        prior_strength = F.softplus(self.prior_strength)
        similarity_strength = F.softplus(self.similarity_strength)
        temperature = F.softplus(self.temperature) + 1e-4

        logits = edge_logits + prior_strength * prior + similarity_strength * cosine_similarity

        logits = logits + eye * F.softplus(self.self_loop_logit)
        adj = torch.softmax(logits / temperature, dim=-1)
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj

    


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return torch.einsum("bnm,bsmd->bsnd", adj, x)


class GraphGCNReconstructor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        graph_hidden_dim: int = 32,
        gcn_hidden_dim: int = 32,
        physical_distance_matrix: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.graph = GraphLearner(
            num_nodes=num_nodes,
            hidden_dim=graph_hidden_dim,
            physical_distance_matrix=physical_distance_matrix,
        )
        self.gcn_in = GCNLayer(1, gcn_hidden_dim)
        self.gcn_mid = GCNLayer(gcn_hidden_dim, gcn_hidden_dim)
        self.gcn_out = GCNLayer(gcn_hidden_dim, 1)
        self.norm_in = nn.LayerNorm(gcn_hidden_dim)
        self.norm_mid = nn.LayerNorm(gcn_hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adj = self.graph(x, mask)

        h = self.gcn_in(x.unsqueeze(-1), adj)
        h = self.norm_in(F.relu(h))
        h = self.dropout(h)

        h = self.gcn_mid(h, adj)
        h = self.norm_mid(F.relu(h))
        h = self.dropout(h)

        x_rec = self.gcn_out(h, adj).squeeze(-1)
        return x_rec, adj


def masked_reconstruction_loss(
    x_rec: torch.Tensor,
    x_true: torch.Tensor,
    missing_mask: torch.Tensor,
) -> torch.Tensor:
    observed_mask = 1.0 - missing_mask
    squared_error = ((x_rec - x_true) * observed_mask).pow(2)
    return squared_error.sum() / observed_mask.sum().clamp_min(1.0)


def adjacency_balance_loss(adj: torch.Tensor, target_diag: float) -> torch.Tensor:
    diag_values = torch.diagonal(adj, dim1=-2, dim2=-1)
    return (diag_values - target_diag).pow(2).mean()


def train_model(
    model: nn.Module,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    adj_reg_weight: float,
    diag_target: float,
) -> nn.Module:
    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_graph_loss = 0.0

        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            x_input = x.masked_fill(mask.bool(), 0.0)

            x_rec, adj = model(x_input, mask)
            recon_loss = masked_reconstruction_loss(x_rec, x, mask)
            graph_loss = adjacency_balance_loss(adj, target_diag=diag_target)
            loss = recon_loss + adj_reg_weight * graph_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_graph_loss += graph_loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        avg_recon_loss = total_recon_loss / max(len(loader), 1)
        avg_graph_loss = total_graph_loss / max(len(loader), 1)
        print(
            f"Epoch {epoch + 1:02d}/{epochs}  "
            f"Loss: {avg_loss:.6f}  "
            f"Recon: {avg_recon_loss:.6f}  "
            f"Graph: {avg_graph_loss:.6f}"
        )

    return model


def reconstruct_sequence(
    model: nn.Module,
    windows: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )

    last_step_reconstruction = []
    adj_samples = []
    model.eval()
    with torch.no_grad():
        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            x_input = x.masked_fill(mask.bool(), 0.0)

            x_rec, adj = model(x_input, mask)
            last_step_reconstruction.append(x_rec[:, -1, :].cpu().numpy())
            adj_samples.append(adj.cpu().numpy())

    reconstructed = np.concatenate(last_step_reconstruction, axis=0).astype(np.float32)
    adjacency = np.concatenate(adj_samples, axis=0).astype(np.float32)
    return reconstructed, adjacency


def build_feature_limits(original: np.ndarray, reconstructed: np.ndarray) -> list[tuple[float, float]]:
    limits = []
    for idx in range(original.shape[1]):
        valid_original = original[:, idx][np.isfinite(original[:, idx])]
        valid_reconstructed = reconstructed[:, idx][np.isfinite(reconstructed[:, idx])]
        merged = np.concatenate([valid_original, valid_reconstructed], axis=0) if len(valid_original) or len(valid_reconstructed) else np.array([0.0])
        y_min = float(np.min(merged))
        y_max = float(np.max(merged))
        if np.isclose(y_min, y_max):
            pad = 1.0 if np.isclose(y_min, 0.0) else abs(y_min) * 0.1
            y_min -= pad
            y_max += pad
        else:
            pad = 0.05 * (y_max - y_min)
            y_min -= pad
            y_max += pad
        limits.append((y_min, y_max))
    return limits


def interpolate_for_plot(values: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.float32)

    observed_x = np.flatnonzero(finite_mask)
    observed_y = values[finite_mask].astype(np.float32)
    if len(observed_x) == 1:
        return np.full_like(values, observed_y[0], dtype=np.float32)

    interp_y = np.interp(np.arange(len(values)), observed_x, observed_y)
    return interp_y.astype(np.float32)


def plot_series_grid(
    data: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
    limits: list[tuple[float, float]],
    max_points: int,
) -> None:
    num_features = data.shape[1]
    cols = 3
    rows = int(np.ceil(num_features / cols))
    time_axis = np.arange(min(len(data), max_points))
    view = data[: len(time_axis)]

    fig, axes = plt.subplots(rows, cols, figsize=(18, max(3 * rows, 6)))
    axes = np.atleast_1d(axes).reshape(-1)
    fig.suptitle(title, fontsize=18)

    for idx, ax in enumerate(axes):
        if idx >= num_features:
            ax.axis("off")
            continue

        values = view[:, idx]
        finite_mask = np.isfinite(values)
        if finite_mask.all():
            ax.plot(time_axis, values, color="#1f77b4", linewidth=1.0)
        elif finite_mask.any():
            interp_values = interpolate_for_plot(values)
            ax.plot(
                time_axis,
                interp_values,
                color="#4c78a8",
                linewidth=0.9,
                alpha=0.75,
            )
            ax.scatter(
                time_axis[finite_mask],
                values[finite_mask],
                color="#e45756",
                s=12,
                alpha=0.9,
                zorder=3,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "全为空值",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )

        ax.set_title(feature_names[idx], fontsize=11)
        ax.set_xlabel("时间步", fontsize=9)
        ax.set_ylabel("数值", fontsize=9)
        ax.set_ylim(*limits[idx])
        ax.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_adjacency_heatmap(
    adjacency: np.ndarray,
    output_path: Path,
    *,
    suppress_diagonal: bool = False,
) -> None:
    mean_adjacency = adjacency.mean(axis=0)
    if suppress_diagonal:
        mean_adjacency = mean_adjacency.copy()
        np.fill_diagonal(mean_adjacency, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    display_matrix = np.ma.masked_invalid(mean_adjacency)
    if suppress_diagonal:
        finite_values = mean_adjacency[np.isfinite(mean_adjacency)]
        vmin = float(finite_values.min()) if finite_values.size else 0.0
        vmax = float(finite_values.max()) if finite_values.size else 1.0
        image = ax.imshow(display_matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title("平均自适应邻接矩阵（去对角线）", fontsize=15)
    else:
        image = ax.imshow(display_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_title("平均自适应邻接矩阵", fontsize=15)
    ax.set_xlabel("节点")
    ax.set_ylabel("节点")
    ax.set_xticks(np.arange(mean_adjacency.shape[0]))
    ax.set_yticks(np.arange(mean_adjacency.shape[0]))
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_adjacency(adjacency: np.ndarray) -> None:
    mean_adjacency = adjacency.mean(axis=0)
    eye_mask = np.eye(mean_adjacency.shape[0], dtype=bool)
    diag_values = mean_adjacency[eye_mask]
    offdiag_values = mean_adjacency[~eye_mask]
    print(
        "邻接矩阵统计: "
        f"diag_mean={diag_values.mean():.4f}, "
        f"offdiag_mean={offdiag_values.mean():.4f}, "
        f"offdiag_max={offdiag_values.max():.4f}"
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_chinese_font()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data, missing_mask, feature_names = load_csv_series(args.input_glob)
    scaler = ObservedStandardScaler().fit(raw_data, missing_mask)
    scaled_data = scaler.transform(raw_data, missing_mask)
    distance_prior_matrix, distance_prior_source = resolve_distance_prior_matrix(
        data=scaled_data,
        missing_mask=missing_mask,
        coords_path=args.physical_coords_path,
        distance_path=args.physical_distance_path,
    )

    windows, window_masks = build_windows(
        data=scaled_data,
        mask=missing_mask.astype(np.float32),
        seq_len=args.seq_len,
        stride=args.stride,
    )
    print(f"构建窗口完成: windows={windows.shape}, masks={window_masks.shape}")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = GraphGCNReconstructor(
        num_nodes=scaled_data.shape[1],
        graph_hidden_dim=args.graph_hidden_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
        physical_distance_matrix=distance_prior_matrix,
    ).to(device)
    if distance_prior_source == "external_physical":
        print("图先验: 使用物理距离构造的欧氏距离相似度先验")
    else:
        print("图先验: 根据训练数据自动计算欧氏距离相似度先验")

    train_model(
        model=model,
        train_windows=windows,
        train_masks=window_masks,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adj_reg_weight=args.adj_reg_weight,
        diag_target=args.diag_target,
    )

    reconstructed_scaled, adjacency = reconstruct_sequence(
        model=model,
        windows=windows,
        masks=window_masks,
        device=device,
        batch_size=args.batch_size,
    )
    reconstructed = scaler.inverse_transform(reconstructed_scaled).astype(np.float32)

    original_csv = output_dir / "original_data.csv"
    reconstructed_csv = output_dir / "reconstructed_data.csv"
    pd.DataFrame(raw_data, columns=feature_names).to_csv(original_csv, index=False)
    pd.DataFrame(reconstructed, columns=feature_names).to_csv(reconstructed_csv, index=False)

    limits = build_feature_limits(raw_data, reconstructed)
    original_plot = output_dir / "original_data.png"
    reconstructed_plot = output_dir / "reconstructed_data.png"
    adjacency_plot = output_dir / "mean_adjacency.png"
    adjacency_offdiag_plot = output_dir / "mean_adjacency_offdiag.png"

    plot_series_grid(
        data=raw_data,
        feature_names=feature_names,
        output_path=original_plot,
        title="原始数据",
        limits=limits,
        max_points=args.plot_points,
    )
    plot_series_grid(
        data=reconstructed,
        feature_names=feature_names,
        output_path=reconstructed_plot,
        title="GCN 重构数据",
        limits=limits,
        max_points=args.plot_points,
    )
    save_adjacency_heatmap(adjacency, adjacency_plot)
    save_adjacency_heatmap(adjacency, adjacency_offdiag_plot, suppress_diagonal=True)
    summarize_adjacency(adjacency)

    print("输出完成:")
    print(f"  原始数据 CSV: {original_csv}")
    print(f"  重构数据 CSV: {reconstructed_csv}")
    print(f"  原始数据图  : {original_plot}")
    print(f"  重构数据图  : {reconstructed_plot}")
    print(f"  邻接矩阵图  : {adjacency_plot}")
    print(f"  去对角线图  : {adjacency_offdiag_plot}")


if __name__ == "__main__":
    main()
