#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from freq_reconstruction import FrequencyOnlyReconstructor
from graph_gcn_reconstruction import GraphGCNReconstructor, adjacency_balance_loss


@dataclass(frozen=True)
class LossWeights:
    fusion: float = 1.0
    gcn: float = 0.4
    freq: float = 0.4
    detector: float = 0.5
    graph: float = 0.1


@dataclass(frozen=True)
class AblationConfig:
    name: str
    description: str
    fusion_mode: str = "gate"
    gate_use_rate: bool = True
    detector_use_rate: bool = True
    detector_use_phase: bool = True
    use_ewaf: bool = True


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_csv(data: np.ndarray, feature_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=feature_names).to_csv(output_path, index=False)


def infer_rate_metadata(missing_mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    missing_ratio = missing_mask.mean(axis=0)
    observed_ratio = np.clip(1.0 - missing_ratio, 1e-6, 1.0)
    inferred_stride = np.rint(1.0 / observed_ratio).astype(np.int64)
    unique_stride = sorted(np.unique(inferred_stride).tolist())
    rate_lookup = {stride_value: idx for idx, stride_value in enumerate(unique_stride)}
    rate_id = np.asarray([rate_lookup[item] for item in inferred_stride], dtype=np.int64)
    return torch.tensor(rate_id, dtype=torch.long), torch.tensor(
        inferred_stride,
        dtype=torch.long,
    )


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = ((prediction - target) * target_mask).pow(2)
    return squared_error.sum() / target_mask.sum().clamp_min(1.0)


def sample_holdout_mask(missing_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    observed = ~missing_mask.bool()
    holdout = ((torch.rand_like(missing_mask) < ratio) & observed).view(
        missing_mask.size(0),
        -1,
    )
    observed_flat = observed.view(missing_mask.size(0), -1)

    for batch_idx in range(holdout.size(0)):
        if observed_flat[batch_idx].sum() == 0:
            continue
        if holdout[batch_idx].any():
            continue
        candidate_idx = torch.nonzero(observed_flat[batch_idx], as_tuple=False).squeeze(
            -1
        )
        picked = candidate_idx[
            torch.randint(0, len(candidate_idx), (1,), device=missing_mask.device)
        ]
        holdout[batch_idx, picked] = True

    return holdout.view_as(missing_mask).float()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.pe[:length].to(device=device, dtype=dtype)


class TwoExpertGatedImputer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        num_rates: int,
        graph_hidden_dim: int,
        gcn_hidden_dim: int,
        gate_hidden_dim: int,
        rate_embed_dim: int,
        ablation: AblationConfig,
    ) -> None:
        super().__init__()
        self.ablation = ablation
        self.has_gcn = ablation.fusion_mode != "single_freq"
        self.has_freq = ablation.fusion_mode != "single_gcn"
        self.uses_gate = ablation.fusion_mode == "gate"
        self.rate_embed_dim = rate_embed_dim if ablation.gate_use_rate else 0

        if self.has_gcn:
            self.gcn = GraphGCNReconstructor(
                num_nodes=num_nodes,
                graph_hidden_dim=graph_hidden_dim,
                gcn_hidden_dim=gcn_hidden_dim,
            )
        else:
            self.gcn = None

        if self.has_freq:
            self.freq = FrequencyOnlyReconstructor(
                num_features=num_nodes,
                seq_len=seq_len,
            )
        else:
            self.freq = None

        if ablation.gate_use_rate:
            self.rate_embedding = nn.Embedding(num_rates, rate_embed_dim)
        else:
            self.rate_embedding = None

        if self.uses_gate:
            gate_input_dim = 6 + self.rate_embed_dim
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 2),
            )
        else:
            self.gate = None

    def _fixed_gate_weights(
        self,
        batch_size: int,
        time_steps: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.ablation.fusion_mode == "single_gcn":
            weights = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        elif self.ablation.fusion_mode == "single_freq":
            weights = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        else:
            weights = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
        return weights.view(1, 1, 1, 2).expand(batch_size, time_steps, num_nodes, 2)

    def forward(
        self,
        x_input: torch.Tensor,
        model_missing_mask: torch.Tensor,
        rate_id: torch.Tensor,
    ) -> dict[str, torch.Tensor | bool | None]:
        x_seed = x_input
        adjacency: torch.Tensor | None = None

        if self.has_gcn and self.gcn is not None:
            x_gcn, adjacency = self.gcn(x_seed, model_missing_mask)
        else:
            x_gcn = torch.zeros_like(x_seed)

        if self.has_freq and self.freq is not None:
            x_freq = self.freq(x_seed)
        else:
            x_freq = torch.zeros_like(x_seed)

        if not self.has_gcn:
            x_gcn = x_freq
        if not self.has_freq:
            x_freq = x_gcn

        batch_size, time_steps, num_nodes = x_input.shape
        if self.uses_gate and self.gate is not None:
            gate_features = torch.stack(
                [
                    x_seed,
                    model_missing_mask,
                    x_gcn,
                    x_freq,
                    x_gcn - x_seed,
                    x_freq - x_seed,
                ],
                dim=-1,
            )
            gate_inputs = [gate_features]
            if self.rate_embedding is not None:
                rate_embed = self.rate_embedding(rate_id).unsqueeze(0).unsqueeze(0)
                rate_embed = rate_embed.expand(batch_size, time_steps, -1, -1)
                gate_inputs.append(rate_embed)
            gate_input = torch.cat(gate_inputs, dim=-1)
            gate_logits = self.gate(gate_input)
            gate_logits = gate_logits - gate_logits.max(dim=-1, keepdim=True).values
            gate_weights = torch.softmax(gate_logits, dim=-1)
        else:
            gate_weights = self._fixed_gate_weights(
                batch_size=batch_size,
                time_steps=time_steps,
                num_nodes=num_nodes,
                device=x_input.device,
                dtype=x_input.dtype,
            )

        experts = torch.stack([x_gcn, x_freq], dim=-1)
        x_imputed = (gate_weights * experts).sum(dim=-1)
        x_complete = torch.where(model_missing_mask.bool(), x_imputed, x_input)

        return {
            "x_seed": x_seed,
            "x_gcn": x_gcn,
            "x_freq": x_freq,
            "x_imputed": x_imputed,
            "x_complete": x_complete,
            "gate_weights": gate_weights,
            "adjacency": adjacency,
            "has_gcn": self.has_gcn,
            "has_freq": self.has_freq,
        }


class SamplingAwareTransformerAD(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_rates: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        use_rate: bool,
        use_phase: bool,
        per_variable_dim: int = 16,
        phase_dim: int = 8,
        variable_embed_dim: int = 8,
        rate_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.use_rate = use_rate
        self.use_phase = use_phase

        self.variable_embedding = nn.Embedding(num_nodes, variable_embed_dim)
        if use_rate:
            self.rate_embedding = nn.Embedding(num_rates, rate_embed_dim)
        else:
            self.rate_embedding = None

        if use_phase:
            self.phase_mlp = nn.Sequential(
                nn.Linear(1, phase_dim),
                nn.GELU(),
                nn.Linear(phase_dim, phase_dim),
            )
            phase_feature_dim = phase_dim
        else:
            self.phase_mlp = None
            phase_feature_dim = 0

        rate_feature_dim = rate_embed_dim if use_rate else 0
        per_variable_input_dim = (
            1 + 1 + variable_embed_dim + rate_feature_dim + phase_feature_dim
        )
        self.per_variable_proj = nn.Sequential(
            nn.Linear(per_variable_input_dim, per_variable_dim),
            nn.GELU(),
            nn.Linear(per_variable_dim, per_variable_dim),
        )

        token_input_dim = num_nodes * 2 + num_nodes * per_variable_dim
        self.token_proj = nn.Linear(token_input_dim, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_nodes),
        )

    def forward(
        self,
        x_complete: torch.Tensor,
        observed_mask: torch.Tensor,
        rate_id: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, num_nodes = x_complete.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"期望 {self.num_nodes} 个变量，收到 {num_nodes}")

        variable_embed = self.variable_embedding(
            torch.arange(num_nodes, device=x_complete.device)
        )
        variable_embed = (
            variable_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1, -1)
        )

        per_variable_parts = [
            x_complete.unsqueeze(-1),
            observed_mask.unsqueeze(-1),
            variable_embed,
        ]

        if self.rate_embedding is not None:
            rate_embed = self.rate_embedding(rate_id)
            rate_embed = (
                rate_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1, -1)
            )
            per_variable_parts.append(rate_embed)

        if self.phase_mlp is not None:
            time_index = torch.arange(
                time_steps,
                device=x_complete.device,
                dtype=x_complete.dtype,
            ).view(time_steps, 1)
            stride_value = (
                stride.to(device=x_complete.device, dtype=x_complete.dtype)
                .view(1, num_nodes)
                .clamp_min(1.0)
            )
            phase = torch.remainder(time_index, stride_value) / stride_value
            phase_embed = self.phase_mlp(phase.unsqueeze(-1))
            phase_embed = phase_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)
            per_variable_parts.append(phase_embed)

        per_variable_input = torch.cat(per_variable_parts, dim=-1)
        rate_aware_features = self.per_variable_proj(per_variable_input)

        token_input = torch.cat(
            [
                x_complete,
                observed_mask,
                rate_aware_features.reshape(batch_size, time_steps, -1),
            ],
            dim=-1,
        )
        tokens = self.token_proj(token_input)
        tokens = tokens + self.position_encoding(
            time_steps,
            x_complete.device,
            x_complete.dtype,
        ).unsqueeze(0)

        encoded = self.encoder(tokens)
        delta = self.reconstruction_head(encoded)
        return x_complete + delta


class FusionAnomalyModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        num_rates: int,
        graph_hidden_dim: int,
        gcn_hidden_dim: int,
        gate_hidden_dim: int,
        rate_embed_dim: int,
        detector_d_model: int,
        detector_heads: int,
        detector_layers: int,
        dropout: float,
        ablation: AblationConfig,
    ) -> None:
        super().__init__()
        self.ablation = ablation
        self.imputer = TwoExpertGatedImputer(
            num_nodes=num_nodes,
            seq_len=seq_len,
            num_rates=num_rates,
            graph_hidden_dim=graph_hidden_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gate_hidden_dim=gate_hidden_dim,
            rate_embed_dim=rate_embed_dim,
            ablation=ablation,
        )
        self.detector = SamplingAwareTransformerAD(
            num_nodes=num_nodes,
            num_rates=num_rates,
            d_model=detector_d_model,
            num_heads=detector_heads,
            num_layers=detector_layers,
            dropout=dropout,
            use_rate=ablation.detector_use_rate,
            use_phase=ablation.detector_use_phase,
            per_variable_dim=max(detector_d_model // 8, 8),
        )

    def forward(
        self,
        x_input: torch.Tensor,
        model_missing_mask: torch.Tensor,
        structural_missing_mask: torch.Tensor,
        rate_id: torch.Tensor,
        stride: torch.Tensor,
    ) -> dict[str, torch.Tensor | bool | None]:
        outputs = self.imputer(x_input, model_missing_mask, rate_id)
        observed_mask = 1.0 - structural_missing_mask
        detector_reconstruction = self.detector(
            outputs["x_complete"],
            observed_mask,
            rate_id,
            stride,
        )
        detector_error = torch.abs(detector_reconstruction - outputs["x_complete"])

        outputs["detector_reconstruction"] = detector_reconstruction
        outputs["detector_error"] = detector_error
        return outputs


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def compute_losses(
    outputs: dict[str, torch.Tensor | bool | None],
    x_true: torch.Tensor,
    structural_missing_mask: torch.Tensor,
    holdout_mask: torch.Tensor,
    loss_weights: LossWeights,
    diag_target: float,
) -> dict[str, torch.Tensor]:
    x_imputed = outputs["x_imputed"]
    x_complete = outputs["x_complete"]
    detector_reconstruction = outputs["detector_reconstruction"]
    assert isinstance(x_imputed, torch.Tensor)
    assert isinstance(x_complete, torch.Tensor)
    assert isinstance(detector_reconstruction, torch.Tensor)

    fusion_loss = masked_mse(x_imputed, x_true, holdout_mask)

    if bool(outputs["has_gcn"]):
        x_gcn = outputs["x_gcn"]
        assert isinstance(x_gcn, torch.Tensor)
        gcn_loss = masked_mse(x_gcn, x_true, holdout_mask)
    else:
        gcn_loss = _zero_like(x_complete)

    if bool(outputs["has_freq"]):
        x_freq = outputs["x_freq"]
        assert isinstance(x_freq, torch.Tensor)
        freq_loss = masked_mse(x_freq, x_true, holdout_mask)
    else:
        freq_loss = _zero_like(x_complete)

    adjacency = outputs["adjacency"]
    if adjacency is not None:
        assert isinstance(adjacency, torch.Tensor)
        graph_loss = adjacency_balance_loss(adjacency, target_diag=diag_target)
    else:
        graph_loss = _zero_like(x_complete)

    observed_mask = 1.0 - structural_missing_mask
    detector_target = x_complete.detach()
    detector_loss = masked_mse(
        detector_reconstruction,
        detector_target,
        observed_mask,
    )

    total_loss = (
        loss_weights.fusion * fusion_loss
        + loss_weights.gcn * gcn_loss
        + loss_weights.freq * freq_loss
        + loss_weights.detector * detector_loss
        + loss_weights.graph * graph_loss
    )
    return {
        "fusion_loss": fusion_loss,
        "gcn_loss": gcn_loss,
        "freq_loss": freq_loss,
        "detector_loss": detector_loss,
        "graph_loss": graph_loss,
        "total_loss": total_loss,
    }


def train_model(
    model: FusionAnomalyModel,
    train_windows: np.ndarray,
    train_masks: np.ndarray,
    rate_id: torch.Tensor,
    stride: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    holdout_ratio: float,
    diag_target: float,
    loss_weights: LossWeights,
) -> list[dict[str, float]]:
    loader = DataLoader(
        TensorDataset(torch.tensor(train_windows), torch.tensor(train_masks)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rate_id = rate_id.to(device)
    stride = stride.to(device)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        meter = {
            "fusion_loss": 0.0,
            "gcn_loss": 0.0,
            "freq_loss": 0.0,
            "detector_loss": 0.0,
            "graph_loss": 0.0,
            "total_loss": 0.0,
        }

        for x_true, structural_missing_mask in loader:
            x_true = x_true.to(device)
            structural_missing_mask = structural_missing_mask.to(device)
            holdout_mask = sample_holdout_mask(structural_missing_mask, holdout_ratio)
            model_missing_mask = torch.maximum(structural_missing_mask, holdout_mask)
            x_input = x_true.masked_fill(model_missing_mask.bool(), 0.0)

            outputs = model(
                x_input,
                model_missing_mask,
                structural_missing_mask,
                rate_id,
                stride,
            )
            losses = compute_losses(
                outputs=outputs,
                x_true=x_true,
                structural_missing_mask=structural_missing_mask,
                holdout_mask=holdout_mask,
                loss_weights=loss_weights,
                diag_target=diag_target,
            )

            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for key in meter:
                meter[key] += float(losses[key].detach().cpu())

        epoch_stats = {key: value / max(len(loader), 1) for key, value in meter.items()}
        history.append(epoch_stats)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"total={epoch_stats['total_loss']:.5f} "
            f"fusion={epoch_stats['fusion_loss']:.5f} "
            f"det={epoch_stats['detector_loss']:.5f} "
            f"graph={epoch_stats['graph_loss']:.5f}"
        )

    return history


@torch.no_grad()
def reconstruct_full_sequence(
    model: FusionAnomalyModel,
    windows: np.ndarray,
    masks: np.ndarray,
    rate_id: torch.Tensor,
    stride: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )
    rate_id = rate_id.to(device)
    stride = stride.to(device)

    completed = []
    gate_weights = []
    adj_samples = []
    model.eval()

    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = model(
            x_input,
            structural_missing_mask,
            structural_missing_mask,
            rate_id,
            stride,
        )
        x_complete = outputs["x_complete"]
        gate = outputs["gate_weights"]
        adjacency = outputs["adjacency"]
        assert isinstance(x_complete, torch.Tensor)
        assert isinstance(gate, torch.Tensor)

        completed.append(x_complete[:, -1, :].cpu().numpy())
        gate_weights.append(gate[:, -1, :, :].cpu().numpy())
        if adjacency is not None:
            assert isinstance(adjacency, torch.Tensor)
            adj_samples.append(adjacency.cpu().numpy())

    adjacency_array: np.ndarray | None = None
    if adj_samples:
        adjacency_array = np.concatenate(adj_samples, axis=0).astype(np.float32)

    return (
        np.concatenate(completed, axis=0).astype(np.float32),
        np.concatenate(gate_weights, axis=0).astype(np.float32),
        adjacency_array,
    )


@torch.no_grad()
def collect_window_statistics(
    model: FusionAnomalyModel,
    windows: np.ndarray,
    masks: np.ndarray,
    rate_id: torch.Tensor,
    stride: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
        shuffle=False,
    )
    rate_id = rate_id.to(device)
    stride = stride.to(device)

    collected = {
        "detector_score": [],
        "disagreement_score": [],
        "gate_entropy": [],
    }

    model.eval()
    for x_true, structural_missing_mask in loader:
        x_true = x_true.to(device)
        structural_missing_mask = structural_missing_mask.to(device)
        observed_mask = 1.0 - structural_missing_mask
        x_input = x_true.masked_fill(structural_missing_mask.bool(), 0.0)
        outputs = model(
            x_input,
            structural_missing_mask,
            structural_missing_mask,
            rate_id,
            stride,
        )

        detector_reconstruction = outputs["detector_reconstruction"]
        x_complete = outputs["x_complete"]
        x_gcn = outputs["x_gcn"]
        x_freq = outputs["x_freq"]
        gate = outputs["gate_weights"]
        assert isinstance(detector_reconstruction, torch.Tensor)
        assert isinstance(x_complete, torch.Tensor)
        assert isinstance(x_gcn, torch.Tensor)
        assert isinstance(x_freq, torch.Tensor)
        assert isinstance(gate, torch.Tensor)

        detector_sq = ((detector_reconstruction - x_complete) * observed_mask).pow(2)
        detector_score = detector_sq.sum(dim=(1, 2)) / observed_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        disagreement_score = torch.abs(x_gcn - x_freq).mean(dim=(1, 2))
        gate = gate.clamp_min(1e-8)
        gate_entropy = -(gate * gate.log()).sum(dim=-1).mean(dim=(1, 2))

        collected["detector_score"].append(
            detector_score.cpu().numpy().astype(np.float32)
        )
        collected["disagreement_score"].append(
            disagreement_score.cpu().numpy().astype(np.float32)
        )
        collected["gate_entropy"].append(gate_entropy.cpu().numpy().astype(np.float32))

    return {
        key: np.concatenate(value, axis=0).astype(np.float32)
        for key, value in collected.items()
    }


def normalize_scores(
    train_values: np.ndarray,
    eval_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    std = max(std, 1e-6)
    return (
        ((train_values - mean) / std).astype(np.float32),
        ((eval_values - mean) / std).astype(np.float32),
    )


def combine_scores(
    train_stats: dict[str, np.ndarray],
    eval_stats: dict[str, np.ndarray],
    disagreement_weight: float,
    shift_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_det, eval_det = normalize_scores(
        train_stats["detector_score"],
        eval_stats["detector_score"],
    )
    train_gap, eval_gap = normalize_scores(
        train_stats["disagreement_score"],
        eval_stats["disagreement_score"],
    )
    train_shift, eval_shift = normalize_scores(
        train_stats["shift_score"],
        eval_stats["shift_score"],
    )
    train_score = (
        np.abs(train_det) + disagreement_weight * train_gap + shift_weight * train_shift
    )
    eval_score = (
        np.abs(eval_det) + disagreement_weight * eval_gap + shift_weight * eval_shift
    )
    return train_score.astype(np.float32), eval_score.astype(np.float32)


def apply_min_anomaly_duration(
    prediction: np.ndarray,
    min_duration: int,
) -> np.ndarray:
    if min_duration <= 1:
        return prediction.astype(np.int64)

    filtered = np.zeros_like(prediction, dtype=np.int64)
    run_start: int | None = None

    for idx in range(len(prediction) + 1):
        current = int(prediction[idx]) if idx < len(prediction) else 0
        if current == 1 and run_start is None:
            run_start = idx
            continue
        if current == 0 and run_start is not None:
            if idx - run_start >= min_duration:
                filtered[run_start:idx] = 1
            run_start = None

    return filtered


def save_gate_statistics(
    gate_weights: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    mean_gate = gate_weights.mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "gcn_weight": mean_gate[:, 0],
            "freq_weight": mean_gate[:, 1],
        }
    )
    df.to_csv(output_path, index=False)


def build_window_feature_matrix(windows: np.ndarray) -> np.ndarray:
    features = []
    for column_slice in (slice(0, 6), slice(6, 12), slice(12, 18), slice(0, 18)):
        block = windows[:, :, column_slice]
        features.append(block.mean(axis=1))
        features.append(block.std(axis=1))

    features.append(np.abs(np.diff(windows, axis=1)).mean(axis=1))
    fft_feature = (
        np.abs(np.fft.rfft(windows, axis=1))[:, 1:6, :].mean(axis=1).astype(np.float32)
    )
    features.append(fft_feature)
    return np.concatenate(features, axis=1).astype(np.float32)


def compute_distribution_shift_scores(
    train_windows: np.ndarray,
    eval_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = build_window_feature_matrix(train_windows)
    eval_features = build_window_feature_matrix(eval_windows)
    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1e-6)

    train_z = (train_features - feature_mean) / feature_std
    eval_z = (eval_features - feature_mean) / feature_std
    train_score = np.sqrt((train_z**2).mean(axis=1))
    eval_score = np.sqrt((eval_z**2).mean(axis=1))
    return train_score.astype(np.float32), eval_score.astype(np.float32)
