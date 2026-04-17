"""
MRA-LSTM anomaly detection adapted to the current project data pipeline.

This implementation follows the standard paper structure more closely:
- infer multirate groups from observation rates in the training set
- preserve NaN masks through preprocessing and window construction
- reconstruct the full input window instead of forecasting future steps
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from _project_root import PROJECT_ROOT
from utils.methods.data_loading import load_csv_dir_values
from utils.methods.display import (
    compute_binary_classification_metrics,
    plot_detection_scores,
    print_metrics,
)
from utils.methods.postprocess import (
    apply_ewaf_by_segments,
    choose_threshold,
    infer_segment_lengths,
    split_index_from_labels,
)
from utils.methods.windowing import (
    build_front_padded_windows,
    build_prompt_test_windows_values,
)

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = ["SimSun"]
plt.rcParams["font.sans-serif"] = ["SimSun", "SimSun-ExtB", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

USE_EWAF = True
EWAF_ALPHA = 0.15
EPS = 1e-6


def fit_minmax_ignore_nan(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mins = np.nanmin(train_data, axis=0)
        maxs = np.nanmax(train_data, axis=0)

    invalid = ~np.isfinite(mins) | ~np.isfinite(maxs)
    mins[invalid] = 0.0
    maxs[invalid] = 1.0

    scales = maxs - mins
    scales[~np.isfinite(scales) | (np.abs(scales) < EPS)] = 1.0
    return mins.astype(np.float32), scales.astype(np.float32)


def transform_minmax_preserve_nan(
    data: np.ndarray,
    mins: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    normalized = (data - mins) / scales
    normalized[np.isnan(data)] = np.nan
    return normalized.astype(np.float32)


def infer_chunking_idx(train_data: np.ndarray) -> list[list[int]]:
    sampling_map = ~np.isnan(train_data)
    sampling_rate = sampling_map.mean(axis=0)
    rate_groups: dict[float, list[int]] = {}

    for feature_idx, rate in enumerate(sampling_rate):
        key = round(float(rate), 6)
        rate_groups.setdefault(key, []).append(feature_idx)

    return [rate_groups[key] for key in sorted(rate_groups.keys(), reverse=True)]


def masked_window_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_filled = torch.nan_to_num(target, nan=0.0)
    observed_mask = (~torch.isnan(target)).float()
    squared_error = (pred - target_filled).pow(2) * observed_mask
    denom = observed_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return squared_error.sum(dim=(1, 2)) / denom


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return masked_window_mse(pred, target).mean()


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray | None = None):
        self.windows = windows.astype(np.float32)
        if labels is None:
            labels = np.zeros((len(self.windows),), dtype=np.int64)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(self.windows[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return window, label


class HierarchicalStateExcitation(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        total_hidden = num_layers * hidden_size

        self.Q_list = nn.ParameterList(
            [nn.Parameter(torch.randn(total_hidden, hidden_size) * 0.02) for _ in range(num_layers)]
        )
        self.R_list = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02) for _ in range(num_layers)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [K, B, S, H]
        batch_size = h.shape[1]
        seq_len = h.shape[2]

        q_r_k = torch.stack([param for param in self.Q_list], dim=0)
        r_r_k = torch.stack([param for param in self.R_list], dim=0)

        h_by_group = h.permute(1, 0, 2, 3).contiguous()  # [B, K, S, H]
        h_concat = h_by_group.permute(0, 2, 1, 3).reshape(
            batch_size,
            seq_len,
            self.num_layers * self.hidden_size,
        )

        r_t_k = torch.sigmoid(
            torch.matmul(
                h_concat.unsqueeze(1),
                q_r_k.unsqueeze(0),
            )
        )  # [B, K, S, H]
        mul1 = torch.matmul(r_t_k, r_r_k.unsqueeze(0))  # [B, K, S, H]
        mul2 = mul1 * h_by_group  # [B, K, S, H]
        return torch.relu(torch.sum(mul2, dim=1))  # [B, S, H]


class StandardMRALSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        seq_len: int,
        chunking_idx: list[list[int]],
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.chunking_idx = [list(group) for group in chunking_idx]
        self.num_layers = len(self.chunking_idx)

        self.initial_boundary = nn.Parameter(torch.rand(self.num_layers, seq_len))

        self.W_list = nn.ParameterList()
        self.Z_list = nn.ParameterList()
        self.U_list = nn.ParameterList()
        self.V_list = nn.ParameterList()
        self.J_list = nn.ParameterList()
        self.b_list = nn.ParameterList()

        for group in self.chunking_idx:
            group_size = len(group)
            self.b_list.append(nn.Parameter(torch.zeros(5 * hidden_size)))
            self.W_list.append(nn.Parameter(torch.randn(5 * hidden_size, hidden_size) * 0.02))
            self.Z_list.append(nn.Parameter(torch.randn(5 * hidden_size, hidden_size) * 0.02))
            self.U_list.append(
                nn.Parameter(torch.randn(5 * hidden_size, hidden_size + group_size) * 0.02)
            )
            self.V_list.append(nn.Parameter(torch.randn(5 * hidden_size, hidden_size) * 0.02))
            self.J_list.append(nn.Parameter(torch.randn(5 * hidden_size, group_size) * 0.02))

        self.hse = HierarchicalStateExcitation(self.num_layers, hidden_size)
        self.output_layer = nn.Linear(hidden_size, input_size)

    @staticmethod
    def _step(value: torch.Tensor) -> torch.Tensor:
        return (value > 0.5).to(dtype=torch.float32)

    @staticmethod
    def _hard_sigmoid(value: torch.Tensor) -> torch.Tensor:
        return torch.clamp((value + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _select_operator(s_tprev_k: torch.Tensor, s_t_kprev: torch.Tensor) -> int:
        prev_flag = float(s_tprev_k.item())
        lower_flag = float(s_t_kprev.item())

        if prev_flag < 0.5 and lower_flag >= 0.5:
            return 1
        if prev_flag >= 0.5 and lower_flag < 0.5:
            return 2
        if prev_flag >= 0.5 and lower_flag >= 0.5:
            return 3
        return 4

    def _compute_gate(
        self,
        W: torch.Tensor,
        Z: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        J: torch.Tensor,
        bias: torch.Tensor,
        x_t_k: torch.Tensor,
        z_t_k: torch.Tensor,
        h_tprev_k: torch.Tensor,
        h_t_kprev: torch.Tensor,
        h_tprev_knext: torch.Tensor,
        operator: int,
        is_forget_gate: bool,
    ) -> torch.Tensor:
        term1 = torch.zeros_like(h_tprev_k)
        term2 = torch.zeros_like(h_tprev_k)

        if operator == 1:
            term1 = torch.matmul(h_tprev_k, W.T)
            with_input = torch.matmul(torch.cat((h_t_kprev, x_t_k), dim=1), U.T)
            without_input = torch.matmul(h_t_kprev, V.T)
            term2 = torch.where(z_t_k, with_input, without_input)

        if not is_forget_gate:
            if operator == 2:
                term1 = torch.matmul(h_tprev_knext, Z.T)
                term2 = torch.where(z_t_k, torch.matmul(x_t_k, J.T), torch.zeros_like(term1))
            elif operator == 3:
                term1 = torch.matmul(h_tprev_knext, Z.T)
                with_input = torch.matmul(torch.cat((h_t_kprev, x_t_k), dim=1), U.T)
                without_input = torch.matmul(h_t_kprev, V.T)
                term2 = torch.where(z_t_k, with_input, without_input)

        return term1 + term2 + bias

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input_window.shape
        if seq_len > self.seq_len:
            raise ValueError(f"seq_len={seq_len} 超过模型初始化长度 {self.seq_len}")

        observed_mask = ~torch.isnan(input_window)
        x = torch.nan_to_num(input_window, nan=0.0)
        boundary_template = self._step(self.initial_boundary[:, :seq_len])

        zero_state = x.new_zeros((batch_size, self.hidden_size))
        h_prev_time = [zero_state.clone() for _ in range(self.num_layers)]
        c_prev_time = [zero_state.clone() for _ in range(self.num_layers)]
        s_prev_time = [boundary_template[k, 0].clone() for k in range(self.num_layers)]
        h_history: list[torch.Tensor] = []

        for t in range(seq_len):
            h_curr = [zero_state.clone() for _ in range(self.num_layers)]
            c_curr = [zero_state.clone() for _ in range(self.num_layers)]
            s_curr = [boundary_template[k, t].clone() for k in range(self.num_layers)]

            for k in range(self.num_layers):
                k_prev = max(k - 1, 0)
                k_next = min(k + 1, self.num_layers - 1)

                group_idx = self.chunking_idx[k]
                x_t_k = x[:, t, group_idx]
                z_t_k = observed_mask[:, t, group_idx].any(dim=1, keepdim=True)

                h_tprev_k = h_prev_time[k]
                h_t_kprev = h_curr[k_prev]
                h_tprev_knext = h_prev_time[k_next]
                c_tprev_k = c_prev_time[k]

                bias = self.b_list[k]
                W = self.W_list[k]
                Z = self.Z_list[k]
                U = self.U_list[k]
                V = self.V_list[k]
                J = self.J_list[k]

                s_tprev_k = s_prev_time[k] if t > 0 else boundary_template[k, 0]
                operator = self._select_operator(s_tprev_k, s_curr[k_prev])

                start = 0 * self.hidden_size
                end = 1 * self.hidden_size
                f_t_k = torch.sigmoid(
                    self._compute_gate(
                        W[start:end],
                        Z[start:end],
                        U[start:end],
                        V[start:end],
                        J[start:end],
                        bias[start:end],
                        x_t_k,
                        z_t_k,
                        h_tprev_k,
                        h_t_kprev,
                        h_tprev_knext,
                        operator,
                        True,
                    )
                )

                start = 1 * self.hidden_size
                end = 2 * self.hidden_size
                g_t_k = torch.tanh(
                    self._compute_gate(
                        W[start:end],
                        Z[start:end],
                        U[start:end],
                        V[start:end],
                        J[start:end],
                        bias[start:end],
                        x_t_k,
                        z_t_k,
                        h_tprev_k,
                        h_t_kprev,
                        h_tprev_knext,
                        operator,
                        False,
                    )
                )

                start = 2 * self.hidden_size
                end = 3 * self.hidden_size
                i_t_k = torch.sigmoid(
                    self._compute_gate(
                        W[start:end],
                        Z[start:end],
                        U[start:end],
                        V[start:end],
                        J[start:end],
                        bias[start:end],
                        x_t_k,
                        z_t_k,
                        h_tprev_k,
                        h_t_kprev,
                        h_tprev_knext,
                        operator,
                        False,
                    )
                )

                start = 3 * self.hidden_size
                end = 4 * self.hidden_size
                o_t_k = torch.sigmoid(
                    self._compute_gate(
                        W[start:end],
                        Z[start:end],
                        U[start:end],
                        V[start:end],
                        J[start:end],
                        bias[start:end],
                        x_t_k,
                        z_t_k,
                        h_tprev_k,
                        h_t_kprev,
                        h_tprev_knext,
                        operator,
                        False,
                    )
                )

                start = 4 * self.hidden_size
                end = 5 * self.hidden_size
                s_t_k = self._hard_sigmoid(
                    self._compute_gate(
                        W[start:end],
                        Z[start:end],
                        U[start:end],
                        V[start:end],
                        J[start:end],
                        bias[start:end],
                        x_t_k,
                        z_t_k,
                        h_tprev_k,
                        h_t_kprev,
                        h_tprev_knext,
                        operator,
                        False,
                    )
                )
                s_curr[k] = self._step(torch.mean(s_t_k))

                if operator == 1:
                    c_t_k = f_t_k * c_tprev_k + i_t_k * g_t_k
                elif operator in (2, 3):
                    c_t_k = i_t_k * g_t_k
                else:
                    c_t_k = c_tprev_k

                if operator == 4:
                    h_t_k = o_t_k * torch.tanh(c_t_k)
                else:
                    h_t_k = h_tprev_k

                h_curr[k] = h_t_k
                c_curr[k] = c_t_k

            h_history.append(torch.stack(h_curr, dim=0))
            h_prev_time = h_curr
            c_prev_time = c_curr
            s_prev_time = s_curr

        h = torch.stack(h_history, dim=2)
        excited_state = self.hse(h)
        return self.output_layer(excited_state)


def load_and_preprocess_data():
    data_dir = PROJECT_ROOT / "data"
    train_pattern = "train_*.csv"
    test_pattern = "test_*.csv"

    print("Loading training data...")
    train_raw, num_features = load_csv_dir_values(data_dir / "train", train_pattern)
    print(f"Training data: {train_raw.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_raw, _ = load_csv_dir_values(data_dir / "test", test_pattern)
    print(f"Test data: {test_raw.shape}")

    chunking_idx = infer_chunking_idx(train_raw)
    print(f"\nDetected {len(chunking_idx)} multirate groups: {[len(group) for group in chunking_idx]}")

    mins, scales = fit_minmax_ignore_nan(train_raw)
    train_data_norm = transform_minmax_preserve_nan(train_raw, mins, scales)
    test_data_norm = transform_minmax_preserve_nan(test_raw, mins, scales)

    return train_data_norm, test_data_norm, num_features, chunking_idx


def train_model(
    model: StandardMRALSTM,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float,
    lr_gamma: float,
    weight_decay: float,
    device: torch.device,
) -> list[float]:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    losses: list[float] = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            reconstruction = model(x)
            loss = masked_mse_loss(reconstruction, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        losses.append(avg_loss)

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] "
                f"Loss: {avg_loss:.6f}  LR: {current_lr:.6e}"
            )

    return losses


def compute_anomaly_scores(
    model: StandardMRALSTM,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_errors = []
    all_labels = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, labels in data_loader:
            x = x.to(device)
            reconstruction = model(x)

            sample_errors = masked_window_mse(reconstruction, x)
            all_errors.extend(sample_errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.append(reconstruction.cpu().numpy())
            all_targets.append(x.cpu().numpy())

    return (
        np.asarray(all_errors, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        np.vstack(all_predictions),
        np.vstack(all_targets),
    )


def plot_training_loss(losses: list[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sequence_length = 100
    batch_size = 50
    hidden_size = 80
    num_epochs = 10
    lr = 2.0e-4
    lr_gamma = 0.8
    weight_decay = 1.0e-6

    print("Loading and preprocessing data...")
    train_data, test_data, num_features, chunking_idx = load_and_preprocess_data()

    train_windows = build_front_padded_windows(
        train_data,
        sequence_length,
        stride=1,
    )
    test_windows, test_labels = build_prompt_test_windows_values(
        test_data,
        sequence_length,
        stride=1,
    )

    train_dataset = WindowDataset(train_windows)
    test_dataset = WindowDataset(test_windows, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Training windows: {len(train_dataset)}, Test windows: {len(test_dataset)}")

    model = StandardMRALSTM(
        input_size=num_features,
        hidden_size=hidden_size,
        seq_len=sequence_length,
        chunking_idx=chunking_idx,
    ).to(device)

    print("\nInitializing standard MRA-LSTM model...")
    print(f"Input size: {num_features}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of multirate groups: {len(chunking_idx)}")
    print(f"Group sizes: {[len(group) for group in chunking_idx]}")
    print(f"Sequence length: {sequence_length}")

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nTraining model...")
    losses = train_model(
        model,
        train_loader,
        num_epochs=num_epochs,
        lr=lr,
        lr_gamma=lr_gamma,
        weight_decay=weight_decay,
        device=device,
    )

    print("\nComputing training anomaly scores for threshold...")
    train_errors, _, _, _ = compute_anomaly_scores(model, train_eval_loader, device)
    if USE_EWAF:
        train_errors = apply_ewaf_by_segments(train_errors, EWAF_ALPHA)
    print(
        "Training error stats - "
        f"mean: {train_errors.mean():.6f}, std: {train_errors.std():.6f}, max: {train_errors.max():.6f}"
    )

    # threshold_train = choose_threshold(train_errors, method="mean")
    threshold_train = choose_threshold(train_errors, method="gaussian_quantile_max")
    print(f"Training-based threshold (mean train score): {threshold_train:.6f}")

    print("\nComputing test anomaly scores...")
    test_errors, labels, predictions, targets = compute_anomaly_scores(model, test_loader, device)
    split_idx = split_index_from_labels(labels)
    if USE_EWAF:
        test_errors = apply_ewaf_by_segments(
            test_errors,
            EWAF_ALPHA,
            infer_segment_lengths(labels),
        )
    print(
        "Test error stats - "
        f"mean: {test_errors.mean():.6f}, std: {test_errors.std():.6f}, max: {test_errors.max():.6f}"
    )

    y_pred = (test_errors > threshold_train).astype(int)
    metrics = compute_binary_classification_metrics(
        labels,
        y_pred,
        threshold=threshold_train,
        include_specificity=True,
        include_counts=True,
    )
    print("\nDetection Performance (using training-based threshold):")
    print_metrics(
        "",
        metrics,
        order=[
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
        ],
    )
    print(f"  Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_errors)}) anomaly")

    print("\nPlotting anomaly detection results...")
    plot_detection_scores(
        test_errors,
        threshold_train,
        split_idx,
        PROJECT_ROOT / "outputs" / "mra_lstm_detection.png",
        ylabel="重构误差",
        color_scheme="mra",
        show=True,
    )

    if losses:
        plot_training_loss(losses)

    return model, test_errors, labels, threshold_train, predictions, targets


if __name__ == "__main__":
    main()
