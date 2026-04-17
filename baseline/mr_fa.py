from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, f

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

plt.rcParams["font.family"] = ["SimSun"]
plt.rcParams["font.sans-serif"] = ["SimSun", "SimSun-ExtB", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


EPS = 1e-8
USE_EWAF = True
EWAF_ALPHA = 0.15


@dataclass
class RateGroup:
    name: str
    columns: np.ndarray
    interval: int
    train_mask: np.ndarray

def fit_standardizer(train_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(train_data, axis=0)
    stds = np.nanstd(train_data, axis=0)
    stds[~np.isfinite(stds) | (stds < EPS)] = 1.0
    return means, stds


def transform_observed(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    transformed = (data - means) / stds
    transformed[np.isnan(data)] = np.nan
    return transformed


def infer_rate_groups(train_data: np.ndarray) -> list[RateGroup]:
    mask = ~np.isnan(train_data)
    grouped_columns: dict[bytes, list[int]] = {}

    for column_idx in range(mask.shape[1]):
        grouped_columns.setdefault(mask[:, column_idx].tobytes(), []).append(column_idx)

    groups: list[RateGroup] = []
    for columns in grouped_columns.values():
        ref_mask = mask[:, columns[0]]
        for column_idx in columns[1:]:
            if not np.array_equal(ref_mask, mask[:, column_idx]):
                raise ValueError("Columns in the same rate group do not share the same observation mask.")

        observed_idx = np.flatnonzero(ref_mask)
        if len(observed_idx) <= 1:
            interval = 1
        else:
            interval = int(np.gcd.reduce(np.diff(observed_idx)))

        groups.append(
            RateGroup(
                name=f"rate_{interval}",
                columns=np.array(columns, dtype=int),
                interval=interval,
                train_mask=ref_mask.copy(),
            )
        )

    groups.sort(key=lambda item: (item.interval, item.columns[0]))
    return groups


def infer_row_patterns(data: np.ndarray, groups: list[RateGroup]) -> tuple[np.ndarray, dict[tuple[bool, ...], np.ndarray]]:
    phi = np.column_stack([~np.isnan(data[:, group.columns[0]]) for group in groups])
    pattern_to_indices: dict[tuple[bool, ...], list[int]] = {}

    for row_idx, pattern in enumerate(phi):
        pattern_to_indices.setdefault(tuple(bool(flag) for flag in pattern.tolist()), []).append(row_idx)

    return phi, {pattern: np.asarray(indices, dtype=int) for pattern, indices in pattern_to_indices.items()}

def prepare_data(
    seq_len: int = 60,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[RateGroup], int]:
    data_dir = PROJECT_ROOT / "data"
    train_pattern = "train_*.csv"
    test_pattern = "test_*.csv"

    print("Loading training data...")
    train_raw, _ = load_csv_dir_values(
        str(data_dir / "train"),
        train_pattern,
        dtype=np.float64,
    )
    print(f"Training data: {train_raw.shape}, num_features={train_raw.shape[1]}")

    print("\nLoading test data...")
    test_raw, _ = load_csv_dir_values(
        str(data_dir / "test"),
        test_pattern,
        dtype=np.float64,
    )
    print(f"Test data: {test_raw.shape}")

    means, stds = fit_standardizer(train_raw)
    train_scaled = transform_observed(train_raw, means, stds)
    test_scaled = transform_observed(test_raw, means, stds)

    x_train = build_front_padded_windows(
        train_scaled,
        seq_len=seq_len,
        stride=stride,
    )
    x_test, test_labels = build_prompt_test_windows_values(
        test_scaled,
        seq_len=seq_len,
        stride=stride,
    )
    groups = infer_rate_groups(train_scaled)

    return train_scaled, test_scaled, x_train, x_test, test_labels, groups, train_raw.shape[1]


class MultirateFactorAnalysis:
    def __init__(
        self,
        n_factors: int,
        max_iter: int = 100,
        tol: float = 1e-5,
        min_noise: float = 1e-5,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.min_noise = min_noise
        self.random_state = random_state
        self.groups: list[RateGroup] = []
        self.loadings_: list[np.ndarray] = []
        self.noise_vars_: list[np.ndarray] = []
        self.train_loss_history_: list[float] = []
        self.loglik_history_: list[float] = []

    def fit(self, data: np.ndarray, groups: list[RateGroup]) -> "MultirateFactorAnalysis":
        train_rows = self._prepare_training_rows(data)
        self.groups = groups
        self._initialize_parameters(train_rows)
        self.train_loss_history_.clear()
        self.loglik_history_.clear()

        prev_loss = np.inf
        for iteration in range(1, self.max_iter + 1):
            posterior = self._posterior(train_rows)
            reconstruction = self._reconstruct_from_means(posterior["means"])
            train_loss = self._masked_mse(train_rows, reconstruction)
            self.train_loss_history_.append(train_loss)
            self._m_step(train_rows, posterior["means"], posterior["second_moments"])

            improvement = prev_loss - train_loss
            print(
                f"    EM iter {iteration:03d}/{self.max_iter}"
                f"  train_loss={train_loss:.8f}  delta={improvement:.8f}"
            )

            if iteration > 1 and abs(improvement) < self.tol * max(1.0, abs(prev_loss)):
                break

            prev_loss = train_loss

        return self

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        scores = np.zeros(data.shape[0], dtype=np.float64)
        _, pattern_to_indices = infer_row_patterns(data, self.groups)

        for pattern, indices in pattern_to_indices.items():
            observed_groups = [group_idx for group_idx, is_observed in enumerate(pattern) if is_observed]
            if not observed_groups:
                continue

            observed_matrix = self._stack_observed_rows(data[indices], observed_groups)
            covariance = self._pattern_covariance(observed_groups)
            covariance += np.eye(covariance.shape[0]) * self.min_noise

            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise np.linalg.LinAlgError("Observed covariance is not positive definite.")

            covariance_inv = np.linalg.inv(covariance)
            quad_form = np.sum((observed_matrix @ covariance_inv) * observed_matrix, axis=1)
            dim = observed_matrix.shape[1]
            scores[indices] = -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad_form)

        return scores

    def monitor(self, data: np.ndarray) -> dict[str, np.ndarray]:
        posterior = self._posterior(data)
        means = posterior["means"]
        precisions = posterior["precisions"]
        covariances = posterior["covariances"]
        phi = posterior["phi"]

        # Paper Eq. (18): T^2 = t_hat^T Sigma(k)^{-1} t_hat, where Sigma(k)
        # in Eq. (5) is the precision matrix. Therefore, the quadratic form
        # is weighted by the posterior covariance.
        t2 = np.einsum("bi,bij,bj->b", means, covariances, means)
        spe = np.full((data.shape[0], len(self.groups)), np.nan, dtype=np.float64)

        for group_idx, group in enumerate(self.groups):
            row_idx = np.flatnonzero(phi[:, group_idx])
            if len(row_idx) == 0:
                continue

            x_group = data[np.ix_(row_idx, group.columns)]
            reconstruction = means[row_idx] @ self.loadings_[group_idx].T
            residual = x_group - reconstruction
            spe[row_idx, group_idx] = np.sum(residual * residual, axis=1)

        return {
            "t2": t2,
            "spe": spe,
            "means": means,
            "precisions": precisions,
            "covariances": covariances,
            "phi": phi,
        }

    def _prepare_training_rows(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            num_rows = data.shape[0] * data.shape[1]
            return data.reshape(num_rows, data.shape[2])
        raise ValueError(f"Unsupported training data ndim={data.ndim}, expected 2 or 3.")

    def _initialize_parameters(self, data: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)

        self.loadings_ = []
        self.noise_vars_ = []

        for group in self.groups:
            d_group = len(group.columns)
            self.loadings_.append(rng.normal(size=(d_group, self.n_factors)))
            self.noise_vars_.append(np.ones(d_group, dtype=np.float64))

    def _posterior(self, data: np.ndarray) -> dict[str, np.ndarray]:
        num_rows = data.shape[0]
        means = np.zeros((num_rows, self.n_factors), dtype=np.float64)
        precisions = np.zeros((num_rows, self.n_factors, self.n_factors), dtype=np.float64)
        covariances = np.zeros((num_rows, self.n_factors, self.n_factors), dtype=np.float64)
        second_moments = np.zeros((num_rows, self.n_factors, self.n_factors), dtype=np.float64)
        phi, pattern_to_indices = infer_row_patterns(data, self.groups)

        for pattern, indices in pattern_to_indices.items():
            observed_groups = [group_idx for group_idx, is_observed in enumerate(pattern) if is_observed]
            precision = np.eye(self.n_factors, dtype=np.float64)

            for group_idx in observed_groups:
                loading = self.loadings_[group_idx]
                noise_inv = 1.0 / self.noise_vars_[group_idx]
                precision += (loading.T * noise_inv) @ loading

            precision += np.eye(self.n_factors) * self.min_noise
            covariance = np.linalg.inv(precision)

            rhs = np.zeros((len(indices), self.n_factors), dtype=np.float64)
            for group_idx in observed_groups:
                group = self.groups[group_idx]
                x_group = data[np.ix_(indices, group.columns)]
                rhs += (x_group / self.noise_vars_[group_idx]) @ self.loadings_[group_idx]

            means[indices] = rhs @ covariance
            precisions[indices] = precision
            covariances[indices] = covariance
            second_moments[indices] = covariance + np.einsum(
                "bi,bj->bij", means[indices], means[indices]
            )

        return {
            "means": means,
            "precisions": precisions,
            "covariances": covariances,
            "second_moments": second_moments,
            "phi": phi,
        }

    def _m_step(self, data: np.ndarray, means: np.ndarray, second_moments: np.ndarray) -> None:
        eye = np.eye(self.n_factors) * self.min_noise

        for group_idx, group in enumerate(self.groups):
            obs_idx = self._observed_row_indices(data, group)
            if len(obs_idx) == 0:
                continue
            x_group = data[np.ix_(obs_idx, group.columns)]
            mean_group = means[obs_idx]
            second_sum = np.sum(second_moments[obs_idx], axis=0) + eye

            loading = x_group.T @ mean_group @ np.linalg.inv(second_sum)
            sample_cov = (x_group.T @ x_group) / len(obs_idx)
            cross_term = loading @ (mean_group.T @ x_group / len(obs_idx))
            noise_var = np.diag(sample_cov - cross_term)
            noise_var = np.clip(noise_var, self.min_noise, None)

            self.loadings_[group_idx] = loading
            self.noise_vars_[group_idx] = noise_var

    def _pattern_covariance(self, observed_groups: list[int]) -> np.ndarray:
        loadings = [self.loadings_[group_idx] for group_idx in observed_groups]
        block_loadings = np.vstack(loadings)
        block_noise = np.concatenate([self.noise_vars_[group_idx] for group_idx in observed_groups])
        covariance = block_loadings @ block_loadings.T + np.diag(block_noise)
        return covariance

    def _observed_row_indices(self, data: np.ndarray, group: RateGroup) -> np.ndarray:
        return np.flatnonzero(~np.isnan(data[:, group.columns[0]]))

    def _reconstruct_from_means(self, means: np.ndarray) -> np.ndarray:
        num_rows = means.shape[0]
        num_features = sum(len(group.columns) for group in self.groups)
        reconstruction = np.zeros((num_rows, num_features), dtype=np.float64)
        for group_idx, group in enumerate(self.groups):
            reconstruction[:, group.columns] = means @ self.loadings_[group_idx].T
        return reconstruction

    def _masked_mse(self, data: np.ndarray, reconstruction: np.ndarray) -> float:
        observed = ~np.isnan(data)
        if not np.any(observed):
            return 0.0
        residual = np.nan_to_num(data, nan=0.0) - reconstruction
        return float(np.mean(np.square(residual[observed])))

    def _stack_observed_rows(self, data: np.ndarray, observed_groups: list[int]) -> np.ndarray:
        blocks = [data[:, self.groups[group_idx].columns] for group_idx in observed_groups]
        return np.concatenate(blocks, axis=1)


def t2_control_limit(num_factors: int, num_samples: int, alpha: float) -> float:
    if num_samples <= num_factors + 1:
        return np.inf
    scale = num_factors * (num_samples**2 - 1.0) / (num_samples * (num_samples - num_factors))
    return float(scale * f.ppf(alpha, num_factors, num_samples - num_factors))


def spe_control_limit(scores: np.ndarray, alpha: float) -> float:
    valid = scores[np.isfinite(scores)]
    if len(valid) == 0:
        return np.inf

    mean = float(np.mean(valid))
    var = float(np.var(valid))

    if mean <= EPS or var <= EPS:
        return float(np.quantile(valid, alpha))

    h = 2.0 * mean * mean / var
    g = var / (2.0 * mean)
    return float(g * chi2.ppf(alpha, df=h))


def score_window_dataset(
    model: MultirateFactorAnalysis,
    windows: np.ndarray,
    t2_limit: float,
    spe_limits: np.ndarray,
) -> np.ndarray:
    num_windows, seq_len, num_features = windows.shape
    flat_windows = windows.reshape(num_windows * seq_len, num_features)
    monitor = model.monitor(flat_windows)

    t2_ratio = monitor["t2"] / max(t2_limit, EPS)
    spe_ratio = np.nanmax(monitor["spe"] / spe_limits, axis=1)
    row_scores = np.maximum(t2_ratio, spe_ratio)

    return row_scores.reshape(num_windows, seq_len).mean(axis=1)

def train_model() -> None:
    seq_len = 50
    stride = 1
    latent_dim = 3
    alpha = 0.99
    max_iter = 80

    train_data, test_data, x_train, x_test, test_labels, groups, num_features = prepare_data(
        seq_len=seq_len,
        stride=stride,
    )

    print("\nInferred multirate groups:")
    for group in groups:
        sample_count = int(group.train_mask.sum())
        columns = ", ".join(str(column) for column in group.columns.tolist())
        print(
            f"  {group.name}: columns=[{columns}], interval={group.interval},"
            f" training_samples={sample_count}"
        )

    print("\n开始训练 MR-FA 模型...")
    model = MultirateFactorAnalysis(
        n_factors=latent_dim,
        max_iter=max_iter,
        tol=1e-5,
        random_state=42,
    )
    print(
        f"Using standard-style training samples: windows={x_train.shape[0]}, "
        f"seq_len={x_train.shape[1]}, flattened_rows={x_train.shape[0] * x_train.shape[1]}"
    )
    model.fit(x_train, groups)

    train_monitor = model.monitor(train_data)
    t2_limit = t2_control_limit(latent_dim, train_data.shape[0], alpha)
    spe_limits = np.array(
        [spe_control_limit(train_monitor["spe"][:, group_idx], alpha) for group_idx in range(len(groups))]
    )

    train_scores = score_window_dataset(model, x_train, t2_limit, spe_limits)
    test_scores = score_window_dataset(model, x_test, t2_limit, spe_limits)
    if USE_EWAF:
        train_scores = apply_ewaf_by_segments(train_scores, EWAF_ALPHA)
    threshold = choose_threshold(train_scores, method="mean")

    y_true = test_labels
    split_idx = split_index_from_labels(y_true)
    if USE_EWAF:
        test_scores = apply_ewaf_by_segments(
            test_scores,
            EWAF_ALPHA,
            infer_segment_lengths(y_true),
        )
    y_pred = (test_scores > threshold).astype(int)

    print("\n--- 异常检测评估结果 ---")
    print(f"Latent factors: {latent_dim}")
    print(f"Window size: {seq_len}, stride: {stride}, num_features={num_features}")
    print(f"Train score: mean={np.mean(train_scores):.6f}, std={np.std(train_scores):.6f}")
    print(f"Threshold (mean train score): {threshold:.6f}")
    print(f"Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores)}) anomaly")
    print(f"Anomalies detected: {(y_pred == 1).sum()} / {len(y_pred)}")

    metrics = compute_binary_classification_metrics(y_true, y_pred)
    print_metrics(
        "\nClassification Metrics:",
        metrics,
        order=["accuracy", "precision", "recall", "fdr", "fra", "f1"],
    )

    plot_detection_scores(
        test_scores,
        threshold,
        split_idx,
        PROJECT_ROOT / "outputs" / "mr_fa_detection.png",
        ylabel="异常分数",
        color_scheme="mra",
    )


if __name__ == "__main__":
    train_model()
