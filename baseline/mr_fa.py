import glob
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from window_utils import build_prompt_test_windows

plt.rcParams["font.sans-serif"] = ["SimHei"]


EPS = 1e-8
WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000


@dataclass
class RateGroup:
    name: str
    columns: np.ndarray
    interval: int
    train_mask: np.ndarray


def load_csv_dir(dir_path: str, file_pattern: str = "*.csv") -> np.ndarray:
    csv_files = sorted(glob.glob(os.path.join(dir_path, file_pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {dir_path}")

    arrays = []
    for file_path in csv_files:
        df = pd.read_csv(file_path, header=None)
        print(f"  Loaded {file_path}: {len(df)} rows, {df.shape[1]} cols")
        arrays.append(df.to_numpy(dtype=np.float64))

    return np.concatenate(arrays, axis=0)


def create_windows(data: np.ndarray, seq_len: int = 60, stride: int = 1) -> np.ndarray:
    """Follow baseline/cnn.py and build one front-padded window per sample."""
    n = len(data)
    if n == 0:
        return np.zeros((0, seq_len, data.shape[1]), dtype=np.float64)

    stop_idx = min(n, WINDOW_START_INDEX + WINDOW_SAMPLE_COUNT * stride)
    if stop_idx <= WINDOW_START_INDEX:
        return np.zeros((0, seq_len, data.shape[1]), dtype=np.float64)

    windows = []
    for i in range(WINDOW_START_INDEX, stop_idx, stride):
        if i < seq_len:
            pad_len = seq_len - i - 1
            window_data = np.concatenate(
                [np.tile(data[0:1], (pad_len, 1)), data[0 : i + 1]],
                axis=0,
            )
        else:
            window_data = data[i - seq_len + 1 : i + 1]
        windows.append(window_data)

    return np.stack(windows)


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
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_pattern = "train_*.csv"
    test_pattern = "test_*.csv"

    print("Loading training data...")
    train_raw = load_csv_dir(str(data_dir / "train"), train_pattern)
    print(f"Training data: {train_raw.shape}, num_features={train_raw.shape[1]}")

    print("\nLoading test data...")
    test_raw = load_csv_dir(str(data_dir / "test"), test_pattern)
    print(f"Test data: {test_raw.shape}")

    means, stds = fit_standardizer(train_raw)
    train_scaled = transform_observed(train_raw, means, stds)
    test_scaled = transform_observed(test_raw, means, stds)

    x_train = create_windows(train_scaled, seq_len=seq_len, stride=stride)
    x_test, test_labels = build_prompt_test_windows(
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
        self.loglik_history_: list[float] = []

    def fit(self, data: np.ndarray, groups: list[RateGroup]) -> "MultirateFactorAnalysis":
        self.groups = groups
        self._initialize_parameters(data)

        prev_loglik = -np.inf
        for iteration in range(1, self.max_iter + 1):
            posterior = self._posterior(data)
            self._m_step(data, posterior["means"], posterior["second_moments"])

            loglik = float(np.sum(self.score_samples(data)))
            self.loglik_history_.append(loglik)
            improvement = loglik - prev_loglik
            print(
                f"    EM iter {iteration:03d}/{self.max_iter}"
                f"  loglik={loglik:.4f}  delta={improvement:.6f}"
            )

            if iteration > 1 and abs(improvement) < self.tol * max(1.0, abs(prev_loglik)):
                break

            prev_loglik = loglik

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

    def _initialize_parameters(self, data: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        filled = np.nan_to_num(data, nan=0.0)

        u, s, _ = np.linalg.svd(filled, full_matrices=False)
        rank = min(self.n_factors, s.shape[0])
        latent = u[:, :rank] * s[:rank]
        if rank < self.n_factors:
            extra = rng.normal(scale=0.05, size=(data.shape[0], self.n_factors - rank))
            latent = np.hstack([latent, extra])

        latent -= latent.mean(axis=0, keepdims=True)
        cov = np.atleast_2d(np.cov(latent, rowvar=False, bias=True))
        cov += np.eye(cov.shape[0]) * self.min_noise
        eigvals, eigvecs = np.linalg.eigh(cov)
        whitening = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, self.min_noise, None))) @ eigvecs.T
        latent = latent @ whitening

        self.loadings_ = []
        self.noise_vars_ = []
        eye = np.eye(self.n_factors) * self.min_noise

        for group in self.groups:
            obs_idx = np.flatnonzero(group.train_mask)
            x_group = data[np.ix_(obs_idx, group.columns)]
            latent_obs = latent[obs_idx]
            sum_tt = latent_obs.T @ latent_obs + eye
            loading = x_group.T @ latent_obs @ np.linalg.inv(sum_tt)
            residual = x_group - latent_obs @ loading.T
            noise_var = np.var(residual, axis=0)
            noise_var = np.clip(noise_var, self.min_noise, None)
            self.loadings_.append(loading)
            self.noise_vars_.append(noise_var)

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
            obs_idx = np.flatnonzero(group.train_mask)
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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    normal_mask = y_true == 0
    fault_mask = y_true == 1
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fdr": float(np.mean(y_pred[fault_mask] == 1)) if np.any(fault_mask) else 0.0,
        "fra": float(np.mean(y_pred[normal_mask] == 1)) if np.any(normal_mask) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  FDR:       {metrics['fdr']:.4f}")
    print(f"  FRA:       {metrics['fra']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")


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


def plot_results(
    scores: np.ndarray,
    threshold: float,
    split_idx: int,
    save_path: Path | None = None,
) -> None:
    if save_path is None:
        save_path = Path(__file__).resolve().parent.parent / "outputs" / "mr_fa_detection.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(scores, label="测试异常分数", alpha=0.7)
    plt.axhline(y=threshold, color="r", linestyle="--", label=f"阈值 ({threshold:.4f})")
    plt.axvline(x=split_idx, color="g", linestyle=":", label="测试集分界")
    plt.xlabel("测试样本索引")
    plt.ylabel("异常分数")
    plt.title("MR-FA异常检测")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {save_path}")


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
    model.fit(train_data, groups)

    train_monitor = model.monitor(train_data)
    t2_limit = t2_control_limit(latent_dim, train_data.shape[0], alpha)
    spe_limits = np.array(
        [spe_control_limit(train_monitor["spe"][:, group_idx], alpha) for group_idx in range(len(groups))]
    )

    train_scores = score_window_dataset(model, x_train, t2_limit, spe_limits)
    test_scores = score_window_dataset(model, x_test, t2_limit, spe_limits)
    threshold = float(np.mean(train_scores))

    y_true = test_labels
    split_idx = int(np.sum(y_true == 0))
    y_pred = (test_scores > threshold).astype(int)

    print("\n--- 异常检测评估结果 ---")
    print(f"Latent factors: {latent_dim}")
    print(f"Window size: {seq_len}, stride: {stride}, num_features={num_features}")
    print(f"Train score: mean={np.mean(train_scores):.6f}, std={np.std(train_scores):.6f}")
    print(f"Threshold (mean train score): {threshold:.6f}")
    print(f"Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores)}) anomaly")
    print(f"Anomalies detected: {(y_pred == 1).sum()} / {len(y_pred)}")

    metrics = evaluate_predictions(y_true, y_pred)
    print_metrics("\nClassification Metrics:", metrics)

    plot_results(
        scores=test_scores,
        threshold=threshold,
        split_idx=split_idx,
    )


if __name__ == "__main__":
    train_model()
