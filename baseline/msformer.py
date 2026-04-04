import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from model.mstransformer import MSTransformer
from _project_root import PROJECT_ROOT
from utils.methods.data_loading import load_csv_dir_with_mask
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
    build_front_padded_windows_with_mask,
    build_prompt_test_windows,
)

import random


WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000
USE_EWAF = True
EWAF_ALPHA = 0.15


def seed_everything(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(40)


def build_type_index(seq_len, batch_size, sampling_rate, device):
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) % sampling_rate


def mask_input(x, missing_mask):
    return x.masked_fill(missing_mask.bool(), 0.0)


def train_model(model, train_loader, device, sampling_rate, epochs=50, lr=1e-3, mask_ratio=0.15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x, missing_mask in train_loader:
            x = x.to(device)
            missing_mask = missing_mask.to(device)

            observed = ~missing_mask.bool()
            random_hide = (torch.rand_like(x) < mask_ratio) & observed
            loss_mask = random_hide
            if not loss_mask.any():
                loss_mask = observed

            input_mask = missing_mask.clone()
            input_mask[random_hide] = 1.0
            x_input = mask_input(x, input_mask)

            enc_types = build_type_index(x.size(1), x.size(0), sampling_rate, device)
            pred_types = torch.empty((x.size(0), 0), dtype=torch.long, device=device)

            reconstructed = model(
                x_input, x_input, input_mask, input_mask,
                enc_types, pred_types,
                IFALL=0,
            )

            loss_weights = loss_mask.float()
            loss = ((reconstructed - x).pow(2) * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1:03d}/{epochs}  Loss: {avg_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss


def score_dataset(model, windows, masks, device, sampling_rate, batch_size=32):
    scores = []
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size
    )

    model.eval()
    with torch.no_grad():
        for x, missing_mask in loader:
            x = x.to(device)
            missing_mask = missing_mask.to(device)
            observed = (~missing_mask.bool()).float()
            x_input = mask_input(x, missing_mask)

            enc_types = build_type_index(x.size(1), x.size(0), sampling_rate, device)
            pred_types = torch.empty((x.size(0), 0), dtype=torch.long, device=device)

            reconstructed = model(
                x_input, x_input, missing_mask, missing_mask,
                enc_types, pred_types,
                IFALL=0,
            )

            sq_err = ((reconstructed - x) * observed).pow(2).sum(dim=[1, 2])
            obs_cnt = observed.sum(dim=[1, 2]).clamp_min(1e-8)
            scores.extend((sq_err / obs_cnt).cpu().numpy())

    return np.array(scores)

# ==========================================
# 2. 核心检测与绘图逻辑
# ==========================================
def run_full_detection():
    # --- A. 准备数据 ---
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_PATTERN = "train_*.csv"
    TEST_PATTERN  = "test_*.csv"
    CKPT_PATH = PROJECT_ROOT / "mstransformer_model.pth"

    print("Loading training data...")
    train_data, train_mask, num_features = load_csv_dir_with_mask(
        DATA_DIR / "train",
        TRAIN_PATTERN,
    )
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, test_mask, _ = load_csv_dir_with_mask(
        DATA_DIR / "test",
        TEST_PATTERN,
    )
    print(f"Test data: {test_data.shape}")

    # 处理 NaN
    train_filled = np.nan_to_num(train_data, nan=0.0)
    test_filled = np.nan_to_num(test_data, nan=0.0)

    # 标准化 (fit on training data only)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled).astype(np.float32)
    test_scaled = scaler.transform(test_filled).astype(np.float32)

    # --- B. 模型参数配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = num_features
    window_size = 50
    s_rate = 6
    train_epochs = 10
    batch_size = 32
    mask_ratio = 0.15

    # --- C. 创建滑动窗口 ---
    X_train, M_train = build_front_padded_windows_with_mask(
        train_scaled,
        train_mask,
        seq_len=window_size,
        stride=1,
        start_index=WINDOW_START_INDEX,
        max_window_count=WINDOW_SAMPLE_COUNT,
    )
    X_test, M_test, test_labels = build_prompt_test_windows(
        test_scaled,
        test_mask,
        seq_len=window_size,
        stride=1,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(M_train)),
        batch_size=batch_size,
        shuffle=True
    )

    # --- D. 初始化模型 ---
    model = MSTransformer(
        enc_in=feature_dim,
        dec_in=feature_dim,
        c_out=feature_dim,
        d_model=64,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        dff=256,
        query_size=16,
        value_size=16,
        sampling_rate=s_rate,
    ).to(device)

    print(f"开始训练并检测... 设备: {device}")
    print("Mask convention: 1 = missing, 0 = observed")

    model, best_loss = train_model(
        model,
        train_loader,
        device=device,
        sampling_rate=s_rate,
        epochs=train_epochs,
        lr=1e-3,
        mask_ratio=mask_ratio,
    )
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Best training loss: {best_loss:.6f}")
    print(f"Model saved to {CKPT_PATH}")

    # --- E. 计算训练集异常分数 (用于确定阈值) ---
    train_scores = score_dataset(model, X_train, M_train, device=device, sampling_rate=s_rate, batch_size=batch_size)
    if USE_EWAF:
        train_scores = apply_ewaf_by_segments(train_scores, EWAF_ALPHA)
    threshold = choose_threshold(train_scores, method="mean_std", std_factor=1.0)

    print(f"\nTraining Set Score Stats:")
    print(f"  Mean: {np.mean(train_scores):.6f}")
    print(f"  Std:  {np.std(train_scores):.6f}")
    print(f"  Threshold (mean train score): {threshold:.6f}")

    # --- F. 计算测试集异常分数 ---
    test_scores_arr = score_dataset(model, X_test, M_test, device=device, sampling_rate=s_rate, batch_size=batch_size)
    split_idx = split_index_from_labels(test_labels)
    if USE_EWAF:
        test_scores_arr = apply_ewaf_by_segments(
            test_scores_arr,
            EWAF_ALPHA,
            infer_segment_lengths(test_labels),
        )

    print(f"\nAnomaly Detection Results:")
    print(f"  Mean Score: {np.mean(test_scores_arr):.6f}")
    print(f"  Std Score:  {np.std(test_scores_arr):.6f}")
    print(f"  Threshold (from train): {threshold:.6f}")
    print(f"  Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores_arr)}) anomaly")
    print(f"  Anomalies detected: {(test_scores_arr > threshold).sum()} / {len(test_scores_arr)}")

    # Classification Metrics
    y_true = test_labels
    y_pred = (test_scores_arr > threshold).astype(int)

    metrics = compute_binary_classification_metrics(y_true, y_pred)
    print_metrics(
        "\nClassification Metrics:",
        metrics,
        order=["accuracy", "precision", "recall", "fdr", "fra", "f1"],
    )

    # --- G. 结果可视化 ---
    plot_detection_scores(
        test_scores_arr,
        threshold,
        split_idx,
        PROJECT_ROOT / "anomaly_detection_metrics.png",
        title="Multirate Former 异常检测",
        ylabel="重构误差",
        show=True,
    )


if __name__ == "__main__":
    run_full_detection()
