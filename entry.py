import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from model.mstransformer import MSTransformer

import random


def seed_everything(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(40)


# ==========================================
# 1. 数据加载与预处理模块
# ==========================================
def generate_mask_matrix():
    """读取之前生成的块状分布 CSV 数据"""
    try:
        # 读取 CSV，保留表头以便确认列名
        df = pd.read_csv("./TEP_3000_Block_Split.csv")
        # 提取 xmeas_1 到 xmeas_41 (丢弃后面的 xmv)
        data_df = df.filter(like="xmeas_").iloc[:, :41]
        data = data_df.astype(float).to_numpy()
        # 1 代表缺失 (NaN), 0 代表观测值
        mask = np.isnan(data).astype(int)
        return data, mask
    except FileNotFoundError:
        print("找不到数据文件，请确保 TEP_3000_Block_Split.csv 在当前目录下")
        return None, None


def create_windows(data, mask, seq_len, stride=1):
    """创建滑动窗口，与 mra.py 的 create_windows 逻辑一致"""
    X, M = [], []
    n = len(data)

    if n == 0:
        return np.zeros((0, seq_len, data.shape[1])), np.zeros((0, seq_len, data.shape[1]))

    for i in range(0, n, stride):
        if i < seq_len:
            # Front-pad by repeating the first sample
            pad_len = seq_len - i - 1
            window_data = np.concatenate([
                np.tile(data[0:1], (pad_len, 1)),  # repeat first sample
                data[0 : i + 1]
            ], axis=0)
            window_mask = np.concatenate([
                np.tile(mask[0:1], (pad_len, 1)),
                mask[0 : i + 1]
            ], axis=0)
        else:
            # Normal lookback: take the previous seq_len samples
            window_data = data[i - seq_len + 1 : i + 1]
            window_mask = mask[i - seq_len + 1 : i + 1]

        X.append(window_data)
        M.append(window_mask)

    return np.stack(X), np.stack(M)


# ==========================================
# 2. 核心检测与绘图逻辑
# ==========================================
def run_full_detection():
    # --- A. 准备数据 ---
    raw_data, raw_mask = generate_mask_matrix()
    if raw_data is None:
        return

    # 处理 NaN 以便进行标准化 (StandardScaler 不接受 NaN)
    temp_data = np.nan_to_num(raw_data, nan=0.0)
    scaler = StandardScaler()
    scaler.fit(temp_data[:1500])
    scaled_data = scaler.transform(temp_data)

    # --- B. 模型参数配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 41
    window_size = 60  # 滑动窗口长度
    s_rate = 3  # 采样类型

    # --- C. 创建滑动窗口 (与 mra.py 一致) ---
    # 将 mask 转换为观测掩码: 1=观测到, 0=缺失 (与 mra.py 一致)
    obs_mask = 1 - raw_mask.astype(np.float32)
    X_windows, M_windows = create_windows(scaled_data.astype(np.float32), obs_mask, seq_len=window_size, stride=1)

    split = 1500  # 正常/故障分界点
    X_train, M_train = X_windows[:split], M_windows[:split]
    X_test, M_test = X_windows[split:], M_windows[split:]

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

    print(f"正在开始检测... 设备: {device}")

    # 强制开启 eval 模式
    model.eval()

    # --- E. 计算训练集异常分数 (用于确定阈值) ---
    train_scores = []
    with torch.no_grad():
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(M_train)),
            batch_size=32
        )
        for x, m in train_loader:
            x, m = x.to(device), m.to(device)

            # 构造 enc_types
            enc_types = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1) % s_rate
            pred_types = torch.randint(0, s_rate, (x.size(0), 0), dtype=torch.long).to(device)

            # 推理
            reconstructed = model(
                x, x, m, m,
                enc_types, pred_types,
                IFALL=1,
            )

            # 计算重构误差
            sq_err = ((reconstructed - x) * m).pow(2).sum(dim=[1, 2])
            obs_cnt = m.sum(dim=[1, 2]).clamp_min(1e-8)
            train_scores.extend((sq_err / obs_cnt).cpu().numpy())

    train_scores = np.array(train_scores)
    threshold = np.mean(train_scores) + 3 * np.std(train_scores)

    print(f"\nTraining Set Score Stats:")
    print(f"  Mean: {np.mean(train_scores):.6f}")
    print(f"  Std:  {np.std(train_scores):.6f}")
    print(f"  Threshold (mean + 3*std): {threshold:.6f}")

    # --- F. 计算测试集异常分数 ---
    test_scores = []
    with torch.no_grad():
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(M_test)),
            batch_size=32
        )
        for x, m in test_loader:
            x, m = x.to(device), m.to(device)

            # 构造 enc_types
            enc_types = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1) % s_rate
            pred_types = torch.randint(0, s_rate, (x.size(0), 0), dtype=torch.long).to(device)

            # 推理
            reconstructed = model(
                x, x, m, m,
                enc_types, pred_types,
                IFALL=1,
            )

            # 计算重构误差
            sq_err = ((reconstructed - x) * m).pow(2).sum(dim=[1, 2])
            obs_cnt = m.sum(dim=[1, 2]).clamp_min(1e-8)
            test_scores.extend((sq_err / obs_cnt).cpu().numpy())

    test_scores_arr = np.array(test_scores)

    # Splice train scores to front of test scores for evaluation
    all_scores = np.concatenate([train_scores, test_scores_arr])

    print(f"\nAnomaly Detection Results:")
    print(f"  Mean Score: {np.mean(all_scores):.6f}")
    print(f"  Std Score:  {np.std(all_scores):.6f}")
    print(f"  Threshold (from train): {threshold:.6f}")
    print(f"  Anomalies detected: {(all_scores > threshold).sum()} / {len(all_scores)}")

    # Classification Metrics
    # Labels: 0 for train (normal), 1 for test (anomaly)
    y_true = np.concatenate([np.zeros(len(train_scores), dtype=int), np.ones(len(test_scores_arr), dtype=int)])
    y_pred = (all_scores > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # --- G. 结果可视化 ---
    plot_results(all_scores, threshold)


def plot_results(scores, threshold):
    """可视化异常检测结果"""
    plt.figure(figsize=(6, 5))

    plt.plot(scores, label='异常分数', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    plt.xlabel('样本索引')
    plt.ylabel('重构误差')
    plt.title('AGF-ADNet异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/akira/codespace/mra-detection/anomaly_detection_metrics.png', dpi=150)
    plt.show()
    print("图表已保存为 anomaly_detection_metrics.png")


if __name__ == "__main__":
    run_full_detection()

