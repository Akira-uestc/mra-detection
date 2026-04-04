import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from pathlib import Path

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

plt.rcParams['font.sans-serif'] = ['SimHei']

WINDOW_START_INDEX = 49
WINDOW_SAMPLE_COUNT = 4000
TEST_SPLIT_INDEX = 2000
USE_EWAF = True
EWAF_ALPHA = 0.15

# 加载数据
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATTERN = "train_*.csv"
TEST_PATTERN  = "test_*.csv"

print("Loading training data...")
train_data_raw, num_features = load_csv_dir_values(str(DATA_DIR / "train"), TRAIN_PATTERN)
print(f"Training data: {train_data_raw.shape}, num_features={num_features}")

print("\nLoading test data...")
test_data_raw, _ = load_csv_dir_values(str(DATA_DIR / "test"), TEST_PATTERN)
print(f"Test data: {test_data_raw.shape}")

# 处理缺失值
train_data_raw = np.nan_to_num(train_data_raw, nan=0.0)
test_data_raw = np.nan_to_num(test_data_raw, nan=0.0)

# 归一化 (fit on training data only)
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data_raw)
test_data_norm = scaler.transform(test_data_raw)

# 创建时间序列窗口
SEQ_LENGTH = 50

X_train = build_front_padded_windows(
    train_data_norm,
    SEQ_LENGTH,
    stride=1,
    start_index=WINDOW_START_INDEX,
    max_window_count=WINDOW_SAMPLE_COUNT,
)
X_test, test_labels = build_prompt_test_windows_values(
    test_data_norm,
    SEQ_LENGTH,
    stride=1,
)

# 转为 PyTorch Tensor
train_tensor = torch.FloatTensor(X_train)
test_tensor = torch.FloatTensor(X_test)

print(f"训练集形状: {train_tensor.shape}")
print(f"测试集形状: {test_tensor.shape}")

# 创建 DataLoader
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=32, shuffle=True)

# ==========================================
# 3. 定义 LSTM 自编码器模型
# ==========================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder: 压缩信息
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Decoder: 还原信息
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        
        # Encoder
        # enc_out: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        _, (hidden_n, _) = self.encoder(x)
        
        # 将 Encoder 最后的隐状态复制 seq_len 次，作为 Decoder 的输入
        # 这一步是为了让 Decoder 根据压缩的特征重构整个序列
        # hidden_n 形状: (1, batch, hidden) -> squeeze -> (batch, hidden)
        repeated_hidden = hidden_n.squeeze(0).unsqueeze(1).repeat(1, x.shape[1], 1)
        
        # Decoder
        dec_out, _ = self.decoder(repeated_hidden)
        
        # 映射回原始特征维度
        reconstructed = self.output_layer(dec_out)
        
        return reconstructed

# ==========================================
# 4. 训练模型
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAutoencoder(seq_len=SEQ_LENGTH, n_features=num_features, hidden_dim=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("开始训练模型...)")
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.6f}")

# ==========================================
# 5. 异常检测与评估
# ==========================================
model.eval()
with torch.no_grad():
    train_tensor_eval = train_tensor.to(device)
    train_predictions = model(train_tensor_eval)
    train_loss_dist = torch.mean(
        (train_predictions - train_tensor_eval) ** 2, dim=[1, 2]
    ).cpu().numpy()
    if USE_EWAF:
        train_loss_dist = apply_ewaf_by_segments(train_loss_dist, EWAF_ALPHA)

    test_tensor_eval = test_tensor.to(device)
    test_predictions = model(test_tensor_eval)
    test_loss_dist = torch.mean(
        (test_predictions - test_tensor_eval) ** 2, dim=[1, 2]
    ).cpu().numpy()

test_split_idx = split_index_from_labels(test_labels)
if USE_EWAF:
    test_loss_dist = apply_ewaf_by_segments(
        test_loss_dist,
        EWAF_ALPHA,
        infer_segment_lengths(test_labels),
    )

# 设定阈值
threshold = choose_threshold(train_loss_dist, method="mean")

print(f"\n计算出的异常阈值: {threshold:.6f}")

y_true = test_labels
y_pred = (test_loss_dist > threshold).astype(int)

print(f"测试集分界: [0:{test_split_idx}) normal, [{test_split_idx}:{len(test_loss_dist)}) anomaly")
print(f"检测到的异常数量: {(y_pred == 1).sum()} / {len(y_pred)}")

metrics = compute_binary_classification_metrics(y_true, y_pred)
print_metrics(
    "\nClassification Metrics:",
    metrics,
    order=["accuracy", "precision", "recall", "fdr", "fra", "f1"],
)

plot_detection_scores(
    test_loss_dist,
    threshold,
    test_split_idx,
    "/home/akira/codespace/mra-detection/anomaly_detection_results.png",
    title="LSTM异常检测",
    ylabel="重构误差",
    show=True,
)
