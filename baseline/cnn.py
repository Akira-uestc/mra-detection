import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. 数据加载与预处理 (包含你的原始函数)
# ---------------------------------------------------------

def generate_mask_matrix():
    """读取之前生成的块状分布 CSV 数据"""
    try:
        # 读取 CSV，保留表头以便确认列名
        df = pd.read_csv("./TEP_3000_Block_Split.csv")
        # 提取 xmeas_1 到 xmeas_41 (丢弃后面的 xmv)
        data_df = df.filter(like='xmeas_').iloc[:, :41]
        data = data_df.astype(float).to_numpy()
        # 1 代表缺失 (NaN), 0 代表观测值
        mask = np.isnan(data).astype(int)
        return data, mask
    except FileNotFoundError:
        print("警告: 找不到数据文件。正在生成模拟数据用于演示代码运行...")
        # 生成模拟数据以确保代码可运行
        mock_data = np.random.randn(3000, 41)
        # 随机设置一些 NaN
        mock_data[np.random.rand(*mock_data.shape) < 0.1] = np.nan
        mock_mask = np.isnan(mock_data).astype(int)
        return mock_data, mock_mask

def prepare_data():
    # 1. 加载数据
    data, mask = generate_mask_matrix()
    
    if data is None:
        return None, None, None, None

    # 2. 处理缺失值 (NaN)
    # 神经网络不能输入 NaN。这里我们将 NaN 填充为 0。
    # 由于后续会做标准化 (StandardScaler)，0 通常接近均值，是一个安全的填充值。
    # 如果希望模型感知"缺失"这个特征，可以将 mask 也作为输入拼接到 data 后面，
    # 但为了保持 CNN 简单，这里仅做填充。
    data = np.nan_to_num(data, nan=0.0)

    # 3. 生成标签 (0: 正常, 1: 异常)
    # 前1500正常，后1500异常
    y_normal = np.zeros(1500)
    y_faulty = np.ones(1500)
    y = np.concatenate([y_normal, y_faulty])

    # 4. 数据分割 (训练集/测试集)
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 5. 标准化 (Standardization) - 对神经网络非常重要
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # 使用训练集的参数转换测试集

    # 6. 重塑数据以适应 1D-CNN
    # PyTorch Conv1d 输入形状: (Batch_Size, Channels, Length)
    # 我们将 41 个特征视为长度为 41 的序列，通道数为 1
    X_train = X_train.reshape(-1, 1, 41)
    X_test = X_test.reshape(-1, 1, 41)

    # 7. 转为 PyTorch Tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) # shape变为 (N, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# ---------------------------------------------------------
# 2. 定义 1D-CNN 模型
# ---------------------------------------------------------

class AnomalyDetectorCNN(nn.Module):
    def __init__(self):
        super(AnomalyDetectorCNN, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积: 输入通道1, 输出通道16, 卷积核大小3
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 41 -> 20
            
            # 第二层卷积
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 20 -> 10
        )
        
        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10, 64), # 32通道 * 长度10
            nn.ReLU(),
            nn.Dropout(0.5),        # 防止过拟合
            nn.Linear(64, 1),
            nn.Sigmoid()            # 二分类输出 0~1 概率
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 3. 训练与评估流程
# ---------------------------------------------------------

def train_model():
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data()
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型、损失函数、优化器
    model = AnomalyDetectorCNN()
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 20
    print(f"开始训练，共 {epochs} 个 Epoch...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 评估模型
    print("\n--- 评估结果 ---")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float() # 阈值 0.5
        
        y_true = y_test.numpy()
        y_pred = predicted.numpy()

        # 计算准确率
        accuracy = (predicted.eq(y_test).sum() / float(y_test.shape[0])).item()
        print(f"测试集准确率: {accuracy * 100:.2f}%")
        
        # 详细报告
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        
if __name__ == "__main__":
    train_model()