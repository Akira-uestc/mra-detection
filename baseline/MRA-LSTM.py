import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 检查设备 (GPU 优先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🐱 运行设备: {device}")

# ---------------------------------------------------------
# 1. 数据集类 (Dataset)
# ---------------------------------------------------------
class TEPDataset(Dataset):
    def __init__(self, X, y):
        # PyTorch Conv1d 需要输入形状为 (Batch, Channels, Length)
        # 原始 X 形状: (Sample, 41) -> 变成 (Sample, 1, 41)
        self.X = torch.FloatTensor(X).unsqueeze(1) 
        # 标签如果是 CrossEntropyLoss，需要 LongTensor 且不需要 One-hot (直接用 0, 1)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    """读取数据，填充缺失值，生成标签并划分"""
    try:
        print("📥 正在读取数据...")
        df = pd.read_csv("./TEP_3000_Block_Split.csv")
        
        # 1. 提取特征
        data_df = df.filter(like='xmeas_').iloc[:, :41]
        
        # 2. 填充 NaN (用 0 填充，对应 Mask 机制)
        data_filled = data_df.fillna(0).astype(float).to_numpy()
        
        # 3. 生成标签 (前1500正常=0, 后1500异常=1)
        # 注意：PyTorch CrossEntropyLoss 期望标签是 1D 数组 [0, 1, 0, ...]，而不是 One-hot
        y = np.concatenate([np.zeros(1500), np.ones(1500)])
        
        print(f"📊 数据形状: {data_filled.shape}")
        
        # 4. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            data_filled, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError:
        print("找不到文件，请检查路径！")
        return None, None, None, None

# ---------------------------------------------------------
# 2. 模型构建 (CNN-LSTM)
# ---------------------------------------------------------
class CNNLSTMModel(nn.Module):
    def __init__(self, input_len=41, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        
        # --- CNN 部分 (提取局部特征) ---
        # 参照论文 Table 5 [cite: 454]，但为了适配 input_len=41 减小了 kernel_size
        
        # Layer 1: Conv -> Tanh -> MaxPool
        # Input: (Batch, 1, 41) -> Padding=2 保持长度 -> (Batch, 16, 41)
        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.act1 = nn.Tanh() # 论文公式 (1) 指定 Tanh [cite: 143]
        self.p1 = nn.MaxPool1d(kernel_size=2)
        
        # Layer 2
        # Input: (Batch, 16, 20)
        self.c2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.act2 = nn.Tanh()
        self.p2 = nn.MaxPool1d(2)
        
        # Layer 3
        # Input: (Batch, 32, 10)
        self.c3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.act3 = nn.Tanh()
        self.p3 = nn.MaxPool1d(2)
        
        # Output after CNN: (Batch, 64, 5)
        
        # --- LSTM 部分 (挖掘时间相关性) ---
        # LSTM 需要输入 (Batch, Sequence, Features)
        # 所以我们需要在 forward 中把 (Batch, 64, 5) 转置为 (Batch, 5, 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        
        # --- 全连接与分类 ---
        # FC (Nodes 32) -> Tanh -> BN [cite: 250, 251]
        self.fc1 = nn.Linear(64, 32)
        self.fc_act = nn.Tanh()
        self.bn = nn.BatchNorm1d(32)
        
        # Output Layer
        self.fc_out = nn.Linear(32, num_classes)
        # 注意：不加 Softmax，因为 CrossEntropyLoss 会自动处理

    def forward(self, x):
        # x shape: (Batch, 1, 41)
        
        # CNN Block
        x = self.p1(self.act1(self.c1(x)))
        x = self.p2(self.act2(self.c2(x)))
        x = self.p3(self.act3(self.c3(x)))
        
        # 准备 LSTM 输入
        # 当前 shape: (Batch, 64, 5) -> 需要 (Batch, 5, 64)
        x = x.permute(0, 2, 1)
        
        # LSTM
        # out: (Batch, Seq_Len, Hidden), (h_n, c_n)
        # 我们取最后一个时间步的输出作为特征
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步: h_n[-1] 或者 out[:, -1, :]
        x = out[:, -1, :] 
        
        # FC -> BN
        x = self.fc_act(self.fc1(x))
        x = self.bn(x)
        
        # Output logits
        x = self.fc_out(x)
        return x

# ---------------------------------------------------------
# 3. 训练与验证流程
# ---------------------------------------------------------
def train_model():
    # 1. 准备数据
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None: return
    
    train_dataset = TEPDataset(X_train, y_train)
    test_dataset = TEPDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. 初始化模型
    model = CNNLSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # [cite: 268]
    
    # 3. 训练循环
    epochs = 100
    train_acc_history = []
    test_acc_history = []
    
    print("\n🚀 开始训练...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        train_acc_history.append(train_acc)
        
        # 验证
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_acc = 100 * correct_test / total_test
        test_acc_history.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print("🎉 训练完成！")
    
    # 4. 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(test_acc_history, label='Test Accuracy')
    plt.title('PyTorch CNN-LSTM Fault Diagnosis')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()