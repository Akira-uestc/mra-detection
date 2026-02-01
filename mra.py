import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据预处理
# ==========================================

class TEPDatasetBuilder:
    def __init__(self, seq_len=60, stride=5):
        self.seq_len = seq_len
        self.stride = stride
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        data = df.filter(like='xmeas_').iloc[:, :41].to_numpy()

        mask = (~np.isnan(data)).astype(np.float32)
        data_filled = np.nan_to_num(data, nan=0.0)

        self.scaler.fit(data_filled[:1500])
        data_scaled = self.scaler.transform(data_filled)

        return data_scaled.astype(np.float32), mask

    def create_windows(self, data, mask):
        X, M = [], []
        for i in range(0, len(data) - self.seq_len + 1, self.stride):
            X.append(data[i:i+self.seq_len])
            M.append(mask[i:i+self.seq_len])
        return np.stack(X), np.stack(M)

# ==========================================
# 2. Graph Learner
# ==========================================

class GraphLearner(nn.Module):
    def __init__(self, num_nodes, embed_dim=16, alpha=3.0):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.alpha = alpha

    def forward(self):
        M1 = torch.tanh(self.alpha * self.E1)
        M2 = torch.tanh(self.alpha * self.E2)
        A = torch.matmul(M1, M2.T)
        A = F.relu(A)
        A = F.softmax(A, dim=-1)

        # row normalization for stability
        A = A / (A.sum(-1, keepdim=True) + 1e-8)
        return A

# ==========================================
# 3. GCN
# ==========================================

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: (B, S, N, F)
        x = self.linear(x)
        out = torch.einsum('nm,bsmd->bsnd', adj, x)
        return out

# ==========================================
# 4. TCN（per-node causal convolution）
# ==========================================

class TCNLayer(nn.Module):
    def __init__(self, num_nodes, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            num_nodes, num_nodes,
            kernel_size=kernel_size,
            padding=kernel_size-1,
            groups=num_nodes
        )

    def forward(self, x):
        # x: (B, N, S)
        out = self.conv(x)
        return out[..., :x.size(-1)]

# ==========================================
# 5. Frequency Imputer（mask-aware）
# ==========================================

class FrequencyImputer(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.freq_len = seq_len // 2 + 1
        self.att = nn.Sequential(
            nn.Linear(self.freq_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_len * 2),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        # x, mask: (B, S, N)
        xf = torch.fft.rfft(x, dim=1)
        real, imag = xf.real, xf.imag

        feat = torch.cat([real, imag], dim=1).permute(0, 2, 1)
        w = self.att(feat).permute(0, 2, 1)
        wr, wi = w[:, :self.freq_len], w[:, self.freq_len:]

        xf_enhanced = torch.complex(real * wr, imag * wi)
        x_rec = torch.fft.irfft(xf_enhanced, n=x.size(1), dim=1)

        # only fill missing positions
        return x * mask + x_rec * (1 - mask)

# ==========================================
# 6. Positional Encoding
# ==========================================

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pe

# ==========================================
# 7. AGF-ADNet
# ==========================================

class AGF_ADNet(nn.Module):
    def __init__(self, num_nodes=41, seq_len=60, d_model=64):
        super().__init__()

        self.graph = GraphLearner(num_nodes)
        self.gcn = GCNLayer(1, d_model)
        self.tcn = TCNLayer(num_nodes)

        self.freq = FrequencyImputer(seq_len)

        self.time_norm = nn.LayerNorm(d_model)
        self.freq_norm = nn.LayerNorm(num_nodes)

        self.fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm([seq_len, num_nodes])
        )

        self.input_proj = nn.Linear(num_nodes, d_model)
        self.pos_enc = LearnablePositionalEncoding(seq_len, d_model)

        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, 2)

        self.output_proj = nn.Linear(d_model, num_nodes)

    def forward(self, x, mask):
        B, S, N = x.shape
        adj = self.graph()

        x_gcn = self.gcn(x.unsqueeze(-1), adj)
        x_gcn = self.time_norm(x_gcn)
        x_gcn = x_gcn.mean(-1)

        x_tcn = self.tcn(x.permute(0,2,1)).permute(0,2,1)
        h_time = x_gcn + x_tcn

        h_freq = self.freq(x, mask)
        h_freq = self.freq_norm(h_freq)

        h = torch.stack([h_time, h_freq], dim=1)
        x_imp = self.fusion(h).squeeze(1)

        x_filled = x * mask + x_imp * (1 - mask)

        z = self.input_proj(x_filled)
        z = self.pos_enc(z)
        z = self.transformer(z)
        x_rec = self.output_proj(z)

        return x_rec, adj, x_filled

# ==========================================
# 8. 训练 & 测试
# ==========================================

def train():
    SEQ_LEN = 60
    builder = TEPDatasetBuilder(SEQ_LEN)

    data, mask = builder.load_data("./TEP_3000_Block_Split.csv")
    Xtr, Mtr = builder.create_windows(data[:1500], mask[:1500])
    Xte, Mte = builder.create_windows(data[1500:], mask[1500:])

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(Mtr)),
        batch_size=32, shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AGF_ADNet(seq_len=SEQ_LEN).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        loss_sum = 0

        for x, m in train_loader:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()

            x_rec, adj, _ = model(x, m)

            recon_loss = ((x_rec - x) * m).pow(2).sum() / (m.sum() + 1e-8)
            imp_loss = ((x_rec - x) * (1 - m)).pow(2).sum() / ((1 - m).sum() + 1e-8)

            entropy = -(adj * torch.log(adj + 1e-8)).sum(-1).mean()

            loss = recon_loss + imp_loss + 0.01 * entropy
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        print(f"Epoch {epoch+1}, Loss {loss_sum/len(train_loader):.6f}")

    # Evaluation
    model.eval()
    scores = []

    with torch.no_grad():
        for x, m in DataLoader(
            TensorDataset(torch.tensor(Xte), torch.tensor(Mte)),
            batch_size=32
        ):
            x, m = x.to(device), m.to(device)
            xr, _, _ = model(x, m)
            err = ((xr - x) * m).pow(2).mean(dim=[1,2])
            scores.extend(err.cpu().numpy())

    plt.plot(scores)
    plt.axvline(len(scores)//2, color='r')
    plt.title("Anomaly Score")
    plt.show()

if __name__ == "__main__":
    train()
