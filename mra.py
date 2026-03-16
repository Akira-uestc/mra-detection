import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import copy

plt.rcParams['font.sans-serif'] = ['SimHei']

# ==========================================
# 1. Dataset Builder (loads from data/ directory)
# ==========================================
class DatasetBuilder:
    def __init__(self, seq_len=60, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.scaler = StandardScaler()
        self.num_features = None

    def load_dir(self, dir_path, file_pattern="*.csv"):
        """Load CSV files matching file_pattern from a directory and concatenate.
        CSVs are headerless with numeric columns.
        Returns (data, mask) where mask uses 1 for missing and 0 for observed.
        """
        import glob
        csv_files = sorted(glob.glob(os.path.join(dir_path, file_pattern)))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {dir_path}")

        dfs = []
        masks = []
        for f in csv_files:
            df = pd.read_csv(f, header=None)
            arr = df.to_numpy(dtype=np.float32)
            dfs.append(arr)
            masks.append(np.isnan(arr).astype(np.float32))
            print(f"  Loaded {f}: {len(df)} rows, {df.shape[1]} cols")

        data = np.concatenate(dfs, axis=0)
        self.num_features = data.shape[1]
        mask = np.concatenate(masks, axis=0)
        return data, mask

    def fit_scaler(self, data):
        """Fit the scaler on (training) data."""
        data_filled = np.nan_to_num(data, nan=0.0)
        self.scaler.fit(data_filled)

    def transform(self, data):
        """Scale data using the fitted scaler."""
        data_filled = np.nan_to_num(data, nan=0.0)
        return self.scaler.transform(data_filled).astype(np.float32)

    def create_windows(self, data, mask):
        X, M = [], []
        n = len(data)
        num_feat = data.shape[1]

        if n == 0:
            shape = (0, self.seq_len, num_feat)
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

        for i in range(0, n, self.stride):
            if i < self.seq_len:
                pad_len = self.seq_len - i - 1
                window_data = np.concatenate([
                    np.tile(data[0:1], (pad_len, 1)),
                    data[0 : i + 1]
                ], axis=0)
                window_mask = np.concatenate([
                    np.tile(mask[0:1], (pad_len, 1)),
                    mask[0 : i + 1]
                ], axis=0)
            else:
                window_data = data[i - self.seq_len + 1 : i + 1]
                window_mask = mask[i - self.seq_len + 1 : i + 1]

            X.append(window_data)
            M.append(window_mask)

        return np.stack(X).astype(np.float32), np.stack(M).astype(np.float32)


# ==========================================
# 2. FIXED: Enhanced Graph Learner
# ==========================================
class GraphLearner(nn.Module):
    """
    FIXED: Removed softmax to allow true sparsity enforcement.
    Uses row-wise normalization after ReLU instead.
    """
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
        
        # FIXED: Row-wise normalization instead of softmax
        # This allows L1 sparsity to have effect while maintaining normalized weights
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        return A

# ==========================================
# 3. GCN Layer (Unchanged)
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: (B, S, N, F)
        x = self.linear(x)
        out = torch.einsum("nm,bsmd->bsnd", adj, x)
        return out

# ==========================================
# 4. FIXED: Multi-Scale TCN with Proper Causal Padding
# ==========================================
class MultiScaleTCN(nn.Module):
    """
    FIXED: Uses proper causal padding (left-side only) to prevent seeing future.
    For reconstruction tasks, you may want bidirectional; this implements causal version.
    """
    def __init__(self, num_nodes, kernel_sizes=[3, 5, 7], causal=True):
        super().__init__()
        self.causal = causal
        self.convs = nn.ModuleList()
        self.kernel_sizes = kernel_sizes
        
        for k in kernel_sizes:
            # No padding in conv, we'll handle it manually for causal
            self.convs.append(
                nn.Conv1d(
                    num_nodes, num_nodes, 
                    kernel_size=k, 
                    padding=0,  # Manual padding
                    groups=num_nodes
                )
            )
        
        # Fusion of multi-scale outputs
        self.fusion = nn.Linear(len(kernel_sizes), 1)

    def forward(self, x):
        # x: (B, N, S)
        outputs = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            if self.causal:
                # Left-side padding only (causal)
                padded = F.pad(x, (k-1, 0))
            else:
                # Symmetric padding (non-causal, for reconstruction)
                padded = F.pad(x, ((k-1)//2, k//2))
            
            out = conv(padded)
            outputs.append(out)  # (B, N, S)
        
        # Stack: (B, N, S, K)
        out_stack = torch.stack(outputs, dim=-1)
        # Weighted sum of scales: (B, N, S)
        out = self.fusion(out_stack).squeeze(-1)
        return out

# ==========================================
# 5. FIXED: Frequency Imputer with Proper Attention
# ==========================================
class FrequencyImputer(nn.Module):
    """
    FIXED: Implements proper attention mechanism over frequency features.
    Attention weights are computed and applied to the magnitude spectrum.
    """
    def __init__(self, seq_len, num_nodes=18):
        super().__init__()
        self.freq_len = seq_len // 2 + 1
        self.num_nodes = num_nodes
        
        # Attention network: learns which frequencies are important
        self.attention = nn.Sequential(
            nn.Linear(self.freq_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_len),
            nn.Sigmoid(),  # Attention weights in [0, 1]
        )
        
        # Frequency enhancement network
        self.freq_enhance = nn.Sequential(
            nn.Linear(self.freq_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_len * 2),
        )

    def forward(self, x):
        # x: (B, S, N) -> permute to (B, N, S) for FFT
        x_perm = x.permute(0, 2, 1)  # (B, N, S)
        
        # FFT to frequency domain
        xf = torch.fft.rfft(x_perm, dim=2)  # (B, N, F) complex
        
        # Extract magnitude and phase
        magnitude = torch.abs(xf)  # (B, N, F)
        phase = torch.angle(xf)  # (B, N, F)
        
        # Concatenate real and imaginary for feature extraction
        real, imag = xf.real, xf.imag
        feat = torch.cat([real, imag], dim=-1)  # (B, N, 2F)
        
        # Compute attention weights per node
        att_weights = self.attention(feat)  # (B, N, F)
        
        # Enhance frequency features
        feat_enhanced = self.freq_enhance(feat)  # (B, N, 2F)
        real_enh = feat_enhanced[..., :self.freq_len]
        imag_enh = feat_enhanced[..., self.freq_len:]
        
        # Apply attention to enhanced features
        real_attended = real_enh * att_weights
        imag_attended = imag_enh * att_weights
        
        # Residual spectrum refinement is more stable than replacing the FFT outright.
        xf_enhanced = xf + torch.complex(real_attended, imag_attended)
        
        # IFFT back to time domain
        x_rec = torch.fft.irfft(xf_enhanced, n=x.size(1), dim=2)  # (B, N, S)
        
        # Permute back to (B, S, N)
        return x_rec.permute(0, 2, 1)

# ==========================================
# 6. FIXED: Gated Fusion with Temporal Context
# ==========================================
class GatedFusion(nn.Module):
    """
    FIXED: Uses temporal convolution for gate computation to capture temporal context.
    """
    def __init__(self, num_nodes, seq_len):
        super().__init__()
        # Use 1D conv to capture temporal patterns in gate computation
        self.gate_net = nn.Sequential(
            nn.Conv1d(num_nodes * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_nodes, kernel_size=1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(num_nodes)

    def forward(self, h_time, h_freq):
        # h_time, h_freq: (B, S, N)
        # Permute to (B, N, S) for conv1d
        h_time_perm = h_time.permute(0, 2, 1)
        h_freq_perm = h_freq.permute(0, 2, 1)
        
        combined = torch.cat([h_time_perm, h_freq_perm], dim=1)  # (B, 2N, S)
        z = self.gate_net(combined)  # (B, N, S)
        z = z.permute(0, 2, 1)  # (B, S, N)
        
        # Gated combination
        h = z * h_time + (1 - z) * h_freq
        return self.norm(h)

# ==========================================
# 7. AGF-ADNet (Fixed Integration)
# ==========================================
class AGF_ADNet(nn.Module):
    def __init__(self, num_nodes=18, seq_len=60, d_model=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        self.graph = GraphLearner(num_nodes)
        
        # Time Branch
        self.gcn = GCNLayer(1, 1)
        self.tcn = MultiScaleTCN(num_nodes, kernel_sizes=[3, 5, 9], causal=False)  # Non-causal for reconstruction
        self.time_norm = nn.LayerNorm(num_nodes)
        
        # Freq Branch
        self.freq = FrequencyImputer(seq_len, num_nodes)
        self.freq_norm = nn.LayerNorm(num_nodes)
        
        # Fusion - keeping gated fusion as it's more advanced than Conv1x1
        self.fusion = GatedFusion(num_nodes, seq_len)

        # Transformer with better input projection
        # Project each node's time series to d_model dimension
        self.input_proj = nn.Linear(num_nodes, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.output_proj = nn.Linear(d_model, num_nodes)

    def forward(self, x, mask):
        # mask: 1 for missing, 0 for observed
        # x: (B, S, N)
        adj = self.graph()

        # 1. Time Branch
        x_gcn = self.gcn(x.unsqueeze(-1), adj).squeeze(-1)  # (B, S, N)
        x_gcn = self.time_norm(x_gcn)
        
        x_tcn = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, S, N)
        h_time = x_gcn + x_tcn

        # 2. Freq Branch
        h_freq = self.freq(x)  # (B, S, N)
        h_freq = self.freq_norm(h_freq)

        # 3. Gated Fusion
        x_imp = self.fusion(h_time, h_freq)  # (B, S, N)

        # 4. Imputation: Fill missing with imputed values
        observed_mask = 1.0 - mask
        x_filled = x * observed_mask + x_imp * mask

        # 5. Transformer Reconstruction
        z = self.input_proj(x_filled) + self.pos_enc  # (B, S, d_model)
        z = self.transformer(z)
        x_rec = self.output_proj(z)  # (B, S, N)

        return x_rec, adj, x_imp

# ==========================================
# 8. FIXED: Dual-Domain Loss
# ==========================================
def dual_domain_loss(x_rec, x_true, missing_mask, target_mask, adj, freq_weight=0.1, sparsity_weight=0.1):
    """
    missing_mask: 1 for missing, 0 for observed
    target_mask: 1 where reconstruction error should be supervised
    """
    target_mask = target_mask.float()
    recon_loss = ((x_rec - x_true) * target_mask).pow(2).sum() / target_mask.sum().clamp_min(1.0)
    
    observed_ratio = (1.0 - missing_mask).mean(dim=[1, 2])
    valid_samples = observed_ratio > 0.5

    # Apply a masked target so the FFT branch is supervised only where ground truth is valid.
    freq_target = x_rec.detach() * (1.0 - target_mask) + x_true * target_mask

    if valid_samples.sum() > 0:
        x_rec_valid = x_rec[valid_samples].permute(0, 2, 1)  # (B', N, S)
        x_true_valid = freq_target[valid_samples].permute(0, 2, 1)
        
        fft_rec = torch.fft.rfft(x_rec_valid, dim=2)
        fft_true = torch.fft.rfft(x_true_valid, dim=2)
        
        # Compare magnitude spectra
        freq_loss = (fft_rec.abs() - fft_true.abs()).pow(2).mean()
    else:
        freq_loss = torch.tensor(0.0, device=x_rec.device)
    
    # Row-normalized non-negative adjacency makes L1 nearly constant, so use entropy instead.
    sparsity_loss = -(adj * torch.log(adj + 1e-8)).sum(dim=-1).mean()
    
    return recon_loss + freq_weight * freq_loss + sparsity_weight * sparsity_loss


def apply_missing_mask(x, missing_mask):
    return x.masked_fill(missing_mask.bool(), 0.0)


def anomaly_scores(model, windows, masks, device, batch_size=32):
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

            observed_mask = 1.0 - missing_mask
            x_input = apply_missing_mask(x, missing_mask)
            xr, _, _ = model(x_input, missing_mask)

            sq_err = ((xr - x) * observed_mask).pow(2).sum(dim=[1, 2])
            obs_cnt = observed_mask.sum(dim=[1, 2]).clamp_min(1e-8)
            scores.extend((sq_err / obs_cnt).cpu().numpy())

    return np.array(scores)


def build_test_labels(num_scores):
    labels = np.zeros(num_scores, dtype=int)
    labels[num_scores // 2 :] = 1
    return labels


def plot_results(scores, threshold, split_idx, save_path='/home/akira/codespace/mra-detection/anomaly_detection_results.png'):
    plt.figure(figsize=(6, 5))
    plt.plot(scores, label='测试异常分数', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    plt.axvline(x=split_idx, color='g', linestyle=':', label='测试集分界')
    plt.xlabel('测试样本索引')
    plt.ylabel('重构误差')
    plt.title('AGF-ADNet异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

# ==========================================
# 9. FIXED: Training Pipeline
# ==========================================
def train():
    SEQ_LEN = 60
    DATA_DIR = "./data"
    builder = DatasetBuilder(SEQ_LEN)

    # Load train and test data from separate directories
    # Use file_pattern to select only files with consistent column counts
    TRAIN_PATTERN = "train_*.csv"      # 18-col files
    TEST_PATTERN  = "test_*.csv"        # 18-col files

    print("Loading training data...")
    train_data, train_mask = builder.load_dir(os.path.join(DATA_DIR, "train"), TRAIN_PATTERN)
    num_features = builder.num_features
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    print("\nLoading test data...")
    test_data, test_mask = builder.load_dir(os.path.join(DATA_DIR, "test"), TEST_PATTERN)
    print(f"Test data: {test_data.shape}")

    # Fit scaler on training data, then transform both
    builder.fit_scaler(train_data)
    train_data_scaled = builder.transform(train_data)
    test_data_scaled = builder.transform(test_data)

    # Create sliding windows
    Xtr, Mtr = builder.create_windows(train_data_scaled, train_mask)
    Xte, Mte = builder.create_windows(test_data_scaled, test_mask)

    if len(Xtr) == 0: 
        print("No training data available!")
        return

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(Mtr)), 
        batch_size=32, shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = AGF_ADNet(num_nodes=num_features, seq_len=SEQ_LEN).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    best_state = None
    best_loss = float("inf")

    print("Starting training...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        
        for x, m in train_loader:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            
            # Self-supervised masking over currently observed values.
            observed = ~m.bool()
            rand_drop_prob = 0.1
            rand_drop = (torch.rand_like(x) < rand_drop_prob) & observed
            target_mask = rand_drop.float()
            if not rand_drop.any():
                target_mask = observed.float()

            m_input = m.clone()
            m_input[rand_drop] = 1.0
            x_input = apply_missing_mask(x, m_input)
            
            # Forward pass
            x_rec, adj, _ = model(x_input, m_input)

            loss = dual_domain_loss(x_rec, x, m, target_mask, adj)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, LR: {opt.param_groups[0]['lr']:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nTraining completed. Computing threshold from training set...")
    
    # Compute anomaly scores on the TRAINING set to establish the threshold
    train_scores = anomaly_scores(model, Xtr, Mtr, device=device, batch_size=32)
    threshold = float(np.mean(train_scores) + np.std(train_scores))
    
    print(f"\nTraining Set Score Stats:")
    print(f"  Mean: {np.mean(train_scores):.6f}")
    print(f"  Std:  {np.std(train_scores):.6f}")
    print(f"  Threshold (mean train score): {threshold:.6f}")
    
    # Evaluate on the TEST set using the training-derived threshold
    print("\nEvaluating on test set...")
    test_scores_arr = anomaly_scores(model, Xte, Mte, device=device, batch_size=32)
    test_labels = build_test_labels(len(test_scores_arr))
    split_idx = len(test_scores_arr) // 2
    
    print(f"\nAnomaly Detection Results:")
    print(f"  Mean Score: {np.mean(test_scores_arr):.6f}")
    print(f"  Std Score:  {np.std(test_scores_arr):.6f}")
    print(f"  Threshold (from train): {threshold:.6f}")
    print(f"  Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores_arr)}) anomaly")
    print(f"  Anomalies detected: {(test_scores_arr > threshold).sum()} / {len(test_scores_arr)}")

    # Classification Metrics
    y_true = test_labels
    y_pred = (test_scores_arr > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Visualization
    plot_results(test_scores_arr, threshold, split_idx)
    
    return model, test_scores_arr

if __name__ == "__main__":
    model, scores = train()
