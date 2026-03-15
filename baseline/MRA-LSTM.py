"""
MRA-LSTM for Anomaly Detection on TEP Dataset
Based on "Unmanned Aerial Vehicle Flight Data Anomaly Detection
Based on Multirate-Aware LSTM"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']

class TEPDataset(Dataset):
    """Dataset for Tennessee Eastman Process data.

    Uses front-padding sliding window (following mra.py methodology):
    - For each data point i, the input window covers [i - seq_len + 1, i].
    - When i < seq_len, the window is front-padded by repeating the first sample.
    - This produces exactly one window per data point (stride=1), so every
      sample gets a corresponding anomaly score.
    """
    def __init__(self, data, feature_cols, sequence_length=30,
                 prediction_horizon=1, stride=1, training=True):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.training = training
        self.feature_cols = feature_cols

        values = data[feature_cols].values.astype(np.float32)
        n = len(values)

        # Build windows with front-padding (mra.py style)
        self.x_windows = []
        self.y_windows = []
        self.labels = []

        for i in range(0, n, stride):
            # --- input window (length = sequence_length) ---
            if i < sequence_length:
                pad_len = sequence_length - i - 1
                window = np.concatenate([
                    np.tile(values[0:1], (pad_len, 1)),   # repeat first sample
                    values[0:i + 1]
                ], axis=0)
            else:
                window = values[i - sequence_length + 1:i + 1]

            # --- target window (length = prediction_horizon) ---
            y_start = i + 1
            y_end = y_start + prediction_horizon
            if y_end <= n:
                target = values[y_start:y_end]
            else:
                # Pad with the last sample when at the boundary
                available = values[y_start:n] if y_start < n else values[-1:]
                pad_needed = prediction_horizon - len(available)
                target = np.concatenate([
                    available,
                    np.tile(values[-1:], (pad_needed, 1))
                ], axis=0)

            self.x_windows.append(window)
            self.y_windows.append(target)

            # Label
            if training:
                self.labels.append(0)
            else:
                if 'faultNumber' in data.columns:
                    self.labels.append(int(data['faultNumber'].iloc[i]))
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.x_windows)

    def __getitem__(self, idx):
        x_tensor = torch.FloatTensor(self.x_windows[idx])
        y_tensor = torch.FloatTensor(self.y_windows[idx])
        label = self.labels[idx]
        return x_tensor, y_tensor, label


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for discrete boundaries"""
    @staticmethod
    def forward(ctx, x):
        # Forward: use step function
        ctx.save_for_backward(x)
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: use hard sigmoid (differentiable)
        x, = ctx.saved_tensors
        # Hard sigmoid: max(0, min(1, (x + 1) / 2))
        grad_x = torch.where((x > -1) & (x < 1), grad_output * 0.5, torch.zeros_like(grad_output))
        return grad_x


class MRALSTMCell(nn.Module):
    """Multirate-Aware LSTM Cell"""
    def __init__(self, hidden_size, lower_hidden_size, higher_hidden_size, input_size=None):
        super(MRALSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.lower_hidden_size = lower_hidden_size
        self.higher_hidden_size = higher_hidden_size
        self.input_size = input_size

        # Gate parameters (f: forget, i: input, o: output, g: cell proposal, s: boundary)
        # W: connection from own hidden state
        self.W_f = nn.Linear(hidden_size, hidden_size)
        self.W_i = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.W_g = nn.Linear(hidden_size, hidden_size)
        self.W_s = nn.Linear(hidden_size, 1)

        # U: connection from input (only for lowest layer)
        if input_size is not None:
            self.U_f = nn.Linear(input_size, hidden_size)
            self.U_i = nn.Linear(input_size, hidden_size)
            self.U_o = nn.Linear(input_size, hidden_size)
            self.U_g = nn.Linear(input_size, hidden_size)
            self.U_s = nn.Linear(input_size, 1)

        # V: connection from lower layer (higher rate)
        if lower_hidden_size is not None:
            self.V_f = nn.Linear(lower_hidden_size, hidden_size)
            self.V_i = nn.Linear(lower_hidden_size, hidden_size)
            self.V_o = nn.Linear(lower_hidden_size, hidden_size)
            self.V_g = nn.Linear(lower_hidden_size, hidden_size)
            self.V_s = nn.Linear(lower_hidden_size, 1)

        # Z: connection from higher layer (lower rate)
        if higher_hidden_size is not None:
            self.Z_f = nn.Linear(higher_hidden_size, hidden_size)
            self.Z_i = nn.Linear(higher_hidden_size, hidden_size)
            self.Z_o = nn.Linear(higher_hidden_size, hidden_size)
            self.Z_g = nn.Linear(higher_hidden_size, hidden_size)
            self.Z_s = nn.Linear(higher_hidden_size, 1)

    def forward(self, h_prev, c_prev, h_lower=None, h_higher=None, x=None, s_prev=None, s_lower=None):
        """
        h_prev: hidden state from previous time step (batch_size, hidden_size)
        c_prev: cell state from previous time step (batch_size, hidden_size)
        h_lower: hidden state from lower layer (higher rate) (batch_size, lower_hidden_size)
        h_higher: hidden state from higher layer (lower rate) (batch_size, higher_hidden_size)
        x: input at current time step (batch_size, input_size)
        s_prev: boundary from previous time step (batch_size,)
        s_lower: boundary from lower layer (batch_size,)
        """
        batch_size = h_prev.size(0)

        # Initialize s_prev and s_lower if None
        if s_prev is None:
            s_prev = torch.zeros(batch_size, device=h_prev.device)
        if s_lower is None:
            s_lower = torch.zeros(batch_size, device=h_prev.device)

        # Determine operator type based on mean of boundaries for the whole batch
        s_prev_mean = s_prev.float().mean().item()
        s_lower_mean = s_lower.float().mean().item()

        # Calculate gates based on operator type
        has_input = x is not None
        has_lower = h_lower is not None

        if s_prev_mean < 0.5 and s_lower_mean >= 0.5:
            # Operator 1: Input and lower layer available
            f = torch.sigmoid(self.W_f(h_prev))
            i = torch.sigmoid(self.W_i(h_prev))
            o = torch.sigmoid(self.W_o(h_prev))
            g = torch.tanh(self.W_g(h_prev))
            s_logits = self.W_s(h_prev)

            if has_input:
                f = f + torch.sigmoid(self.U_f(x))
                i = i + torch.sigmoid(self.U_i(x))
                o = o + torch.sigmoid(self.U_o(x))
                g = g + torch.tanh(self.U_g(x))
                s_logits = s_logits + self.U_s(x)

            if has_lower:
                f = f + torch.sigmoid(self.V_f(h_lower))
                i = i + torch.sigmoid(self.V_i(h_lower))
                o = o + torch.sigmoid(self.V_o(h_lower))
                g = g + torch.tanh(self.V_g(h_lower))
                s_logits = s_logits + self.V_s(h_lower)

            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)

        elif s_prev_mean >= 0.5 and s_lower_mean < 0.5 and h_higher is not None:
            # Operator 2: Only higher layer available
            f = torch.zeros_like(h_prev)
            i = torch.sigmoid(self.Z_i(h_higher))
            o = torch.sigmoid(self.Z_o(h_higher))
            g = torch.tanh(self.Z_g(h_higher))
            s_logits = self.Z_s(h_higher)

        elif s_prev_mean >= 0.5 and s_lower_mean >= 0.5:
            # Operator 3: Higher layer and input available
            if h_higher is not None:
                i = torch.sigmoid(self.Z_i(h_higher))
                o = torch.sigmoid(self.Z_o(h_higher))
                g = torch.tanh(self.Z_g(h_higher))
                s_logits = self.Z_s(h_higher)
            else:
                i = torch.zeros_like(h_prev)
                o = torch.zeros_like(h_prev)
                g = torch.zeros_like(h_prev)
                s_logits = torch.zeros(batch_size, 1, device=h_prev.device)

            f = torch.zeros_like(h_prev)

            if has_input:
                i = i + torch.sigmoid(self.U_i(x))
                o = o + torch.sigmoid(self.U_o(x))
                g = g + torch.tanh(self.U_g(x))
                s_logits = s_logits + self.U_s(x)

            i = torch.sigmoid(i)
            o = torch.sigmoid(o)

        else:
            # Operator 4: No update, only calculate boundary
            f = torch.zeros_like(h_prev)
            i = torch.zeros_like(h_prev)
            o = torch.zeros_like(h_prev)
            g = torch.zeros_like(h_prev)

            if has_lower:
                s_logits = self.V_s(h_lower)
            else:
                s_logits = self.W_s(h_prev)

        # Apply straight-through estimator for boundary
        s = StraightThroughEstimator.apply(s_logits).squeeze(-1)

        # Update cell state
        if s_prev_mean < 0.5:
            c = f * c_prev + i * g
        else:
            c = i * g

        # Update hidden state
        if s_prev_mean < 0.5 or s_lower_mean < 0.5:
            h = o * torch.tanh(c)
        else:
            h = h_prev

        return h, c, s


class HierarchicalStateExcitation(nn.Module):
    """Hierarchical State Excitation module"""
    def __init__(self, hidden_sizes):
        super(HierarchicalStateExcitation, self).__init__()
        self.hidden_sizes = hidden_sizes
        total_hidden = sum(hidden_sizes)

        # Calculate modulation coefficients - one coefficient per layer
        # This learns to weight the importance of each layer's output
        self.Q_r = nn.ModuleList([nn.Linear(total_hidden, 1) for _ in range(len(hidden_sizes))])

        # Projection layer to output final hidden size
        self.output_size = hidden_sizes[-1]

    def forward(self, states):
        """
        states: list of hidden states from each layer [h^1_t, h^2_t, ..., h^N_t]
        """
        batch_size = states[0].size(0)
        device = states[0].device

        # Concatenate all states
        h_concat = torch.cat(states, dim=-1)

        # Calculate modulation coefficients for each layer
        r_list = []
        for idx, q_r in enumerate(self.Q_r):
            r_k = torch.sigmoid(q_r(h_concat))  # (batch_size, 1)
            r_list.append(r_k)

        # Project each state to output size and aggregate with modulation coefficients
        # First, project all states to the same output size
        projected_states = []
        for h, hs in zip(states, self.hidden_sizes):
            if hs != self.output_size:
                # Simple projection using linear layer (learned during training)
                if not hasattr(self, f'proj_{hs}'):
                    setattr(self, f'proj_{hs}', nn.Linear(hs, self.output_size).to(device))
                proj_layer = getattr(self, f'proj_{hs}')
                projected = proj_layer(h)
            else:
                projected = h
            projected_states.append(projected)

        # Aggregate with modulation coefficients
        e = torch.zeros(batch_size, self.output_size, device=device)
        for proj_h, r in zip(projected_states, r_list):
            e = e + r * proj_h
        e = torch.relu(e)

        return e


class MRALSTM(nn.Module):
    """Multirate-Aware LSTM Network"""
    def __init__(self, input_size, hidden_sizes, num_layers, output_size, prediction_horizon=1):
        super(MRALSTM, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.output_size = output_size
        self.prediction_horizon = prediction_horizon

        # Create LSTM cells for each layer
        self.cells = nn.ModuleList()
        for idx in range(num_layers):
            # Determine sizes for connections
            lower_hidden = hidden_sizes[idx - 1] if idx > 0 else None
            higher_hidden = hidden_sizes[idx + 1] if idx < num_layers - 1 else None
            inp_size = input_size if idx == 0 else None

            cell = MRALSTMCell(hidden_sizes[idx], lower_hidden, higher_hidden, inp_size)
            self.cells.append(cell)

        # Hierarchical State Excitation
        self.hse = HierarchicalStateExcitation(hidden_sizes)

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size * prediction_horizon)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize states
        h_states = [torch.zeros(batch_size, hs, device=x.device) for hs in self.hidden_sizes]
        c_states = [torch.zeros(batch_size, hs, device=x.device) for hs in self.hidden_sizes]
        s_states = [torch.zeros(batch_size, device=x.device) for _ in range(self.num_layers)]

        # Process sequence
        h_list = [[] for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # Process each layer (top-down for multirate structure)
            for k in reversed(range(self.num_layers)):
                h_prev = h_states[k]
                c_prev = c_states[k]
                s_prev = s_states[k]

                h_lower = h_states[k - 1] if k > 0 else None
                h_higher = h_states[k + 1] if k < self.num_layers - 1 else None

                s_lower = s_states[k - 1] if k > 0 else None

                # Input for current layer (only lowest layer gets input)
                x_k = x_t if k == 0 else None

                # Forward pass through cell
                h_k, c_k, s_k = self.cells[k](
                    h_prev, c_prev, h_lower, h_higher, x_k, s_prev, s_lower
                )

                h_states[k] = h_k
                c_states[k] = c_k
                s_states[k] = s_k

            # Store hidden states for each layer at this time step
            for k in range(self.num_layers):
                h_list[k].append(h_states[k])

        # Transpose to get [time_steps, batch_size, hidden_size] for each layer
        h_list = [torch.stack(h, dim=0) for h in h_list]

        # Get final hidden states for HSE
        final_states = [h[-1] for h in h_list]

        # Apply HSE
        e = self.hse(final_states)

        # Predict future
        predictions = self.output_layer(e)
        predictions = predictions.view(batch_size, self.prediction_horizon, self.output_size)

        return predictions


def load_and_preprocess_data(csv_path, n_train=1500):
    """Load and preprocess TEP dataset"""
    # Load data
    df = pd.read_csv(csv_path)

    # Separate train (normal) and test data
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    # Get feature columns (exclude faultNumber, simulationRun, sample)
    feature_cols = [c for c in df.columns if c not in ['faultNumber', 'simulationRun', 'sample']]

    # Handle NaN values - forward fill then backward fill
    for col in feature_cols:
        train_df[col] = train_df[col].ffill().bfill()
        test_df[col] = test_df[col].ffill().bfill()

    # If still NaN, fill with column mean
    for col in feature_cols:
        mean_val = train_df[col].mean()
        if pd.isna(mean_val):
            mean_val = 0
        train_df[col] = train_df[col].fillna(mean_val)
        test_df[col] = test_df[col].fillna(mean_val)

    # Normalize to [0, 1] using train statistics
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, feature_cols, scaler


def train_model(model, train_loader, num_epochs=50, lr=0.001, device='cpu'):
    """Train the MRA-LSTM model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            predictions = model(x)

            # Reshape predictions and targets
            batch_size = x.size(0)
            predictions = predictions.view(batch_size, -1)
            y = y.view(batch_size, -1)

            loss = criterion(predictions, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}')

    return losses


def compute_anomaly_scores(model, test_loader, device='cpu'):
    """Compute anomaly scores (reconstruction errors) for test data"""
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    all_errors = []
    all_labels = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, y, labels in test_loader:
            x, y = x.to(device), y.to(device)

            predictions = model(x)

            # Compute error for each sample
            batch_size = x.size(0)
            predictions = predictions.view(batch_size, -1)
            y = y.view(batch_size, -1)

            errors = criterion(predictions, y)
            sample_errors = errors.mean(dim=1)

            all_errors.extend(sample_errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    return np.array(all_errors), np.array(all_labels), np.vstack(all_predictions), np.vstack(all_targets)


def compute_anomaly_scores_train(model, train_loader, device='cpu'):
    """Compute anomaly scores on training data for threshold calculation"""
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    all_errors = []

    with torch.no_grad():
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)

            predictions = model(x)

            # Compute error for each sample
            batch_size = x.size(0)
            predictions = predictions.view(batch_size, -1)
            y = y.view(batch_size, -1)

            errors = criterion(predictions, y)
            sample_errors = errors.mean(dim=1)

            all_errors.extend(sample_errors.cpu().numpy())

    return np.array(all_errors)


def apply_ewaf(errors, alpha=0.3):
    """Apply Exponentially Weighted Average Filter to smooth errors"""
    smoothed = np.zeros_like(errors)
    smoothed[0] = errors[0]
    for i in range(1, len(errors)):
        smoothed[i] = alpha * errors[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def compute_threshold_static(errors, percentile=95):
    """Compute static threshold based on percentile"""
    return np.percentile(errors, percentile)


def evaluate_detection(errors, labels, threshold):
    """Evaluate anomaly detection performance"""
    predictions = (errors > threshold).astype(int)
    true_labels = (labels > 0).astype(int)

    TP = np.sum((predictions == 1) & (true_labels == 1))
    TN = np.sum((predictions == 0) & (true_labels == 0))
    FP = np.sum((predictions == 1) & (true_labels == 0))
    FN = np.sum((predictions == 0) & (true_labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    specificity = TN / (TN + FP + 1e-10)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }


def plot_anomaly_detection(errors, threshold, save_path='/home/akira/codespace/mra-detection/anomaly_detection_results.png'):
    """Plot anomaly detection results (reconstruction error + threshold) in mra.py style."""
    plt.figure(figsize=(6, 5))
    plt.plot(errors, label='异常分数', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    plt.xlabel('样本索引')
    plt.ylabel('重构误差')
    plt.title('MRA-LSTM异常检测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def plot_training_loss(losses, save_path='training_loss.png'):
    """Plot training loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MRA-LSTM Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"Training loss plot saved to {save_path}")
    plt.show()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    csv_path = '../TEP_3000_Block_Split.csv'
    n_train = 1500
    sequence_length = 30
    prediction_horizon = 1
    batch_size = 32
    num_epochs = 50
    lr = 0.001

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, test_df, feature_cols, scaler = load_and_preprocess_data(csv_path, n_train)
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Number of features: {len(feature_cols)}")

    # Create datasets and dataloaders
    train_dataset = TEPDataset(train_df, feature_cols, sequence_length, prediction_horizon, training=True)
    test_dataset = TEPDataset(test_df, feature_cols, sequence_length, prediction_horizon, training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"Training sequences: {len(train_dataset)}, Test sequences: {len(test_dataset)}")

    # Model parameters
    input_size = len(feature_cols)
    hidden_sizes = [64, 32]
    num_layers = len(hidden_sizes)
    output_size = len(feature_cols)

    # Initialize model
    print("\nInitializing MRA-LSTM model...")
    model = MRALSTM(input_size, hidden_sizes, num_layers, output_size, prediction_horizon)
    model = model.to(device)

    # Print model summary
    print(f"Model: MRA-LSTM")
    print(f"Input size: {input_size}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Number of layers: {num_layers}")
    print(f"Output size: {output_size}")
    print(f"Prediction horizon: {prediction_horizon}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\nTraining model...")
    losses = train_model(model, train_loader, num_epochs, lr, device)

    # Plot training loss
    plot_training_loss(losses)

    # Compute anomaly scores on training data for threshold calculation
    print("\nComputing training anomaly scores for threshold...")
    train_errors = compute_anomaly_scores_train(model, train_loader, device)
    print(f"Training error stats - mean: {train_errors.mean():.6f}, std: {train_errors.std():.6f}, max: {train_errors.max():.6f}")

    # Compute threshold based on training data (e.g., 95th percentile of training errors)
    threshold_train = np.percentile(train_errors, 95)
    print(f"Training-based threshold (95th percentile): {threshold_train:.6f}")

    # Compute anomaly scores on test data
    print("\nComputing test anomaly scores...")
    test_errors, labels, predictions, targets = compute_anomaly_scores(model, test_loader, device)
    print(f"Test error stats - mean: {test_errors.mean():.6f}, std: {test_errors.std():.6f}, max: {test_errors.max():.6f}")

    # Splice train errors to front of test errors for evaluation
    all_errors = np.concatenate([train_errors, test_errors])

    # Apply EWAF to smooth errors
    smoothed_all_errors = apply_ewaf(all_errors, alpha=0.3)

    # Labels: 0 for train (normal), 1 for test (anomaly)
    all_labels = np.concatenate([np.zeros(len(train_errors), dtype=int), np.ones(len(test_errors), dtype=int)])

    # Evaluate detection using training-based threshold
    metrics = evaluate_detection(smoothed_all_errors, all_labels, threshold_train)
    print("\nDetection Performance (using training-based threshold):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

    # Plot anomaly detection results
    print("\nPlotting anomaly detection results...")
    plot_anomaly_detection(smoothed_all_errors, threshold_train)

    # Save model
    torch.save(model.state_dict(), 'mra_lstm_model.pth')
    print("Model saved to mra_lstm_model.pth")

    return model, test_errors, labels, threshold_train


if __name__ == '__main__':
    model, errors, labels, threshold = main()
