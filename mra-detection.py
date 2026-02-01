import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DatasetBuilder:
    def __init__(self,seq_len=60,stride=5) -> None:
        self.seq_len=seq_len
        self.stride=stride
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

