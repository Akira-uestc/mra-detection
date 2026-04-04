import numpy as np
import torch
import torch.nn as nn

from utils import _get_sampling_map


__all__ = ['_chunking']


class _chunking(nn.Module):
    def __init__(self, dataset, data_path):
        super().__init__()
        all_data = np.loadtxt(f'./_dataset/{dataset}/train/{data_path}', np.float32, delimiter=',')
        if all_data.shape[0] < all_data.shape[1]:
            all_data = all_data.T
        all_data = torch.tensor(all_data)

        self.sampling_map = _get_sampling_map(all_data)


    def get_chunking_idx(self):
        sampling_rate = torch.sum(self.sampling_map, dim=0) / self.sampling_map.shape[0]
        rate_groups = {}

        for idx, rate in enumerate(sampling_rate):
            key = round(rate.item(), 6) # 保留6位小数
            if key not in rate_groups:
                rate_groups[key] = []
            rate_groups[key].append(idx)

        sorted_rates = sorted(rate_groups.keys(), reverse=True) # 降序排列

        chunking_idx = []
        for rate in sorted_rates:
            chunking_idx.append(rate_groups[rate])

        return chunking_idx
