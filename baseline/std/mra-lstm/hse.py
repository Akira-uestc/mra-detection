import torch
import torch.nn.functional as F


__all__ = ['_hse']


def _hse(K, B, S, h_size, h, Q_list, R_list):
    # h:[K, B, S, h_size]
    Q_r_k = torch.stack([param for param in Q_list]) # [K, K*h_size, h_size]
    R_r_k = torch.stack([param for param in R_list]) # [K, h_size, h_size]
    Q_r_k = Q_r_k.unsqueeze(0).expand(B, -1, -1, -1) # [B, K, K*h_size, h_size]
    R_r_k = R_r_k.unsqueeze(0).expand(B, -1, -1, -1) # [B, K, h_size, h_size]
    h = h.view(B, 1, S, K*h_size) # [B, 1, S, K*h_size]

    r_t_k = F.sigmoid(torch.matmul(h, Q_r_k)) # [B, K, S, h_size]
    mul1 = torch.matmul(r_t_k, R_r_k) # [B, K, S, h_size]
    mul2 = mul1 * h.view(B, K, S, h_size) # [B, K, S, h_size]
    e_t = F.relu(torch.sum(mul2, dim=1))  # [B, S, h_size]

    return e_t