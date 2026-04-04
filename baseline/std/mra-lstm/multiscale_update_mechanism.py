import torch
import torch.nn.functional as F


__all__ = ['_multiscale_update_mechanism']


def _multiscale_update_mechanism(input, chunking_idx, K, S, h_size, s, h, c, b_list, W_list, Z_list, U_list, V_list, J_list):
    s = step(s)

    for t in range(S):
        for k in range(K):
            t_prev = max(t-1, 0)
            k_prev = max(k-1, 0)
            k_next = min(k+1, K-1)

            x_t_k = input[:, t, chunking_idx[k]] # [B, Dk]
            z_t_k = 1 if torch.any(x_t_k != 0) else 0

            h_tprev_k = h[k, :, t_prev] # [B, h_size]
            h_t_kprev = h[k_prev, :, t] # [B, h_size]
            h_tprev_knext = h[k_next, :, t_prev] # [B, h_size]
            c_tprev_k = c[k, :, t_prev] # [B, h_size]

            bias = b_list[k]
            W = W_list[k] # [5*h_size, h_size]
            Z = Z_list[k] # [5*h_size, h_size]
            U = U_list[k] # [5*h_size, h_size+6]
            V = V_list[k] # [5*h_size, h_size]
            J = J_list[k] # [5*h_size, 6]
            
            operator = operator_select(s[k, t_prev], s[k_prev, t])

            # f_t_k: [B, h_size]
            start = 0*h_size
            end = 1*h_size
            f_t_k = F.sigmoid(compute_gate(W[start:end], Z[start:end], U[start:end], V[start:end], J[start:end], bias[start:end],
                                           x_t_k, z_t_k, h_tprev_k, h_t_kprev, h_tprev_knext, operator, 1))
            # g_t_k: [B, h_size]
            start = 1*h_size
            end = 2*h_size
            g_t_k = F.tanh(compute_gate(W[start:end], Z[start:end], U[start:end], V[start:end], J[start:end], bias[start:end],
                                        x_t_k, z_t_k, h_tprev_k, h_t_kprev, h_tprev_knext, operator, 0))
            # i_t_k: [B, h_size]
            start = 2*h_size
            end = 3*h_size
            i_t_k = F.sigmoid(compute_gate(W[start:end], Z[start:end], U[start:end], V[start:end], J[start:end], bias[start:end],
                                           x_t_k, z_t_k, h_tprev_k, h_t_kprev, h_tprev_knext, operator, 0))
            # o_t_k: [B, h_size]
            start = 3*h_size
            end = 4*h_size
            o_t_k = F.sigmoid(compute_gate(W[start:end], Z[start:end], U[start:end], V[start:end], J[start:end], bias[start:end],
                                           x_t_k, z_t_k, h_tprev_k, h_t_kprev, h_tprev_knext, operator, 0))

            # s_t_k: [B, h_size]
            start = 4*h_size
            end = 5*h_size
            s_t_k = hard_sigm(compute_gate(W[start:end], Z[start:end], U[start:end], V[start:end], J[start:end], bias[start:end],
                                 x_t_k, z_t_k, h_tprev_k, h_t_kprev, h_tprev_knext, operator, 0))
            s_t_k = step(torch.mean(s_t_k))

            # c_t_k: [B, h_size]
            if operator == 1:
                c_t_k = f_t_k * c_tprev_k + i_t_k * g_t_k
            elif operator == 2 or operator == 3:
                c_t_k = i_t_k * g_t_k
            else:
                c_t_k = c_tprev_k

            # h_t_k: [B, h_size]
            if operator == 4:
                h_t_k = o_t_k * F.tanh(c_t_k)
            else:
                h_t_k = h_tprev_k

            h[k, :, t] = h_t_k.detach()
            c[k, :, t] = c_t_k.detach()
            s[k, t] = s_t_k.detach()

    return h, c, s


def step(input):
    return (input > 0.5).float()


def hard_sigm(input):
    return torch.clamp((input + 1) / 2, 0, 1)


def operator_select(s_tprev_k, s_t_kprev):
    if s_tprev_k == 0 and s_t_kprev == 1:
        return 1
    elif s_tprev_k == 1 and s_t_kprev == 0:
        return 2
    elif s_tprev_k == 1 and s_t_kprev == 1:
        return 3
    elif s_tprev_k == 0 and s_t_kprev == 0:
        return 4
    else:
        raise ValueError(f'unknown operator')


def compute_gate(W, Z, U, V, J, bias,
                 x_t_k, z_t_k, h_tprev_k, h_t_kprev, h_tprev_knext,
                 operator, type):
    # W & Z & V: [h_size, h_size]
    # U: [h_size, h_size+6]
    # J: [h_size, 6]
    # x_t_k: [B, 6]
    term1 = torch.zeros_like(h_tprev_k)
    term2 = torch.zeros_like(h_tprev_k)

    # Case 1: operator == 1
    if operator == 1:
        term1 = torch.matmul(h_tprev_k, W.T)
        if z_t_k:
            term2 = torch.matmul(torch.cat((h_t_kprev, x_t_k), dim=1), U.T)
        else:
            term2 = torch.matmul(h_t_kprev, V.T)

    if type == 0:
        # Case 2: operator == 2
        if operator == 2:
            term1 = torch.matmul(h_tprev_knext, Z.T)
            if z_t_k:
                term2 = torch.matmul(x_t_k, J.T)

        # Case 3: operator == 3
        elif operator == 3:
            term1 = torch.matmul(h_tprev_knext, Z.T)
            if z_t_k:
                term2 = torch.matmul(torch.cat((h_t_kprev, x_t_k), dim=1), U.T)
            else:
                term2 = torch.matmul(h_t_kprev, V.T)

    output = term1 + term2 + bias  # [B, h_size]

    return output