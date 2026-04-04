from headers import *
from utils import _BaseModel, _ConfigLoader, _mask_mse, _statistic_mask_norm2
from .chunking import _chunking
from .multiscale_update_mechanism import _multiscale_update_mechanism
from .hse import _hse


__all__ = ["_Model"]


class _Model(_BaseModel):
    def __init__(self, configs: _ConfigLoader):
        super().__init__(configs)

        chunking = _chunking(configs.dataset, configs.train_path)
        self.chunking_idx = chunking.get_chunking_idx()
        self.K = len(self.chunking_idx)

        self.B = configs.batch_size
        self.S = configs.seq_len
        self.h_size = configs.hidden_size

        self.h = torch.zeros(self.K, self.B, self.S, self.h_size)
        self.c = torch.zeros(self.K, self.B, self.S, self.h_size)
        self.s = nn.Parameter(torch.rand(self.K, self.S))

        self.W_list = nn.ParameterList()
        self.Z_list = nn.ParameterList()
        self.U_list = nn.ParameterList()
        self.V_list = nn.ParameterList()
        self.J_list = nn.ParameterList()
        self.Q_list = nn.ParameterList()
        self.R_list = nn.ParameterList()
        self.b_list = nn.ParameterList()

        for k in range(self.K):
            self.b_list.append(nn.Parameter(torch.randn(5 * self.h_size)))
            self.W_list.append(nn.Parameter(torch.randn(5 * self.h_size, self.h_size)))
            self.Z_list.append(nn.Parameter(torch.randn(5 * self.h_size, self.h_size)))
            self.U_list.append(nn.Parameter(torch.randn(5 * self.h_size, self.h_size + len(self.chunking_idx[k]))))
            self.V_list.append(nn.Parameter(torch.randn(5 * self.h_size, self.h_size)))
            self.J_list.append(nn.Parameter(torch.randn(5 * self.h_size, len(self.chunking_idx[k]))))
            self.Q_list.append(nn.Parameter(torch.randn(self.K * self.h_size, self.h_size)))
            self.R_list.append(nn.Parameter(torch.randn(self.h_size, self.h_size)))

        # fully connected
        self.fc = nn.Linear(self.h_size, configs.input_size)

    def forward(self, input):
        input = torch.where(torch.isnan(input), torch.tensor(0), input)

        h, c, s = _multiscale_update_mechanism(
            input,
            self.chunking_idx,
            self.K,
            self.S,
            self.h_size,
            self.s,
            self.h,
            self.c,
            self.b_list,
            self.W_list,
            self.Z_list,
            self.U_list,
            self.V_list,
            self.J_list,
        )
        self.h = h
        self.c = c
        self.s.data = s

        x2 = _hse(self.K, self.B, self.S, self.h_size, self.h, self.Q_list, self.R_list)

        output = self.fc(x2)

        return output

    def _cal_loss(self, target, pred):
        return _mask_mse(target, pred)

    def _cal_statistic(self, target, pred):
        return _statistic_mask_norm2(target, pred)
