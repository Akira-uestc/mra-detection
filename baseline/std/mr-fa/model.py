from headers import *
from utils import _ConfigLoader, _BaseModel, _mask_mse, _statistic_mask_norm2, _get_sampling_map
from .chunking import _chunking


__all__ = ["_Model"]


class _Model(_BaseModel):
    def __init__(self, configs: _ConfigLoader):
        super().__init__(configs)
        self.to(configs.device)
        self.Df = configs.d_factor

        chunking = _chunking(configs.dataset, configs.train_path)
        self.chunking_idx = chunking.get_chunking_idx()
        self.n_factors = len(self.chunking_idx)

        self.list_Psi = nn.ParameterList()
        self.list_Omega = nn.ParameterList()
        for group in self.chunking_idx:
            d_group = len(group)
            self.list_Psi.append(nn.Parameter(torch.randn(d_group, self.Df)))
            self.list_Omega.append(nn.Parameter(torch.zeros(d_group)))  # Omega_n 是对角阵

    def forward(self, input, return_extras=False):
        # input:[B, S, Dd]
        B, S, Dd = input.shape
        BS = B * S
        input = input.reshape(BS, Dd)  # [BS, Dd]
        sampling_map = _get_sampling_map(input)  # [BS, Dd]
        x_k = torch.nan_to_num(input, nan=0.0)  # [BS, Dd]

        # 单位阵 I
        I_k = torch.eye(self.Df).expand(BS, -1, -1)  # [BS, Df, Df]
        # E-step 累加和
        sum_PsiT_invOmega_Psi = torch.zeros(BS, self.Df, self.Df)
        sum_PsiT_invOmega_x = torch.zeros(BS, self.Df)
        # E-step
        with torch.no_grad():
            for n in range(self.n_factors):
                idxs = self.chunking_idx[n]
                Psi_n = self.list_Psi[n]  # [Dn, Df]
                Omega_n = torch.exp(self.list_Omega[n])  # [Dn]
                invOmega_n = 1.0 / (Omega_n + 1e-8)  # [Dn]

                # 因 Omega 是对角阵, 所以可用逐元素乘法代替矩阵乘法, 不影响计算结果
                PsiT_invOmega_n = Psi_n.T * invOmega_n  # [Df, Dn]
                x_kn = x_k[:, idxs]  # [BS, Dn]
                phi_kn = torch.any(sampling_map[:, idxs], dim=1).float()  # [BS]

                PsiT_invOmega_Psi_n = PsiT_invOmega_n @ Psi_n  # [Df, Df]
                sum_PsiT_invOmega_Psi += phi_kn.view(BS, 1, 1) * PsiT_invOmega_Psi_n.unsqueeze(0)  # [BS, Df, Df]

                PsiT_invOmega_x_n = (PsiT_invOmega_n @ x_kn.unsqueeze(-1)).squeeze(-1)  # [BS, Df]
                sum_PsiT_invOmega_x += phi_kn.view(BS, 1) * PsiT_invOmega_x_n  # [BS, Df]

            Sigma_k = sum_PsiT_invOmega_Psi + I_k  # [BS, Df, Df]
            invSigma_k = torch.inverse(Sigma_k)  # [BS, Df, Df]
            tHat_k = (invSigma_k @ sum_PsiT_invOmega_x.unsqueeze(-1)).squeeze(-1)  # [BS, Df]

        x_recon = torch.zeros_like(x_k)  # [BS, Dd]
        for n in range(self.n_factors):
            idxs = self.chunking_idx[n]
            Psi_n = self.list_Psi[n]  # [Dn, Df]
            x_recon[:, idxs] = tHat_k @ Psi_n.T

        output = x_recon.reshape(B, S, Dd)

        if return_extras:
            E_t_tT_k = invSigma_k + (tHat_k.unsqueeze(-1) @ tHat_k.unsqueeze(-2))  # [BS, Df, Df]
            return output, tHat_k, E_t_tT_k, sampling_map, x_k
        else:
            return output

    def _cal_loss(self, target, pred):
        return _mask_mse(target, pred)

    def _cal_statistic(self, target, pred):
        return _statistic_mask_norm2(target, pred)

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        configs: _ConfigLoader,
        model: str,
    ):
        interval = 1
        early_stop_patience = 10
        best_loss = 1e36
        early_stop_counter = 0

        self.train_loss_history.clear()
        self.val_loss_history.clear()
        start = time.time()

        for epoch in range(round(configs.epochs)):
            # E-step & M-step 累加
            self.train()

            # 初始化 M-step 累加器
            sum_S_n_list = [torch.zeros(len(group), len(group)) for group in self.chunking_idx]
            sum_x_tHatT_n_list = [torch.zeros(len(group), self.Df) for group in self.chunking_idx]
            sum_E_t_tT_n_list = [torch.zeros(self.Df, self.Df) for _ in self.chunking_idx]
            K_n_list = [0.0 for _ in self.chunking_idx]

            train_loss = 0.0

            for batch, data in enumerate(train_loader):
                target = data[..., self.target_slice]

                # E-step 计算期望
                # pred: [D, S, Dd]
                # tHat_k: [BS, Df]
                # E_t_tT_k: [BS, Df, Df]
                # sampling_map: [BS, Dd]
                # x_k: [BS, Dd]
                pred, tHat_k, E_t_tT_k, sampling_map, x_k = self.forward(target, return_extras=True)

                # M-step 累加
                for n in range(self.n_factors):
                    idxs = self.chunking_idx[n]
                    phi_kn = torch.any(sampling_map[:, idxs], dim=1).float()  # [BS]
                    K_n_list[n] += torch.sum(phi_kn)
                    x_kn = x_k[:, idxs]  # [BS, Dn]

                    sum_S_n = torch.einsum("k,ki,kj->ij", phi_kn, x_kn, x_kn)
                    sum_S_n_list[n] += sum_S_n

                    sum_x_tHatT_n = torch.einsum("k,ki,kj->ij", phi_kn, x_kn, tHat_k)
                    sum_x_tHatT_n_list[n] += sum_x_tHatT_n

                    sum_E_t_tT_n = torch.einsum("k,kij->ij", phi_kn, E_t_tT_k)
                    sum_E_t_tT_n_list[n] += sum_E_t_tT_n

                loss = self._cal_loss(target, pred)
                train_loss += float(loss)

            train_loss /= batch + 1
            self.train_loss_history.append(train_loss)

            # M-step 参数更新
            with torch.no_grad():
                for n in range(self.n_factors):
                    K_n = K_n_list[n]
                    sum_S_n = sum_S_n_list[n]
                    sum_x_tHatT_n = sum_x_tHatT_n_list[n]
                    sum_E_t_tT_n = sum_E_t_tT_n_list[n]
                    # 更新 Psi_n
                    sum_invE_t_tT_n = torch.inverse(
                        sum_E_t_tT_n + torch.eye(self.Df) * 1e-6
                    )  # 增加扰动以保证数值稳定性
                    PsiHat_n = sum_x_tHatT_n @ sum_invE_t_tT_n
                    self.list_Psi[n].data.copy_(PsiHat_n)
                    # 更新 Omega_n
                    S_n = sum_S_n / K_n
                    OmegaHat_n = torch.diag(S_n - (PsiHat_n @ sum_x_tHatT_n.T) / K_n)
                    OmegaHat_n = torch.clamp(OmegaHat_n, min=1e-6)  # 保证方差为正
                    self.list_Omega[n].data.copy_(torch.log(OmegaHat_n))

            val_loss = self.val_impl(val_loader)

            print(
                f"{model} | Epoch {epoch+1:0>2d}/{configs.epochs:0>2d} | Train_loss={train_loss:8e} | Val_loss={val_loss:8e}"
            )

            if (epoch + 1) % interval == 0:
                if best_loss - val_loss >= configs.min_delta:
                    best_loss = val_loss
                    print(f"--> new is better, save it <--")
                    self.save(f"{model}/models/model.pth")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

            # early stop
            if early_stop_counter > early_stop_patience:
                print(f"--> INFO: Early stop <--")
                break

        end = time.time()
        print(
            f"End train, time:{int((end-start)/60)} min, min_train_loss={min(self.train_loss_history)}, min_val_loss={min(self.val_loss_history)}"
        )
        self.save_loss_hitory(model)
