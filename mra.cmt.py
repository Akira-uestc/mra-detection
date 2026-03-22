# -*- coding: utf-8 -*-
"""
`mra.py` 的中文注释版。

目标：
1. 保持原始实现逻辑不变。
2. 为各个模块、函数、关键代码块补充中文说明。
3. 尽量按逐行阅读的方式解释，方便后续维护和学习。
"""

# 导入 PyTorch 主库，负责张量运算与自动求导。
import torch
# 导入神经网络模块别名 `nn`，后续定义模型层会频繁使用。
import torch.nn as nn
# 导入函数式接口 `F`，常用于激活、填充等无状态操作。
import torch.nn.functional as F
# 导入优化器模块，这里主要使用 Adam。
import torch.optim as optim
# 导入数据加载器和张量数据集封装工具。
from torch.utils.data import DataLoader, TensorDataset
# 导入 Pandas，用于读取 CSV 文件。
import pandas as pd
# 导入 NumPy，负责数组处理与缺失值辅助操作。
import numpy as np
# 导入标准化器，用训练集拟合后统一缩放数据。
from sklearn.preprocessing import StandardScaler
# 导入分类评估指标，用于异常检测结果统计。
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 导入 Matplotlib 进行结果可视化。
import matplotlib.pyplot as plt
# 导入 `os` 以处理路径拼接。
import os
# 导入 `copy`，用于深拷贝最佳模型参数。
import copy

# 设置 Matplotlib 中文字体，避免图中中文乱码。
plt.rcParams['font.sans-serif'] = ['SimHei']


# ==========================================
# 1. Dataset Builder
# 作用：从 `data/` 目录读取 CSV，记录缺失掩码，完成标准化，并切分滑动窗口。
# ==========================================
class DatasetBuilder:
    """数据构建器：负责读取、标准化和窗口化时序数据。"""

    def __init__(self, seq_len=60, stride=1):
        # 保存滑动窗口长度。
        self.seq_len = seq_len
        # 保存窗口步长，决定窗口之间的移动间隔。
        self.stride = stride
        # 创建标准化器，通常只在训练集上拟合。
        self.scaler = StandardScaler()
        # 先占位特征维度，实际读取数据后再设置。
        self.num_features = None

    def load_dir(self, dir_path, file_pattern="*.csv"):
        """
        从目录中读取匹配模式的 CSV 文件并拼接。

        约定：
        1. CSV 无表头。
        2. 所有列都是数值列。
        3. 返回 `(data, mask)`，其中 `mask=1` 表示缺失，`mask=0` 表示观测到。
        """
        # 只在需要时导入 `glob`，用于按模式检索文件。
        import glob

        # 找到目录下所有匹配文件，并排序保证读取顺序稳定。
        csv_files = sorted(glob.glob(os.path.join(dir_path, file_pattern)))
        # 如果没有匹配文件，直接抛错，避免后续拼接时出现更隐蔽的问题。
        if not csv_files:
            raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {dir_path}")

        # `dfs` 用于保存每个文件的数据数组。
        dfs = []
        # `masks` 用于保存每个文件对应的缺失掩码。
        masks = []
        # 逐个读取匹配到的 CSV 文件。
        for f in csv_files:
            # 按无表头格式读取 CSV。
            df = pd.read_csv(f, header=None)
            # 转成 `float32` 数组，便于后续神经网络计算。
            arr = df.to_numpy(dtype=np.float32)
            # 保存当前文件数据。
            dfs.append(arr)
            # 缺失值位置记为 1，非缺失位置记为 0。
            masks.append(np.isnan(arr).astype(np.float32))
            # 打印加载信息，方便确认每个文件的规模与列数。
            print(f"  Loaded {f}: {len(df)} rows, {df.shape[1]} cols")

        # 在时间维上将多个文件顺序拼接起来。
        data = np.concatenate(dfs, axis=0)
        # 记录特征数量，也就是每一行的列数。
        self.num_features = data.shape[1]
        # 同样把所有掩码在时间维拼接。
        mask = np.concatenate(masks, axis=0)
        # 返回完整数据和对应掩码。
        return data, mask

    def fit_scaler(self, data):
        """在训练数据上拟合标准化器。"""
        # 标准化器不能直接处理 NaN，这里先用 0 填充缺失值。
        data_filled = np.nan_to_num(data, nan=0.0)
        # 拟合均值和方差。
        self.scaler.fit(data_filled)

    def transform(self, data):
        """使用已经拟合好的标准化器变换数据。"""
        # 同样先把 NaN 临时替换为 0，再送入标准化器。
        data_filled = np.nan_to_num(data, nan=0.0)
        # 输出转为 `float32`，与模型参数精度保持一致。
        return self.scaler.transform(data_filled).astype(np.float32)

    def create_windows(self, data, mask):
        """将连续时序切成固定长度的滑动窗口。"""
        # `X` 存放窗口化后的数据。
        X = []
        # `M` 存放窗口化后的掩码。
        M = []
        # `n` 是总时间步数。
        n = len(data)
        # `num_feat` 是特征维度数。
        num_feat = data.shape[1]

        # 如果数据为空，直接返回形状正确的空数组。
        if n == 0:
            shape = (0, self.seq_len, num_feat)
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

        # 按步长生成窗口，这里的 `i` 表示当前窗口的结束位置。
        for i in range(0, n, self.stride):
            # 当样本还不足一个完整窗口时，使用首行做左侧填充。
            if i < self.seq_len:
                # 需要补多少步，减 1 是因为窗口中还包含当前第 `i` 行。
                pad_len = self.seq_len - i - 1
                # 生成数据窗口：左侧复制首行，右侧接上真实片段。
                window_data = np.concatenate(
                    [
                        np.tile(data[0:1], (pad_len, 1)),
                        data[0 : i + 1],
                    ],
                    axis=0,
                )
                # 生成掩码窗口：保持与数据窗口完全同样的拼接方式。
                window_mask = np.concatenate(
                    [
                        np.tile(mask[0:1], (pad_len, 1)),
                        mask[0 : i + 1],
                    ],
                    axis=0,
                )
            else:
                # 数据足够长时，直接截取最近 `seq_len` 个时间步。
                window_data = data[i - self.seq_len + 1 : i + 1]
                # 掩码也同步截取对应区间。
                window_mask = mask[i - self.seq_len + 1 : i + 1]

            # 保存当前窗口数据。
            X.append(window_data)
            # 保存当前窗口掩码。
            M.append(window_mask)

        # 将列表堆叠为三维数组，形状为 `(样本数, 序列长度, 特征数)`。
        return np.stack(X).astype(np.float32), np.stack(M).astype(np.float32)


# ==========================================
# 2. Enhanced Graph Learner
# 作用：学习变量之间的图结构邻接矩阵。
# 说明：这里使用可学习嵌入构造邻接矩阵，并做逐行归一化。
# ==========================================
class GraphLearner(nn.Module):
    """
    图结构学习器。

    与常见 softmax 版本不同，这里先 ReLU 再做逐行归一化，
    目的是让稀疏约束更容易产生效果。
    """

    def __init__(self, num_nodes, embed_dim=16, alpha=3.0):
        # 调用父类初始化，注册参数与模块。
        super().__init__()
        # 第一个节点嵌入矩阵，形状为 `(节点数, 嵌入维度)`。
        self.E1 = nn.Parameter(torch.ones(num_nodes, embed_dim))
        # 第二个节点嵌入矩阵，用于与 `E1` 做双线性关联。
        self.E2 = nn.Parameter(torch.ones(num_nodes, embed_dim))
        # `alpha` 控制 `tanh` 前的缩放强度。
        self.alpha = alpha

    def forward(self):
        # 对 `E1` 做缩放后 `tanh`，限制数值范围并增强非线性。
        M1 = torch.tanh(self.alpha * self.E1)
        # 对 `E2` 做同样处理。
        M2 = torch.tanh(self.alpha * self.E2)
        # 通过矩阵乘法构造节点两两关系分数。
        A = torch.matmul(M1, M2.T)
        # 用 ReLU 去掉负权重，只保留非负边。
        A = F.relu(A)

        # 使用逐行归一化代替 softmax。
        # 这样每一行和为 1，但又不会像 softmax 那样过于平滑。
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        # 返回学习到的邻接矩阵。
        return A


# ==========================================
# 3. GCN Layer
# 作用：在图结构上进行节点间信息传播。
# 输入形状：`(B, S, N, F)`。
# ==========================================
class GCNLayer(nn.Module):
    """一个简单的图卷积层：先做线性映射，再按邻接矩阵聚合。"""

    def __init__(self, in_dim, out_dim):
        # 调用父类初始化。
        super().__init__()
        # 对每个节点特征做线性变换。
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # `x` 的形状是 `(批大小, 序列长度, 节点数, 特征维度)`。
        x = self.linear(x)
        # 使用爱因斯坦求和在节点维进行邻接加权聚合。
        out = torch.einsum("nm,bsmd->bsnd", adj, x)
        # 返回图卷积后的结果。
        return out


# ==========================================
# 4. Multi-Scale TCN
# 作用：使用多种卷积核长度提取不同时间尺度的时序模式。
# 说明：支持因果和非因果两种填充模式。
# ==========================================
class MultiScaleTCN(nn.Module):
    """
    多尺度时序卷积模块。

    `causal=True` 时只做左填充，保证当前位置不会看到未来信息。
    对于纯重构任务，也可以用 `causal=False` 使用对称填充。
    """

    def __init__(self, num_nodes, kernel_sizes=[3, 5, 7], causal=True):
        # 调用父类初始化。
        super().__init__()
        # 保存是否使用因果卷积。
        self.causal = causal
        # 保存所有卷积层。
        self.convs = nn.ModuleList()
        # 保存卷积核列表，便于前向时同步遍历。
        self.kernel_sizes = kernel_sizes

        # 为每个卷积核大小构建一个深度可分离风格的一维卷积。
        for k in kernel_sizes:
            # 这里 `groups=num_nodes`，表示每个节点各自沿时间卷积，不混通道。
            self.convs.append(
                nn.Conv1d(
                    num_nodes,
                    num_nodes,
                    kernel_size=k,
                    padding=0,
                    groups=num_nodes,
                )
            )

        # 使用线性层把多尺度输出融合成单一路径。
        self.fusion = nn.Linear(len(kernel_sizes), 1)

    def forward(self, x):
        # 输入 `x` 的形状为 `(B, N, S)`，即每个节点是一条长度为 `S` 的序列。
        outputs = []
        # 同时遍历卷积层和对应卷积核大小。
        for conv, k in zip(self.convs, self.kernel_sizes):
            # 因果模式下，只在左侧补 `k-1` 个时间步。
            if self.causal:
                padded = F.pad(x, (k - 1, 0))
            else:
                # 非因果模式下，左右两侧尽量对称补齐。
                padded = F.pad(x, ((k - 1) // 2, k // 2))

            # 对补齐后的序列做一维卷积。
            out = conv(padded)
            # 保存当前尺度输出，形状仍为 `(B, N, S)`。
            outputs.append(out)

        # 在最后一维堆叠多尺度结果，得到 `(B, N, S, K)`。
        out_stack = torch.stack(outputs, dim=-1)
        # 通过线性层学习各尺度权重，并压缩掉最后一维。
        out = self.fusion(out_stack).squeeze(-1)
        # 返回多尺度融合结果。
        return out


# ==========================================
# 5. Frequency Imputer
# 作用：在频域上增强时序信号，用于补全缺失信息。
# 核心思想：FFT -> 频域特征建模 -> IFFT 回到时域。
# ==========================================
class FrequencyImputer(nn.Module):
    """
    频域插补模块。

    它先把时序变换到频域，再对实部和虚部进行建模，
    最后通过逆傅里叶变换回到时域。
    """

    def __init__(self, seq_len, num_nodes=18):
        # 调用父类初始化。
        super().__init__()
        # `rfft` 只保留实信号的一半频谱，因此长度是 `seq_len // 2 + 1`。
        self.freq_len = seq_len // 2 + 1
        # 保存节点数量。
        self.num_nodes = num_nodes

        # 注意力网络：输入实部和虚部拼接后的特征，输出每个频率点的重要性。
        self.attention = nn.Sequential(
            nn.Linear(self.freq_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_len),
            nn.Sigmoid(),
        )

        # 频域增强网络：学习对实部和虚部的残差修正。
        self.freq_enhance = nn.Sequential(
            nn.Linear(self.freq_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_len * 2),
        )

    def forward(self, x):
        # 输入 `x` 的形状是 `(B, S, N)`，先转成 `(B, N, S)` 方便沿时间维做 FFT。
        x_perm = x.permute(0, 2, 1)

        # 对时间维做实数快速傅里叶变换，输出是复数张量。
        xf = torch.fft.rfft(x_perm, dim=2)

        # 把复数频谱拆成实部和虚部。
        real, imag = xf.real, xf.imag
        # 在最后一维拼接实部和虚部，形成可送入全连接层的实数特征。
        feat = torch.cat([real, imag], dim=-1)

        # 计算每个节点每个频率点的注意力权重。
        att_weights = self.attention(feat)

        # 对频域特征做增强，输出仍然是实部和虚部拼接的形式。
        feat_enhanced = self.freq_enhance(feat)
        # 切出增强后的实部。
        real_enh = feat_enhanced[..., :self.freq_len]
        # 切出增强后的虚部。
        imag_enh = feat_enhanced[..., self.freq_len:]

        # 用注意力对增强后的实部加权。
        real_attended = real_enh * att_weights
        # 用注意力对增强后的虚部加权。
        imag_attended = imag_enh * att_weights

        # 采用残差方式修正原始频谱，通常比完全替换更稳定。
        xf_enhanced = xf + torch.complex(real_attended, imag_attended)

        # 把增强后的频谱变回时域，长度指定为原始序列长度。
        x_rec = torch.fft.irfft(xf_enhanced, n=x.size(1), dim=2)

        # 转回 `(B, S, N)`，与其它模块的时域表示对齐。
        return x_rec.permute(0, 2, 1)


# ==========================================
# 6. Gated Fusion
# 作用：融合时域分支和频域分支的输出。
# 核心思想：学习一个门控系数 `z`，动态决定两条分支各占多少权重。
# ==========================================
class GatedFusion(nn.Module):
    """带时序上下文感知的门控融合模块。"""

    def __init__(self, num_nodes, seq_len):
        # 调用父类初始化。
        super().__init__()
        # 门控网络输入是时域和频域特征拼接后的 `(2N)` 通道。
        self.gate_net = nn.Sequential(
            nn.Conv1d(num_nodes * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, num_nodes, kernel_size=1),
            nn.Sigmoid(),
        )
        # 融合后再做层归一化，稳定不同节点维度的分布。
        self.norm = nn.LayerNorm(num_nodes)

    def forward(self, h_time, h_freq):
        # `h_time` 和 `h_freq` 的形状都是 `(B, S, N)`。

        # 转成 `(B, N, S)`，以适配 Conv1d 期望的通道优先格式。
        h_time_perm = h_time.permute(0, 2, 1)
        # 频域分支也做相同变换。
        h_freq_perm = h_freq.permute(0, 2, 1)

        # 在通道维拼接两条分支，得到 `(B, 2N, S)`。
        combined = torch.cat([h_time_perm, h_freq_perm], dim=1)
        # 通过卷积门控网络生成每个节点每个时间步的融合权重。
        z = self.gate_net(combined)
        # 再转回 `(B, S, N)`。
        z = z.permute(0, 2, 1)

        # `z` 越接近 1，越偏向时域分支；越接近 0，越偏向频域分支。
        h = z * h_time + (1 - z) * h_freq
        # 返回归一化后的融合结果。
        return self.norm(h)


# ==========================================
# 7. AGF-ADNet
# 作用：整体异常检测模型，整合图学习、时域分支、频域分支、门控融合和 Transformer。
# ==========================================
class AGF_ADNet(nn.Module):
    """主模型：Adaptive Graph Fusion Anomaly Detection Network。"""

    def __init__(self, num_nodes=18, seq_len=60, d_model=64):
        # 调用父类初始化。
        super().__init__()
        # 保存节点数量。
        self.num_nodes = num_nodes
        # 保存序列长度。
        self.seq_len = seq_len

        # 图结构学习器，负责输出邻接矩阵。
        self.graph = GraphLearner(num_nodes)

        # 时域分支：先图卷积，再时序卷积。
        self.gcn = GCNLayer(1, 1)
        # 这里设为 `causal=False`，因为重构任务允许利用双侧上下文。
        self.tcn = MultiScaleTCN(num_nodes, kernel_sizes=[3, 5, 9], causal=False)
        # 对时域分支结果做层归一化。
        self.time_norm = nn.LayerNorm(num_nodes)

        # 频域分支：做频谱建模后再归一化。
        self.freq = FrequencyImputer(seq_len, num_nodes)
        self.freq_norm = nn.LayerNorm(num_nodes)

        # 融合模块：根据上下文动态融合时域与频域信息。
        self.fusion = GatedFusion(num_nodes, seq_len)

        # Transformer 前先把节点维映射到统一隐藏维度。
        self.input_proj = nn.Linear(num_nodes, d_model)
        # 可学习位置编码，形状为 `(1, 序列长度, 隐藏维度)`。
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 构造单层 Transformer 编码器配置。
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1,
        )
        # 堆叠两层编码器。
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        # 把 Transformer 输出重新映射回节点维度。
        self.output_proj = nn.Linear(d_model, num_nodes)

    def forward(self, x, mask):
        # `mask=1` 表示缺失，`mask=0` 表示观测值。
        # `x` 的形状是 `(B, S, N)`。

        # 先学习当前批次共享的图结构邻接矩阵。
        adj = self.graph()

        # ------------------------------
        # 1. 时域分支
        # ------------------------------
        # 图卷积期望最后一维是特征维，因此先扩一维再在最后 squeeze 回来。
        x_gcn = self.gcn(x.unsqueeze(-1), adj).squeeze(-1)
        # 图卷积结果做归一化。
        x_gcn = self.time_norm(x_gcn)

        # TCN 期望输入 `(B, N, S)`，因此先转置，卷积后再转回来。
        x_tcn = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        # 将图卷积和时序卷积结果相加，形成时域综合表示。
        h_time = x_gcn + x_tcn

        # ------------------------------
        # 2. 频域分支
        # ------------------------------
        # 使用频域插补模块提取另一种视角的时序表示。
        h_freq = self.freq(x)
        # 频域结果做归一化。
        h_freq = self.freq_norm(h_freq)

        # ------------------------------
        # 3. 门控融合
        # ------------------------------
        # 融合时域和频域表示，得到插补结果。
        x_imp = self.fusion(h_time, h_freq)

        # ------------------------------
        # 4. 用插补值填补缺失位置
        # ------------------------------
        # 计算观测掩码：观测位置为 1，缺失位置为 0。
        observed_mask = 1.0 - mask
        # 观测位置保留原值，缺失位置使用插补值填充。
        x_filled = x * observed_mask + x_imp * mask

        # ------------------------------
        # 5. Transformer 重构
        # ------------------------------
        # 先投影到 Transformer 隐空间，并加上位置编码。
        z = self.input_proj(x_filled) + self.pos_enc
        # 通过 Transformer 编码器建模全局时序依赖。
        z = self.transformer(z)
        # 再映射回原始节点维度，作为最终重构结果。
        x_rec = self.output_proj(z)

        # 返回重构结果、图结构和融合后的插补结果。
        return x_rec, adj, x_imp


# ==========================================
# 8. Loss And Utilities
# 作用：定义训练损失、输入掩码处理、异常分数计算和可视化辅助函数。
# ==========================================
def dual_domain_loss(x_rec, x_true, missing_mask, target_mask, adj, freq_weight=0.1, sparsity_weight=0.1):
    """
    双域损失函数。

    组成：
    1. 时域重构损失。
    2. 频域谱幅损失。
    3. 图邻接矩阵的熵式稀疏约束。

    参数说明：
    - `missing_mask=1` 表示原始缺失位置。
    - `target_mask=1` 表示当前需要监督误差的位置。
    """
    # 统一转成浮点，避免布尔张量参与除法或乘法时出现类型不一致。
    target_mask = target_mask.float()
    # 只在 `target_mask` 指定的位置计算均方重构误差。
    recon_loss = ((x_rec - x_true) * target_mask).pow(2).sum() / target_mask.sum().clamp_min(1.0)

    # 计算每个样本的观测比例，用于筛除缺失过多、频域监督不可靠的样本。
    observed_ratio = (1.0 - missing_mask).mean(dim=[1, 2])
    # 只保留观测比例大于 0.5 的样本。
    valid_samples = observed_ratio > 0.5

    # 频域监督只在有真实值的位置用 `x_true`，其它位置用 `x_rec.detach()` 补齐。
    # 这样做可以避免频谱损失在无标签区域传播不稳定梯度。
    freq_target = x_rec.detach() * (1.0 - target_mask) + x_true * target_mask

    # 只有在存在有效样本时才计算频谱损失。
    if valid_samples.sum() > 0:
        # 选出有效样本，并转成 `(B', N, S)` 以便沿时间维做 FFT。
        x_rec_valid = x_rec[valid_samples].permute(0, 2, 1)
        # 同样处理对应的目标序列。
        x_true_valid = freq_target[valid_samples].permute(0, 2, 1)

        # 对重构结果做频域变换。
        fft_rec = torch.fft.rfft(x_rec_valid, dim=2)
        # 对目标序列做频域变换。
        fft_true = torch.fft.rfft(x_true_valid, dim=2)

        # 这里比较的是幅值谱，而不是相位或复数整体差异。
        freq_loss = (fft_rec.abs() - fft_true.abs()).pow(2).mean()
    else:
        # 若无有效样本，则频域损失记为 0，并放在同一设备上。
        freq_loss = torch.tensor(0.0, device=x_rec.device)

    # 邻接矩阵是逐行归一化的，L1 范数几乎恒定，因此使用熵作为稀疏度正则。
    # 熵越低，通常表示分布越尖锐、越接近稀疏。
    sparsity_loss = -(adj * torch.log(adj + 1e-8)).sum(dim=-1).mean()

    # 返回三项损失的加权和。
    return recon_loss + freq_weight * freq_loss + sparsity_weight * sparsity_loss


def apply_missing_mask(x, missing_mask):
    """把缺失位置直接置为 0，作为模型输入。"""
    return x.masked_fill(missing_mask.bool(), 0.0)


def anomaly_scores(model, windows, masks, device, batch_size=32):
    """计算每个窗口的异常分数，这里使用观测位置上的平均重构误差。"""
    # 保存所有窗口对应的异常分数。
    scores = []
    # 构建批量数据加载器。
    loader = DataLoader(
        TensorDataset(torch.tensor(windows), torch.tensor(masks)),
        batch_size=batch_size,
    )

    # 切换到评估模式，关闭 Dropout 等训练行为。
    model.eval()
    # 推理阶段关闭梯度，节省显存并提升速度。
    with torch.no_grad():
        # 逐批处理窗口数据。
        for x, missing_mask in loader:
            # 把输入放到目标设备上。
            x = x.to(device)
            # 把缺失掩码也放到同一设备。
            missing_mask = missing_mask.to(device)

            # 构造观测掩码，用于只统计真实观测位置的重构误差。
            observed_mask = 1.0 - missing_mask
            # 把缺失位置置 0，得到模型输入。
            x_input = apply_missing_mask(x, missing_mask)
            # 前向推理，拿到重构输出。
            xr, _, _ = model(x_input, missing_mask)

            # 只在观测位置计算平方误差，并按样本汇总。
            sq_err = ((xr - x) * observed_mask).pow(2).sum(dim=[1, 2])
            # 统计每个样本的观测元素数量，避免除 0。
            obs_cnt = observed_mask.sum(dim=[1, 2]).clamp_min(1e-8)
            # 使用平均误差作为异常分数，转回 NumPy 后加入结果列表。
            scores.extend((sq_err / obs_cnt).cpu().numpy())

    # 返回最终的一维分数数组。
    return np.array(scores)


def build_test_labels(num_scores):
    """构造测试标签：前半段视为正常，后半段视为异常。"""
    # 先全部初始化为 0，表示正常。
    labels = np.zeros(num_scores, dtype=int)
    # 后半段改为 1，表示异常。
    labels[num_scores // 2 :] = 1
    # 返回标签数组。
    return labels


def plot_results(
    scores,
    threshold,
    split_idx,
    save_path='/home/akira/codespace/mra-detection/anomaly_detection_results.png',
):
    """绘制测试分数、阈值和测试集分界线。"""
    # 创建画布。
    plt.figure(figsize=(6, 5))
    # 绘制异常分数曲线。
    plt.plot(scores, label='测试异常分数', alpha=0.7)
    # 绘制阈值水平线。
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    # 绘制测试集正常/异常分界位置。
    plt.axvline(x=split_idx, color='g', linestyle=':', label='测试集分界')
    # 设置横轴名称。
    plt.xlabel('测试样本索引')
    # 设置纵轴名称。
    plt.ylabel('重构误差')
    # 设置图标题。
    plt.title('AGF-ADNet异常检测')
    # 显示图例。
    plt.legend()
    # 打开网格，透明度设低一些避免抢视觉。
    plt.grid(True, alpha=0.3)

    # 调整布局，防止标签被截断。
    plt.tight_layout()
    # 保存图片到指定路径。
    plt.savefig(save_path, dpi=150)
    # 打印保存位置。
    print(f"\nPlot saved to: {save_path}")
    # 显示图像窗口。
    plt.show()


# ==========================================
# 9. Training Pipeline
# 作用：加载数据、训练模型、确定阈值、评估测试集并可视化结果。
# ==========================================
def train():
    """训练并评估整个异常检测流程。"""
    # 定义窗口长度。
    SEQ_LEN = 60
    # 定义数据根目录。
    DATA_DIR = "./data"
    # 创建数据构建器。
    builder = DatasetBuilder(SEQ_LEN)

    # 指定训练集文件匹配模式，这里假设都是 18 列。
    TRAIN_PATTERN = "train_*.csv"
    # 指定测试集文件匹配模式，这里假设也都是 18 列。
    TEST_PATTERN = "test_*.csv"

    # 提示即将读取训练数据。
    print("Loading training data...")
    # 从训练目录读取数据和缺失掩码。
    train_data, train_mask = builder.load_dir(os.path.join(DATA_DIR, "train"), TRAIN_PATTERN)
    # 读取完后，从构建器中拿到特征维度。
    num_features = builder.num_features
    # 打印训练数据形状和特征数。
    print(f"Training data: {train_data.shape}, num_features={num_features}")

    # 提示即将读取测试数据。
    print("\nLoading test data...")
    # 从测试目录读取数据和缺失掩码。
    test_data, test_mask = builder.load_dir(os.path.join(DATA_DIR, "test"), TEST_PATTERN)
    # 打印测试数据形状。
    print(f"Test data: {test_data.shape}")

    # 只在训练集上拟合标准化器，避免数据泄漏。
    builder.fit_scaler(train_data)
    # 对训练数据做标准化。
    train_data_scaled = builder.transform(train_data)
    # 对测试数据使用同一标准化器做变换。
    test_data_scaled = builder.transform(test_data)

    # 把标准化后的数据切成滑动窗口。
    Xtr, Mtr = builder.create_windows(train_data_scaled, train_mask)
    # 测试集也切成滑动窗口。
    Xte, Mte = builder.create_windows(test_data_scaled, test_mask)

    # 如果没有得到任何训练窗口，则直接退出。
    if len(Xtr) == 0:
        print("No training data available!")
        return

    # 构建训练数据加载器，启用随机打乱。
    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(Mtr)),
        batch_size=32,
        shuffle=True,
    )

    # 自动选择 GPU 或 CPU。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 打印实际使用设备。
    print(f"Device: {device}")

    # 初始化主模型，并把它放到目标设备。
    model = AGF_ADNet(num_nodes=num_features, seq_len=SEQ_LEN).to(device)
    # 使用 Adam 优化器。
    opt = optim.Adam(model.parameters(), lr=1e-3)
    # 使用验证式的自适应学习率调度器，这里按训练均值损失调整。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    # 用于保存最好的一份模型参数。
    best_state = None
    # 最优损失先初始化为正无穷。
    best_loss = float("inf")

    # 打印训练开始提示。
    print("Starting training...")
    # 这里只训练 3 个 epoch。
    for epoch in range(10):
        # 切换到训练模式。
        model.train()
        # 累计本轮所有 batch 的损失。
        total_loss = 0

        # 逐批读取训练窗口和其缺失掩码。
        for x, m in train_loader:
            # 把数据和掩码送到设备。
            x, m = x.to(device), m.to(device)
            # 清空上一步梯度。
            opt.zero_grad()

            # 原本可观测的位置为 True，缺失位置为 False。
            observed = ~m.bool()
            # 设定自监督随机遮挡概率。
            rand_drop_prob = 0.1
            # 仅在当前可观测位置上随机再遮挡一部分，制造监督目标。
            rand_drop = (torch.rand_like(x) < rand_drop_prob) & observed
            # 把这部分被随机遮挡的位置作为当前监督目标。
            target_mask = rand_drop.float()
            # 如果本 batch 恰好一个都没抽中，则退化为监督所有原始观测位置。
            if not rand_drop.any():
                target_mask = observed.float()

            # 复制一份原始缺失掩码，避免原地改动影响原始 `m`。
            m_input = m.clone()
            # 将随机遮挡位置也标记为缺失，形成模型输入掩码。
            m_input[rand_drop] = 1.0
            # 用缺失掩码把输入中缺失位置清零。
            x_input = apply_missing_mask(x, m_input)

            # 前向传播，得到重构结果和图结构。
            x_rec, adj, _ = model(x_input, m_input)

            # 计算双域损失。
            loss = dual_domain_loss(x_rec, x, m, target_mask, adj)

            # 反向传播。
            loss.backward()
            # 做梯度裁剪，防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 更新参数。
            opt.step()
            # 累加当前 batch 损失值。
            total_loss += loss.item()

        # 求当前 epoch 的平均损失。
        avg_loss = total_loss / len(train_loader)
        # 用平均损失驱动学习率调度器。
        scheduler.step(avg_loss)
        # 如果本轮更优，就保存当前参数快照。
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

        # 打印当前轮数、平均损失和学习率。
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, LR: {opt.param_groups[0]['lr']:.6f}")

    # 训练结束后，如果记录过最佳参数，就恢复到最佳状态。
    if best_state is not None:
        model.load_state_dict(best_state)

    # 提示开始基于训练集估计阈值。
    print("\nTraining completed. Computing threshold from training set...")

    # 在训练集上计算异常分数，用于建立阈值。
    train_scores = anomaly_scores(model, Xtr, Mtr, device=device, batch_size=32)
    # 这里使用 `均值 + 标准差` 作为经验阈值。
    threshold = float(np.mean(train_scores) + np.std(train_scores))

    # 打印训练集分数统计。
    print(f"\nTraining Set Score Stats:")
    print(f"  Mean: {np.mean(train_scores):.6f}")
    print(f"  Std:  {np.std(train_scores):.6f}")
    print(f"  Threshold (mean train score): {threshold:.6f}")

    # 使用训练集得到的阈值，在测试集上评估。
    print("\nEvaluating on test set...")
    # 计算测试集每个窗口的异常分数。
    test_scores_arr = anomaly_scores(model, Xte, Mte, device=device, batch_size=32)
    # 构造测试标签，默认前半正常后半异常。
    test_labels = build_test_labels(len(test_scores_arr))
    # 记录分界位置，便于画图与打印说明。
    split_idx = len(test_scores_arr) // 2

    # 打印异常检测结果概览。
    print(f"\nAnomaly Detection Results:")
    print(f"  Mean Score: {np.mean(test_scores_arr):.6f}")
    print(f"  Std Score:  {np.std(test_scores_arr):.6f}")
    print(f"  Threshold (from train): {threshold:.6f}")
    print(f"  Test split: [0:{split_idx}) normal, [{split_idx}:{len(test_scores_arr)}) anomaly")
    print(f"  Anomalies detected: {(test_scores_arr > threshold).sum()} / {len(test_scores_arr)}")

    # 真实标签就是刚才构造的测试标签。
    y_true = test_labels
    # 预测标签由“分数是否超过阈值”得到。
    y_pred = (test_scores_arr > threshold).astype(int)

    # 计算准确率。
    acc = accuracy_score(y_true, y_pred)
    # 计算精确率，若分母为 0 则返回 0。
    prec = precision_score(y_true, y_pred, zero_division=0)
    # 计算召回率。
    rec = recall_score(y_true, y_pred, zero_division=0)
    # 计算 F1 分数。
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 打印分类指标。
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # 绘制并保存测试结果图。
    plot_results(test_scores_arr, threshold, split_idx)

    # 返回训练完成的模型和测试分数。
    return model, test_scores_arr


# 当脚本被直接运行时，执行训练入口。
if __name__ == "__main__":
    # 启动训练流程，并接收返回的模型和分数。
    model, scores = train()
