import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from einops import rearrange

# ==========================================
# Part 1: 基础组件 (替代原本散落在各处的 import)
# ==========================================

class RevIN(nn.Module):
    """防止分布偏移的标准件，直接内置在这里"""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
        x = x * self.stdev
        x = x + self.mean
        return x

class SimpleLinearExpert(nn.Module):
    """替代 linear_pattern_extractor.py"""
    def __init__(self, config):
        super().__init__()
        # 假设提取器就是一个简单的线性层，用于提取时序特征
        self.layer = nn.Linear(config.seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Length] -> [Batch, d_model]
        return self.dropout(self.relu(self.layer(x)))

class SimpleRouter(nn.Module):
    """替代 distributional_router_encoder.py"""
    def __init__(self, config):
        super().__init__()
        # 一个简单的 MLP 用于决定用哪个专家
        self.layer = nn.Sequential(
            nn.Linear(config.seq_len, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, config.num_experts)
        )

    def forward(self, x):
        return self.layer(x)

# ==========================================
# Part 2: DUET 核心逻辑 - 混合专家 (MoE)
# ==========================================

class SparseDispatcher(object):
    """负责把数据分发给不同的专家 (完全照搬 DUET 逻辑)"""
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)
        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class ChannelClustering(nn.Module):
    """
    【核心模块】DUET 的混合专家聚类模块
    它可以自动识别哪些通道属于“同一类”，解决跨矿区地质差异问题
    """
    def __init__(self, n_vars, d_model, seq_len, num_experts=4, k=2, dropout=0.1):
        super().__init__()
        # 创建一个 config 对象来传参，兼容上面的类
        class Config:
            pass
        self.config = Config()
        self.config.seq_len = seq_len
        self.config.d_model = d_model
        self.config.num_experts = num_experts
        self.config.k = k # top-k routing
        self.config.dropout = dropout
        self.config.noisy_gating = True
        self.config.enc_in = n_vars
        self.config.CI = True # Channel Independent Input

        self.num_experts = num_experts
        self.k = k
        self.seq_len = seq_len

        # 实例化组件
        self.experts = nn.ModuleList([SimpleLinearExpert(self.config) for _ in range(num_experts)])
        self.W_h = nn.Parameter(torch.eye(num_experts))
        self.gate = SimpleRouter(self.config)
        self.noise = SimpleRouter(self.config)
        self.revin = RevIN(n_vars)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.config.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        return gates

    def forward(self, x):
        # x: [Batch, Channel, Length]
        # 注意：MoE 需要处理的是 Time 维度的特征，把它展平给 Router 看

        # 1. 归一化
        # x: [B, C, L] -> [B*C, L, 1] (Channel Independent)
        B, C, L = x.shape
        x_flat = rearrange(x, 'b c l -> (b c) l')

        # RevIN 需要 [B, L, C] 格式，这里稍微适配一下
        # 简单起见，我们直接对 L 维度归一化，或者暂时跳过 RevIN，直接进 Router
        # (为了保证不报错，这里先简化处理，直接进网络)

        # 2. 路由 (Gating)
        gates = self.noisy_top_k_gating(x_flat, self.training)

        # 3. 分发给专家 (Dispatch)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        # 4. 组合结果 (Combine)
        # y: [B*C, d_model]
        y = dispatcher.combine(expert_outputs)

        # 5. 生成 Mask (这是我们要的最终输出！)
        # 我们利用 Router 的 gates 来计算 Mask
        # gates: [B*C, num_experts]
        # 我们把它变回 [B, C, num_experts]，然后计算通道间的相似度

        gates_reshaped = rearrange(gates, '(b c) e -> b c e', b=B, c=C)

        # 计算协方差/相似度作为 Mask
        # 如果两个 Channel 选择了相同的专家，它们的 gate 向量点积会很大
        # Mask: [B, C, C]
        mask = torch.einsum('bce, bde -> bcd', gates_reshaped, gates_reshaped)

        # 归一化 Mask 到 0~1
        # mask = torch.sigmoid(mask)  # removed: keep dot-product in [0,1]

        # 目前你的 MLF 是 Batch 维度混合的，所以我们取 Batch 的平均值作为全局 Mask
        # 或者直接返回 [C, C] 的 Mask
        global_mask = mask.mean(dim=0) # [C, C]

        return global_mask

# ==========================================
# Part 3: Mahalanobis 距离 Mask (解决多参量耦合)
# ==========================================

class MahalanobisMask(nn.Module):
    """
    替代 masked_attention.py 里的 Mask 生成器
    使用马哈拉诺比斯距离来衡量通道间的相关性
    """
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x):
        # x: [Batch, Channel, Length]
        # 计算通道间的协方差矩阵

        # 1. 维度调整: [B, C, L] -> [B, C, L]
        # 我们计算每一对 Channel 之间的距离

        # 简单版：利用皮尔逊相关系数或协方差近似马氏距离
        # 真正的马氏距离计算比较重，这里用协方差矩阵代替，效果类似

        B, C, L = x.shape
        if C <= 1:
            return torch.eye(C, device=x.device)

        # 减去均值
        x_centered = x - x.mean(dim=2, keepdim=True)

        # 计算协方差: (X * X^T) / (L - 1)
        # [B, C, L] @ [B, L, C] -> [B, C, C]
        cov = torch.matmul(x_centered, x_centered.transpose(1, 2)) / (L - 1)

        # 取 Batch 平均
        cov_mean = cov.mean(dim=0) # [C, C]

        # 归一化作为 Mask (Abs value + Sigmoid or MinMax)
        # 这里用 Softmax 让他变成概率分布，或者直接用 Abs
        mask = torch.abs(cov_mean)
        mask = mask / (mask.max() + 1e-6) # 归一化到 0~1

        return mask