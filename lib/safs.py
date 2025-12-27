"""
SFD 特征合成模块 (SAFS - Synthetic Feature-based Decoupled training)

本模块实现了 SFD 论文中的特征合成算法，包括：
1. MeanCov Aligner: 将未对齐的特征变换为与全局统计量对齐的特征 (Eq. 11-14)
2. 特征合成循环: 通过优化 Memory Bank 生成合成特征 (Algorithm 1 & Eq. 15-17)

参考来源: SFD/src/flexp/sfd/nn.py, SFD/src/flexp/sfd/safs.py
"""

import math
import torch
import torch.nn as nn
from tqdm import tqdm

from lib.sfd_utils import RFF


class MeanCovAligner(nn.Module):
    """
    MeanCov Aligner 类，实现论文公式 Eq. 11-14。
    
    目标：将未对齐的特征 Z_{raw}^c 变换为 Z_{syn}^c，
    使其均值和协方差与全局统计量 μ_g^c, Σ_g^c 对齐。
    
    步骤：
    1. 计算当前 batch 合成特征 Z_{raw}^c 的均值 μ_{raw}^c 和协方差 Σ_{raw}^c
    2. Cholesky 分解 (Eq. 11 & 12): Σ_g^c = L_g L_g^T, Σ_{raw}^c = L_{raw} L_{raw}^T
    3. 计算变换矩阵 A (Eq. 13): A = L_g L_{raw}^{-1}
    4. 执行仿射变换 (Eq. 14): Z_{syn}^c = (Z_{raw}^c - μ_{raw}^c) A^T + μ_g^c
    """
    
    def __init__(
        self,
        target_mean: torch.Tensor,
        target_cov: torch.Tensor,
        target_cov_eps: float = 1e-5,
    ):
        """
        初始化 MeanCov Aligner。
        
        参数:
            target_mean: 目标均值，形状 (d,)，对应全局统计量 μ_g^c
            target_cov: 目标协方差矩阵，形状 (d, d)，对应全局统计量 Σ_g^c
            target_cov_eps: 协方差矩阵对角线的 jitter，用于数值稳定性（默认 1e-5）
        """
        super().__init__()
        
        # 注册目标均值为 buffer（不参与梯度更新，但会随模型保存/加载）
        self.register_buffer('target_mean', target_mean)
        self.target_mean: torch.Tensor
        
        # 注册 jitter 参数
        self.register_buffer('target_cov_eps', torch.tensor(target_cov_eps))
        self.target_cov_eps: torch.Tensor
        
        # 对目标协方差矩阵进行 Cholesky 分解（Eq. 11）
        # 添加 jitter 以保证数值稳定性
        d = target_cov.shape[0]
        target_cov_stable = (
            target_cov.to(dtype=torch.float64, device='cpu')
            + target_cov_eps * torch.eye(d, dtype=torch.float64)
        )
        
        try:
            # Cholesky 分解: Σ_g^c = L_g L_g^T
            L_g = torch.linalg.cholesky_ex(target_cov_stable, check_errors=True)[0]
        except RuntimeError as e:
            raise RuntimeError(
                f'目标协方差矩阵的 Cholesky 分解失败，eps={target_cov_eps}。'
                f'请检查输入是否有效或尝试更大的 target_cov_eps。'
            ) from e
        
        assert torch.isfinite(L_g).all(), \
            f'L_g 中存在非有限值: {L_g}，请检查输入是否有效或尝试更大的 target_cov_eps。'
        
        # 注册 L_g 为 buffer
        self.register_buffer('L_g', L_g)
        self.L_g: torch.Tensor
        assert self.L_g.dtype == torch.float64
    
    def forward(
        self,
        data: torch.Tensor,
        *,
        decompose_dtype: torch.dtype = torch.float64,
        input_cov_eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        前向传播：执行 MeanCov 对齐变换。
        
        参数:
            data: 输入特征张量，形状 (n, d)，对应 Z_{raw}^c
            decompose_dtype: Cholesky 分解使用的数据类型（默认 float64 以提高精度）
            input_cov_eps: 输入协方差矩阵对角线的 jitter（默认 1e-5）
        
        返回:
            aligned_data: 对齐后的特征张量，形状 (n, d)，对应 Z_{syn}^c
        """
        d = data.shape[1]
        
        # 步骤 1: 计算当前 batch 的均值 μ_{raw}^c
        mean_raw = data.mean(dim=0)  # shape: (d,)
        
        # 步骤 2: 计算当前 batch 的协方差 Σ_{raw}^c
        # 使用 torch.cov 计算协方差矩阵
        cov_raw = data.T.cov()  # shape: (d, d)
        
        # 添加 jitter 以保证数值稳定性
        cov_raw_stable = cov_raw.to(decompose_dtype) + input_cov_eps * torch.eye(
            d, dtype=decompose_dtype, device=cov_raw.device
        )
        
        try:
            # Cholesky 分解: Σ_{raw}^c = L_{raw} L_{raw}^T (Eq. 12)
            L_raw = torch.linalg.cholesky_ex(cov_raw_stable, check_errors=True)[0]
        except RuntimeError as e:
            raise RuntimeError(
                f'输入协方差矩阵的 Cholesky 分解失败，eps={input_cov_eps}。'
                f'请检查输入是否有效或尝试更大的 input_cov_eps。'
            ) from e
        
        assert torch.isfinite(L_raw).all(), \
            f'L_raw 中存在非有限值: {L_raw}，请检查输入是否有效或尝试更大的 input_cov_eps。'
        assert L_raw.dtype == decompose_dtype
        
        # 步骤 3: 计算变换矩阵 A (Eq. 13)
        # A = L_g @ L_{raw}^{-1}
        # 使用 torch.linalg.solve_triangular 求解下三角线性系统，更高效且数值稳定
        # 求解 L_{raw} @ A^T = L_g^T，即 A = (L_g @ L_{raw}^{-1})
        # 等价于求解 L_{raw}^T @ A = L_g，然后转置
        A = torch.linalg.solve_triangular(
            L_raw, 
            self.L_g.to(L_raw.dtype), 
            upper=False, 
            left=False
        ).to(data.dtype)  # shape: (d, d)
        
        assert torch.isfinite(A).all(), f'A 中存在非有限值: {A}'
        
        # 步骤 4: 执行仿射变换 (Eq. 14)
        # Z_{syn}^c = (Z_{raw}^c - μ_{raw}^c) A^T + μ_g^c
        aligned_data = (data - mean_raw) @ A.T + self.target_mean
        
        return aligned_data


def make_syn_nums(class_sizes: list[int], max_num: int, min_num: int) -> list[int]:
    """
    根据类别大小生成每个类别的合成特征数量。
    
    根据论文，最小类生成 max_num 个，最大类生成 min_num 个，中间类线性插值。
    论文中：最小类生成 2000 个，最大类生成 600 个。
    
    参数:
        class_sizes: 每个类别的样本数列表（仅考虑顺序，不考虑实际值）
        max_num: 最小类别的合成特征数量（默认 2000）
        min_num: 最大类别的合成特征数量（默认 600）
    
    返回:
        每个类别的合成特征数量列表
    """
    assert len(class_sizes) > 0, 'class_sizes 不能为空'
    assert max_num >= min_num, 'max_num 必须大于等于 min_num'
    
    # 获取唯一的大小值并排序
    unique_sorted_sizes = sorted(list(set(class_sizes)))
    num_unique_sizes = len(unique_sorted_sizes)
    size_to_syn_num_map: dict[int, int] = {}
    
    if num_unique_sizes == 1:
        # 如果所有类别大小相同，都使用 max_num
        size_to_syn_num_map[unique_sorted_sizes[0]] = max_num
    else:
        # 线性插值：最小类（rank=0）得到 max_num，最大类（rank=num_unique_sizes-1）得到 min_num
        scaling_denominator = float(num_unique_sizes - 1)
        for rank, size in enumerate(unique_sorted_sizes):
            syn_num_float = max_num - (rank / scaling_denominator) * (max_num - min_num)
            size_to_syn_num_map[size] = int(round(syn_num_float))
    
    # 根据原始顺序返回结果
    result_syn_nums = [size_to_syn_num_map[size] for size in class_sizes]
    return result_syn_nums


def train_memory_bank(
    *,
    feature_dim: int,
    syn_num: int,
    device: 'torch.device | str',
    aligner: MeanCovAligner,
    rf_model: RFF,
    real_rf_mean: torch.Tensor,
    steps: int,
    lr: float,
    input_cov_eps: float = 1e-5,
) -> torch.Tensor:
    """
    训练 Memory Bank（合成原始特征 Z_{raw}^c）。
    
    通过优化损失函数 L_syn = L_MMD + L_ARR 来更新 Memory Bank。
    
    参数:
        feature_dim: 特征维度
        syn_num: 该类别的合成特征数量 M_c
        device: 计算设备
        aligner: MeanCov Aligner 实例
        rf_model: RFF 模型，用于计算随机特征
        real_rf_mean: 真实的 RFF 均值 φ_g^c（目标值）
        steps: 优化步数
        lr: 学习率
        input_cov_eps: 输入协方差矩阵的 jitter
    
    返回:
        训练好的 Memory Bank（合成原始特征），形状 (M_c, feature_dim)
    """
    torch.cuda.empty_cache()
    
    # 初始化 Memory Bank（需要梯度）
    memory_bank = nn.Parameter(
        torch.randn((syn_num, feature_dim), device=device),
        requires_grad=True
    )
    
    real_rf_mean = real_rf_mean.to(device)
    rf_model = rf_model.to(device)
    aligner = aligner.to(device)
    
    # 使用 SGD 优化器（论文使用 SGD）
    optimizer = torch.optim.SGD([memory_bank], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    progress_bar = tqdm(range(steps), desc='训练 Memory Bank', leave=False)
    
    for step in progress_bar:
        # 获取当前 batch 的合成原始特征
        syn_z_raw = memory_bank  # shape: (M_c, feature_dim)
        
        # 通过 MeanCov Aligner 对齐得到合成特征
        syn_z = aligner.forward(syn_z_raw, input_cov_eps=input_cov_eps)
        assert torch.isfinite(syn_z).all(), f'syn_z 中存在非有限值: {syn_z}'
        
        # 计算合成特征的 RFF 映射
        syn_rf = rf_model.forward(syn_z)  # shape: (M_c, D_rf)
        syn_rf_mean = syn_rf.mean(dim=0)  # shape: (D_rf,)
        
        # 计算 MMD 损失 (Eq. 15)
        # L_MMD = || φ_g^c - (1/M_c) Σ_j φ(z_{syn,j}^c) ||_1
        # 论文指出使用 L1 范数收敛更快
        mmd_loss = (syn_rf_mean - real_rf_mean).abs().sum()
        
        # 计算 ARR 正则项 (Eq. 16)
        # L_ARR = - Σ_j min(0, z_{syn,j}^c)
        # 假设 Encoder 使用 ReLU 激活，特征应非负
        range_reg = -torch.min(torch.tensor(0.0, device=device), syn_z).sum(dim=1).mean(dim=0)
        
        # 总损失 (Eq. 17)
        loss = mmd_loss + range_reg
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 返回训练好的 Memory Bank（分离梯度）
    trained_memory_bank = memory_bank.detach().cpu()
    torch.cuda.empty_cache()
    
    return trained_memory_bank


def feature_synthesis(
    *,
    feature_dim: int,
    class_num: int,
    device,
    aligners: list[MeanCovAligner],
    rf_model: RFF,
    class_rf_means: list[torch.Tensor],
    steps: int,
    lr: float,
    syn_num_per_class: list[int],
    input_cov_eps: float = 1e-5,
):
    """
    特征合成主函数，实现 Algorithm 1。
    
    对每个类别 c，通过优化 Memory Bank 生成合成特征。
    
    参数:
        feature_dim: 特征维度
        class_num: 类别数量
        device: 计算设备
        aligners: 每个类别的 MeanCov Aligner 列表
        rf_model: RFF 模型
        class_rf_means: 每个类别的全局 RFF 均值列表
        steps: 优化步数
        lr: 学习率
        syn_num_per_class: 每个类别的合成特征数量列表
        input_cov_eps: 输入协方差矩阵的 jitter
    
    返回:
        合成特征数据集列表，每个元素是一个字典，包含：
            - 'class_index': 类别索引
            - 'synthetic_raw_features': 合成原始特征 Z_{raw}^c
            - 'synthetic_features': 对齐后的合成特征 Z_{syn}^c
    """
    memory_banks: list[torch.Tensor] = []  # 每个类别的合成原始特征
    class_syn_z: list[torch.Tensor] = []  # 每个类别的对齐后合成特征
    
    rf_model = rf_model.to(device)
    
    for c in tqdm(range(class_num), desc='特征合成', leave=False):
        aligner = aligners[c].to(device)
        torch.cuda.empty_cache()
        
        # 训练 Memory Bank（合成原始特征）for class c
        syn_zc_raw = train_memory_bank(
            feature_dim=feature_dim,
            syn_num=syn_num_per_class[c],
            device=device,
            aligner=aligners[c],
            rf_model=rf_model,
            real_rf_mean=class_rf_means[c],
            steps=steps,
            lr=lr,
            input_cov_eps=input_cov_eps,
        )
        memory_banks.append(syn_zc_raw)
        torch.cuda.empty_cache()
        
        # 通过 MeanCov Aligner 生成对齐后的合成特征
        syn_zc = aligner.forward(syn_zc_raw.to(device), input_cov_eps=input_cov_eps).cpu()
        assert torch.isfinite(syn_zc).all(), f'syn_zc 中存在非有限值: {syn_zc}'
        class_syn_z.append(syn_zc)
        
        aligner.cpu()
        torch.cuda.empty_cache()
    
    rf_model.cpu()
    torch.cuda.empty_cache()
    
    # 返回结果列表
    return [
        {
            'class_index': c,
            'synthetic_raw_features': memory_banks[c],
            'synthetic_features': class_syn_z[c],
        }
        for c in range(class_num)
    ]

