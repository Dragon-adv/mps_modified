"""
SFD 工具函数移植模块

本模块从 SFD 项目中移植了 Random Fourier Features (RFF) 相关的核心组件，
用于在 FedMPS 项目中使用。

参考来源: SFD/src/flexp/sfd/nn.py
"""

import math
import torch
import torch.nn as nn


def make_orf_matrix(dim_in: int, dim_out: int, std: float, device):
    """
    生成正交随机特征矩阵 (Orthogonal Random Features Matrix)。

    该方法通过以下步骤生成正交随机矩阵：
    1. 计算需要的 batch 数量
    2. 使用 torch.randn 生成高斯随机矩阵
    3. 使用 torch.linalg.qr 进行 QR 分解获取正交矩阵 Q
    4. 使用 torch.distributions.Chi2 采样缩放因子 S
    5. 最终权重矩阵 W = std * S * Q

    Args:
        dim_in (int): 输入维度
        dim_out (int): 输出维度
        std (float): 标准差，用于缩放最终矩阵
        device: PyTorch 设备 (torch.device 或 str)

    Returns:
        torch.Tensor: 形状为 (dim_out, dim_in) 的正交随机特征矩阵
    """
    # 计算需要的 batch 数量
    num_batches = dim_out // dim_in + 1
    
    # 使用 torch.randn 生成高斯随机矩阵
    # Shape: (num_batches, dim_in, dim_in)
    rand_matrices = torch.randn((num_batches, dim_in, dim_in), device=device)
    
    # 使用 torch.linalg.qr 进行 QR 分解获取正交矩阵 Q
    # Q_matrices shape: (num_batches, dim_in, dim_in)
    Q_matrices = torch.linalg.qr(rand_matrices).Q
    
    # 使用 torch.distributions.Chi2 采样缩放因子 S
    chi2_dist = torch.distributions.Chi2(df=torch.tensor(dim_in, device=device))
    # 从 Chi-squared 分布采样并取平方根得到 Chi 分布
    # s_batch shape: (num_batches, dim_in)
    s_batch = chi2_dist.sample(torch.Size([num_batches, dim_in])).sqrt()
    
    # 为每个正交矩阵创建对角缩放矩阵
    # diag_s_batch shape: (num_batches, dim_in, dim_in)
    diag_s_batch = torch.stack([torch.diag(s) for s in s_batch])
    
    # 最终权重矩阵 W = std * S * Q
    # V_matrices shape: (num_batches, dim_in, dim_in)
    V_matrices = std * torch.bmm(diag_s_batch, Q_matrices)
    
    # 将多个 batch 的结果拼接并重塑为最终矩阵
    # W_full shape: (dim_in, num_batches * dim_in)
    W_full = V_matrices.transpose(0, 1).reshape(dim_in, -1)
    
    # 截取前 dim_out 列，并转置得到 (dim_out, dim_in) 形状
    return W_full[:, :dim_out].T


class RFF(nn.Module):
    """
    Random Fourier Features (RFF) 模块。

    该类实现了随机傅里叶特征映射，用于近似 RBF (Radial Basis Function) 核。
    通过将输入映射到高维空间，可以高效地计算核函数。

    参考: SFD/src/flexp/sfd/nn.py
    """

    def __init__(self, d: int, D: int, gamma: float, device, rf_type: str = 'orf'):
        """
        初始化 RFF 模块。

        Args:
            d (int): 输入特征维度
            D (int): 映射后的特征维度，必须是偶数
            gamma (float): RBF 核参数，gamma = 1 / (2 * sigma^2)
            device: PyTorch 设备 (torch.device 或 str)
            rf_type (str): 随机特征类型，默认为 'orf' (正交随机特征)
        """
        super(RFF, self).__init__()
        assert D % 2 == 0, "D must be an even number"
        
        self.d = d
        self.D = D
        self.gamma = gamma
        
        # 计算 sigma = 1 / sqrt(2 * gamma)
        # 其中 gamma = 1 / (2 * sigma^2)
        sigma = 1 / math.sqrt(2 * gamma)
        
        # 调用 make_orf_matrix 初始化权重 w
        # 注意: make_orf_matrix 返回 (dim_out, dim_in)，这里需要 (D//2, d)
        w = make_orf_matrix(d, D // 2, std=1 / sigma, device=device)
        
        # 注册为 buffer (不参与梯度更新，但会随模型保存/加载)
        self.register_buffer('w', w)
        self.w: torch.Tensor

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            X (torch.Tensor): 输入张量，形状为 (batch_size, d)

        Returns:
            torch.Tensor: 变换后的张量，形状为 (batch_size, D)
        """
        # 计算 projection: Xw = X @ w.T
        # X shape: (batch_size, d)
        # w shape: (D//2, d)
        # Xw shape: (batch_size, D//2)
        Xw = torch.matmul(X, self.w.T)
        
        # 计算 cos 和 sin 特征
        Z_cos = torch.cos(Xw)
        Z_sin = torch.sin(Xw)
        
        # 拼接 cos 和 sin 特征
        # Z shape: (batch_size, D)
        Z = torch.cat([Z_cos, Z_sin], dim=-1)
        
        # 归一化: Z = Z * sqrt(2 / D)
        Z = Z * math.sqrt(2 / self.D)
        
        return Z


def aggregate_global_statistics(client_responses, class_num, stats_level='high'):
    """
    在服务器端聚合客户端上传的统计量，计算全局统计量。

    该方法参考了 SFD/src/flexp/sfd/stat_agg.py 中的 compute_global_stats 函数。
    支持双层级（high-level 和 low-level）的统计量聚合。

    参数:
        client_responses: 客户端响应列表，每个元素是一个字典，包含：
            - 'high': 包含 high-level 统计量的字典（如果 stats_level 为 'high' 或 'both'）
                - 'class_means': 每个类别的特征均值列表
                - 'class_outers': 每个类别的特征外积均值列表
                - 'class_rf_means': 每个类别的随机特征均值列表
            - 'low': 包含 low-level 统计量的字典（如果 stats_level 为 'low' 或 'both'，结构同上）
            - 'sample_per_class': 每个类别的样本数
        class_num: 类别总数
        stats_level: 统计量聚合层级选择，可选值：
                    - 'high': 仅聚合 high-level 统计量（默认）
                    - 'low': 仅聚合 low-level 统计量
                    - 'both': 聚合 high-level 和 low-level 统计量

    返回:
        dict: 包含以下结构的字典
            - 'high': 包含 high-level 全局统计量的字典（如果 stats_level 为 'high' 或 'both'）
                - 'class_means': 全局类别均值列表
                - 'class_covs': 全局类别协方差列表
                - 'class_rf_means': 全局类别随机特征均值列表
            - 'low': 包含 low-level 全局统计量的字典（如果 stats_level 为 'low' 或 'both'，结构同上）
            - 'sample_per_class': 全局每个类别的样本数
    """
    assert len(client_responses) > 0, 'No client stats responses to aggregate.'
    
    # 确定需要聚合的层级
    aggregate_high = (stats_level == 'high' or stats_level == 'both')
    aggregate_low = (stats_level == 'low' or stats_level == 'both')
    
    # 初始化：读取第一个响应以获取 feature_dim 和 rf_dim
    first_response = client_responses[0]
    
    # 获取需要聚合的层级的维度信息
    high_feature_dim = None
    high_rf_dim = None
    low_feature_dim = None
    low_rf_dim = None
    ori_dtype = None
    
    if aggregate_high:
        if 'high' not in first_response:
            raise ValueError("stats_level 包含 'high'，但客户端响应中缺少 'high' 统计量")
        high_feature_dim = first_response['high']['class_means'][0].shape[0]
        high_rf_dim = first_response['high']['class_rf_means'][0].shape[0]
        ori_dtype = first_response['high']['class_means'][0].dtype
    
    if aggregate_low:
        if 'low' not in first_response:
            raise ValueError("stats_level 包含 'low'，但客户端响应中缺少 'low' 统计量")
        low_feature_dim = first_response['low']['class_means'][0].shape[0]
        low_rf_dim = first_response['low']['class_rf_means'][0].shape[0]
        if ori_dtype is None:
            ori_dtype = first_response['low']['class_means'][0].dtype
    
    # 初始化全局样本数
    global_sample_per_class = torch.zeros(class_num)
    
    # 为需要聚合的层级分别初始化用于累加的全局统计量容器（使用 float64 精度）
    levels = []
    if aggregate_high:
        levels.append('high')
    if aggregate_low:
        levels.append('low')
    
    global_stats_accum = {}
    
    for level in levels:
        if level == 'high':
            feature_dim = high_feature_dim
            rf_dim = high_rf_dim
        else:  # level == 'low'
            feature_dim = low_feature_dim
            rf_dim = low_rf_dim
        
        global_stats_accum[level] = {
            'class_means': [torch.zeros(feature_dim, dtype=torch.float64) for _ in range(class_num)],
            'class_outers': [torch.zeros((feature_dim, feature_dim), dtype=torch.float64) for _ in range(class_num)],
            'class_rf_means': [torch.zeros(rf_dim, dtype=torch.float64) for _ in range(class_num)]
        }
    
    # 累加循环 (Aggregation)
    for response in client_responses:
        # 累加样本数
        global_sample_per_class += response['sample_per_class']
        
        # 遍历每个层级
        for level in levels:
            # 遍历每个类别
            for c in range(class_num):
                # 获取该客户端该类别的样本数
                n_k = response['sample_per_class'][c]
                
                if n_k > 0:
                    # 累加统计量（加权累加）
                    global_stats_accum[level]['class_means'][c] += n_k * response[level]['class_means'][c].to(torch.float64)
                    global_stats_accum[level]['class_outers'][c] += n_k * response[level]['class_outers'][c].to(torch.float64)
                    global_stats_accum[level]['class_rf_means'][c] += n_k * response[level]['class_rf_means'][c].to(torch.float64)
    
    # 计算全局统计量 (Normalization & Covariance)
    global_stats = {
        'sample_per_class': global_sample_per_class
    }
    
    # 遍历每个需要聚合的层级
    for level in levels:
        # 初始化结果列表
        means = []
        covs = []
        rf_means = []
        
        # 遍历每个类别
        for c in range(class_num):
            # 获取该类全局总样本数
            N_c = global_sample_per_class[c]
            
            if N_c > 0:
                # 均值: mu_g = global_means[level][c] / N_c
                mu_g = global_stats_accum[level]['class_means'][c] / N_c
                
                # 外积均值: outer_g = global_outers[level][c] / N_c
                outer_g = global_stats_accum[level]['class_outers'][c] / N_c
                
                # RFF 均值: phi_g = global_rf_means[level][c] / N_c
                phi_g = global_stats_accum[level]['class_rf_means'][c] / N_c
                
                # 协方差 (关键): cov_g = outer_g - torch.outer(mu_g, mu_g)
                # 公式: Cov[X] = E[XX^T] - E[X]E[X]^T
                cov_g = outer_g - torch.outer(mu_g, mu_g)
                
                # 转回原始精度（float32）
                means.append(mu_g.to(ori_dtype))
                covs.append(cov_g.to(ori_dtype))
                rf_means.append(phi_g.to(ori_dtype))
            else:
                # 如果 N_c == 0，保持为零张量
                if level == 'high':
                    feature_dim = high_feature_dim
                    rf_dim = high_rf_dim
                else:  # level == 'low'
                    feature_dim = low_feature_dim
                    rf_dim = low_rf_dim
                
                means.append(torch.zeros(feature_dim, dtype=ori_dtype))
                covs.append(torch.zeros((feature_dim, feature_dim), dtype=ori_dtype))
                rf_means.append(torch.zeros(rf_dim, dtype=ori_dtype))
        
        # 将列表存入 global_stats[level] 中
        global_stats[level] = {
            'class_means': means,
            'class_covs': covs,
            'class_rf_means': rf_means
        }
    
    return global_stats

