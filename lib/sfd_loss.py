#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
SFD 核心损失函数模块

该模块包含从 SFD (Synthetic Feature-Based Decoupled Training) 算法中移植的核心损失函数，
用于支持 FedMPS 项目中的 ABBL (Adaptive Bi-Branch Learning) 功能。

主要包含以下损失函数：
1. make_pi_sample_per_class: 计算带有平滑处理的类别分布 π
2. logit_adjustment_ce: 实现 L_ACE (Adaptive Cross-Entropy) 损失
3. a_scl_loss: 实现 L_A-SCL (Adaptive Supervised Contrastive Loss) 损失
"""

import torch
import torch.nn.functional as F


def make_pi_sample_per_class(real_sample_per_class: torch.Tensor, beta_pi: float) -> torch.Tensor:
    """
    计算带有平滑处理的类别分布 π，用于解决 Non-IID 场景下本地数据类缺失（missing classes）导致梯度消失的问题。
    
    该函数通过将缺失类别的样本数设置为最小类别样本数的 beta_pi 倍，从而避免在计算损失时出现 log(0) 的情况。
    这是 SFD 算法中处理长尾分布和类别不平衡的关键技术。
    
    参数:
        real_sample_per_class: torch.Tensor, shape (num_classes,)
            每个类别的真实样本数量。对于缺失的类别，该值为 0。
        beta_pi: float
            平滑系数，用于设置缺失类别的虚拟样本数。通常取值为 0.1 到 1.0 之间。
    
    返回:
        pi_sample_per_class: torch.Tensor, shape (num_classes,)
            平滑处理后的类别分布，所有类别（包括缺失类别）都有非零的样本数。
    
    在 FedMPS 中的用途:
        用于计算 L_ACE 和 L_A-SCL 损失时，确保即使某些类别在本地数据中缺失，
        也能正确计算损失函数，避免梯度消失问题。
    """
    class_size_min = real_sample_per_class[real_sample_per_class > 0].min()
    pi_sample_per_class = real_sample_per_class.clone()
    pi_sample_per_class[pi_sample_per_class == 0] = class_size_min * beta_pi
    return pi_sample_per_class


def logit_adjustment_ce(
    logit: torch.Tensor,
    target: torch.Tensor,
    sample_per_class: torch.Tensor,
    gamma: float,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    实现 L_ACE (Adaptive Cross-Entropy) 损失函数。
    
    该损失函数通过调整 logits 来适应类别不平衡问题。具体地，对于每个类别 c，
    将 logit 调整为 logit + gamma * log(π_c)，其中 π_c 是该类别的样本比例。
    这样可以增强对尾部类别的学习，提高模型在长尾分布数据上的性能。
    
    **重要提示**：此函数的第一个参数 `logit` 必须是**原始 Logits**（未经过 Softmax/LogSoftmax），
    函数内部会执行 `logit + gamma * log(pi)` 然后再做 CrossEntropy。
    
    参数:
        logit: torch.Tensor, shape (batch_size, num_classes)
            **原始 logits**（未经过 Softmax/LogSoftmax），直接从分类器输出获得。
        target: torch.Tensor, shape (batch_size,)
            真实标签，每个元素是类别索引。
        sample_per_class: torch.Tensor, shape (num_classes,)
            每个类别的样本数（通常使用 make_pi_sample_per_class 处理后的平滑分布 π）。
        gamma: float
            调整系数，控制类别不平衡调整的强度。通常取值为 0.1 到 1.0 之间。
        reduction: str, 可选 {'mean', 'sum', 'none'}, 默认 'mean'
            损失函数的归约方式。
    
    返回:
        loss: torch.Tensor
            计算得到的 L_ACE 损失值。
    
    在 FedMPS 中的用途:
        用于替代传统的 CrossEntropy 损失，在联邦学习的 Non-IID 场景下，
        通过自适应调整 logits 来平衡不同类别的学习，特别适用于长尾分布数据。
    
    数学公式:
        L_ACE = CrossEntropy(logit + gamma * log(π), target)
    """
    # 将 sample_per_class 扩展为与 logit 相同的形状，以便进行广播运算
    sample_per_class = (sample_per_class
        .type_as(logit)
        .unsqueeze(dim=0)
        .expand(logit.shape[0], -1))
    
    # 对 logits 进行自适应调整：logit + gamma * log(π)
    logit = logit + gamma * sample_per_class.log()
    
    # 计算交叉熵损失
    loss = F.cross_entropy(logit, target, reduction=reduction)
    return loss


def a_scl_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    temperature: float,
    sample_per_class: torch.Tensor,
    gamma: float = 1.0
) -> torch.Tensor:
    """
    实现 L_A-SCL (Adaptive Supervised Contrastive Loss) 损失函数。
    
    该损失函数是监督对比学习（Supervised Contrastive Learning）的自适应版本。
    与标准的 SCL 不同，A-SCL 通过基于类别样本数量调整负样本对的相似度矩阵，
    增强对尾部类别的特征学习，从而在长尾分布数据上取得更好的性能。
    
    核心思想：
    1. 对于正样本对（相同类别），使用标准的对比学习损失。
    2. 对于负样本对（不同类别），根据类别样本数量调整相似度：
       adjusted_similarity = similarity + gamma * log(n_i)
       其中 n_i 是样本所属类别的样本数。
    3. 这样可以让尾部类别（样本数少）的负样本对获得更大的惩罚，从而增强特征区分度。
    
    参数:
        z: torch.Tensor, shape (batch_size, feature_dim)
            投影后的特征向量（未归一化），通常来自 projector 的输出。
        y: torch.Tensor, shape (batch_size,)
            样本对应的标签，每个元素是类别索引。
        temperature: float
            温度系数，用于缩放相似度矩阵。通常取值为 0.07 到 0.1 之间。
        sample_per_class: torch.Tensor, shape (num_classes,)
            每个类别的样本数（通常使用 make_pi_sample_per_class 处理后的平滑分布 π）。
        gamma: float, 默认 1.0
            自适应调整系数，控制类别不平衡调整的强度。
    
    返回:
        loss: torch.Tensor
            计算得到的 L_A-SCL 损失值（标量）。
    
    在 FedMPS 中的用途:
        用于对比学习阶段，通过投影特征（projected_features）进行自适应对比学习，
        增强模型对尾部类别的特征学习能力，特别适用于联邦学习中的 Non-IID 和长尾分布场景。
    
    数学公式:
        L_A-SCL = -mean(log(exp(sim_pos) / sum(exp(sim_neg + gamma * log(n_i)))))
        其中 sim_pos 是正样本对的相似度，sim_neg 是负样本对的相似度（已调整）。
    """
    # 1. 归一化特征向量
    z = F.normalize(z, p=2, dim=1)  # shape: (batch_size, feature_dim)
    
    # 2. 计算相似度矩阵 (z_i · z_j / temperature)
    similarity_matrix = (z @ z.T) / temperature  # shape: (batch_size, batch_size)
    
    # 3. 获取正样本对掩码矩阵（相同标签的位置为 True）
    labels_equal = y.unsqueeze(0) == y.unsqueeze(1)  # shape: (batch_size, batch_size)
    mask = labels_equal.float()  # 转换为浮点数类型，用于后续计算
    
    # 4. 计算 logit 调整项
    # 获取每个样本所属类别的样本数
    sample_class_counts = sample_per_class[y]  # shape: (batch_size,)
    log_class_counts = torch.log(sample_class_counts.float())  # 计算 log(n_i)
    # 扩展为矩阵形式，用于广播运算
    logit_adjustment_matrix = gamma * log_class_counts.unsqueeze(0)  # shape: (1, batch_size)
    
    # 5. 生成负样本掩码（不同标签的位置为 1）
    negative_mask = 1 - labels_equal.float()  # shape: (batch_size, batch_size)
    
    # 6. 对负样本对的相似度矩阵应用 logit 调整
    adjusted_similarity_matrix = similarity_matrix + logit_adjustment_matrix * negative_mask
    
    # 7. 数值稳定性处理：减去最大值以防止数值溢出
    logits_max = torch.max(adjusted_similarity_matrix, dim=1, keepdim=True).values
    logits = adjusted_similarity_matrix - logits_max.detach()
    
    # 8. 排除自比较（对角线元素），计算 softmax 分母
    exp_logits = torch.exp(logits) * (1 - torch.eye(z.shape[0], device=z.device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    
    # 9. 计算每个样本的正样本对平均对数概率
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    
    # 10. 返回最终的监督对比学习损失（取负号并求平均）
    loss = -mean_log_prob_pos.mean()
    return loss

