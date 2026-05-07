"""
损失函数模块
—— 包含内容损失、风格损失（Gram 矩阵）和总变差损失

=== 核心修复（与 Gatys et al. 2015 论文对齐）===
v1.0 原始实现：Gram 矩阵 = F·Fᵀ / (C·H·W)，然后取 MSE loss
    问题：各层风格损失的归一化系数不一致，浅层被过度放大，深层被严重压制

v2.0 论文公式：E_l = (1 / (4 · N_l² · M_l²)) · Σ(G_raw - A_raw)²
    其中 N_l = 通道数 C, M_l = 空间尺寸 H·W, G_raw = F·Fᵀ (未归一化)
    修复：将 MSE loss 除以 (4 · M²)，使各层风格损失归一化与论文一致
    新增：等权分层（w_l = 1/5），每层对总风格损失贡献相同
"""
import torch
import torch.nn as nn


# ==================== 内容损失 ====================

class ContentLoss(nn.Module):
    """
    内容损失 L_content

    计算生成图像与内容图像在 VGG 某一层特征图上的均方误差（MSE）。
    它强制生成图像保留内容图像的高层语义结构。

    数学公式:
        L_content = MSE(F_gen, F_content) = mean((F_gen - F_content)²)

    其中:
        F_gen     —— 生成图像在指定层的特征图
        F_content —— 内容图像在同一层的特征图
    """

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)


# ==================== Gram 矩阵 ====================

def gram_matrix(features):
    """
    计算特征图的 Gram 矩阵 G（未归一化）

    Gram 矩阵描述特征图各通道之间的相关性，捕捉图像的纹理和风格信息。

    数学推导:
        设 F ∈ R^{C × H × W} 为某层特征图，将其展平为 F ∈ R^{C × (H·W)}
        Gram 矩阵 G_raw = F · Fᵀ ∈ R^{C × C}

    参数:
        features: 特征图张量，形状 (N, C, H, W)
    返回:
        gram: 未归一化的 Gram 矩阵，形状 (N, C, C)
    """
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2))


# ==================== 风格损失（v2.0 — 论文公式） ====================

class StyleLoss(nn.Module):
    """
    风格损失 L_style（v2.0 论文对齐版）

    使用 Gatys et al. (2015) 原始论文中的风格损失归一化公式：
        E_l = (1 / (4 · N_l² · M_l²)) · Σ_{i,j} (G_{ij}^l - A_{ij}^l)²

    其中:
        N_l = C    (通道数)
        M_l = H·W  (空间尺寸)
        G^l = F·Fᵀ (未归一化的 Gram 矩阵)

    与 v1.0 的关键区别:
        - v1.0: Gram 预先除以 (C·H·W) 归一化，损失系数与 C 强相关
        - v2.0: 使用论文的 1/(4·N²·M²) 归一化，各层贡献由外部 w_l 等权控制
    """

    def __init__(self, target_features):
        """
        参数:
            target_features: 风格图像的特征图，用于计算目标 Gram 矩阵
        """
        super(StyleLoss, self).__init__()
        b, c, h, w = target_features.size()
        target_features_flat = target_features.view(b, c, h * w).detach()
        self.target_gram = torch.bmm(
            target_features_flat,
            target_features_flat.transpose(1, 2)
        )
        self.c = c
        self.m = h * w

    def forward(self, input_features):
        b, c, h, w = input_features.size()
        input_features_flat = input_features.view(b, c, h * w)
        input_gram = torch.bmm(
            input_features_flat,
            input_features_flat.transpose(1, 2)
        )
        mse = nn.functional.mse_loss(input_gram, self.target_gram)
        return mse / (4.0 * self.m * self.m)


# ==================== 总变差损失（正则化项） ====================

def total_variation_loss(image):
    """
    总变差损失 L_TV

    鼓励生成图像的相邻像素值平滑变化，减少高频噪点和伪影。
    对水平方向和垂直方向的相邻像素差求绝对值的均值。

    数学公式:
        L_TV = mean(|x_{i,j+1} - x_{i,j}|) + mean(|x_{i+1,j} - x_{i,j}|)

    参数:
        image: 图像张量，形状 (N, C, H, W)，值域 [0, 1]
    返回:
        标量损失值
    """
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    loss += torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss
