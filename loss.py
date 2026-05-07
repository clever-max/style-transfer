"""
损失函数模块
—— 包含内容损失、风格损失（Gram 矩阵）和总变差损失
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
        L_content = (1 / N) * ||F_gen - F_content||²₂

    其中:
        F_gen     —— 生成图像在指定层的特征图
        F_content —— 内容图像在同一层的特征图
    """

    def __init__(self, target):
        """
        参数:
            target: 内容图像的特征图，会被 detach() 固定，不参与梯度
        """
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return nn.functional.mse_loss(input, self.target)


# ==================== Gram 矩阵 ====================

def gram_matrix(features):
    """
    计算特征图的 Gram 矩阵 G

    Gram 矩阵描述特征图各通道之间的相关性，捕捉图像的纹理和风格信息。
    它丢弃了空间结构，只保留通道间的统计关系。

    数学推导:
        设 F ∈ R^{C × H × W} 为某层特征图，将其展平为 F ∈ R^{C × (H•W)}
        Gram 矩阵 G = (1 / (C • H • W)) * F • Fᵀ

        G_{ij} 表示第 i 个通道和第 j 个通道的特征激活在空间上的相关性。

    参数:
        features: 特征图张量，形状 (N, C, H, W)
    返回:
        gram: Gram 矩阵，形状 (N, C, C)
    """
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


# ==================== 风格损失 ====================

class StyleLoss(nn.Module):
    """
    风格损失 L_style

    计算生成图像与风格图像在 VGG 多个层上 Gram 矩阵的均方误差。
    Gram 矩阵的匹配使得生成图像具有与风格图相似的纹理和色彩分布。

    数学公式:
        L_style = (1 / N) * ||G_gen - G_style||²₂

    其中:
        G_gen   —— 生成图像特征图的 Gram 矩阵
        G_style —— 风格图像特征图的 Gram 矩阵
    """

    def __init__(self, target):
        """
        参数:
            target: 风格图像的特征图（detach 后计算 Gram 矩阵并固定）
        """
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target.detach())

    def forward(self, input):
        g = gram_matrix(input)
        return nn.functional.mse_loss(g, self.target)


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
