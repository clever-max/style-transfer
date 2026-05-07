"""
VGG19 特征提取器模块
—— 用于神经风格迁移中提取内容特征和风格特征
"""
import torch.nn as nn
from torchvision import models

# VGG19 各层名称，按顺序排列（共 35 个子层）
vgg19_layers = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
    'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
    'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
    'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
]

# 内容层：使用 conv4_2 的特征图作为内容表示
#   —— 较深层保留语义信息，舍弃像素级细节
content_layer = ['conv4_2']

# 风格层：使用 5 个不同深度的卷积层提取多尺度风格纹理
#   —— 浅层捕获简单纹理，深层捕获复杂结构
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


class VGG19FeatureExtractor(nn.Module):
    """
    VGG19 特征提取器

    加载预训练的 VGG19 模型，冻结参数后作为固定特征提取器。
    在风格迁移中，只用于前向传播提取特征，不参与梯度更新。
    """

    def __init__(self, pretrained=True):
        """
        初始化特征提取器

        参数:
            pretrained: 是否加载 ImageNet 预训练权重（默认 True）
        """
        super(VGG19FeatureExtractor, self).__init__()

        # 加载 VGG19 的卷积层部分（不含全连接层）
        try:
            vgg19 = models.vgg19(
                weights='IMAGENET1K_V1' if pretrained else None
            ).features
        except Exception:
            vgg19 = models.vgg19(pretrained=pretrained).features

        # 将 VGG19 的各层按名称注册到 Sequential 中，便于按名称提取中间层特征
        self.model = nn.Sequential()
        for i, layer in enumerate(vgg19):
            name = vgg19_layers[i]
            self.model.add_module(name, layer)

        # 冻结所有参数 —— 风格迁移中 VGG19 只做特征提取，不更新权重
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, out_keys=None):
        """
        前向传播，返回指定层的特征图

        参数:
            x: 输入图像张量，形状 (N, 3, H, W)，需已做 ImageNet 标准化
            out_keys: 需要提取的层名称列表，默认取 content_layer + style_layers
        返回:
            features: 字典，键为层名，值为对应该层的特征图张量
        """
        if out_keys is None:
            out_keys = content_layer + style_layers
        features = {}
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in out_keys:
                features[name] = x
        return features
