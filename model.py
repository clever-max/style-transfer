"""
VGG19 特征提取器模块
—— 用于神经风格迁移中提取内容特征和风格特征

=== v2.0 改进 ===
1. MaxPool2d → AvgPool2d：减少棋盘格伪影，产生更平滑的特征图
2. content_layer conv4_2 → relu4_2：与 Caffe VGG19 中的 'conv4_2' 语义对齐
   （Caffe 中 conv 层名称包含 ReLU 输出，PyTorch 中 conv 与 relu 分开命名）
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

# 内容层（v2.0 修正）：使用 relu4_2 的输出作为内容表示
#   在 Caffe 的命名约定中，'conv4_2' 指的是 conv+relu 之后的输出
#   因此 PyTorch 中应使用 'relu4_2' 而非 'conv4_2'
content_layer = ['relu4_2']

# 风格层：使用 5 个不同深度的卷积层提取多尺度风格纹理
#   —— 浅层捕获简单纹理，深层捕获复杂结构
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


class VGG19FeatureExtractor(nn.Module):
    """
    VGG19 特征提取器（v2.0）

    加载预训练的 VGG19 模型，冻结参数后作为固定特征提取器。
    在风格迁移中，只用于前向传播提取特征，不参与梯度更新。

    v2.0 改进：将 MaxPool2d 替换为 AvgPool2d
        - MaxPool 取最大值，产生锐利边缘和棋盘格伪影
        - AvgPool 取平均值，特征图更平滑，风格迁移效果更自然
        - 注意：预训练权重是在 MaxPool 下训练的，替换后会有轻微分布偏移，
          但风格迁移中对颜色/纹理的感知鲁棒性使此影响可忽略
    """

    def __init__(self, pretrained=True):
        super(VGG19FeatureExtractor, self).__init__()

        try:
            vgg19 = models.vgg19(
                weights='IMAGENET1K_V1' if pretrained else None
            ).features
        except Exception:
            vgg19 = models.vgg19(pretrained=pretrained).features

        self.model = nn.Sequential()
        for i, layer in enumerate(vgg19):
            name = vgg19_layers[i]
            if isinstance(layer, nn.MaxPool2d):
                layer = nn.AvgPool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                )
            self.model.add_module(name, layer)

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
