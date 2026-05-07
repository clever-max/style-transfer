"""
图像预处理、后处理工具模块
"""
import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 内容图最长边的上限（像素），风格图会缩放至与内容图相同尺寸
MAX_SIZE = 512

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


def get_image_size(image_path):
    """
    快速获取图像原始宽高（不加载为 tensor）

    返回:
        (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def load_image(image_path, target_size=None):
    """
    从文件路径加载图像，转为 tensor

    - 若 target_size=None: 保持原图宽高比，将最长边缩放到 MAX_SIZE
    - 若 target_size=(H, W): 缩放到指定尺寸（用于风格图匹配内容图尺寸）

    参数:
        image_path:   图像文件路径
        target_size:  可选的目标尺寸 (H, W) 元组
    返回:
        tensor: 形状为 (1, 3, H, W) 的 PyTorch 张量，值域 [0, 1]
    """
    image = Image.open(image_path).convert('RGB')

    if target_size is not None:
        h, w = target_size
    else:
        orig_w, orig_h = image.size
        if orig_w >= orig_h:
            w = MAX_SIZE
            h = int(round(orig_h * MAX_SIZE / orig_w))
        else:
            h = MAX_SIZE
            w = int(round(orig_w * MAX_SIZE / orig_h))

    image = T.Resize((h, w))(image)
    image = T.ToTensor()(image)
    return image.unsqueeze(0).to(device, torch.float)


def normalize(tensor):
    """
    对图像 tensor 做 ImageNet 标准化，使其可输入 VGG19

    参数:
        tensor: 值域 [0, 1] 的 RGB 图像张量
    返回:
        tensor: 标准化后的张量，值域约 [-2, 2]
    """
    return (tensor - imagenet_mean) / imagenet_std


def tensor_to_image(tensor):
    """
    将 PyTorch 张量转换为 PIL Image，便于显示和保存

    参数:
        tensor: (1, 3, H, W) 张量，值域 [0, 1]
    返回:
        PIL.Image: RGB 图像
    """
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = T.ToPILImage()(image)
    return image


def save_image(tensor, save_path):
    """
    将张量保存为 PNG 图像文件

    参数:
        tensor: (1, 3, H, W) 图像张量
        save_path: 保存路径
    """
    image = tensor_to_image(tensor)
    image.save(save_path)


def show_images(content, style, output, save_path=None):
    """
    生成内容图、风格图、生成图的横向对比图

    参数:
        content: 内容图像张量
        style:   风格图像张量
        output:  生成图像张量
        save_path: 可选，保存路径
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_image(content))
    plt.title("Content Image", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_image(style))
    plt.title("Style Image", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_image(output))
    plt.title("Generated Image", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curve(losses, save_path=None):
    """
    绘制并保存训练损失收敛曲线

    参数:
        losses: 损失值列表
        save_path: 可选，保存路径
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Total Loss', color='blue', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Style Transfer Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
