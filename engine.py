"""
训练引擎模块
—— 提供可回调的风格迁移训练函数，每次训练自动创建独立文件夹

=== v2.0 参数调整 ===
1. style_weight: 1e6 → 1e2  （配合修正后的风格损失公式，使 α/β ≈ 0.002 接近论文推荐值）
2. tv_weight:   1e-1 → 1e-3 （降低总变差正则化强度，保留更多高频纹理）
3. 新增等权分层：每层风格损失除以 style_layers 数量（5），确保浅层/深层贡献均衡
"""
import os
import time
import json
import shutil
import torch
import torch.optim as optim

from utils import device, load_image, normalize, save_image, tensor_to_image, \
    plot_loss_curve
from model import VGG19FeatureExtractor, content_layer, style_layers
from loss import ContentLoss, StyleLoss, total_variation_loss

BASE_EXPERIMENT_DIR = "experiments"


def run_style_transfer(content_path, style_path,
                       content_weight=1e0, style_weight=1e2,
                       tv_weight=1e-3, num_steps=300, save_interval=50,
                       progress_callback=None,
                       max_size=512):
    """
    执行单次神经风格迁移，所有输出存入独立实验文件夹

    文件夹命名规则:
        experiments/YYYY-MM-DD_HH-MM-SS_<风格文件名>/

    文件夹结构:
        config.json           — 训练参数
        content_original.png  — 内容图副本
        style_original.png    — 风格图副本
        final.png             — 最终生成结果
        loss_curve.png        — 损失收敛曲线
        step_comparison.png   — 逐步对比拼合图（稍后生成）
        summary.txt           — 训练耗时、参数、最终损失
        intermediate/         — 每 save_interval 步的中间图像

    参数:
        content_path:       内容图像路径
        style_path:         风格图像路径
        content_weight:     内容损失权重 α
        style_weight:       风格损失权重 β
        tv_weight:          总变差损失权重 γ
        num_steps:          优化迭代步数
        save_interval:      保存中间结果的间隔步数
        progress_callback:  可选的进度回调函数 callback(step, total, losses_dict, tensor)
        max_size:           内容图最长边的最大像素数（保持宽高比）

    返回:
        experiment_dir: 本次实验的文件夹路径
    """
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(BASE_EXPERIMENT_DIR,
                                  f"{timestamp}_{style_name}")
    inter_dir = os.path.join(experiment_dir, "intermediate")
    os.makedirs(inter_dir, exist_ok=True)

    from PIL import Image as PILImage
    PILImage.open(content_path).convert("RGB").save(
        os.path.join(experiment_dir, "content_original.png"), "PNG")
    PILImage.open(style_path).convert("RGB").save(
        os.path.join(experiment_dir, "style_original.png"), "PNG")

    start_time = time.time()

    # 先以保持宽高比的方式加载内容图，获取其实际 (H, W)
    content_img = load_image(content_path, target_size=None)
    _, _, ch, cw = content_img.shape
    content_size = (ch, cw)

    # 风格图缩放到与内容图相同尺寸
    style_img = load_image(style_path, target_size=content_size)

    content_img_norm = normalize(content_img)
    style_img_norm = normalize(style_img)

    cnn = VGG19FeatureExtractor(pretrained=True).to(device)
    content_features = cnn(content_img_norm, out_keys=content_layer)
    style_features = cnn(style_img_norm, out_keys=style_layers)

    target = content_img.clone().requires_grad_(True)

    content_losses = [
        ContentLoss(content_features[cl_name]) for cl_name in content_layer
    ]
    style_losses_list = [
        StyleLoss(style_features[sl_name]) for sl_name in style_layers
    ]

    optimizer = optim.LBFGS([target], lr=1.0, max_iter=1)
    loss_history = []
    step_count = [0]

    while step_count[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            target.data.clamp_(0, 1)
            target_norm = normalize(target)
            gen_features = cnn(target_norm,
                               out_keys=content_layer + style_layers)

            c_loss = sum(cl_loss(gen_features[cl_name])
                         for cl_name, cl_loss in zip(content_layer, content_losses))
            s_loss = sum(sl_loss(gen_features[sl_name])
                         for sl_name, sl_loss in zip(style_layers, style_losses_list))
            s_loss = s_loss / len(style_layers)
            tv_loss = total_variation_loss(target)

            total_loss = (content_weight * c_loss
                          + style_weight * s_loss
                          + tv_weight * tv_loss)
            total_loss.backward()

            if step_count[0] % save_interval == 0 or step_count[0] == num_steps:
                save_image(target.data.clone(),
                           os.path.join(inter_dir,
                                        f"step_{step_count[0]:04d}.png"))

            loss_history.append(total_loss.item())

            if progress_callback:
                progress_callback(
                    step=step_count[0],
                    total=num_steps,
                    c_loss=c_loss.item(),
                    s_loss=s_loss.item(),
                    tv_loss=tv_loss.item(),
                    total_loss=total_loss.item(),
                    image_tensor=target.data.clone()
                )

            step_count[0] += 1
            return total_loss

        optimizer.step(closure)

    target.data.clamp_(0, 1)
    elapsed = time.time() - start_time

    final_path = os.path.join(experiment_dir, "final.png")
    save_image(target.data, final_path)

    loss_curve_path = os.path.join(experiment_dir, "loss_curve.png")
    plot_loss_curve(loss_history, save_path=loss_curve_path)

    config = {
        "content_image": os.path.basename(content_path),
        "style_image": os.path.basename(style_path),
        "content_weight": content_weight,
        "style_weight": style_weight,
        "tv_weight": tv_weight,
        "num_steps": num_steps,
        "save_interval": save_interval,
        "output_size": f"{cw}x{ch}",
        "device": str(device),
    }
    with open(os.path.join(experiment_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    summary_lines = [
        f"训练开始: {timestamp}",
        f"风格图片: {os.path.basename(style_path)}",
        f"内容图片: {os.path.basename(content_path)}",
        f"输出尺寸: {cw} x {ch}",
        f"迭代步数: {num_steps}",
        f"训练耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)",
        f"设备: {device}",
        f"初始损失: {loss_history[0]:.4e}",
        f"最终损失: {loss_history[-1]:.4e}",
        f"收敛比: {loss_history[0]/loss_history[-1]:.1f}x",
    ]
    with open(os.path.join(experiment_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return experiment_dir
