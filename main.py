"""
神经风格迁移（Neural Style Transfer）主入口

参考论文: "A Neural Algorithm of Artistic Style" (Gatys et al., 2015)

运行方式:
    python gui.py        — 图形化训练界面（推荐）
    python main.py       — 命令行批量训练
"""
import os
import sys
import torch
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils import device, load_image, normalize, save_image, show_images, plot_loss_curve
from model import VGG19FeatureExtractor, content_layer, style_layers
from loss import ContentLoss, StyleLoss, total_variation_loss
from engine import run_style_transfer
from step_comparison import generate_step_comparison

CONTENT_DIR = "images/content"
STYLE_DIR = "images/style"

EXPERIMENTS = [
    {"content": "content1.png", "style": "style_starry.png", "name": "starry",
     "content_weight": 1e0, "style_weight": 1e6, "tv_weight": 1e-1, "steps": 300},
    {"content": "content1.png", "style": "style_waves.png", "name": "waves",
     "content_weight": 1e0, "style_weight": 1e6, "tv_weight": 1e-1, "steps": 300},
    {"content": "content1.png", "style": "style_mosaic.png", "name": "mosaic",
     "content_weight": 1e0, "style_weight": 1e6, "tv_weight": 1e-1, "steps": 300},
]


def main():
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Total experiments: {len(EXPERIMENTS)}")

    for i, exp in enumerate(EXPERIMENTS, 1):
        content_path = os.path.join(CONTENT_DIR, exp["content"])
        style_path = os.path.join(STYLE_DIR, exp["style"])

        if not os.path.exists(content_path):
            print(f"[{i}/{len(EXPERIMENTS)}] SKIP: 内容图不存在: {content_path}")
            continue
        if not os.path.exists(style_path):
            print(f"[{i}/{len(EXPERIMENTS)}] SKIP: 风格图不存在: {style_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"[{i}/{len(EXPERIMENTS)}] Running: {exp['name']}")
        print(f"{'#'*60}")

        exp_dir = run_style_transfer(
            content_path=content_path,
            style_path=style_path,
            content_weight=exp["content_weight"],
            style_weight=exp["style_weight"],
            tv_weight=exp["tv_weight"],
            num_steps=exp["steps"],
            save_interval=50,
        )

        inter_dir = os.path.join(exp_dir, "intermediate")
        step_files = sorted(os.listdir(inter_dir)) if os.path.exists(inter_dir) else []
        generate_step_comparison(exp_dir, step_files)
        print(f"Experiment folder: {exp_dir}")

    print(f"\n{'#'*60}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
