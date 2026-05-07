"""
逐步对比拼合图生成工具
—— 将内容图、每N步中间结果、最终结果、风格图横向拼接为一张长图
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试配置中文字体，找不到则使用英文标签
_CHINESE_FONT = None
for font_name in ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
    for f in fm.fontManager.ttflist:
        if font_name.lower() in f.name.lower():
            _CHINESE_FONT = f
            break
    if _CHINESE_FONT:
        break

_LABELS_CN = {
    "content": "内容图",
    "final": "最终结果",
    "style": "风格图",
}
_LABELS_EN = {
    "content": "Content",
    "final": "Final Output",
    "style": "Style",
}

_USE_CN = _CHINESE_FONT is not None
_LABELS = _LABELS_CN if _USE_CN else _LABELS_EN

if _USE_CN:
    matplotlib.rcParams['font.sans-serif'] = [_CHINESE_FONT.name, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


def generate_step_comparison(experiment_dir, step_files, save_path=None):
    """
    生成训练过程的逐步对比拼合图

    排列: content | step0 | step50 | step100 | ... | final | style

    参数:
        experiment_dir: 实验文件夹路径
        step_files:     有序的中间步骤文件名列表（不含路径）
        save_path:      保存路径，默认为 experiment_dir/step_comparison.png
    """
    content_path = os.path.join(experiment_dir, "content_original.png")
    style_path = os.path.join(experiment_dir, "style_original.png")
    final_path = os.path.join(experiment_dir, "final.png")
    intermediate_dir = os.path.join(experiment_dir, "intermediate")

    if save_path is None:
        save_path = os.path.join(experiment_dir, "step_comparison.png")

    image_items = []

    if os.path.exists(content_path):
        image_items.append((_LABELS["content"], content_path))

    for fname in sorted(step_files):
        fpath = os.path.join(intermediate_dir, fname)
        step_num = fname.replace("step_", "").replace(".png", "")
        if os.path.exists(fpath):
            image_items.append((f"Step {int(step_num)}", fpath))

    if os.path.exists(final_path):
        image_items.append((_LABELS["final"], final_path))

    if os.path.exists(style_path):
        image_items.append((_LABELS["style"], style_path))

    n = len(image_items)
    if n == 0:
        return

    max_cols = min(n, 8)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(3.5 * cols, 3.2 * rows),
                             squeeze=False)

    for idx, (title, fpath) in enumerate(image_items):
        r = idx // cols
        c = idx % cols
        img = plt.imread(fpath)
        axes[r][c].imshow(img)
        axes[r][c].set_title(title, fontsize=10)
        axes[r][c].axis('off')

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
