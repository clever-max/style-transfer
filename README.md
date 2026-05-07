# 神经风格迁移 (Neural Style Transfer)

基于 PyTorch + VGG19 的图像风格迁移工具，附带图形化训练界面。

## 快速开始

```bash
# 安装依赖
pip install torch torchvision pillow matplotlib

# 启动图形界面
python gui.py

# 或命令行批量训练（使用 images/ 目录下的预设图片）
python main.py
```

## 功能

- **图形化界面**：选择内容图、多选风格图、调节参数、实时进度显示
- **批量训练**：一次选择多张风格图，自动逐个训练
- **实验管理**：每次训练自动创建独立文件夹（含配置、中间结果、损失曲线、对比图）
- **GPU 加速**：自动检测并使用 NVIDIA GPU
- **保持宽高比**：输出图像保持原内容图比例

## 项目结构

```
├── gui.py              # 图形化训练界面
├── main.py             # 命令行入口
├── engine.py           # 训练引擎
├── model.py            # VGG19 特征提取器
├── loss.py             # 损失函数
├── utils.py            # 图像工具
├── step_comparison.py  # 对比图生成
└── images/             # 测试图片
```

## 参数说明

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| 迭代步数 | 优化次数 | 300 | 越大效果越好，耗时越长 |
| α 内容权重 | 内容保留程度 | 1 | 越大越像原图 |
| β 风格权重 | 风格化程度 | 1e6 | 越大风格越强 |
| γ TV 权重 | 平滑程度 | 0.1 | 越大图像越平滑 |

## 参考论文

- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al., 2015)
