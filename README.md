# Cross-View Localization with Pi3 Backbone

基于 Pi3 (DINOv2 + RoPE) 的双向跨视角定位系统，支持多模态 Prompt（Point/BBox/Mask）的无人机定位任务。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 目录

- [模型架构](#模型架构)
- [核心特性](#核心特性)
- [数据处理](#数据处理)
- [快速开始](#快速开始)
- [训练与测试](#训练与测试)
- [项目结构](#项目结构)
- [性能指标](#性能指标)

---

## 🏗️ 模型架构

### 整体 Pipeline

```
输入: Mono View (518×518) + Sat View (518×518) + Prompts (Point/BBox/Mask)
  ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Pi3 Backbone (DINOv2 + Alternative Attention)      │
│  - DINOv2 Encoder: 提取 patch features                      │
│  - Pi3 Decoder: 跨视角特征融合 (RoPE 位置编码)              │
│  输出: F_mono [B, 1369, 2048], F_sat [B, 1369, 2048]       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: SAM Prompt Encoder                                 │
│  - Point: 位置编码 + Type Embedding                         │
│  - BBox: 两个角点的位置编码 + Type Embedding                 │
│  - Mask: CNN 下采样到 37×37                                 │
│  输出: Sparse [B, N, 2048], Dense [B, 2048, 37, 37]        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Prompt Fusion (SAM Two-Way Transformer)           │
│  - Dense: [B, 2048, 37, 37] → [B, 1369, 2048] → 直接相加      │
│  - Sparse: Two-Way Attention + Attention Pooling (N→1)      │
│  输出: Fused Features [B, 1369, 2048] + Target Guidance [B, 2048] │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Unified Query Decoder (DETR-style)                │
│  - Object Queries: 在目标视图上定位 bbox                     │
│  - Location Queries: 在卫星图上定位 camera position         │
│  - Cross-Attention: Queries attend to fused features        │
│  输出: Object Embeddings [B, 10, 2048]                      │
│        Location Embeddings [B, 16, 2048]                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Task Heads                                         │
│  - BBox Head: MLP → [cx, cy, w, h] 归一化坐标               │
│  - Heatmap Head: Dot Product → [H, W] 概率分布              │
│  - Camera Head: MLP → yaw angle (radians)                   │
└─────────────────────────────────────────────────────────────┘
  ↓
输出: pred_boxes, heatmap, yaw_radians
```

### 关键组件说明

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **Pi3 Backbone** | 两个视图图像 | 跨视角特征 | DINOv2 提取特征 + Pi3 融合 |
| **Prompt Encoder** | Point/BBox/Mask | Sparse + Dense Embeddings | SAM 风格的 prompt 编码 |
| **Prompt Fusion** | Features + Prompts | Fused Features | Two-Way Transformer 融合 |
| **Query Decoder** | Queries + Features | Object/Location Embeddings | DETR 风格的解码器 |
| **Task Heads** | Embeddings | Predictions | 多任务预测头 |

---

## ✨ 核心特性

### 1. 双向定位

支持三种定位方向：

- **`mono_to_sat`**: 在街景图上给 prompt，在卫星图上定位目标
  - 用例："这个建筑在卫星图的哪里？"
  
- **`sat_to_mono`**: 在卫星图上给 prompt，在街景图上定位目标
  - 用例："卫星图上的这个区域在街景图中是什么样子？"
  
- **`both`**: 训练时随机选择方向
  - 增强模型的双向定位能力

### 2. 多模态 Prompt 支持

支持任意组合的 prompt 输入：

| Prompt 类型 | 格式 | 用途 |
|------------|------|------|
| **Point** | `[x, y]` 像素坐标 | 指定目标中心点 |
| **BBox** | `[x, y, w, h]` 像素坐标 | 指定目标边界框 |
| **Mask** | `[1, H, W]` 二值 mask | 指定目标精确轮廓 |

训练时随机组合：
- Point only
- BBox only
- Mask only
- Point + BBox
- Point + Mask
- BBox + Mask
- Point + BBox + Mask

### 3. Pi3 跨视角特征提取

- **DINOv2 Encoder**: 强大的视觉特征提取能力
- **RoPE 位置编码**: 相对位置编码，适合跨视角任务
- **Cross-View Attention**: 显式建模两个视图的关联

### 4. DETR 风格的端到端训练

- 无需 NMS 后处理
- 多任务联合优化（BBox + Heatmap + Yaw）
- Hungarian Matching 自动匹配预测和目标

---

## 📊 数据处理

### 数据格式

```json
{
  "city": "London",
  "mono_filename": "front.jpg",
  "sat_filename": "satellite.jpg",
  "mono_point": [256, 200],
  "mono_bbox": [x, y, w, h],
  "mono_segmentation": {"size": [518, 518], "counts": "..."},
  "sate_point": [640, 640],
  "sate_bbox": [x, y, w, h],
  "sate_segmentation": [[x1, y1, x2, y2, ...]],
  "rotation": -45.0,
  "camera_position": [640, 640]
}
```

### 处理流程

| 步骤 | Mono 图 | Sat 图 |
|------|---------|--------|
| **1. 加载** | PIL Image | PIL Image (1280×1280) |
| **2. 变换** | Resize → 518×518 | Random/Center Crop → 518×518 |
| **3. Mask** | RLE → Binary [518, 518] | RLE → Binary → Crop |
| **4. Tensor** | [3, 518, 518], 值域 [0, 1] | [3, 518, 518], 值域 [0, 1] |
| **5. 坐标** | 按比例缩放 | 减去 crop offset |

### 关键设计决策

| 数据类型 | Prompt (输入) | Target (输出) | 原因 |
|---------|--------------|--------------|------|
| **BBox** | `[x, y, w, h]` 像素坐标 | `[cx, cy, w, h]` 归一化 [0,1] | Prompt 需要像素坐标计算位置编码，Target 归一化训练更稳定 |
| **Point** | `[x, y]` 像素坐标 | `[x, y]` 归一化 [0,1] | 同上 |
| **Position** | - | `[x, y]` 归一化 [0,1] | Heatmap 回归任务 |

**详细文档**: 查看 [`data/DATA_PIPELINE.md`](data/DATA_PIPELINE.md) 了解完整的数据处理流程。

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone <repo_url>
cd location

# 安装依赖
pip install -r requirements.txt

# 下载 Pi3 预训练权重
# 放置到 ckpt/pi3_large.pth
```

### 数据准备

```bash
# 数据目录结构
/data/GoogleEarth/
├── London/
│   ├── mono/
│   │   └── *.jpg
│   └── sate/
│       └── *.jpg
├── Moscow/
├── Newyork/
├── Paris/
└── Tokyo/

# 数据标注文件
/data/xhj/location/data/
├── results_filter.json  # 完整数据集 (124,412 条)
└── test_samples.json    # 测试样本 (10 条)
```

**数据集详情**: 查看 [`data/README.md`](data/README.md)

---

## 🎯 训练与测试

### 训练




### 测试

```bash
# 测试模型
python test_detr.py \
    --config configs/default.yaml \
    --checkpoint output/best_model.pth \
    --test_json data/test_samples.json

# 可视化结果
python vis_detr.py \
    --checkpoint output/best_model.pth \
    --test_json data/test_samples.json \
    --output_dir visualizations/
```

### 配置文件

```yaml
# configs/default.yaml
model:
  img_size: 518
  patch_size: 14
  decoder_size: 'large'  # 'small', 'base', 'large'
  num_decoder_layers: 6
  num_object_queries: 10
  num_location_queries: 16
  freeze_backbone: false

data:
  train_json: 'data/results_filter.json'
  val_json: 'data/test_samples.json'
  data_root: '/data/GoogleEarth'
  train_direction: 'both'  # 'mono_to_sat', 'sat_to_mono', 'both'
  val_direction: 'both'
  crop_sat: true
  random_crop: true

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
  warmup_epochs: 5
  
loss:
  weight_bbox: 5.0
  weight_giou: 2.0
  weight_heatmap: 1.0
  weight_yaw: 1.0
```

---

## 📁 项目结构

```
location/
├── configs/              # 配置文件
│   └── default.yaml
├── data/                 # 数据集代码
│   ├── README.md         # 数据集说明
│   ├── DATA_PIPELINE.md  # 数据处理详解
│   ├── dataset.py        # Dataset 实现
│   └── results_filter.json
├── models/               # 模型代码
│   ├── backbone/         # Pi3 Backbone
│   ├── encoder/          # Prompt Encoder & Fusion
│   ├── decoder/          # Query Decoder
│   ├── heads/            # Task Heads
│   └── cross_view_localizer_pi3.py  # 主模型
├── utils/                # 工具函数
│   ├── losses.py         # Loss 函数
│   ├── box_ops.py        # BBox 操作
│   └── prompt_utils.py   # Prompt 处理
├── scripts/              # 训练脚本
│   └── train_detr.sh
├── train_detr.py         # 训练入口
├── test_detr.py          # 测试入口
├── vis_detr.py           # 可视化
└── README.md             # 本文档
```

---

## 📈 性能指标

### 评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| **BBox 定位** | IoU, GIoU | 预测框与真实框的重叠度 |
| **Camera 定位** | Position Error (m) | 预测位置与真实位置的距离 |
| **Yaw 估计** | Angular Error (°) | 预测角度与真实角度的差异 |




---

## 📚 参考文献

- [Pi3: Improved Pose Estimation](https://arxiv.org/abs/xxxx.xxxxx)
- [SAM2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

---

## 📄 License

MIT License

---

## 🙏 致谢

- Pi3 团队提供的预训练模型
- Meta AI 的 SAM2 和 DINOv2
- Facebook Research 的 DETR

---

## 📮 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。