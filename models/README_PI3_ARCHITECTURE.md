# Cross-View Drone Localization with Pi3 Backbone

## 概述

基于 **Pi3** (Pose-conditioned Image-to-Image-to-Image) 的跨视角无人机定位系统。该系统融合前视图像和卫星图像，通过用户提供的几何提示（点/框/掩码）实现精确的无人机定位和姿态估计。

**核心特性**：
- 🔄 **Pi3 Backbone**: 使用 DINOv2 encoder + Pi3 decoder，支持跨视角特征提取
- 🎯 **灵活的提示系统**: 支持点、边界框、掩码的任意组合（基于 SAM2）
- 🎨 **统一查询解码器**: DETR-style 解码器同时处理目标检测和位置预测
- 📍 **多任务输出**: BBox 检测 + 热力图定位 + 相机姿态估计

---

## 架构设计

### 整体流程

```
输入: Front-view Image + Satellite Image + Prompts (point/bbox/mask)
  ↓
[1] Pi3 Backbone (DINOv2 + Pi3 Decoder)
  → Front Features: [B, 1369, 2048]
  → Satellite Features: [B, 1369, 2048]
  ↓
[2] Prompt Encoder (SAM2-style)
  → Sparse Embeddings: [B, N, 2048]
  → Dense Embeddings: [B, 2048, 37, 37] (if mask provided)
  ↓
[3] Prompt Fusion (Two-Way Transformer)
  → Fused Front Features: [B, 1369, 2048]
  → Target Guidance: [B, N, 2048]
  ↓
[4] Unified Query Decoder (DETR-style)
  → Object Features: [B, 10, 2048]  (for bbox)
  → Location Features: [B, 16, 2048] (for heatmap)
  ↓
[5] Task-Specific Heads
  ├─ BBox Head → [B, 10, 4] + [B, 10, 1]
  ├─ Heatmap Head → [B, 1, 518, 518]
  └─ Camera Head (Pi3-style) → [B, 4, 4] SE(3) pose
```

---

## 模块详解

### 1. Pi3 Backbone (`backbone/pi3_backbone.py`)

**作用**: 提取前视图和卫星图的跨视角特征

**结构**:
```python
Encoder: DINOv2-ViT-L/14 (1024-dim)
  ↓
Decoder: Pi3 Blocks with RoPE (2048-dim)
  - 5 layers for 'large' config
  - Flash Attention with RoPE positional encoding
  - Cross-view attention between front & satellite
```

**输出**:
- `front_patch_features`: [B, 1369, 2048] - 前视图 patch tokens
- `sat_patch_features`: [B, 1369, 2048] - 卫星图 patch tokens
- `front_camera_token`: [B, 2048] - 前视图全局 token
- `sat_camera_token`: [B, 2048] - 卫星图全局 token

**关键特性**:
- 不需要固定参考帧（与 VGGT 的主要区别）
- 所有视角平等对待
- RoPE 位置编码支持任意分辨率

---

### 2. Prompt Encoder (`encoder/prompt_encoder.py`)

**作用**: 编码用户几何提示（点/框/掩码）

**基于**: SAM2 的 PromptEncoder

**支持的提示类型**:
```python
# 点提示
points = (coords, labels)  # coords: [B, N, 2], labels: [B, N] (1=前景, 0=背景)

# 框提示
boxes = [B, M, 4]  # (x, y, w, h) 格式

# 掩码提示
masks = [B, 1, 518, 518]  # 二值掩码
```

**输出**:
- `sparse_embeddings`: [B, N, 2048] - 点和框的嵌入
- `dense_embeddings`: [B, 2048, 37, 37] - 掩码的密集嵌入（仅当提供掩码时）

**关键特性**:
- 支持任意组合的提示
- 位置编码基于 Fourier features
- 可学习的点/框类型嵌入

---

### 3. Prompt Fusion (`prompt_fusion.py`)

**作用**: 将提示嵌入融合到前视图特征中

**基于**: SAM 的 Two-Way Transformer

**融合策略**:
```python
# Sparse prompts (点/框)
TwoWayTransformer:
  - Queries: sparse_embeddings [B, N, 2048]
  - Keys/Values: front_features [B, 1369, 2048]
  - 双向注意力: prompts ↔ image features

# Dense prompts (掩码)
Direct Addition:
  front_features = front_features + dense_embeddings
```

**输出**:
- `fused_sparse`: [B, N, 2048] - 融合后的提示嵌入
- `fused_front`: [B, 1369, 2048] - 融合后的前视图特征
- `target_guidance`: [B, N, 2048] - 目标引导信号（用于 decoder）

---

### 4. Unified Query Decoder (`decoder/query_decoder.py`)

**作用**: DETR-style 解码器，统一处理目标检测和位置预测

**查询设计**:
```python
Object Queries: [B, 10, 2048]  # 用于 bbox 检测
  - Learnable embeddings
  - 用于检测前视图中的目标

Location Queries: [B, 16, 2048]  # 用于热力图预测
  - Learnable embeddings
  - 用于预测卫星图上的位置
```

**Decoder 结构**:
```python
TransformerDecoder (6 layers):
  - Self-Attention: queries 之间的交互
  - Cross-Attention: queries attend to satellite features
  - FFN: 特征变换
  - Target Guidance Injection: 在第一层注入 target_guidance
```

**输出**:
- `obj_features`: [B, 10, 2048] - 用于 bbox head
- `loc_features`: [B, 16, 2048] - 用于 heatmap head

---

### 5. Task-Specific Heads

#### 5.1 BBox Head (`heads/bbox_head.py`)

**作用**: 预测前视图中目标的边界框

**结构**:
```python
Input: obj_features [B, 10, 2048]
  ↓
MLP (3 layers)
  ↓
Output:
  - pred_boxes: [B, 10, 4]  # (cx, cy, w, h) normalized
  - bbox_scores: [B, 10, 1]  # confidence scores
```

#### 5.2 Heatmap Head (`heads/heatmap_head.py`)

**作用**: 预测卫星图上的位置热力图

**结构**:
```python
Input: 
  - loc_features: [B, 16, 2048]
  - sat_features: [B, 1369, 2048]
  ↓
Cross-Attention: loc_features attend to sat_features
  ↓
Upsampling: [B, 16, 2048] → [B, 1, 518, 518]
  ↓
Output:
  - heatmap: [B, 1, 518, 518]  # probability map
  - position: [B, 2]  # (x, y) coordinates
  - heatmap_logits: [B, 1, 518, 518]  # raw logits
```

#### 5.3 Camera Head (`heads/pi3_camera_head.py`)

**作用**: 预测相机姿态（基于 Pi3 的设计）

**结构**:
```python
Input: 
  - front_patch_features: [B, 1369, 2048]
  - sat_patch_features: [B, 1369, 2048]
  ↓
Transformer Decoder (5 layers with RoPE)
  ↓
Camera Head Core:
  - ResConv Blocks (2 layers)
  - Global Average Pooling
  - MLP
  - FC layers for translation (3D) and rotation (9D)
  ↓
SVD Orthogonalization: 9D → SO(3)
  ↓
Output:
  - pose: [B, 4, 4]  # SE(3) transformation matrix
  - yaw_radians: [B]
  - yaw_degrees: [B]
  - quaternion: [B, 4]
```

**关键特性**:
- 使用 patch tokens（不是 camera token）
- RoPE 位置编码
- SVD 正交化保证旋转矩阵的有效性

---

## 文件结构

```
models/
├── __init__.py                          # 导出主要接口
├── cross_view_localizer_pi3.py          # 主模型类
│
├── backbone/                            # Pi3 特征提取
│   ├── __init__.py
│   ├── pi3_backbone.py                  # Pi3 backbone wrapper
│   └── layers/                          # Pi3 内部组件
│       ├── dinov2/                      # DINOv2 encoder
│       └── ...
│
├── encoder/                             # 提示编码和位置编码
│   ├── __init__.py
│   ├── prompt_encoder.py                # SAM2-style 提示编码器
│   ├── transformer.py                   # Two-Way Transformer
│   ├── pe.py                            # Fourier 位置编码
│   └── position_encoding.py             # Sine 位置编码
│
├── prompt_fusion.py                     # SAM-style 提示融合
│
├── decoder/                             # DETR 解码器
│   ├── __init__.py
│   ├── query_decoder.py                 # 统一查询解码器
│   ├── detr.py                          # Transformer decoder
│   └── mlp.py                           # MLP 工具
│
├── heads/                               # 任务头
│   ├── __init__.py
│   ├── bbox_head.py                     # 边界框预测
│   ├── heatmap_head.py                  # 热力图预测
│   └── pi3_camera_head.py               # 相机姿态预测
│
├── layers/                              # 共享组件
│   ├── __init__.py
│   ├── block.py                         # Transformer blocks with RoPE
│   ├── attention.py                     # Flash Attention
│   ├── pos_embed.py                     # RoPE2D 实现
│   ├── transformer_head.py              # Transformer decoder for heads
│   └── camera_head.py                   # Camera head core (ResConv + SVD)
│
└── dinov2/                              # DINOv2 实现
    └── ...
```

---

## 使用示例

### 基本使用

```python
import torch
from models import CrossViewLocalizerPi3

# 创建模型
model = CrossViewLocalizerPi3(
    img_size=518,
    decoder_size='large',
    num_object_queries=10,
    num_location_queries=16,
)

# 准备输入
front_view = torch.randn(2, 3, 518, 518)  # [B, 3, H, W]
satellite_view = torch.randn(2, 3, 518, 518)

# 提示（可选，支持任意组合）
points = (
    torch.tensor([[[100, 200], [300, 400]]]),  # coords [B, N, 2]
    torch.tensor([[1, 0]])  # labels [B, N]: 1=前景, 0=背景
)
boxes = torch.tensor([[[50, 50, 100, 100]]])  # [B, M, 4]
masks = torch.randn(2, 1, 518, 518) > 0  # [B, 1, H, W]

# 前向传播
outputs = model(
    front_view=front_view,
    satellite_view=satellite_view,
    points=points,
    boxes=boxes,
    masks=masks,
)

# 输出
print(outputs['pred_boxes'].shape)      # [2, 10, 4]
print(outputs['bbox_scores'].shape)     # [2, 10, 1]
print(outputs['heatmap'].shape)         # [2, 1, 518, 518]
print(outputs['position'].shape)        # [2, 2]
print(outputs['yaw_degrees'].shape)     # [2]
print(outputs['quaternion'].shape)      # [2, 4]
```



