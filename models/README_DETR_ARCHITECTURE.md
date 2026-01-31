# Cross-View Drone Localization with DETR Architecture

基于 VGGT、DETR 和 SAM 的跨视角无人机定位系统

## 架构概述

```
输入: Front-View Image + Satellite Image + Prompts (Point/BBox/Mask)
  │
  ├─> 1. VGGT Backbone (特征提取)
  │     ├─> Front-View Features: F_f [B, P, 2048]
  │     └─> Satellite Features: F_s [B, P, 2048]
  │
  ├─> 2. SAM Prompt Encoder (提示编码)
  │     ├─> Sparse Embeddings: E_p [B, N, 2048] (points/boxes)
  │     └─> Dense Embeddings: E_d [B, 2048, H, W] (masks)
  │
  ├─> 3. Prompt Fusion (SAM-style Two-Way Transformer)
  │     └─> Target-Aware Features: F_target [B, P, 2048]
  │
  ├─> 4. DETR Decoder (多任务预测)
  │     ├─> Object Queries (N_obj=100) → BBox Head
  │     │     └─> BBox Predictions [B, 100, 4] + Scores [B, 100]
  │     │
  │     └─> Location Queries (32×32 grid) → Heatmap Head
  │           └─> Position Heatmap [B, 518, 518] 
  │
  └─> 5. Camera Head (姿态预测)
        └─> Yaw Angle [B] (from camera tokens)
```

## 核心组件

### 1. VGGT Backbone (`vggt_aggregator.py`)

**功能**: 提取前视图和卫星图的跨视角融合特征

**架构**:
- DINOv2 ViT-L/14 作为 patch embedding
- 24 层交替注意力 (Alternating Attention):
  - Frame Attention: 单视角内部自注意力
  - Global Attention: 跨视角交互注意力
- 输出: `[B, 2, P_total, 2*C]` 其中 P_total = 1374 (1 camera + 4 register + 1369 patches)

**特征提取**:
```python
# 输入
images = torch.stack([satellite_view, front_view], dim=1)  # [B, 2, 3, 518, 518]

# VGGT 处理
vggt_outputs, patch_start_idx = vggt(images)
features = vggt_outputs[-1]  # [B, 2, 1374, 2048]

# 分离特征
sat_features = features[:, 0, patch_start_idx:]     # [B, 1369, 2048]
front_features = features[:, 1, patch_start_idx:]   # [B, 1369, 2048]
sat_camera_token = features[:, 0, 0]                # [B, 2048]
front_camera_token = features[:, 1, 0]              # [B, 2048]
```

### 2. SAM Prompt Encoder (`prompt_encoder.py`)

**功能**: 将用户提示 (点击/框选/涂抹) 编码为嵌入向量

**支持的提示类型**:
- **Points**: 正负点击 `(coords [B, N, 2], labels [B, N])`
- **Boxes**: 矩形框 `[B, M, 4]` in (x1, y1, x2, y2) format
- **Masks**: 二值掩码 `[B, 1, H, W]`

**输出**:
- Sparse Embeddings: `[B, N_sparse, 2048]` - 点和框的嵌入
- Dense Embeddings: `[B, 2048, H', W']` - 掩码的嵌入

### 3. Prompt Fusion Module (`prompt_fusion.py`)

**功能**: 将提示嵌入与前视图特征融合，生成目标感知特征

**方法**: SAM-style Two-Way Transformer
1. Sparse prompts 通过 Cross-Attention 与前视图特征交互
2. Dense prompts (masks) 直接相加到特征图

**实现**:
```python
# SimplePromptFusion (轻量级版本)
fused_sparse, fused_front = prompt_fusion(
    sparse_embeddings=E_p,      # [B, N, 2048]
    dense_embeddings=E_d,       # [B, 2048, H, W]
    front_features=F_f,         # [B, P, 2048]
)
# fused_sparse: [B, N, 2048] - 目标感知的提示嵌入
# fused_front: [B, P, 2048] - F_target (提示引导的前视图特征)
```

### 4. DETR Decoder (`detr_decoder.py`)

**功能**: 标准 DETR Transformer Decoder

**架构**:
- 6 层 Transformer Decoder Layer
- 每层包含:
  1. Self-Attention on queries
  2. Cross-Attention to memory (satellite features)
  3. Feed-Forward Network

**两种查询类型**:

#### 4.1 Object Queries (目标检测)
- 可学习的稀疏查询: `[B, N_obj, 2048]` (N_obj=100)
- 用于预测卫星图中的物体边界框
- 通过 F_target 引导: `Q_final = Q_init + Linear(F_target)`

#### 4.2 Location Queries (位置定位)
- 密集网格查询: `[B, G×G, 2048]` (G=32)
- 每个查询对应卫星图的一个物理位置
- 输出标量分数，形成热力图

### 5. Heatmap Location Head (`heads/heatmap_location_head.py`)

**功能**: 基于查询的相机位置热力图预测

**架构**:
```
Location Queries (32×32) 
  │
  ├─> Add F_target guidance
  │
  ├─> DETR Decoder (attend to satellite features)
  │
  ├─> Linear Head: [2048] → [1] (每个查询输出一个分数)
  │
  ├─> Reshape to [32, 32]
  │
  ├─> Bilinear Upsample to [518, 518]
  │
  └─> Softmax → Probability Heatmap
```

**输出**:
- `heatmap`: `[B, 518, 518]` - 概率分布图
- `position`: `[B, 2]` - 通过 soft-argmax 提取的坐标
- `heatmap_logits`: `[B, 32, 32]` - 原始 logits

### 6. Camera Head (`heads/yaw_head.py`)

**功能**: 预测相机姿态 (主要是 yaw 角)

**方法**: 沿用 VGGT 的 CameraHead
- 使用前视图和卫星图的 camera tokens
- Cross-view fusion + 迭代细化 (4 次迭代)
- 输出 9 维 pose: Translation(3) + Quaternion(4) + FoV(2)
- 从 quaternion 提取 yaw 角度

## 损失函数 (`utils/losses.py`)

### MultiTaskLoss

支持 5 种监督信号:

1. **BBox Loss**: L1 + GIoU
   ```python
   loss_bbox = F.l1_loss(pred, target)
   loss_giou = 1 - diag(generalized_box_iou(pred, target))
   ```

2. **Heatmap Loss**: Penalty-reduced Focal Loss
   ```python
   # 生成高斯热力图作为监督信号
   target_heatmap = generate_gaussian_heatmap(camera_position, size, sigma=2.0)
   
   # Focal Loss with penalty reduction
   pos_loss = -log(pred) * (1 - pred)^α * pos_mask
   neg_loss = -log(1 - pred) * pred^α * (1 - target)^β * neg_mask
   ```
   - α = 2.0 (focusing parameter)
   - β = 4.0 (penalty reduction parameter)

3. **Yaw Loss**: 周期性角度损失
   ```python
   diff = atan2(sin(pred - target), cos(pred - target))
   loss = (diff^2).mean()
   ```

4. **Position Loss** (可选): MSE Loss
   ```python
   loss_position = F.mse_loss(pred_position, target_position)
   ```

5. **Mask Loss** (可选): BCE + Dice Loss

## 使用方法

### 基本使用

```python
from models import CrossViewLocalizerDETR

# 创建模型
model = CrossViewLocalizerDETR(
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    vggt_depth=24,
    num_heads=16,
    num_decoder_layers=6,
    num_object_queries=100,      # 目标查询数量
    location_grid_size=32,       # 位置查询网格大小
    freeze_vggt=False,
    use_prompt_fusion=True,      # 使用 SAM-style 融合
)

# 准备输入
front_view = torch.randn(B, 3, 518, 518)
satellite_view = torch.randn(B, 3, 518, 518)

# 点提示
point_coords = torch.tensor([[[256, 200]]])  # [B, N, 2]
point_labels = torch.tensor([[1]])           # [B, N]
points = (point_coords, point_labels)

# 前向传播
outputs = model(
    front_view=front_view,
    satellite_view=satellite_view,
    points=points,
)

# 输出
print(outputs['pred_boxes'])    # [B, 100, 4] - BBox 预测
print(outputs['bbox_scores'])   # [B, 100] - 置信度
print(outputs['heatmap'])       # [B, 518, 518] - 位置热力图
print(outputs['position'])      # [B, 2] - 相机位置
print(outputs['yaw_radians'])   # [B] - Yaw 角度
```

### 训练

```python
from utils.losses import MultiTaskLoss

# 创建损失函数
criterion = MultiTaskLoss(
    weight_bbox=5.0,
    weight_giou=2.0,
    weight_yaw=1.0,
    weight_heatmap=1.0,
)

# 准备目标
targets = {
    'sat_bbox': torch.rand(B, 4),           # 归一化的 (cx, cy, w, h)
    'camera_position': torch.rand(B, 2),    # 归一化的 (x, y)
    'yaw_radians': torch.rand(B) * 2 * π - π,
}

# 计算损失
losses = criterion(outputs, targets)
total_loss = losses['loss']

# 反向传播
total_loss.backward()
optimizer.step()
```

### 加载预训练权重

```python
from models import build_cross_view_localizer_detr

model = build_cross_view_localizer_detr(
    pretrained_vggt='path/to/vggt.pth',
    freeze_vggt=True,  # 冻结 VGGT backbone
)
```

## 数据格式

### 输入数据 (参考 `data/single.json`)

```json
{
  "city": "Tokyo",
  "mono_filename": "front_view.jpg",
  "mono_point": [511.5, 170.5],           // 前视图中的点
  "mono_bbox": [505, 59, 13, 223],        // [x, y, w, h]
  "sat_filename": "satellite.jpg",
  "sate_bbox": [515.6, 605.1, 75, 72],    // 卫星图中的 bbox
  "rotation": -149.0,                      // Yaw 角度 (度)
  "camera_position": [640.0, 640.0]       // 相机在卫星图中的位置
}
```

### 模型输入

- **Images**: `[B, 3, 518, 518]` 归一化到 [0, 1]
- **Points**: `(coords [B, N, 2], labels [B, N])` 像素坐标 + 标签 (1=正, 0=负)
- **Boxes**: `[B, M, 4]` in (x1, y1, x2, y2) 像素坐标
- **Masks**: `[B, 1, H, W]` 二值掩码

### 模型输出

```python
{
    'pred_boxes': [B, N_obj, 4],      # BBox (cx, cy, w, h) 归一化
    'bbox_scores': [B, N_obj],        # 置信度分数
    'heatmap': [B, H, W],             # 位置概率热力图
    'position': [B, 2],               # 相机位置 (x, y) 归一化
    'heatmap_logits': [B, G, G],      # 原始 logits
    'yaw_radians': [B],               # Yaw 角度 (弧度)
    'yaw_degrees': [B],               # Yaw 角度 (度)
    'quaternion': [B, 4],             # 四元数
    'pose_enc': [B, 9],               # 完整姿态编码
}
```

## 文件结构

```
models/
├── cross_view_localizer_detr.py    # 主模型 (DETR 架构)
├── vggt_aggregator.py              # VGGT backbone
├── prompt_encoder.py               # SAM prompt encoder
├── prompt_fusion.py                # SAM-style fusion module
├── detr_decoder.py                 # DETR transformer decoder
├── heads/
│   ├── bbox_head.py                # BBox 检测头
│   ├── heatmap_location_head.py    # 热力图位置头 (NEW)
│   ├── yaw_head.py                 # Camera/Yaw 头
│   └── ...
└── example_usage_detr.py           # 使用示例

utils/
└── losses.py                       # 损失函数 (含 Heatmap Loss)
```

## 关键改进

相比原始架构 (`cross_view_localizer_v2.py`):

1. **DETR-style Decoder**: 使用标准 DETR transformer decoder，更好的可扩展性
2. **Query-based Heatmap**: 位置预测使用密集查询网格，更精确的空间定位
3. **SAM-style Fusion**: 提示与特征的融合采用 SAM 的双向 transformer
4. **Penalty-reduced Focal Loss**: 专门为热力图设计的损失函数
5. **Target Guidance**: F_target 引导所有下游任务，提升目标感知能力

## 性能优化建议

1. **冻结 VGGT**: 训练初期冻结 VGGT backbone，只训练 heads
2. **学习率策略**: Backbone 用较小学习率 (1e-5)，Heads 用较大学习率 (1e-4)
3. **混合精度训练**: 使用 AMP 加速训练
4. **梯度累积**: 小 batch size 时使用梯度累积
5. **数据增强**: 随机裁剪、颜色抖动、旋转等

## 参考文献

- **VGGT**: Visual Geometry Grounded Deep Structure From Motion
- **DETR**: End-to-End Object Detection with Transformers
- **SAM**: Segment Anything Model
- **DINOv2**: Learning Robust Visual Features without Supervision
