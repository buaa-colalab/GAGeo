# Cross-View Localizer

跨视角目标定位：前视图 → 卫星图多任务预测

## 文件结构

```
models/
├── cross_view_localizer_v2.py  # 主模型
├── prompt_encoder.py           # 几何提示编码器
├── vggt_aggregator.py          # VGGT backbone (含DINOv2)
├── heads/
│   ├── bbox_head.py            # BBox检测
│   ├── mask_head.py            # Mask预测 (DPT-style)
│   ├── yaw_head.py             # CameraHead (pose预测)
│   ├── position_head.py        # 位置预测
│   └── multi_task_head.py      # 多任务整合
├── utils/
│   └── weight_loader.py        # 权重加载工具
└── layers/                     # VGGT层实现 (含DINOv2 ViT)
```

## Backbone架构

VGGT Aggregator内部已包含DINOv2：

```
Image [B, 2, 3, 518, 518]
         │
         ▼
┌─────────────────────────────────────┐
│  self.patch_embed (DINOv2 ViT-L)    │  ← 完整DINOv2, 24层ViT
│  - patch embedding + position embed │
│  - 24层 self-attention blocks       │
│  输出: [B*2, 1369, 1024]            │
└─────────────────────────────────────┘
         │
         ▼
    + camera_token [1, 1024]
    + register_tokens [4, 1024]
         │
         ▼
┌─────────────────────────────────────┐
│  VGGT Alternating Attention (24层)  │
│  for i in range(24):                │
│    frame_blocks[i]  → 单视角内attn  │
│    global_blocks[i] → 跨视角attn    │
│  输出: [B, 2, 1374, 2048]           │
└─────────────────────────────────────┘
```

**关键点**：`patch_embed="dinov2_vitl14_reg"` 时，`self.patch_embed` 是完整的DINOv2 ViT-Large。

## 数据流

```
Front View [B,3,518,518] ──┐
                          ├──> VGGT Aggregator
Satellite View [B,3,518,518]┘
                               │
                               ▼
                    features [B, 2, P, 2048]
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   front_features         sat_features         camera_tokens
   [B, 1369, 2048]        [B, 1369, 2048]      [B, 2048] x 2
        │                      │                      │
        ▼                      │                      │
  PromptEncoder                │                      │
  (points/boxes/masks)         │                      │
        │                      │                      │
        ▼                      ▼                      ▼
   ┌─────────────────────────────────────────────────────┐
   │                    MultiTaskHead                     │
   ├──────────┬──────────┬──────────────┬────────────────┤
   │ BBoxHead │ MaskHead │  CameraHead  │  PositionHead  │
   │ [B,N,4]  │[B,1,H,W] │ yaw [B]      │ (x,y) [B,2]    │
   └──────────┴──────────┴──────────────┴────────────────┘
```

## 使用方法

### 基本使用

```python
from models import CrossViewLocalizer
from models.utils import load_vggt_weights, freeze_backbone, get_param_groups

# 创建模型
model = CrossViewLocalizer(
    enable_bbox=True,
    enable_seg=True,        # mask预测
    enable_camera=True,     # yaw预测
    enable_position=True,   # 位置预测
)

# 加载VGGT预训练权重
load_vggt_weights(model, "vggt.pth", load_heads=False)

# 冻结DINOv2 backbone
freeze_backbone(model, freeze_patch_embed=True)

# 不同学习率
param_groups = get_param_groups(model, lr_backbone=1e-5, lr_heads=1e-4)
optimizer = torch.optim.AdamW(param_groups)
```

### 前向传播

```python
# 点提示
outputs = model(
    front_view,      # [B, 3, 518, 518]
    satellite_view,  # [B, 3, 518, 518]
    points=(coords, labels),  # coords: [B, N, 2], labels: [B, N]
)

# 输出
outputs['pred_boxes']          # [B, N, 4] bbox
outputs['masks']               # [B, 1, 518, 518] mask
outputs['yaw_radians']         # [B] 相机yaw角
outputs['position']            # [B, 2] 相机在卫星图位置
outputs['position_confidence'] # [B] 置信度
```

## 权重加载

```python
from models.utils import load_dinov2_weights, load_vggt_weights

# 方式1: 从DINOv2开始 (只有patch_embed)
load_dinov2_weights(model, dinov2_model_name="dinov2_vitl14_reg")

# 方式2: 从VGGT开始 (推荐，包含跨视角attention)
load_vggt_weights(model, "vggt.pth", load_heads=False)
```

## 输入格式

| 输入 | Shape | 说明 |
|------|-------|------|
| front_view | `[B, 3, 518, 518]` | RGB, 范围[0,1] |
| satellite_view | `[B, 3, 518, 518]` | RGB, 范围[0,1] |
| points | `([B,N,2], [B,N])` | 像素坐标 + 标签(1正/-1负) |
| boxes | `[B, N, 4]` | (x1, y1, x2, y2) 像素坐标 |
| masks | `[B, 1, H, W]` | 二值mask |

## Head详解

### BBoxHead
**文件**: `heads/bbox_head.py`

DETR风格的检测头，使用prompt embeddings作为query，cross-attend到卫星图特征。

```
prompt_embeddings [B, N, 2048]  ──┐
                                 ├──> Cross-Attention (6层) ──> bbox [B, N, 4] + score [B, N]
sat_features [B, 1369, 2048]  ───┘
```

- **输入**: prompt embeddings (来自PromptEncoder) + 卫星图patch特征
- **输出**: `(cx, cy, w, h)` 归一化bbox + 置信度分数

### MaskHead
**文件**: `heads/mask_head.py`

DPT风格的dense预测头，融合VGGT多层特征生成高分辨率mask。

```
VGGT多层特征 [layer 5, 11, 17, 23]
         │
         ▼
    多尺度特征融合 (DPT-style)
         │
         ▼
    上采样 + prompt条件调制
         │
         ▼
    mask [B, 1, 518, 518]
```

- **输入**: VGGT中间层特征 + prompt embeddings
- **输出**: 像素级mask预测

### CameraHead
**文件**: `heads/yaw_head.py`

参考VGGT的CameraHead，使用跨视角camera token融合 + 迭代细化预测相机pose。

```
front_camera_token [B, 2048] ──┐
                               ├──> Cross-Attention ──> 迭代细化(4次) ──> pose [B, 9]
sat_camera_token [B, 2048]  ───┘
```

- **输入**: 前视图和卫星图的camera token (VGGT输出的index 0)
- **输出**: 9维pose (translation[3] + quaternion[4] + fov[2])，从quaternion提取yaw
- **训练**: 只对yaw做监督

### PositionHead
**文件**: `heads/position_head.py`

预测相机在卫星图中的(x, y)位置，支持regression和heatmap两种模式。

```
front_features [B, 1369, 2048] ──┐
                                 ├──> Cross-Attention ──> position [B, 2] + confidence [B]
sat_features [B, 1369, 2048]  ───┘
```

- **输入**: 前视图和卫星图的patch特征
- **输出**: 归一化(x, y)坐标 [0,1] + 置信度
- **模式**: `regression` (直接回归) 或 `heatmap` (概率热力图 + soft-argmax)
