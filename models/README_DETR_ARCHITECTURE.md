# Cross-View Localization - DETR Architecture

## 架构

```
Front-View + Satellite + Prompts
       │
       ├─> VGGT Backbone ─> F_front, F_sat [B, 1369, 2048]
       │
       ├─> Prompt Encoder ─> Sparse [B, N, 2048], Dense [B, 2048, 37, 37]
       │
       ├─> Prompt Fusion (TwoWayTransformer) ─> target_guidance [B, 2048]
       │
       └─> Unified DETR Decoder
             │
             ├─> Object Queries (100) ─> BBox Head ─> [B, 100, 4]
             │
             └─> Location Queries (32×32) ─> Heatmap ─> [B, 518, 518]
                                          └─> Position [B, 2]
       │
       └─> Camera Head ─> yaw_radians [B]
```

## 核心组件

| 组件 | 文件 | 输出 |
|-----|------|-----|
| VGGT Backbone | `vggt_aggregator.py` | `[B, 1369, 2048]` 特征 |
| Prompt Encoder | `encoder/prompt_encoder.py` | Sparse + Dense embeddings |
| Prompt Fusion | `prompt_fusion.py` | target_guidance `[B, 2048]` |
| DETR Decoder | `decoder/detr.py` | 统一处理两种 queries |
| Camera Head | `heads/yaw_head.py` | yaw 角度 |

## 使用示例

```python
from models import CrossViewLocalizerDETR
from utils import DETRCriterion

model = CrossViewLocalizerDETR(
    num_object_queries=100,
    location_grid_size=32,
)

outputs = model(front_view, satellite_view, points=(coords, labels))

# 输出
outputs['pred_boxes']    # [B, 100, 4]
outputs['heatmap']       # [B, 518, 518]
outputs['position']      # [B, 2]
outputs['yaw_radians']   # [B]
```

## 损失函数

```python
criterion = DETRCriterion(
    weight_bbox=5.0,
    weight_giou=2.0,
    weight_heatmap=1.0,
    weight_yaw=1.0,
)

losses = criterion(outputs, {
    'sat_bbox': target_bbox,        # [B, 4]
    'camera_position': target_pos,  # [B, 2]
    'yaw_radians': target_yaw,      # [B]
})
```

## 文件结构

```
models/
├── cross_view_localizer_detr.py  # 主模型
├── vggt_aggregator.py            # VGGT backbone
├── encoder/                      # PE, TwoWayTransformer, PromptEncoder
├── decoder/                      # TransformerDecoder, MLP
├── prompt_fusion.py              # PromptFusionWithDense
└── heads/yaw_head.py             # CameraHead
```
