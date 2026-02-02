# Cross-View Localization - DETR Architecture

## 架构

支持两种 Backbone：**VGGT** 和 **Pi3**（推荐）

```
Front-View + Satellite + Prompts
       │
       ├─> Backbone (VGGT or Pi3) ─> F_front, F_sat [B, 1369, 2048]
       │
       ├─> Prompt Encoder ─> Sparse [B, N, 2048], Dense [B, 2048, 37, 37]
       │
       ├─> Prompt Fusion (TwoWayTransformer) ─> target_guidance [B, 2048]
       │
       └─> Unified DETR Decoder
             │
             ├─> Object Queries (10) ─> BBox Head ─> [B, 10, 4]
             │
             └─> Location Queries (16) ─> Heatmap ─> [B, 518, 518]
                                        └─> Position [B, 2]
       │
       └─> Camera Head ─> yaw_radians [B]
```

## Backbone 对比

| 特性 | VGGT | Pi3 (推荐) |
|-----|------|-----------|
| 参考帧 | 需要固定第一帧 | 无需固定参考帧 |
| 位置编码 | Sinusoidal | RoPE (旋转位置编码) |
| 参数量 | 1527M | 1376M |
| 文件 | `vggt_aggregator.py` | `pi3_backbone.py` |

## 核心组件

| 组件 | 文件 | 输出 |
|-----|------|-----|
| Pi3 Backbone | `pi3_backbone.py` | `[B, 1369, 2048]` 特征 |
| VGGT Backbone | `vggt_aggregator.py` | `[B, 1369, 2048]` 特征 |
| Prompt Encoder | `encoder/prompt_encoder.py` | Sparse + Dense embeddings |
| Prompt Fusion | `prompt_fusion.py` | target_guidance `[B, 2048]` |
| Query Decoder | `decoder/query_decoder.py` | 统一处理两种 queries |
| BBox Head | `heads/bbox_head.py` | BBox 预测 |
| Heatmap Head | `heads/heatmap_head.py` | 位置热力图 |
| Camera Head | `heads/yaw_head.py` | yaw 角度 |

## 使用示例

### Pi3 版本（推荐）

```python
from models import CrossViewLocalizerPi3

model = CrossViewLocalizerPi3(
    img_size=518,
    decoder_size='large',
    num_object_queries=10,
    num_location_queries=16,
)

outputs = model(front_view, satellite_view, points=(coords, labels))
```

### VGGT 版本

```python
from models import CrossViewLocalizerDETR

model = CrossViewLocalizerDETR(
    img_size=518,
    num_object_queries=10,
    num_location_queries=16,
)

outputs = model(front_view, satellite_view, points=(coords, labels))
```

### 输出

```python
outputs['pred_boxes']    # [B, 10, 4] - BBox (cx, cy, w, h)
outputs['bbox_scores']   # [B, 10] - 置信度
outputs['heatmap']       # [B, 518, 518] - 位置热力图
outputs['position']      # [B, 2] - 相机位置
outputs['yaw_radians']   # [B] - 相机朝向
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


