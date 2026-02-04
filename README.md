# Cross-View Localization (DETR)

基于 VGGT + DETR + SAM 的跨视角定位系统。

## 快速开始

```python
from models import CrossViewLocalizerDETR
from utils import DETRCriterion

# 创建模型
model = CrossViewLocalizerDETR(
    img_size=518,
    num_object_queries=100,
    location_grid_size=32,
)

# 前向传播
outputs = model(
    front_view=front_img,      # [B, 3, 518, 518]
    satellite_view=sat_img,    # [B, 3, 518, 518]
    points=(coords, labels),   # ([B, N, 2], [B, N])
)

# 输出
print(outputs['pred_boxes'])   # [B, 100, 4] BBox
print(outputs['heatmap'])      # [B, 518, 518] 位置热力图
print(outputs['yaw_radians'])  # [B] 相机朝向
```

## 训练

```bash
python train_detr.py --config configs/detr.yaml
```

## 目录结构

```
models/
├── cross_view_localizer_detr.py  # 主模型
├── vggt_aggregator.py            # VGGT backbone
├── encoder/                      # Prompt Encoder, PE, Transformer
├── decoder/                      # DETR Decoder
└── heads/                        # CameraHead

utils/
├── losses.py                     # DETRCriterion
└── box_ops.py                    # BBox 操作
```

## 数据格式

```json
{
  "mono_filename": "front.jpg",
  "sat_filename": "satellite.jpg",
  "mono_point": [256, 200],
  "sate_bbox": [0.5, 0.5, 0.1, 0.1],
  "rotation": -45.0,
  "camera_position": [0.5, 0.5]
}
```