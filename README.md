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

Step 1: Pi3 Backbone
输入: Mono View, Satellite View
DINOv2 Encoder + Pi3 Decoder (提取patch的3D特征)

输出:
  F_mono [B, 1369, 2048]
  F_sat  [B, 1369, 2048]


Step 2: SAM Prompt Encoder                                 
 Point: 位置编码 + Type Embedding                         
 BBox: 两个角点的位置编码 + Type Embedding                 
 Mask: CNN 下采样到 37×37                                 
 输出: Sparse [B, N, 2048], Dense [B, 2048, 37, 37]        

 ↓
Step 3: Intent Formation          
 Sparse Prompt Tokens
 Dense Prompt Tokens (flatten → [B, 1369, 2048])
 Intent Queries [Q_intent, 2048]  (learnable)     
  T0 = concat(Sparse Prompt Tokens, Dense Prompt Tokens, Intent Queries)
  Multi-layer Self-Attention (T0)
  multi-layer cross-attention(query=Q_intent, key/value=F_mono)
  FFN
  输出:
  Z_intent = Intent Queries [B, Q_intent, 2048]

Step 4: View Conditioning  

Inputs: 
  - Z_intent
  - Object Queries: 在卫星视图上定位 bbox    (learnable)                 
  - Location Queries: 在卫星视图上定位 camera position  (learnable)  

T2 = concat(Z_intent, Object Queries, Location Queries)
Multi-layer Self-Attention (T2)
Z_obj = Cross-Attention(query = Object Queries,Location Queries,key/value = F_sat)
FFN
输出: Object Queries  [B, 50, 2048]                      
Location Queries [B, 50, 2048]    

Step 5: Task Heads  
Inputs: 
  - Z_intent
  - Object Queries: 在卫星视图上定位 bbox                  
  - Location Queries: 在卫星视图上定位 camera position                                         
BBox Head: MLP → [cx, cy, w, h] 归一化坐标               
Heatmap Head: Dot Product → [H, W] 概率分布              
Camera Head: MLP → pose (radians)                   

输出: pred_boxes, heatmap, pose


Step 6: contrative learning 
Inputs: 
  F_mono [B, 1369, 2048]
  F_sat  [B, 1369, 2048]
  mono_mask
  sat_mask                                      

用moco维护队列的方式，经过Average Pooling 和mlp，对齐两者损失

输出: loss
```
ground-sate,drone-sate