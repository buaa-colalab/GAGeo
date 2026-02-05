# Cross-View Drone Localization with Pi3 Backbone

## 概述

基于 **Pi3** (Pose-conditioned Image-to-Image-to-Image) 的跨视角无人机定位系统。该系统使用**单向定位**方式：基于前视图（视野较小），在卫星图上进行定位。通过用户提供的几何提示（点/框/掩码）标注前视图中的目标，系统预测该目标在卫星图上的位置和相机姿态。

**核心特性**：
- 🔄 **Pi3 Backbone**: 使用 DINOv2 encoder + Pi3 decoder，支持跨视角特征提取
- 🎯 **灵活的提示系统**: 支持点、边界框、掩码的任意组合（基于 SAM2），在前视图上标注目标
- 🎨 **统一查询解码器**: DETR-style 解码器处理卫星图上的位置预测
- 📍 **单向定位**: 前视图 → 卫星图定位 + 相机姿态估计
- 🌍 **热力图输出**: 在卫星图上生成位置概率热力图

---

## 架构设计

### 整体流程

```
Step 1: Pi3 Backbone
输入: Mono View, Satellite View
DINOv2 Encoder + Pi3 Decoder (RoPE)

输出:
  F_mono [B, 1369, 2048]
  F_sat  [B, 1369, 2048]
  ↓

Step 2: SAM Prompt Encoder                                 
 Point: 位置编码 + Type Embedding                         
 BBox: 两个角点的位置编码 + Type Embedding                 
 Mask: CNN 下采样到 37×37                                 
 输出: Sparse Prompt Tokens [B, N, 2048], Dense Prompt Tokens [B, 2048, 37, 37]        

 ↓
Step 3: Intent Formation          
 Sparse Prompt Tokens
 Dense Prompt Tokens (flatten → [B, 1369, 2048])
 Intent Queries [Q_intent, 2048]  (learnable)     
  T0 = concat(Sparse Prompt Tokens, Dense Prompt Tokens, Intent Queries)
  T1 = Multi-layer Self-Attention (T0)
  multi-layer cross-attention(query=T1, key/value=F_mono)
  FFN
  输出:
  Z_intent = Intent Queries [B, Q_intent, 2048]

Step 4: View Conditioning  

Inputs: 
  - Z_intent
  - Object Queries: 在卫星视图上定位 bbox    (learnable)                 
  - Location Queries: 在卫星视图上定位 camera position  (learnable)  

T2 = concat(Z_intent, Object Queries, Location Queries)
T3 = Multi-layer Self-Attention (T2)
Z_obj = Cross-Attention(query = T3 ,key/value = F_sat)
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
Camera Head: MLP → yaw angle (radians)                   

输出: pred_boxes, heatmap, yaw_radians

  
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
│   ├── prompt_fusion.py                 # Two-Stage Cross-Attention Fusion
│   │                                    # - IntentFormationModule: 可学习Intent Queries
│   │                                    # - TwoStageCrossAttentionFusion: 双阶段融合
│   ├── pe.py                            # Fourier 位置编码
│   └── position_encoding.py             # Sine 位置编码
│
├── decoder/                             # DETR 解码器
│   ├── __init__.py
│   ├── query_decoder.py                 # 统一查询解码器 + IntentConditioningModule
│   │                                    # - Stage 2: View Conditioning with Intent
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





