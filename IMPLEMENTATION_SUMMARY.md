# Cross-View Drone Localization System - Implementation Summary

## 项目完成情况

已完成基于 Pi3、DETR 和 SAM 的跨视角无人机定位系统的代码实现。

**定位方式**: **单向定位** - 基于前视图（视野较小），在卫星图上进行定位。

## 实现的核心组件

### 1. DETR-style Transformer Decoder
**文件**: `models/detr_decoder.py`

- ✅ 标准 DETR Transformer Decoder 实现
- ✅ 支持 Object Queries (稀疏查询) 和 Location Queries (密集查询)
- ✅ 6 层 decoder，包含 self-attention 和 cross-attention
- ✅ 可配置的 heads、layers、dropout 等参数

### 2. Query-based Heatmap Location Head
**文件**: `models/heads/heatmap_location_head.py`

- ✅ 基于密集网格查询 (G×G，默认 32×32) 的位置预测
- ✅ 每个查询对应卫星图的一个物理位置
- ✅ 输出标量分数 → 双线性插值上采样到目标分辨率
- ✅ Softmax 生成概率分布热力图
- ✅ Soft-argmax 提取精确坐标
- ✅ 支持 F_target 引导 (前视图目标特征)
- ✅ 额外实现了 HybridLocationHead (结合热力图和直接回归)

### 3. SAM-style Prompt Fusion Module
**文件**: `models/prompt_fusion.py`

- ✅ `TwoWayAttentionBlock`: SAM 的双向注意力机制
  - Self-attention on prompts
  - Cross-attention: prompts → image features
  - MLP on prompts
  - Cross-attention: image features → prompts
- ✅ `PromptFusionModule`: 完整的 SAM-style 融合
- ✅ `SimplePromptFusion`: 轻量级版本 (仅 cross-attention)
- ✅ Sparse prompts (点/框): Cross-Attention 融合
- ✅ Dense prompts (掩码): 直接相加到特征图

### 4. Heatmap Loss Function
**文件**: `utils/losses.py`

- ✅ Penalty-reduced Focal Loss 实现
- ✅ 高斯热力图生成函数 `generate_gaussian_heatmap`
- ✅ 可配置的 α (focusing) 和 β (penalty reduction) 参数
- ✅ 集成到 `MultiTaskLoss` 中
- ✅ 支持与其他损失 (bbox, yaw, position) 联合训练

### 5. 集成主模型
**文件**: `models/cross_view_localizer_detr.py`

完整的端到端模型，整合所有组件：

```
Pipeline (单向定位: 前视图 → 卫星图):
1. Pi3 Backbone → F_f, F_s (前视图和卫星图特征)
2. SAM Prompt Encoder → E_p (sparse), E_d (dense)
   - 在前视图上标注目标（点/框/掩码）
3. Prompt Fusion → F_target (目标感知特征)
   - 将前视图的提示信息融合
4. DETR Decoder:
   - Location Queries → Position heatmap (在卫星图上定位)
5. Camera Head → Camera pose (相机姿态)
```

**特性**:
- ✅ 支持点/框/掩码三种提示类型（在前视图上标注）
- ✅ F_target 引导卫星图定位任务
- ✅ 单向定位输出: Heatmap (卫星图) + Camera Pose
- ✅ 可选的 prompt fusion (可开关)
- ✅ 支持加载预训练 Pi3 权重
- ✅ 灵活的冻结/解冻 backbone

## 文件清单

### 新增文件

1. **`models/detr_decoder.py`** (298 行)
   - TransformerDecoder
   - TransformerDecoderLayer
   - MLP

2. **`models/heads/heatmap_location_head.py`** (287 行)
   - HeatmapLocationHead
   - HybridLocationHead

3. **`models/prompt_fusion.py`** (319 行)
   - TwoWayAttentionBlock
   - PromptFusionModule
   - SimplePromptFusion

4. **`models/cross_view_localizer_detr.py`** (345 行)
   - CrossViewLocalizerDETR (主模型)
   - build_cross_view_localizer_detr

5. **`models/example_usage_detr.py`** (340 行)
   - 5 个完整的使用示例
   - 训练、推理、可视化代码

6. **`models/README_DETR_ARCHITECTURE.md`** (详细文档)
   - 架构说明
   - 组件详解
   - 使用方法
   - 数据格式

7. **`IMPLEMENTATION_SUMMARY.md`** (本文件)

### 修改文件

1. **`utils/losses.py`**
   - ✅ 添加 `_heatmap_loss` 方法
   - ✅ 添加 `generate_gaussian_heatmap` 函数
   - ✅ 添加 `weight_heatmap` 参数
   - ✅ 更新文档字符串

2. **`models/heads/__init__.py`**
   - ✅ 导出 `HeatmapLocationHead` 和 `HybridLocationHead`

3. **`models/__init__.py`**
   - ✅ 导出所有新组件

## 架构对比

### 当前架构 (单向定位)
```
Pi3 Backbone → Prompt Encoder (前视图提示)
            → Prompt Fusion (SAM-style)
            → DETR Decoder:
                - Location Queries → Heatmap (卫星图定位)
            → Camera Head (相机姿态)
```

**主要特性**:
1. ✅ 单向定位: 前视图 → 卫星图
2. ✅ SAM-style prompt fusion (双向 transformer)
3. ✅ DETR-style decoder (标准化架构)
4. ✅ Query-based heatmap (精确的位置预测)
5. ✅ Penalty-reduced focal loss (专门的热力图损失)
6. ✅ F_target guidance (前视图目标引导卫星图定位)

## 使用示例

### 快速开始

```python
from models import CrossViewLocalizerDETR
from utils.losses import MultiTaskLoss

# 创建模型
model = CrossViewLocalizerDETR(
    num_object_queries=100,
    location_grid_size=32,
    use_prompt_fusion=True,
)

# 准备数据
front_view = torch.randn(2, 3, 518, 518)
satellite_view = torch.randn(2, 3, 518, 518)
points = (torch.rand(2, 5, 2) * 518, torch.ones(2, 5))

# 前向传播
outputs = model(front_view, satellite_view, points=points)

# 输出 (单向定位)
outputs['heatmap']       # [2, 518, 518] - 卫星图上的位置热力图
outputs['position']      # [2, 2] - 卫星图上的预测位置
outputs['yaw_radians']   # [2] - 相机偏航角
```

### 训练

```python
# 损失函数
criterion = MultiTaskLoss(
    weight_bbox=5.0,
    weight_giou=2.0,
    weight_yaw=1.0,
    weight_heatmap=1.0,
)

# 目标
targets = {
    'sat_bbox': torch.rand(B, 4),
    'camera_position': torch.rand(B, 2),
    'yaw_radians': torch.rand(B) * 2 * π - π,
}

# 计算损失
losses = criterion(outputs, targets)
losses['loss'].backward()
```

## 数据流详解

### 1. 输入处理
```python
# 输入
front_view: [B, 3, 518, 518]
satellite_view: [B, 3, 518, 518]
points: (coords [B, N, 2], labels [B, N])

# Stack for VGGT
images = stack([satellite_view, front_view], dim=1)  # [B, 2, 3, 518, 518]
```

### 2. VGGT 特征提取
```python
vggt_outputs, patch_start_idx = vggt(images)
features = vggt_outputs[-1]  # [B, 2, 1374, 2048]

# 分离特征
sat_features = features[:, 0, 5:]     # [B, 1369, 2048]
front_features = features[:, 1, 5:]   # [B, 1369, 2048]
sat_camera_token = features[:, 0, 0]  # [B, 2048]
front_camera_token = features[:, 1, 0] # [B, 2048]
```

### 3. Prompt 编码与融合
```python
# 编码
sparse_emb, dense_emb = prompt_encoder(points, boxes, masks)
# sparse_emb: [B, N, 2048]
# dense_emb: [B, 2048, H, W]

# 融合 (SAM-style)
fused_sparse, fused_front = prompt_fusion(
    sparse_emb, dense_emb, front_features
)
# fused_front: [B, 1369, 2048] - F_target

# 池化为引导向量
target_guidance = fused_front.mean(dim=1)  # [B, 2048]
```

### 4. Object Detection
```python
# 初始化查询
obj_queries = object_queries.weight  # [100, 2048]
obj_queries = obj_queries + Linear(target_guidance)  # 添加引导

# DETR decoder
decoder_out = object_decoder(obj_queries, sat_features)  # [B, 100, 2048]

# 预测
pred_boxes = bbox_head(decoder_out).sigmoid()  # [B, 100, 4]
scores = score_head(decoder_out).sigmoid()     # [B, 100]
```

### 5. Position Heatmap
```python
# Location queries (32×32 grid)
loc_queries = location_queries.weight  # [1024, 2048]
loc_queries = loc_queries + Linear(target_guidance)

# Decoder
decoder_out = heatmap_decoder(loc_queries, sat_features)  # [B, 1024, 2048]

# 预测分数
logits = Linear(decoder_out)  # [B, 1024, 1] → [B, 32, 32]

# 上采样
heatmap = interpolate(logits, size=(518, 518))  # [B, 518, 518]
heatmap = softmax(heatmap.flatten(), dim=-1).view(B, 518, 518)

# 提取位置
position = soft_argmax(heatmap)  # [B, 2]
```

### 6. Camera Yaw
```python
# 使用 camera tokens
camera_output = camera_head(front_camera_token, sat_camera_token)
yaw_radians = camera_output['yaw_radians']  # [B]
```

## 与您的需求对照

### ✅ 已实现的所有要求

1. **输入数据** ✅
   - Front-View Image, Satellite Image
   - Prompts: Point, BBox, Mask (在前视图上标注目标)
   - GT Labels: Camera Position (卫星图坐标), Yaw

2. **Pi3 Backbone** ✅
   - 提取 F_f 和 F_s (前视图和卫星图特征)
   - DINOv2 + Pi3 Decoder 架构

3. **SAM Prompt Encoder** ✅
   - Point/BBox/Mask 编码为 E_p (在前视图上标注)
   - 与 F_f 融合生成 F_target (前视图目标特征)
   - Sparse prompts: Cross-Attention
   - Dense prompts: 直接相加

4. **DETR Transformer** ✅
   - Encoder: 增强卫星图特征 (已在 Pi3 中完成)
   - Decoder: Location Queries (G×G) → Heatmap (卫星图定位)
   - F_target 引导: `Q_final = Q_init + Linear(F_target)`
   - 单向定位: 前视图目标信息 → 卫星图位置

5. **预测头** ✅
   - Heatmap Head: Query-based，双线性插值上采样（卫星图定位）
   - Camera Head: Pi3 原生 Head（相机姿态）

6. **损失函数** ✅
   - BBox Loss: L1 + GIoU ✅
   - Heatmap Loss: Penalty-reduced Focal Loss ✅
   - Angle Loss: MSE Loss ✅

## 下一步建议

### 1. 数据准备
- 实现 DataLoader 加载 `data/single.json`
- 图像预处理和归一化
- 数据增强 (随机裁剪、旋转等)

### 2. 训练脚本
```python
# train.py 示例结构
- 加载数据
- 创建模型
- 定义优化器 (不同学习率)
- 训练循环
- 验证和保存
```

### 3. 评估指标
- Position: 像素误差、距离误差（卫星图上）
- Yaw: 角度误差 (度)
- Heatmap: Peak accuracy, AUC

### 4. 可视化工具
- 热力图可视化
- BBox 预测可视化
- 注意力图可视化

## 参考文档

- **架构文档**: `models/README_DETR_ARCHITECTURE.md`
- **使用示例**: `models/example_usage_detr.py`
- **原始文档**: `models/README_MODELS.md`

## 技术栈

- PyTorch
- Pi3 (Pose-conditioned Image-to-Image-to-Image)
- DINOv2 (Self-supervised Vision Transformer)
- DETR (End-to-End Object Detection with Transformers)
- SAM2 (Segment Anything Model 2)

---

**最后更新日期**: 2026-02-04

**定位方式**: 单向定位（前视图 → 卫星图），因为前视图视野较小，不方便进行双向定位。

所有核心组件已按照单向定位需求实现完毕，可以直接使用或根据实际需求进行调整。
