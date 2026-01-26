# Cross-View Localization Architecture

## 核心设计理念

本系统的关键创新在于**直接利用VGGT的Alternating Attention机制处理跨视角对应问题**，而不是添加额外的融合模块。

## VGGT Alternating Attention 详解

### 为什么VGGT能处理3D和跨视角问题？

VGGT的核心是**交替注意力（Alternating Attention）**机制：

```python
# 输入: [B, S, 3, H, W]  其中 S=2 (front_view + satellite_view)

for block in range(depth):
    # 1. Frame Attention (Spatial)
    tokens = tokens.view(B*S, P, C)  # 每个视角独立处理
    tokens = frame_attention(tokens)  # 学习单视角内的空间关系
    
    # 2. Global Attention (Cross-view)
    tokens = tokens.view(B, S*P, C)  # 所有视角联合处理
    tokens = global_attention(tokens)  # 学习跨视角的对应关系
```

### 两种注意力的作用

#### Frame Attention (S-Attn)
- **形状**: `[B*S, P, C]` - 将S个视角展平为独立的batch
- **作用范围**: 单张图片内部
- **学习内容**: 
  - 物体的形状、纹理、局部结构
  - 单视角的空间关系
  - 视觉特征提取

#### Global Attention (T-Attn)
- **形状**: `[B, S*P, C]` - 所有视角的所有patch联合处理
- **作用范围**: 跨视角的所有位置
- **学习内容**:
  - 前视图和卫星图的对应关系
  - 同一物体在不同视角的表现
  - 3D几何约束（通过跨视角一致性）

### 为什么不需要额外的Cross-View Fusion？

**Global Attention已经完成了跨视角融合！**

```
前视图patch [i] ←→ 卫星图patch [j]
       ↓ Global Attention ↓
自动学习对应关系，无需显式监督
```

在Global Attention中：
- 前视图的每个patch可以attend到卫星图的所有patch
- 卫星图的每个patch可以attend到前视图的所有patch
- 通过多层alternating，逐步建立精确的跨视角对应

## 数据流详解

### Step 1: 输入准备
```python
front_view: [B, 3, 518, 518]
satellite_view: [B, 3, 518, 518]
images = torch.stack([front_view, satellite_view], dim=1)  # [B, 2, 3, 518, 518]
```

### Step 2: VGGT Alternating Attention
```python
# Patch embedding
patches: [B*2, P, C]  # P = (518/14)^2 ≈ 1369

# Alternating attention (24层)
for i in range(24):
    # Frame attention: 每个视角独立
    patches = frame_blocks[i](patches.view(B*2, P, C))
    
    # Global attention: 跨视角交互
    patches = global_blocks[i](patches.view(B, 2*P, C))

# 输出
vggt_features: List[[B, 2, P, 2*C]]  # 2*C因为concat了frame和global特征
```

### Step 3: 几何提示编码
```python
# 从前视图提取特征
front_features = vggt_features[-1][:, 0]  # [B, P, 2*C]

# 用户点击/框选的位置
user_prompts: Prompt(boxes, points, masks)

# SAM3编码几何提示
geo_embeddings = geometry_encoder(
    prompts=user_prompts,
    img_features=front_features
)  # [N_prompts, B, C]
```

### Step 4: 卫星图检测
```python
# 卫星图特征（已包含跨视角信息）
sat_features = vggt_features[-1][:, 1]  # [B, P, 2*C]

# 检测头预测
bbox_pred, scores = detection_head(
    condition=geo_embeddings,  # "要找什么"
    features=sat_features      # "在哪里找"（已有跨视角对应）
)
```

## 与原始设计的对比

### ❌ 原始设计（过度复杂）
```
VGGT → 分离前视图/卫星图特征 → Cross-View Fusion → Detection
       ↓                        ↓
    独立特征              显式跨视角注意力
```
问题：
- VGGT的Global Attention已经做了跨视角融合
- 额外的Cross-View Fusion是冗余的
- 增加参数量和计算量

### ✅ 新设计（简洁高效）
```
VGGT (Alternating Attention) → Geometry Encoder → Detection
     ↓                              ↓                ↓
  自动跨视角融合              编码用户意图      预测位置
```
优势：
- 充分利用VGGT的核心能力
- 减少参数量
- 更符合VGGT的设计理念

## 训练策略

### 两阶段训练

#### Stage 1: 冻结VGGT
```bash
python train.py --freeze_vggt --num_epochs 50 --lr 1e-4
```
- 固定VGGT的预训练权重
- 只训练Geometry Encoder和Detection Head
- 快速收敛到合理baseline

#### Stage 2: 端到端微调
```bash
python train.py --resume checkpoint.pth --num_epochs 100 --lr 1e-5
```
- 解冻所有参数
- 小学习率微调
- VGGT学习任务特定的跨视角对应

### 为什么这样有效？

1. **VGGT预训练**: 在大规模多视角数据上训练，已学会通用的跨视角对应
2. **任务适配**: 微调时学习特定任务（前视图→卫星图）的对应模式
3. **几何先验**: Geometry Encoder提供强几何约束

## 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `vggt_depth` | 24 | Alternating attention层数，越多跨视角对应越精确 |
| `aa_order` | `["frame", "global"]` | 交替顺序，先空间后跨视角 |
| `aa_block_size` | 1 | 每次交替的block数量 |
| `embed_dim` | 1024 | 特征维度，需与预训练权重匹配 |

## 消融实验建议

可以验证Alternating Attention的重要性：

1. **仅Frame Attention**: `aa_order=["frame"]` - 性能应显著下降
2. **仅Global Attention**: `aa_order=["global"]` - 缺乏空间细节
3. **不同交替频率**: 改变`aa_block_size` - 影响融合粒度

## 参考文献

- VGGT: Visual Geometry Grounded Transformer (CVPR 2025 Best Paper)
- SAM 3: Segment Anything Model 3 (Meta AI)
- DETR: End-to-End Object Detection with Transformers
