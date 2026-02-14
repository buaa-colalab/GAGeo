# Cross-View Localizer V2 组件级详细结构说明

> 版本：当前 `feature/unified-backbone-v2` 代码状态  
> 目标：按“组件/模块”梳理输入输出形状、层数、参数量、内部机制、代码位置。

---

## 0. 总体数据流（主干）

1. 输入两张图：`front_view` 与 `satellite_view`，形状均为 `[B, 3, 518, 518]`
2. Prompt Encoder 编码点/框/mask prompt
3. Pi3BackboneV2 编码并在 decoder 注入：
   - 卫星视角 token：`[register(5) + 1369 patches]`
   - 前视角 token：`[register(5) + 1369 patches + learnable(2) + prompt(K)]`
4. 主任务头：
   - learnable token 0 -> bbox/mask
   - learnable token 1 -> heatmap(position)
   - patch features -> camera rotation / contrastive
5. `DETRCriterionV2` 汇总最终损失 + 深度监督损失

主入口代码：
- [models/cross_view_localizer_v2.py](../models/cross_view_localizer_v2.py) 的 `CrossViewLocalizerV2.forward()`

---

## 1. 顶层模型：CrossViewLocalizerV2

### 代码位置
- [models/cross_view_localizer_v2.py](../models/cross_view_localizer_v2.py)
- 核心符号：`CrossViewLocalizerV2`、`build_cross_view_localizer_v2()`

### 输入 / 输出
- 输入：
  - `front_view`: `[B, 3, 518, 518]`
  - `satellite_view`: `[B, 3, 518, 518]`
  - `points`: `([B, Np, 2], [B, Np])`（可选）
  - `boxes`: `[B, Nb, 4]`（可选）
  - `masks`: `[B, 1, H, W]`（可选）
- 输出（核心）：
  - `pred_boxes`: `[B, 1, 4]`
  - `class_logits`: `[B, 1, 1]`
  - `mask_logits`: `[B, num_mask_tokens, 518, 518]`
  - `position`: `[B, 2]`
  - `rotation_matrix`: `[B, 3, 3]`
  - `intermediate_preds`: `{4/11/17 -> {pred_boxes, class_logits, mask_logits, ...}}`

### 层数
- 这是组装层，不直接定义 transformer 深度；深度主要在 backbone/camera head 中。

### 参数量（组合级）
- 参数主要来自 `backbone` 和 `camera_head`。
- 该文件中额外新增的“深度监督头”有：
  - `inter_bbox_heads`: 3 个 `BBoxHead`
  - `inter_mask_heads`: 3 个 `SAMMaskHead`

### 内部机制
- 先把 prompt 转成 sparse/dense embedding；
- 进入 backbone 做统一 token 交互；
- 将 learnable query 拆分为 bbox/mask 与 heatmap 两条路径；
- 同时使用 patch features 做 camera/contrastive。

---

## 2. 主干：Pi3BackboneV2（统一 token 注入 + 掩码注意力）

### 代码位置
- [models/backbone/pi3_backbone_v2.py](../models/backbone/pi3_backbone_v2.py)
- 核心符号：
  - `Pi3BackboneV2`
  - `MaskedFlashAttentionRope`
  - `BlockRopeWithMask`
  - `decode_with_extra_tokens()`

### 输入 / 输出
- 输入：
  - `front_view`, `satellite_view`: `[B, 3, 518, 518]`
  - `sparse_embeddings`: `[B, K, 1024]`（由 prompt encoder 投影后）
  - `dense_embeddings`: `[B, 1024, 37, 37]`（mask prompt 时）
  - `prompt_coords`: `[B, K, 2]`（归一化坐标）
- 输出：
  - `features`: `[B, 2, 1374, 2048]`
  - `sate_features`: `[B, 1369, 2048]`
  - `front_features`: `[B, 1369, 2048]`
  - `learnable_out`: `[B, 2, 2048]`
  - `intermediate`: `{stage_idx -> {'learnable':[B,2,2048], 'sate_patches':[B,1369,2048]}}`

### 层数（large）
- Encoder：DINOv2 ViT-L/14 reg（标准 24 blocks）
- Decoder：36 个 `BlockRope`（按 pair-layer 计为 18 层）
- 深度监督层：`[4, 11, 17]`（0-based pair-layer）

### 注意力机制（关键）
- **Local block（偶数 block）**：视角内注意力
  - prompt token 之间互相不可见，仅可见自身
- **Global block（奇数 block）**：跨视角注意力
  - prompt 与 sate token 双向屏蔽
  - prompt 与 prompt 互相屏蔽（保留 self）

### Flash Attention with mask
- `MaskedFlashAttentionRope` 使用 SDPA 后端优先级：
  1. `FLASH_ATTENTION`
  2. `EFFICIENT_ATTENTION`
  3. `MATH` 回退
- mask 使用 bool keep-mask（`True=可见`, `False=屏蔽`）以提升兼容性。

### 参数量
> 说明：backbone 总参数非常大（含 DINOv2 encoder + 36 层 decoder），以下给出关键新增/可定位项的精确值与 decoder 近似估算。

- `learnable_queries`: `1*2*1024 = 2,048`
- `register_token`: `1*1*5*1024 = 5,120`
- `intermediate_projs`（3个）: `3 * (1024*2048 + 2048) = 6,297,600`
- `final_proj`: `1024*2048 + 2048 = 2,099,200`
- Decoder 单 block（d=1024, mlp_ratio=4）约：`12.60M`
- 36 个 decoder blocks 约：`453.55M`
- Encoder（ViT-L/14 reg）参数量约 3e8 级别（依具体实现/注册 token配置微调）

---

## 3. Prompt Encoder：GeometryPromptEncoder

### 代码位置
- [models/encoder/prompt_encoder.py](../models/encoder/prompt_encoder.py)
- 核心符号：`GeometryPromptEncoder`、`load_sam_prompt_encoder_weights()`

### 输入 / 输出
- 输入：
  - `points`: `[B, N, 2]` + labels `[B, N]`
  - `boxes`: `[B, M, 4]`
  - `masks`: `[B, 1, H, W]`
- 输出：
  - `sparse_embeddings`: `[B, K, 1024]`（内部256 -> 外部1024）
  - `dense_embeddings`: `[B, 1024, 37, 37]`

### 层数
- mask_downscaling: 3 个卷积层（2个 stride=2 + 1个1x1）
- sparse 投影: `Linear + LayerNorm`
- dense 投影: `Conv1x1 + LayerNorm2d`

### 参数量（`sam_embed_dim=256`, `embed_dim=1024`, `mask_in_chans=16`）
- 精确可计：约 **536,652**（不含 buffer）
  - 其中两个投影层 `sparse_proj + dense_proj` 合计约 **530,432**

### 内部机制
- 点/框：`PositionEmbeddingRandom` + 类型 embedding（正点/负点/两角点）
- mask：下采样到 `[37,37]`
- 若使用 SAM 权重：
  - 内部 256 维层可直接加载
  - 仅投影层从随机初始化训练

---

## 4. BBoxHead

### 代码位置
- [models/heads/bbox_head.py](../models/heads/bbox_head.py)
- 依赖 MLP: [models/layers/mlp.py](../models/layers/mlp.py)

### 输入 / 输出
- 输入：`query_features` `[B, Nq, 2048]`（这里 `Nq=1`）
- 输出：
  - `pred_boxes`: `[B, Nq, 4]`（sigmoid归一化）
  - `class_logits`: `[B, Nq, 1]`
  - `bbox_scores`: `[B, Nq]`

### 层数
- bbox 回归 MLP：3层
- 分类线性层：1层

### 参数量（精确）
- **8,402,949**

---

## 5. SAMMaskHead

### 代码位置
- [models/heads/mask_head.py](../models/heads/mask_head.py)

### 输入 / 输出
- 输入：
  - `query_token`: `[B, 2048]`
  - `spatial_features`: `[B, 1369, 2048]`
  - `spatial_size`: `(37, 37)`
- 输出：
  - `mask_logits`: `[B, num_mask_tokens, 518, 518]`
  - `mask_pred`: 同形状 sigmoid
  - `iou_pred`: `[B, num_mask_tokens]`

### 层数
- 上采样：2 个 `ConvTranspose2d`
- 超网络：每个 mask token 一套 3层 MLP
- IoU head：3层 MLP

### 参数量（`num_mask_tokens=1`，精确）
- **14,949,889**

### 内部机制
- 将 query token 映射为动态卷积核（hypernetwork）
- 与上采样后的空间特征做点积生成 mask logits
- 最后双线性插值到 518x518

---

## 6. HeatmapHead

### 代码位置
- [models/heads/heatmap_head.py](../models/heads/heatmap_head.py)

### 输入 / 输出
- 输入：
  - `query_features`: `[B, Nloc, 2048]`（当前 `Nloc=1`）
  - `spatial_features`: `[B, 1369, 2048]`
- 输出：
  - `heatmap`: `[B, 518, 518]`（softmax 概率）
  - `heatmap_logits`: `[B, 37, 37]`
  - `position`: `[B, 2]`（soft-argmax，归一化坐标）

### 层数
- 1层线性 `query_to_weight`

### 参数量（精确）
- **2,049**

---

## 7. Pi3CameraHead

### 代码位置
- [models/heads/pi3_camera_head.py](../models/heads/pi3_camera_head.py)
- 依赖：
  - [models/layers/transformer_head.py](../models/layers/transformer_head.py)
  - [models/layers/camera_head.py](../models/layers/camera_head.py)

### 输入 / 输出
- 输入：
  - `front_patch_features`: `[B, 1369, 2048]`
  - `sat_patch_features`: `[B, 1369, 2048]`
- 输出：
  - `rotation_matrix`: `[B, 3, 3]`
  - `yaw/pitch/roll`: `[B]`

### 层数
- `TransformerDecoder` depth = 5（同一套 decoder 分别处理 front/sat）
- `CameraHead`：
  - 2个 `ResConvBlock`（每个 block 内 3层 Linear + skip）
  - MLP 两层 + 平移/旋转输出层

### 内部机制
- 先用 RoPE + transformer 提炼 patch token
- 对每个视角预测绝对位姿 `T_front`, `T_sat`
- 相对位姿：`T_rel = T_sat^{-1} @ T_front`
- 从旋转矩阵提取欧拉角

---

## 8. CrossViewContrastiveHead（MoCo）

### 代码位置
- [models/heads/contrastive_head.py](../models/heads/contrastive_head.py)

### 输入 / 输出
- 输入：
  - `mono_features`, `sat_features`: `[B, 1369, 2048]`
  - `mono_mask`, `sat_mask`: `[B, 1, H, W]`
- 输出：
  - `loss` 标量（InfoNCE）

### 层数
- Query encoder：2层 MLP（Linear-ReLU-Linear）
- Key encoder：结构同 query encoder（EMA更新，不反传）

### 参数量
- 总参数（含 `encoder_k`）: **9,441,792**
- 可训练参数（仅 `encoder_q`）: **4,720,896**
- queue / queue_ptr 为 buffer，不计入参数

### 内部机制
- 先做 mask-aware pooling 得到视角级向量
- satellite 作为 query，mono 作为 key
- 与队列负样本做 InfoNCE

---

## 9. Loss 模块：DETRCriterionV2

### 代码位置
- [utils/losses_v2.py](../utils/losses_v2.py)
- 核心符号：`HungarianMatcher`、`DETRCriterionV2`

### 输入 / 输出
- 输入：
  - `outputs`（主输出 + `intermediate_preds`）
  - `targets`（`sat_bbox`, `sat_mask`, `camera_position`, `rotation_matrix`）
- 输出：
  - `losses` dict（含各子损失 + 总 `loss`）

### 子损失
- BBox: `L1 + GIoU`
- Class: `Focal`
- Mask: `BCE + Dice`
- Heatmap: `position MSE`
- Rotation: `SO(3) geodesic`
- Contrastive: passthrough
- Deep supervision: 对 `inter_{layer}_loss_*` 按层权重再加总

### 深度监督层语义
- 当前语义是 pair-layer 0-based（1层=local+global）
- `[4, 11, 17]` 中 `17` 为 large 配置最后一层 pair 输出

---

## 10. 训练与日志（与结构相关）

### 代码位置
- [train_detr_v2.py](../train_detr_v2.py)
- 配置： [configs/default_v2.yaml](../configs/default_v2.yaml)

### 关键结构约定
- `supervision_layers` 使用 pair-layer 索引
- TensorBoard：
  - 最终输出：`train_step/*`
  - 中间监督：`trian_step_{idx}/*`
  - 若包含最后层（large 的 17），中间监督会跳过最后层曲线，避免和最终输出重复

---

## 11. 备注：参数量统计口径

1. 文档中给出的小模块参数量为“按代码定义可精确推导值”。
2. backbone 总参数量受 encoder 版本、decoder_size、是否加载/冻结影响较大，且体量极大（数亿级）。
3. 若需要“当前运行配置+当前权重状态”的一键精确统计，建议在训练环境中执行模块统计脚本并导出到日志。

---

## 12. 快速定位索引（按功能）

- 顶层编排： [models/cross_view_localizer_v2.py](../models/cross_view_localizer_v2.py)
- Backbone+掩码注意力： [models/backbone/pi3_backbone_v2.py](../models/backbone/pi3_backbone_v2.py)
- Prompt 编码： [models/encoder/prompt_encoder.py](../models/encoder/prompt_encoder.py)
- BBox 头： [models/heads/bbox_head.py](../models/heads/bbox_head.py)
- Mask 头： [models/heads/mask_head.py](../models/heads/mask_head.py)
- Heatmap 头： [models/heads/heatmap_head.py](../models/heads/heatmap_head.py)
- Camera 头： [models/heads/pi3_camera_head.py](../models/heads/pi3_camera_head.py)
- Contrastive 头： [models/heads/contrastive_head.py](../models/heads/contrastive_head.py)
- Loss： [utils/losses_v2.py](../utils/losses_v2.py)
- 训练脚本： [train_detr_v2.py](../train_detr_v2.py)
- 配置： [configs/default_v2.yaml](../configs/default_v2.yaml)
