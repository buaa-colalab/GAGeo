# Cross-View Localization V2 — 统一 Backbone 架构说明

## 1. 总体架构

### 1.1 设计理念

新架构的核心思想是**将所有 token 直接注入 Pi3 Backbone**，而非先提取特征再做后处理。
这样 prompt 信息在 backbone 的深层 attention 中就能与视觉特征充分交互，
同时通过精心设计的 attention mask 控制不同 token 之间的交互模式。

### 1.2 Token 组成

在 Pi3 Backbone 中，每个 view 的 token 序列结构如下：

| 位置 | Token 类型 | 数量 | 说明 |
|------|-----------|------|------|
| 0-4 | Register tokens | 5 | Pi3 原始寄存器 token（从预训练权重加载）|
| 5-1373 | Sate patch tokens | 1369 (37×37) | 卫星图像 patch 特征 |
| 0-4 | Register tokens | 5 | Pi3 原始寄存器 token |
| 5-1373 | Front patch tokens | 1369 (37×37) | 前视图像 patch 特征 |
| 1374-1375 | Learnable query tokens | 2 | BBox/Mask query + Heatmap query |
| 1376-N | Prompt tokens | 变长 | Point/BBox/Mask prompt 编码结果 |

其中 Sate view 和 Front view 分别作为两个 "view" 输入 Pi3，
Learnable query 和 Prompt tokens 附加在 Front view 的 token 序列末尾。

### 1.3 Mask Prompt 处理

Mask prompt 不是作为独立 token 输入的。而是：
1. 通过 SAM 的 `mask_downscaling` CNN 将 mask 编码为 `[B, C, 37, 37]` 的 dense embedding
2. 将 dense embedding reshape 为 `[B, 1369, C]` 后 **逐元素加到 front view patch tokens 上**
3. 这样 front view 的 patch tokens 同时携带了视觉信息和 mask 指示信息

在 attention 计算中，这些 "mask-augmented front tokens" 参与正常的 front view token 交互。

## 2. Attention Mask 机制

### 2.1 Local Attention（偶数层，view 内部交互）

```
序列: [reg(5) | sate_patch(1369)] [reg(5) | front_patch(1369) | learnable(2) | prompt(K)]
```

每个 view 内部独立做 self-attention，但有以下 mask 约束：

**Sate View**：所有 sate token 之间正常 self-attention，无额外 mask。

**Front View + Extra Tokens**：

| Q \ K | front_patch | learnable | prompt |
|-------|------------|-----------|--------|
| front_patch | ✅ 正常 | ✅ 可见 | ✅ 可见 |
| learnable | ✅ 可见 | ✅ 自身可见 | ✅ 可见 |
| prompt | ✅ 可见 | ✅ 可见 | ❌ Masked |

**关键规则**：
- Prompt tokens 之间互相 **不可见**（防止 prompt 之间信息泄漏）
- Prompt tokens 可以看到 front tokens 和 learnable tokens
- Front tokens 和 learnable tokens 可以看到所有 token（包括 prompt）

### 2.2 Global Attention（奇数层，跨 view 交互）

在 global attention 中，所有 view 的 token 被拼接成一个长序列：
```
[reg_s(5) | sate_patch(1369) | reg_f(5) | front_patch(1369) | learnable(2) | prompt(K)]
```

Mask 约束：

| Q \ K | sate_patch | front_patch | learnable | prompt |
|-------|-----------|-------------|-----------|--------|
| sate_patch | ✅ 正常 | ✅ 正常 | ✅ 可见 | ❌ Masked |
| front_patch | ✅ 正常 | ✅ 正常 | ✅ 可见 | ✅ 可见 |
| learnable | ✅ 可见 | ✅ 可见 | ✅ 可见 | ✅ 可见 |
| prompt | ❌ Masked (sate) | ✅ 可见 | ✅ 可见 | ❌ Masked |

**关键规则**：
- Prompt tokens 与 sate tokens 之间互相 **不可见**（prompt 只描述 front view 中的目标）
- Prompt tokens 之间互相 **不可见**
- Sate tokens 看不到 prompt tokens
- 其他所有交互正常进行

### 2.3 Mask Prompt 的特殊处理（Global Attention 中的 Front Token）

在 **global attention** 中，front view tokens 已经通过逐元素加法融合了 mask dense embedding。
这意味着当 front tokens 作为 K/V 被其他 token attend 时，
其他 token 能间接感知到 mask 信息，但不会直接看到 mask prompt token。

## 3. 输出使用

经过 Pi3 Backbone 所有层后：
- **2 个 Learnable Query Tokens** 分别输出到：
  - Query 0 → BBox Head + Mask Head（目标检测和分割）
  - Query 1 → Heatmap Head（相机位置定位）
- **Sate + Front Patch Features** → Camera Head（相机旋转预测）
- **Sate Patch Features** → Mask Head 需要的 spatial memory

## 4. 分层监督

在 Pi3 Decoder 的第 4、11、17 层（0-indexed: 3, 10, 16）输出中间特征，
对这些中间特征也应用 BBox + Mask 监督。权重递增：

| 层 | 权重 | 说明 |
|----|------|------|
| 第 4 层 (idx=3) | 0.1 | 浅层：粗糙的空间理解 |
| 第 11 层 (idx=10) | 0.3 | 中层：语义初步形成 |
| 第 17 层 (idx=16) | 0.6 | 深层：接近最终特征 |
| 最终层 (idx=35) | 1.0 | 最终输出 |

分层监督只对 BBox 和 Mask 任务生效。Heatmap 和 Camera 任务只在最终层监督。

## 5. 位置编码对齐

Pi3 使用 **RoPE (Rotary Position Embedding)**，SAM 使用 **Random Fourier PE**。

新架构中统一使用 Pi3 的 RoPE：
- Sate 和 Front patch tokens：使用 Pi3 原始的 2D RoPE
- Learnable Query Tokens：位置设为 (0, 0)（无空间语义）
- Prompt Tokens：
  - Point prompts：使用点的归一化坐标生成 RoPE 位置
  - Box prompts：使用两个角点的归一化坐标
  - Mask prompts：已经通过逐元素加法融入 front tokens，不需要额外位置编码

## 6. 学习率策略

分三组参数：

| 参数组 | 学习率 | 说明 |
|--------|--------|------|
| Pi3 Backbone (encoder + decoder) | `lr_backbone` (1e-5) | 最小 LR，保留预训练特征 |
| 新增 Tokens (learnable + prompt projection) | `lr_new_tokens` (5e-4) | 最大 LR，快速收敛 |
| Task Heads (bbox, mask, heatmap, camera) | `lr_heads` (1e-4) | 中等 LR |

## 7. Mask Head 设计

参考 SAM2 的 MaskDecoder 设计：

1. **输入**：Learnable Query Token (bbox/mask) 的输出 + Sate Patch Features
2. **上采样**：`ConvTranspose2d` 将 37×37 → 74×74 → 148×148
3. **Hypernetwork MLP**：将 query token 映射为动态卷积核
4. **Mask 预测**：动态卷积核与上采样后的 spatial features 做点积
5. **Loss**：BCE + Dice Loss（参考 SAM）

```
query_token [B, C] → MLP → hyper_weights [B, C//8]
sate_features [B, C, 37, 37] → ConvTranspose2d → upscaled [B, C//8, 148, 148]
mask_logits = hyper_weights @ upscaled → [B, 1, 148, 148]
mask_pred = interpolate(mask_logits, 518×518)
```
