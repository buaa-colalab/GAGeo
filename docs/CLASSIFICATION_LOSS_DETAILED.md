# Classification Loss 详细解析

本文档详细解析 V3 模型中 Classification Loss 的具体运作形式、计算公式，以及每个 query 的 loss 计算方式。

## 1. 分类任务定义

### 1.1 类别数量

```python
# 在 BBoxHead 中定义
num_classes: int = 1
```

**关键点**：
- `num_classes=1` 表示这是一个**二分类问题**
- 分类任务：**有目标（object）** vs **无目标（no-object）**
- 这是一个**单类目标检测**任务（single-class object detection）

### 1.2 为什么是二分类？

在 DETR 风格的检测中：
- **正样本（positive）**：query 匹配到 ground truth bbox → 类别 = 1（有目标）
- **负样本（negative）**：query 未匹配到任何 ground truth → 类别 = 0（无目标）

由于是单目标检测（每张图像只有一个 ground truth），所以：
- 每个 batch 中，**只有一个 query 是正样本**（匹配的）
- 其他所有 queries 都是**负样本**（未匹配的）

---

## 2. 模型输出

### 2.1 BBox Head 输出

```python
# BBoxHead.forward()
class_logits = self.class_embed(query_features)  # [B, Q, num_classes]
# 当 num_classes=1 时: [B, Q, 1]
```

**输出形状**：
- `class_logits`: `[B, Q, 1]` - 原始 logits（未经过 sigmoid）
- `B`: batch size
- `Q`: query 数量（`num_bbox_mask_queries`）

**示例**（`Q=2`）：
```python
# 假设 batch_size=2, num_queries=2
class_logits = torch.tensor([
    [[0.5], [-0.3]],   # Batch 0: Query 0 logit=0.5, Query 1 logit=-0.3
    [[-0.8], [1.2]],  # Batch 1: Query 0 logit=-0.8, Query 1 logit=1.2
])  # Shape: [2, 2, 1]
```

### 2.2 置信度分数计算

```python
# 在 BBoxHead 中
if self.num_classes == 1:
    bbox_scores = class_logits.squeeze(-1).sigmoid()  # [B, Q]
```

**计算过程**：
```python
# class_logits: [B, Q, 1]
# 1. squeeze(-1): [B, Q, 1] -> [B, Q]
# 2. sigmoid: [B, Q] -> [B, Q] (概率值，范围 [0, 1])
```

**公式**：
\[
\text{bbox\_score}_{b,q} = \sigma(\text{class\_logit}_{b,q}) = \frac{1}{1 + e^{-\text{class\_logit}_{b,q}}}
\]

**示例**：
```python
class_logits = [[0.5], [-0.3]]  # [2, 1]
bbox_scores = sigmoid([0.5, -0.3]) = [0.622, 0.426]  # [2]
```

---

## 3. Target 标签生成

### 3.1 匹配结果

在 Hungarian Matching 之后，得到匹配结果：

```python
indices = [
    (tensor([1]), tensor([0])),  # Batch 0: Query 1 匹配到 GT 0
    (tensor([0]), tensor([0])),  # Batch 1: Query 0 匹配到 GT 0
]
```

### 3.2 Target 标签设置

```python
# 在 _compute_bbox_loss 中
target_classes = torch.zeros_like(pred_logits)  # [B, Q, 1] - 初始化为全 0
for b_idx, (src, _) in enumerate(indices):
    target_classes[b_idx, src, :] = 1.0  # 匹配的 query 标记为 1.0
```

**示例**（`B=2, Q=2`）：
```python
# 初始状态
target_classes = [
    [[0.0], [0.0]],  # Batch 0: 所有 queries 初始为 0（负样本）
    [[0.0], [0.0]],  # Batch 1: 所有 queries 初始为 0（负样本）
]

# 匹配后（假设 Query 1 匹配到 GT）
target_classes = [
    [[0.0], [1.0]],  # Batch 0: Query 0=负样本(0), Query 1=正样本(1)
    [[1.0], [0.0]],  # Batch 1: Query 0=正样本(1), Query 1=负样本(0)
]
```

**关键点**：
- **匹配的 query**：`target = 1.0`（正样本，有目标）
- **未匹配的 queries**：`target = 0.0`（负样本，无目标）

---

## 4. Loss 计算公式

### 4.1 Sigmoid Focal Loss

Classification Loss 使用 **Sigmoid Focal Loss**，而不是标准的 Binary Cross-Entropy。

#### 4.1.1 基础公式

```python
def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()  # [B*Q, 1] -> 概率值
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()
```

#### 4.1.2 逐步分解

**Step 1: 计算概率**
\[
p_{b,q} = \sigma(\text{class\_logit}_{b,q}) = \frac{1}{1 + e^{-\text{class\_logit}_{b,q}}}
\]

**Step 2: Binary Cross-Entropy Loss**
\[
\text{CE}_{b,q} = -\left[ y_{b,q} \cdot \log(p_{b,q}) + (1 - y_{b,q}) \cdot \log(1 - p_{b,q}) \right]
\]

其中：
- \(y_{b,q} \in \{0, 1\}\) 是 target 标签
- \(p_{b,q} \in [0, 1]\) 是预测概率

**Step 3: 计算 \(p_t\)（预测正确的概率）**
\[
p_t = \begin{cases}
p_{b,q} & \text{if } y_{b,q} = 1 \text{ (正样本)} \\
1 - p_{b,q} & \text{if } y_{b,q} = 0 \text{ (负样本)}
\end{cases}
\]

**Step 4: Focal Weight（困难样本加权）**
\[
\text{focal\_weight} = (1 - p_t)^{\gamma}
\]

其中 \(\gamma = 2.0\)：
- 当 \(p_t\) 接近 1（预测正确）时，权重接近 0（降低简单样本的权重）
- 当 \(p_t\) 接近 0（预测错误）时，权重接近 1（增加困难样本的权重）

**Step 5: Alpha Weighting（类别平衡）**
\[
\alpha_t = \begin{cases}
\alpha = 0.25 & \text{if } y_{b,q} = 1 \text{ (正样本)} \\
1 - \alpha = 0.75 & \text{if } y_{b,q} = 0 \text{ (负样本)}
\end{cases}
\]

**Step 6: 最终 Loss**
\[
L_{\text{class}}(b,q) = \alpha_t \cdot \text{CE}_{b,q} \cdot (1 - p_t)^{\gamma}
\]

**Step 7: 总 Loss（所有 queries 求和后平均）**
\[
L_{\text{class}} = \frac{1}{B} \sum_{b=1}^{B} \sum_{q=1}^{Q} L_{\text{class}}(b,q)
\]

---

## 5. 每个 Query 的 Loss 计算示例

### 5.1 场景设置

假设：
- `B=2`（batch size = 2）
- `Q=2`（query 数量 = 2）
- 匹配结果：
  - Batch 0: Query 1 匹配到 GT
  - Batch 1: Query 0 匹配到 GT

### 5.2 模型输出

```python
# class_logits: [2, 2, 1]
class_logits = torch.tensor([
    [[0.2], [-0.5]],   # Batch 0: Query 0=0.2, Query 1=-0.5
    [[1.0], [-1.0]],   # Batch 1: Query 0=1.0, Query 1=-1.0
])
```

### 5.3 Target 标签

```python
# target_classes: [2, 2, 1]
target_classes = torch.tensor([
    [[0.0], [1.0]],   # Batch 0: Query 0=负样本, Query 1=正样本
    [[1.0], [0.0]],   # Batch 1: Query 0=正样本, Query 1=负样本
])
```

### 5.4 逐步计算

#### Batch 0, Query 0（负样本）

```python
class_logit = 0.2
target = 0.0
prob = sigmoid(0.2) = 0.550  # 预测概率

# Binary CE
ce_loss = -[0.0 * log(0.550) + 1.0 * log(1 - 0.550)]
         = -log(0.450) = 0.799

# p_t (预测正确的概率)
p_t = 1 - prob = 1 - 0.550 = 0.450  # 负样本，预测为负的概率

# Focal weight
focal_weight = (1 - 0.450)^2 = 0.303

# Alpha weight
alpha_t = 1 - alpha = 1 - 0.25 = 0.75  # 负样本

# Final loss
loss = 0.75 * 0.799 * 0.303 = 0.182
```

#### Batch 0, Query 1（正样本）

```python
class_logit = -0.5
target = 1.0
prob = sigmoid(-0.5) = 0.378  # 预测概率较低（错误）

# Binary CE
ce_loss = -[1.0 * log(0.378) + 0.0 * log(1 - 0.378)]
         = -log(0.378) = 0.973

# p_t (预测正确的概率)
p_t = prob = 0.378  # 正样本，预测为正的概率（较低，说明预测错误）

# Focal weight
focal_weight = (1 - 0.378)^2 = 0.387  # 较大的权重（困难样本）

# Alpha weight
alpha_t = alpha = 0.25  # 正样本

# Final loss
loss = 0.25 * 0.973 * 0.387 = 0.094
```

#### Batch 1, Query 0（正样本）

```python
class_logit = 1.0
target = 1.0
prob = sigmoid(1.0) = 0.731  # 预测概率较高（正确）

# Binary CE
ce_loss = -[1.0 * log(0.731) + 0.0 * log(1 - 0.731)]
         = -log(0.731) = 0.313

# p_t (预测正确的概率)
p_t = prob = 0.731  # 正样本，预测为正的概率（较高，说明预测正确）

# Focal weight
focal_weight = (1 - 0.731)^2 = 0.072  # 较小的权重（简单样本）

# Alpha weight
alpha_t = alpha = 0.25  # 正样本

# Final loss
loss = 0.25 * 0.313 * 0.072 = 0.006  # 很小的 loss（简单样本）
```

#### Batch 1, Query 1（负样本）

```python
class_logit = -1.0
target = 0.0
prob = sigmoid(-1.0) = 0.269  # 预测概率较低（正确，应该是负样本）

# Binary CE
ce_loss = -[0.0 * log(0.269) + 1.0 * log(1 - 0.269)]
         = -log(0.731) = 0.313

# p_t (预测正确的概率)
p_t = 1 - prob = 1 - 0.269 = 0.731  # 负样本，预测为负的概率（较高，正确）

# Focal weight
focal_weight = (1 - 0.731)^2 = 0.072  # 较小的权重（简单样本）

# Alpha weight
alpha_t = 1 - alpha = 0.75  # 负样本

# Final loss
loss = 0.75 * 0.313 * 0.072 = 0.017  # 很小的 loss（简单样本）
```

### 5.5 总 Loss

```python
# 所有 queries 的 loss 求和
total_loss = 0.182 + 0.094 + 0.006 + 0.017 = 0.299

# 平均（除以 batch size）
loss_class = 0.299 / 2 = 0.150
```

---

## 6. Focal Loss 的作用机制

### 6.1 为什么使用 Focal Loss？

在单目标检测中，**负样本（未匹配的 queries）远多于正样本（匹配的 query）**，导致：
- 负样本的 loss 占主导地位
- 模型容易忽略正样本的学习

### 6.2 Focal Loss 的优势

1. **困难样本聚焦**：通过 \((1-p_t)^{\gamma}\) 权重，**困难样本（预测错误的）获得更大的权重**
2. **简单样本抑制**：**简单样本（预测正确的）获得更小的权重**，避免负样本 loss 占主导
3. **类别平衡**：通过 \(\alpha_t\) 权重，**正样本权重较小（0.25），负样本权重较大（0.75）**，平衡正负样本的贡献

### 6.3 参数设置

```python
alpha=0.25  # 正样本权重（较小，因为正样本少）
gamma=2.0   # Focal 权重指数（越大，困难样本权重越大）
```

**效果**：
- 当预测错误时（\(p_t\) 小）：\((1-p_t)^2\) 大 → loss 权重大
- 当预测正确时（\(p_t\) 大）：\((1-p_t)^2\) 小 → loss 权重小

---

## 7. 代码实现流程

### 7.1 完整流程

```python
# Step 1: 模型输出
class_logits = bbox_head(bbox_queries)  # [B, Q, 1]

# Step 2: Hungarian Matching
indices = matcher(outputs, targets)
# 例如: [(tensor([1]), tensor([0])), (tensor([0]), tensor([0]))]

# Step 3: 生成 Target 标签
target_classes = torch.zeros_like(class_logits)  # [B, Q, 1]
for b_idx, (src, _) in enumerate(indices):
    target_classes[b_idx, src, :] = 1.0  # 匹配的 query = 1.0

# Step 4: Flatten
pred_logits_flat = class_logits.flatten(0, 1)  # [B*Q, 1]
target_flat = target_classes.flatten(0, 1)    # [B*Q, 1]

# Step 5: 计算 Focal Loss
loss_class = sigmoid_focal_loss(
    pred_logits_flat,
    target_flat,
    alpha=0.25,
    gamma=2.0,
) / B  # 除以 batch size 得到平均 loss
```

### 7.2 关键代码位置

- **模型输出**：`xhj/location_v4/models/heads/bbox_head.py:53`
- **Target 生成**：`xhj/location_v4/utils/losses_v2.py:223-225`
- **Loss 计算**：`xhj/location_v4/utils/losses_v2.py:227-231`
- **Focal Loss 实现**：`xhj/location_v4/utils/losses_v2.py:72-81`

---

## 8. 总结

### 8.1 分类任务

- **类别数**：`num_classes=1`（二分类：有目标 vs 无目标）
- **正样本**：匹配到 GT 的 query（`target=1.0`）
- **负样本**：未匹配的 queries（`target=0.0`）

### 8.2 Loss 公式

\[
L_{\text{class}} = \frac{1}{B} \sum_{b=1}^{B} \sum_{q=1}^{Q} \alpha_t \cdot \text{CE}_{b,q} \cdot (1 - p_t)^{\gamma}
\]

其中：
- \(\text{CE}_{b,q}\)：Binary Cross-Entropy Loss
- \(p_t\)：预测正确的概率
- \(\alpha_t\)：类别平衡权重（正样本=0.25，负样本=0.75）
- \(\gamma=2.0\)：Focal 权重指数

### 8.3 每个 Query 的 Loss

- **所有 queries 都参与 loss 计算**
- **匹配的 query**：正样本，学习预测"有目标"
- **未匹配的 queries**：负样本，学习预测"无目标"
- **Focal Loss** 确保困难样本（预测错误的）获得更大的权重，简单样本（预测正确的）获得更小的权重

### 8.4 设计优势

1. **多 query 学习**：所有 queries 都参与分类学习，即使未匹配也能学习到"无目标"的表示
2. **困难样本聚焦**：Focal Loss 自动聚焦困难样本，提升学习效率
3. **类别平衡**：通过 alpha 权重平衡正负样本的贡献
