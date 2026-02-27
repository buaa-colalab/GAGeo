# Learnable Query 机制详细解析

本文档详细解析 V3 模型中不同数量的 learnable query 如何作用在 bbox head 和 mask head 上，以及 loss 的设计原理。

## 1. Query 分割机制

### 1.1 模型初始化

在 `CrossViewLocalizerV2.__init__` 中，定义了 query 的数量配置：

```python
# 配置参数
num_bbox_mask_queries: Optional[int] = None,  # bbox/mask 分支的 query 数量
num_heatmap_queries: int = 1,                  # heatmap 分支的 query 数量

# 自动推断逻辑
if num_bbox_mask_queries is None:
    inferred_bbox_mask_queries = int(num_learnable_tokens) - self.num_heatmap_queries
    self.num_bbox_mask_queries = max(inferred_bbox_mask_queries, 1)
else:
    self.num_bbox_mask_queries = int(num_bbox_mask_queries)

# 总 query 数量
self.num_learnable_tokens = self.num_bbox_mask_queries + self.num_heatmap_queries
```

**示例配置**：
- `query_len=1`: `num_bbox_mask_queries=1`, `num_heatmap_queries=1` → 总共 2 queries
- `query_len=2`: `num_bbox_mask_queries=2`, `num_heatmap_queries=1` → 总共 3 queries
- `query_len=4`: `num_bbox_mask_queries=4`, `num_heatmap_queries=1` → 总共 5 queries

### 1.2 Forward 中的 Query 分割

在 `forward` 方法中，learnable queries 被分割为两部分：

```python
# 从 backbone 输出获取 learnable queries
learnable_out = backbone_out['learnable_out']  # [B, N_learnable, 2048]

# 分割 queries
bbox_queries = learnable_out[:, :self.num_bbox_mask_queries]      # [B, Q_bbox_mask, C]
heatmap_queries = learnable_out[:, self.num_bbox_mask_queries:]   # [B, Q_heat, C]
heatmap_query = heatmap_queries.mean(dim=1)                        # [B, C] (平均池化)
```

**关键点**：
- `bbox_queries` 用于 bbox 和 mask 预测（共享）
- `heatmap_queries` 用于 heatmap 预测（多个时取平均）

---

## 2. BBox Head 处理机制

### 2.1 BBox Head 架构

`BBoxHead` 接收 `[B, Q, C]` 的 query features，为每个 query 独立预测 bbox：

```python
class BBoxHead(nn.Module):
    def __init__(self, hidden_dim=2048, num_classes=1):
        # 3层 MLP 用于 bbox 回归
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 线性层用于分类/置信度
        self.class_embed = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, query_features: torch.Tensor):
        # query_features: [B, Q, C]
        
        # 预测 bbox: [B, Q, 4] (cx, cy, w, h, normalized to [0,1])
        pred_boxes = self.bbox_embed(query_features).sigmoid()
        
        # 预测置信度: [B, Q, num_classes] -> [B, Q]
        class_logits = self.class_embed(query_features)
        bbox_scores = class_logits.squeeze(-1).sigmoid()
        
        return {
            'pred_boxes': pred_boxes,      # [B, Q, 4]
            'bbox_scores': bbox_scores,     # [B, Q]
            'class_logits': class_logits,   # [B, Q, num_classes]
        }
```

### 2.2 多 Query 的输出

**当 `num_bbox_mask_queries=1` 时**：
- `pred_boxes`: `[B, 1, 4]` - 只有一个 bbox 预测
- `bbox_scores`: `[B, 1]` - 只有一个置信度分数

**当 `num_bbox_mask_queries=2` 时**：
- `pred_boxes`: `[B, 2, 4]` - 两个独立的 bbox 预测
- `bbox_scores`: `[B, 2]` - 两个置信度分数

**当 `num_bbox_mask_queries=4` 时**：
- `pred_boxes`: `[B, 4, 4]` - 四个独立的 bbox 预测
- `bbox_scores`: `[B, 4]` - 四个置信度分数

---

## 3. Mask Head 处理机制

### 3.1 Mask Head 架构

`SAMMaskHead` 接收 `[B, Q, C]` 的 query features，为每个 query 独立预测 mask：

```python
class SAMMaskHead(nn.Module):
    def __init__(self, hidden_dim=2048, output_size=518):
        # 上采样层: 37x37 -> 74x74 -> 148x148
        self.output_upscaling = nn.Sequential(...)
        # 共享的 hypernetwork MLP: query -> dynamic kernel
        self.output_hypernetwork_mlp = MLP(hidden_dim, hidden_dim, hidden_dim // 8, 3)
        # IoU 预测头
        self.iou_prediction_head = MLP(hidden_dim, hidden_dim // 4, 1, 3)
    
    def forward(self, query_tokens, spatial_features, spatial_size):
        # query_tokens: [B, Q, C]
        # spatial_features: [B, P, C] (P = H*W = 37*37 = 1369)
        
        # 1. 上采样空间特征: [B, C, 37, 37] -> [B, C//8, 148, 148]
        src = spatial_features.permute(0, 2, 1).view(B, C, H, W)
        upscaled = self.output_upscaling(src)
        
        # 2. Hypernetwork: query -> dynamic kernel
        hyper_in = self.output_hypernetwork_mlp(query_tokens)  # [B, Q, C//8]
        
        # 3. 点积生成 mask: [B, Q, C//8] @ [B, C//8, H*W] -> [B, Q, 148, 148]
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        
        # 4. 插值到输出尺寸: [B, Q, 518, 518]
        mask_logits = F.interpolate(masks, size=(518, 518), mode='bilinear')
        
        # 5. IoU 预测
        iou_pred = self.iou_prediction_head(query_tokens).squeeze(-1)  # [B, Q]
        
        return {
            'mask_logits': mask_logits,    # [B, Q, 518, 518]
            'mask_pred': mask_logits.sigmoid(),  # [B, Q, 518, 518]
            'iou_pred': iou_pred,          # [B, Q]
        }
```

### 3.2 多 Query 的输出

**当 `num_bbox_mask_queries=1` 时**：
- `mask_logits`: `[B, 1, 518, 518]` - 只有一个 mask 预测

**当 `num_bbox_mask_queries=2` 时**：
- `mask_logits`: `[B, 2, 518, 518]` - 两个独立的 mask 预测

**当 `num_bbox_mask_queries=4` 时**：
- `mask_logits`: `[B, 4, 518, 518]` - 四个独立的 mask 预测

**关键设计**：
- 所有 queries **共享同一个 hypernetwork MLP**（`output_hypernetwork_mlp`）
- 每个 query 通过共享的 MLP 生成不同的 dynamic kernel
- 每个 query 独立预测一个 mask

---

## 4. Loss 设计机制

### 4.1 Hungarian Matching（匈牙利匹配）

在单目标检测场景中（每个图像只有一个 ground truth），需要将多个预测 queries 匹配到唯一的 ground truth。

```python
class HungarianMatcher(nn.Module):
    def forward(self, outputs, targets):
        # outputs: {'pred_boxes': [B, Q, 4], 'class_logits': [B, Q, 1]}
        # targets: {'sat_bbox': [B, 4]}
        
        B, Q = outputs['pred_boxes'].shape[:2]
        indices = []
        
        for b in range(B):
            pred_b = outputs['pred_boxes'][b]  # [Q, 4]
            tgt_b = targets['sat_bbox'][b:b+1]  # [1, 4]
            
            # 计算成本矩阵: [Q, 1]
            cost_class = -logits_b[:, 0:1]  # 分类成本
            cost_bbox = torch.cdist(pred_b, tgt_b, p=1)  # L1 距离
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)  # GIoU 成本
            
            C = cost_bbox * 5.0 + cost_giou * 2.0 + cost_class * 1.0
            
            # 匈牙利算法匹配
            row_ind, col_ind = linear_sum_assignment(C)  # [Q] -> [1]
            
            # 单目标场景：只保留成本最低的匹配
            if len(col_ind) > 1 and targets['sat_bbox'].shape[1] == 1:
                best_match_idx = np.argmin(C[row_ind, col_ind])
                row_ind = np.array([row_ind[best_match_idx]])
                col_ind = np.array([col_ind[best_match_idx]])
            
            indices.append((row_ind, col_ind))
        
        return indices  # List[(pred_idx, gt_idx)]
```

**匹配结果示例**（`Q=2`，单目标）：
- Query 0 成本: 0.3
- Query 1 成本: 0.1 ← **最佳匹配**
- 返回: `[(tensor([1]), tensor([0]))]` - Query 1 匹配到 GT 0

### 4.2 BBox Loss 计算

```python
def _compute_bbox_loss(self, outputs, targets, indices):
    pred_boxes = outputs['pred_boxes']  # [B, Q, 4]
    target_boxes = targets['sat_bbox']  # [B, 4]
    
    # 提取匹配的预测
    src_idx = torch.cat([src for (src, _) in indices])  # [B] (每个样本一个匹配的 query)
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    
    src_boxes = pred_boxes[batch_idx, src_idx]  # [B, 4] - 只取匹配的 query
    tgt_boxes = target_boxes[batch_idx]         # [B, 4]
    
    # L1 Loss
    loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none').sum() / B
    
    # GIoU Loss
    src_xyxy = box_cxcywh_to_xyxy(src_boxes)
    tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
    giou_matrix = generalized_box_iou(src_xyxy, tgt_xyxy)
    loss_giou = (1 - torch.diag(giou_matrix)).sum() / B
    
    # Classification Loss (所有 queries 都参与)
    pred_logits = outputs['class_logits']  # [B, Q, 1]
    target_classes = torch.zeros_like(pred_logits)  # [B, Q, 1]
    
    # 匹配的 query 标记为正样本 (1.0)，未匹配的标记为负样本 (0.0)
    for b_idx, (src, _) in enumerate(indices):
        target_classes[b_idx, src, :] = 1.0
    
    loss_class = sigmoid_focal_loss(
        pred_logits.flatten(0, 1),
        target_classes.flatten(0, 1),
    ) / B
    
    return {
        'loss_bbox': loss_bbox,
        'loss_giou': loss_giou,
        'loss_class': loss_class,
    }
```

**关键点**：
- **BBox/GIoU Loss**: 只对**匹配的 query** 计算（`src_boxes`）
- **Classification Loss**: **所有 queries** 都参与，匹配的为正样本，未匹配的为负样本

### 4.3 Mask Loss 计算

```python
def _compute_mask_loss(self, outputs, targets, indices):
    mask_logits = outputs['mask_logits']  # [B, Q, 518, 518]
    target_mask = targets['sat_mask']      # [B, 1, H, W]
    
    # 单目标场景：选择匹配的 query 对应的 mask
    if indices is not None and mask_logits.shape[1] > 1:
        selected_masks = []
        for b, (src_idx, _) in enumerate(indices):
            match_idx = int(src_idx[0].item())  # 匹配的 query 索引
            selected_masks.append(mask_logits[b, match_idx])
        mask_logits = torch.stack(selected_masks, dim=0)  # [B, 518, 518]
    else:
        mask_logits = mask_logits[:, 0]  # [B, 518, 518]
    
    # 调整 target mask 尺寸
    target_mask = F.interpolate(target_mask.float(), size=(518, 518))
    target_mask = target_mask.squeeze(1)  # [B, 518, 518]
    target_mask = (target_mask > 0.5).float()
    
    # BCE Loss
    loss_mask_bce = mask_bce_loss(mask_logits, target_mask)
    
    # Dice Loss
    mask_prob = mask_logits.sigmoid()
    loss_mask_dice = dice_loss(mask_prob, target_mask)
    
    return {
        'loss_mask_bce': loss_mask_bce,
        'loss_mask_dice': loss_mask_dice,
    }
```

**关键点**：
- **Mask Loss**: 只对**匹配的 query** 对应的 mask 计算
- 未匹配的 queries 的 mask **不参与 loss 计算**（但它们的 query features 仍然通过 bbox classification loss 学习）

---

## 5. 完整流程示例

### 5.1 Query_len=2 的完整流程

**输入**：
- `learnable_out`: `[B, 3, 2048]` (2 bbox/mask queries + 1 heatmap query)

**Step 1: Query 分割**
```python
bbox_queries = learnable_out[:, :2]      # [B, 2, 2048]
heatmap_queries = learnable_out[:, 2:]   # [B, 1, 2048]
```

**Step 2: BBox Head**
```python
bbox_outputs = bbox_head(bbox_queries)
# pred_boxes: [B, 2, 4]
# bbox_scores: [B, 2]
# class_logits: [B, 2, 1]
```

**Step 3: Mask Head**
```python
mask_outputs = mask_head(bbox_queries, sate_features)
# mask_logits: [B, 2, 518, 518]
# mask_pred: [B, 2, 518, 518]
# iou_pred: [B, 2]
```

**Step 4: Hungarian Matching**
```python
indices = matcher(outputs, targets)
# 假设: [(tensor([1]), tensor([0]))] - Query 1 匹配到 GT
```

**Step 5: Loss 计算**
```python
# BBox Loss: 只对 Query 1 计算
src_boxes = pred_boxes[:, 1]  # [B, 4]
loss_bbox = L1(src_boxes, gt_bbox)
loss_giou = GIoU(src_boxes, gt_bbox)

# Classification Loss: 所有 queries 参与
# Query 0: 负样本 (target=0.0)
# Query 1: 正样本 (target=1.0)
loss_class = FocalLoss([logits_0, logits_1], [0.0, 1.0])

# Mask Loss: 只对 Query 1 的 mask 计算
selected_mask = mask_logits[:, 1]  # [B, 518, 518]
loss_mask_bce = BCE(selected_mask, gt_mask)
loss_mask_dice = Dice(selected_mask, gt_mask)
```

### 5.2 为什么多 Query 能提升性能？

1. **冗余性**：多个 queries 提供多个预测候选，增加找到正确匹配的概率
2. **专业化**：不同 queries 可能学习到不同的特征表示（通过分类 loss 的正负样本区分）
3. **鲁棒性**：即使某个 query 预测失败，其他 queries 仍可能成功
4. **评估时选择最佳**：评估时使用置信度最高的 query，而不是固定的第一个

---

## 6. 评估时的 Query 选择

### 6.1 问题

训练时所有 queries 都参与学习，但评估时需要选择一个最佳预测。

### 6.2 解决方案

在 `evaluate_custom_v2.py` 中，使用置信度分数选择最佳 query：

```python
# 旧代码（错误）：总是使用第一个 query
pred_bbox_norm = outputs["pred_boxes"][:, 0]
pred_mask = outputs["mask_pred"][:, 0]

# 新代码（正确）：选择置信度最高的 query
if pred_boxes.shape[1] > 1 and "bbox_scores" in outputs:
    bbox_scores = outputs["bbox_scores"]  # [B, Q]
    best_query_idx = bbox_scores.argmax(dim=1)  # [B]
    pred_bbox_norm = pred_boxes[torch.arange(B), best_query_idx]
    pred_mask = pred_masks[torch.arange(B), best_query_idx]
else:
    pred_bbox_norm = pred_boxes[:, 0]
    pred_mask = pred_masks[:, 0]
```

**关键点**：
- 评估时使用 `bbox_scores` 选择置信度最高的 query
- 这确保了评估结果反映模型的最佳预测能力

---

## 7. 总结

### 7.1 Query 数量对模型的影响

| Query 数量 | BBox 预测 | Mask 预测 | 匹配策略 | Loss 计算 |
|-----------|----------|-----------|---------|-----------|
| Q=1 | [B, 1, 4] | [B, 1, 518, 518] | 直接匹配 | 全部参与 |
| Q=2 | [B, 2, 4] | [B, 2, 518, 518] | 匈牙利匹配（选最佳） | 匹配的参与 |
| Q=4 | [B, 4, 4] | [B, 4, 518, 518] | 匈牙利匹配（选最佳） | 匹配的参与 |

### 7.2 关键设计原则

1. **共享 vs 独立**：
   - BBox Head: 每个 query 独立预测（独立 MLP）
   - Mask Head: 共享 hypernetwork，但每个 query 生成不同的 kernel

2. **Loss 策略**：
   - BBox/Mask Loss: 只对匹配的 query 计算（避免多 query 学习同一目标）
   - Classification Loss: 所有 queries 参与（区分正负样本）

3. **评估策略**：
   - 使用置信度分数选择最佳 query，而不是固定的第一个

### 7.3 性能提升机制

- **训练时**：多个 queries 通过分类 loss 学习不同的表示（正样本 vs 负样本）
- **评估时**：选择置信度最高的 query，充分利用模型的预测能力
- **结果**：多 query 配置通常比单 query 配置性能更好
