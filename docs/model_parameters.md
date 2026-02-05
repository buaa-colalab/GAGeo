# Model Parameters Guide

## Two-Stage Cross-Attention Architecture Parameters

### Stage 1: Intent Formation (Prompt Fusion)

**`num_intent_queries`** (default: 32)
- 可学习的 Intent Queries 数量
- 这些 queries 通过 cross-attention 从前视图特征中提取目标信息
- 输出: `intent_features [B, num_intent_queries, C]`

**`prompt_fusion_layers`** (default: 3)
- Stage 1 中 TransformerDecoder 的层数
- 每层包含 self-attention + cross-attention + FFN
- 层数越多，模型容量越大，但计算成本也越高

### Stage 2: View Conditioning (Query Decoder)

**`num_object_queries`** (default: 10)
- Object detection queries 数量
- 用于 bbox 预测任务
- 输出: `obj_features [B, num_object_queries, C]`

**`num_location_queries`** (default: 16)
- Location queries 数量
- 用于 heatmap 位置预测任务
- 输出: `loc_features [B, num_location_queries, C]`

**`num_decoder_layers`** (default: 6)
- Stage 2 中 TransformerDecoder 的层数
- Intent Features + Object/Location Queries 通过这些层 cross-attend 到卫星图特征

### Shared Transformer Parameters

**`num_heads`** (default: 8)
- Multi-head attention 的 head 数量
- **注意**: 这是 attention heads 数量，不是可学习 token 数量！
- Stage 1 和 Stage 2 共用此参数
- 必须能整除 `hidden_dim` (2048)，推荐值: 8, 16

**`dropout`** (default: 0.1)
- Dropout rate，用于正则化
- 应用于 attention 和 FFN 层

### Backbone Parameters

**`decoder_size`** (default: 'large')
- Pi3 decoder 大小: 'small', 'base', 'large'
- 决定 `hidden_dim`: small=1024, base=1536, large=2048

**`freeze_dinov2`** (default: true)
- 是否冻结 DINOv2 encoder
- 建议冻结以节省显存和加速训练

**`freeze_decoder`** (default: false)
- 是否冻结 Pi3 decoder
- 建议保持可训练

## Parameter Tuning Guide

### 显存优化
- 减少 `num_intent_queries`: 32 → 16
- 减少 `prompt_fusion_layers`: 3 → 2
- 减少 `num_decoder_layers`: 6 → 4
- 减少 `num_heads`: 8 → 4

### 性能优化
- 增加 `num_intent_queries`: 32 → 64 (更丰富的 intent 表示)
- 增加 `prompt_fusion_layers`: 3 → 4 (更强的前视图特征提取)
- 增加 `num_decoder_layers`: 6 → 8 (更强的卫星图特征融合)

### 任务特定优化
- **Bbox detection**: 增加 `num_object_queries`
- **Location heatmap**: 增加 `num_location_queries`

## Example Configurations

### Lightweight (低显存)
```yaml
model:
  num_intent_queries: 16
  num_object_queries: 5
  num_location_queries: 8
  num_heads: 4
  prompt_fusion_layers: 2
  num_decoder_layers: 4
  dropout: 0.1
```

### Default (平衡)
```yaml
model:
  num_intent_queries: 32
  num_object_queries: 10
  num_location_queries: 16
  num_heads: 8
  prompt_fusion_layers: 3
  num_decoder_layers: 6
  dropout: 0.1
```

### High-capacity (高性能)
```yaml
model:
  num_intent_queries: 64
  num_object_queries: 20
  num_location_queries: 32
  num_heads: 16
  prompt_fusion_layers: 4
  num_decoder_layers: 8
  dropout: 0.1
```
