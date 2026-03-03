# Implementation Details (Current `location_v4` Setup)

本文档总结当前代码与脚本下的**实际可复现实现细节**（对应 `configs/default_v3.yaml` + `scripts/slurm_train_accelerate_v3_ablation_4_all_on.sh` + `train_detr_v2.py`）。

## 1. 训练环境与硬件

- **硬件**：1 节点，8×GPU（SLURM: `--gres=gpu:8`），64 CPU cores，512GB 内存。
- **分布式**：HuggingFace Accelerate + DeepSpeed ZeRO-2（`configs/accelerate_deepspeed_zero2.yaml`）。
- **混合精度**：BF16（Accelerate 和训练配置均为 `bf16`）。
- **环境**：Conda `filtre`，`module load cuda`。
- **随机种子**：训练脚本内设定 `set_seed(42)`。

## 2. 数据与预处理

- **训练集**：`${ROOT_DIR}/data/json/train_all.json`
- **验证集**：`${ROOT_DIR}/data/json/val_all.json`
- **数据根目录**：`${ROOT_DIR}/data/urban`
- **输入分辨率**：`518 × 518`（`img_size=518`, `crop_size=518`）
- **DataLoader**：
  - `batch_size=3`（per-GPU）
  - `num_workers=24`
  - `pin_memory=True`
  - 训练集 `drop_last=True`, 验证集 `drop_last=False`
  - `persistent_workers=True`（当 `num_workers>0`）
  - `prefetch_factor=4`

> 若 8 卡均参与训练，理论全局 batch size 为 `3 × 4 × 8 = 96`（包含 `gradient_accumulation_steps=4` 的等效批量）。

## 3. Prompt 训练策略

训练时调用 `prepare_random_prompt(...)`，并固定为**单一提示类型互斥采样**（每个 batch 三选一）：

- point prompt（来自 `mono_point`）
- bbox prompt（来自 `mono_bbox`，并转换到像素空间 `xywh`）
- mask prompt（来自 `mono_mask`）

验证阶段默认使用单一 point prompt（`prepare_single_prompt` 路径）。

## 4. 模型配置（V4 unified backbone）

- **Backbone**：Pi3 large decoder（`decoder_size=large`）
- **Learnable queries**：
  - `num_bbox_mask_queries=1`
  - `num_heatmap_queries=1`
  - 总 learnable queries = 2
- **深监督**：`supervision_layers=[4,11,17]`, `supervision_weights=[0.1,0.3,0.6]`
- **Prompt encoder**：SAM2 权重 `${ROOT_DIR}/ckpt/sam2.1_hiera_large.pt`，`sam_embed_dim=256`
- **对比学习**（默认开启）：
  - `contrastive_proj_dim=256`
  - `contrastive_queue_size=16384`
  - `contrastive_momentum=0.999`
  - `contrastive_temperature=0.07`

### 冻结策略（当前默认）

- `freeze_dinov2=true`：冻结 DINOv2 encoder
- `freeze_prompt_encoder=true`：冻结 SAM prompt encoder 主体（保留投影层可训练）
- `freeze_mask_conv=true`：冻结 SAM mask downscaling conv
- `freeze_decoder=false`：Pi3 decoder 参与训练

## 5. 优化器与学习率

- **优化器**：AdamW
- **权重衰减**：`1e-4`
- **三组学习率**（`get_param_groups`）：
  1. backbone：`lr_backbone=1e-5`
  2. new tokens（learnable queries / projection 等）：`lr_new_tokens=5e-4`
  3. task heads：`lr_heads=1e-4`
- **梯度裁剪**：`max_norm=1.0`
- **学习率调度**：cosine + warmup
  - `num_epochs=30`
  - `warmup_epochs=5`
  - `min_lr=1e-6`（配置项，调度器名义下限）

## 6. Loss 设计与权重（当前实现）

总损失为多任务加权和（含深监督中间层项）：

- bbox L1：`weight_bbox=5.0`
- GIoU：`weight_giou=2.0`
- classification focal：`weight_class=2.0`
- mask BCE：`weight_mask_bce=2.0`
- mask Dice：`weight_mask_dice=5.0`
- heatmap focal：`weight_heatmap=0.1`
- rotation：`weight_rotation=0.1`
- contrastive：`weight_contrastive=0.1`

### Heatmap loss（当前稳定版实现）

使用 CornerNet 风格逐像素 focal loss，且为数值稳定做了两点实现约束：

1. **FP32 loss path**：在 BF16 训练下，`heatmap_logits` 转 FP32 计算 loss。  
2. **logits-stable 对数项**：
   - `log(sigmoid(x)) = -softplus(-x)`
   - `log(1 - sigmoid(x)) = -softplus(x)`

并显式将目标位置对应最近网格中心置为正样本（`target=1`），确保每个样本至少 1 个正点。默认超参：

- `heatmap_sigma=0.05`
- `heatmap_focal_alpha=2.0`
- `heatmap_focal_beta=4.0`

训练日志额外监控：

- `heatmap_center_prob`（中心正样本概率均值）
- `pos_error` / `pos_error_px`

## 7. Deep Supervision 细节

- 在 stage `4/11/17` 计算中间层 bbox/mask loss。
- 在早中层（`4/11`）额外计算中间层 heatmap + rotation loss。
- 各层 loss 先乘层权重 `0.1/0.3/0.6`，再乘对应任务权重并汇总。

## 8. 训练脚本与实验开关（ablation_4_all_on）

`scripts/slurm_train_accelerate_v3_ablation_4_all_on.sh` 额外显式指定：

- `--use_deep_supervision true`
- `--use_contrastive_loss true`
- `--use_rot_pos_supervision true`

同时输出目录固定为：

- `output_v3/ablation_4_all_on`

## 9. Checkpoint 与验证策略

- 每 `save_freq=5` epoch 存档。
- 每个 epoch 验证（`val_freq=1`）。
- 用**全局 reduce 后的 val loss**决定 best checkpoint（避免多卡分歧）。
- 若 `checkpoint.resume` 路径不存在，会自动回退为从头训练。

## 10. 复现命令（当前推荐）

```bash
export ROOT_DIR=/data/home/scxi704/run/xhj
export WORKSPACE_NAME=location_v4
sbatch /data/home/scxi704/run/xhj/location_v4/scripts/slurm_train_accelerate_v3_ablation_4_all_on.sh
```

