# Cross-View Localization

基于VGGT的跨视角定位系统，从前视图图像定位目标在卫星图像中的位置。

## 目录结构

```
location/
├── models/                 # 模型定义
│   ├── cross_view_localizer_v2.py  # 主模型
│   ├── vggt_aggregator.py          # VGGT backbone
│   ├── heads/                      # 任务头
│   │   ├── bbox_head.py            # BBox预测
│   │   ├── yaw_head.py             # Yaw角度预测
│   │   ├── position_head.py        # 位置预测
│   │   └── multi_task_head.py      # 多任务头
│   └── layers/                     # 基础层 (ViT, Attention等)
├── data/                   # 数据处理
│   ├── dataset.py          # Dataset类 (支持crop增强)
│   └── *.json              # 数据标注文件
├── utils/                  # 工具函数
│   ├── losses.py           # MultiTaskLoss (bbox/mask/yaw/position)
│   ├── metrics.py          # 评估指标 (IoU/AP/距离误差)
│   ├── box_ops.py          # BBox操作
│   └── weight_loader.py    # 权重加载 (VGGT/DINOv2)
├── configs/                # 配置文件
│   ├── default.yaml        # 默认配置（推荐用于DeepSpeed训练）
│   ├── test.yaml           # 过拟合测试配置
│   └── accelerate_deepspeed_zero2.yaml  # Accelerate配置
├── scripts/                # 训练脚本
│   ├── train_accelerate.sh # Accelerate + DeepSpeed训练（推荐）
│   ├── train_ddp.sh        # DDP训练
│   └── train_single.sh     # 单卡训练
├── ckpt/                   # 预训练权重 (手动下载)
├── output/                 # 训练输出
├── train_accelerate.py     # Accelerate训练脚本（推荐）
├── train.py                # DDP训练脚本
└── test.py                 # 测试脚本
```

## 预训练权重

将预训练权重下载到 `ckpt/` 目录：

```bash
# VGGT权重
wget -O ckpt/vggt.pth <vggt_url>

# DINOv2权重 (可选，VGGT已包含)
wget -O ckpt/dinov2_vitl14_reg.pth <dinov2_url>
```

## 数据格式

JSON标注文件格式：
```json
{
  "mono_filename": "city/mono/xxx.jpg",
  "sat_filename": "city/satellite/xxx.jpg",
  "mono_point": [x, y],
  "mono_bbox": [x, y, w, h],
  "sate_bbox": [cx, cy, w, h],
  "rotation": 45.0,
  "camera_position": [cx, cy]
}
```

## 快速开始

### 推荐：Accelerate + DeepSpeed 训练

**适用场景**：不冻结aggregator，需要显存优化

```bash
# 安装依赖
pip install accelerate deepspeed

# 6卡训练（推荐）
bash scripts/train_accelerate.sh configs/default.yaml "0,1,2,3,4,5"

# 4卡训练
bash scripts/train_accelerate.sh configs/default.yaml "0,1,2,3"

# 恢复训练
bash scripts/train_accelerate.sh configs/default.yaml "0,1,2,3,4,5" --resume output/checkpoint_epoch_10
```

**关键配置** (`configs/default.yaml`)：
```yaml
model:
  freeze_aggregator: false  # 必须设为false才需要DeepSpeed

training:
  batch_size: 6              # 每卡batch size（根据显存调整）
  gradient_accumulation_steps: 4
  mixed_precision: bf16      # RTX 4090推荐bf16
```

### 过拟合测试（验证训练流程）

在单样本上快速过拟合，验证模型和训练代码无误：

```bash
# 2卡过拟合测试
bash scripts/train_accelerate.sh configs/test.yaml "0,1"

# 预期：loss快速降到接近0
```

### 单卡训练（仅用于调试）

```bash
# 使用脚本
bash scripts/train_single.sh configs/default.yaml 7

# 或直接运行
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/default.yaml
```

## 多任务Loss

支持四种监督信号，通过配置开关控制：

| 任务 | Loss | 配置开关 |
|------|------|----------|
| BBox | L1 + GIoU | `enable_bbox` |
| Mask | BCE + Dice | `enable_seg` |
| Yaw | 周期性MSE | `enable_camera` |
| Position | MSE | `enable_position` |

## 配置说明

```yaml
model:
  vggt_weights: ckpt/vggt.pth  # 预训练权重
  enable_bbox: true             # 开启bbox预测
  enable_camera: true           # 开启yaw预测
  enable_position: true         # 开启位置预测
  freeze_patch_embed: true      # 冻结DINOv2

training:
  lr_backbone: 1e-5             # backbone学习率
  lr_heads: 1e-4                # heads学习率
  weight_bbox: 5.0              # bbox loss权重
  weight_yaw: 1.0               # yaw loss权重
```

## 训练配置详解

### Batch Size 调优

**显存利用率参考**（RTX 4090, DeepSpeed ZeRO-2, BF16）：

| batch_size | 显存占用 | 利用率 | 推荐 |
|-----------|---------|--------|------|
| 4 | ~14GB | 58% | 保守 |
| **6** | **~18GB** | **75%** | **推荐** |
| 8 | ~22GB | 90% | 激进（可能OOM） |

**有效batch size计算**：
```
有效batch = batch_size × num_gpus × gradient_accumulation_steps
示例: 6 × 6 × 4 = 144
```

**调优原则**：
- ✅ 目标显存利用率：80-85%（留15-20%缓冲）
- ✅ 有效batch size：64-256（视觉任务推荐范围）
- ❌ 不是越大越好：过大batch会降低泛化能力

### 混合精度训练

**推荐使用 BF16**（适用于 RTX 3090/4090, A100, H100）：

| 精度类型 | 优势 | 劣势 | 推荐场景 |
|---------|------|------|---------|
| **BF16** | ✅ 数值稳定，无需loss scaling<br>✅ 动态范围大，不易溢出<br>✅ 代码简洁 | ❌ 需要Ampere+架构 | **RTX 4090 (推荐)** |
| FP16 | ✅ 所有GPU支持 | ❌ 需要loss scaling<br>❌ 易溢出，需要更多dtype转换 | 旧GPU (V100等) |

配置方式（`configs/default.yaml`）：
```yaml
training:
  use_amp: true
  mixed_precision: bf16  # 或 "fp16"
```

### DeepSpeed ZeRO-2 显存优化

- **优化内容**: 优化器状态分片（每个GPU只存储部分优化器状态）
- **适用场景**: 不冻结aggregator，多卡 RTX 4090
- **显存节省**: 相比DDP节省约40%显存

**关键参数**（在训练配置文件中设置）：
```yaml
model:
  freeze_aggregator: false  # DeepSpeed主要用于此场景

training:
  batch_size: 6                      # 每卡batch size
  gradient_accumulation_steps: 4     # 梯度累积步数
  mixed_precision: bf16              # 混合精度类型
```

## 多卡训练 (DDP)

**适用场景**：冻结aggregator时的轻量训练

```bash
# 使用脚本
bash scripts/train_ddp.sh configs/default.yaml "5,6,7"

# 或直接使用 torchrun
CUDA_VISIBLE_DEVICES=5,6,7 torchrun \
    --nproc_per_node=3 \
    train.py --config configs/default.yaml
```

### TensorBoard 监控

训练日志自动保存到 `output/<exp_name>/logs/` 目录：

```bash
tensorboard --logdir ./output/test/logs --port 6006
# 浏览器访问: http://localhost:6006
```

### 常见问题

**Q: 如何调整 batch size？**
```yaml
training:
  batch_size: 2  # 每卡 batch size
# 总 batch size = batch_size × num_gpus
```

**Q: 显存不足？**
1. 降低 `batch_size`
2. 启用混合精度: `use_amp: true`
3. 冻结更多参数: `freeze_aggregator: true`

## 可视化

```bash
# 使用脚本
bash scripts/visualize.sh ./output/best.pth configs/default.yaml 20 7

# 或直接运行
python vis.py \
    --checkpoint ./output/best.pth \
    --config configs/default.yaml \
    --num_samples 20 \
    --gpu 7
```

输出：
- `sample_XXXX.png` - 每个样本的可视化（前视图+卫星图+预测）
- `summary.png` - 误差统计图表


# 同时输出到终端和日志文件
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 train.py --config configs/test.yaml 2>&1 | tee output/test/train.log

# 或者只保存到文件（不显示在终端）
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 train.py --config configs/test.yaml > output/test/train.log 2>&1