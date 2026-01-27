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
│   ├── default.yaml        # 默认配置
│   └── test.yaml           # 测试配置
├── ckpt/                   # 预训练权重 (手动下载)
├── output/                 # 训练输出
├── train.py                # 训练脚本
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

### 单卡训练

```bash
# 使用脚本（推荐）
bash scripts/train_single.sh configs/test.yaml 7

# 或直接运行
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/test.yaml

# 覆盖配置
python train.py --config configs/default.yaml --batch_size 2 --epochs 50

# 恢复训练
python train.py --config configs/default.yaml --resume output/epoch_10.pth
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

## 显存需求

模型约1.5B参数，训练时显存需求：
- batch_size=1: ~15GB
- batch_size=2: ~22GB
- batch_size=4: ~35GB (需要A100)

建议：启用gradient checkpointing或使用混合精度训练。

## 多卡训练 (DDP)

使用 PyTorch DistributedDataParallel 进行多卡训练：

```bash
# 使用脚本（推荐）
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