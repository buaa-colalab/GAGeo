# 数据处理 Pipeline 详解

本文档详细说明跨视角定位系统的数据处理流程，从原始数据到模型输入的完整转换过程。

---

## 概览

数据处理的核心目标是将原始的街景图和卫星图转换为模型可用的输入格式，同时保证：
1. **Prompt 输入**符合 SAM 的位置编码要求（像素坐标）
2. **Target 输出**符合 DETR 的训练范式（归一化坐标）
3. **双向定位**支持任意方向的跨视角匹配

### 关键设计决策

| 数据类型 | 输入格式 | 输出格式 | 用途 |
|---------|---------|---------|------|
| **Prompt BBox** | `[x, y, w, h]` 像素坐标 | `[x, y, w, h]` 像素坐标 | SAM Prompt Encoder |
| **Prompt Point** | `[x, y]` 像素坐标 | `[x, y]` 像素坐标 | SAM Prompt Encoder |
| **Prompt Mask** | `[H, W]` 二值 mask | `[1, H, W]` float tensor | SAM Prompt Encoder |
| **Target BBox** | `[x, y, w, h]` 像素坐标 | `[cx, cy, w, h]` 归一化 [0,1] | DETR Loss 计算 |
| **Camera Position** | `[x, y]` 像素坐标 | `[x, y]` 归一化 [0,1] | Heatmap 回归 |

---

## 数据流程图

```
原始数据 (JSON)
    ├─ mono_filename: "xxx.jpg"
    ├─ sat_filename: "xxx.jpg"
    ├─ mono_point: [x, y]
    ├─ mono_bbox: [x, y, w, h]
    ├─ mono_segmentation: RLE
    ├─ sate_bbox: [x, y, w, h]
    ├─ sate_segmentation: RLE
    ├─ camera_position: [x, y]
    └─ rotation: yaw_degrees

         ↓

Step 1: 加载图像
    ├─ Mono: PIL Image (原始尺寸) → Resize to 518×518
    └─ Sat: PIL Image (1280×1280) → Crop to 518×518

         ↓

Step 2: 解码 Mask
    ├─ Mono: RLE → Binary Mask [518, 518]
    └─ Sat: RLE → Binary Mask [1280, 1280]

         ↓

Step 3: Resize Mono & Crop Sat
    ├─ Mono: Resize 图像到 518×518，缩放坐标、bbox、mask
    └─ Sat:  Crop 图像到 518×518，平移坐标、bbox、mask

         ↓

         ↓

Step 4:  Target_bbox，camara_position归一化（用于训练）
    ├─ Prompt: 保持像素坐标
    └─ Target: 归一化到 [0, 1]

         ↓

Step 5: 根据方向分配 Prompt 和 Target
    ├─ mono_to_sat: Prompt 来自 mono，Target_bbox 在 sat，target_position始终在sat
    └─ sat_to_mono: Prompt 来自 sat，Target_bbox 在 mono，target_position始终在sat

         ↓

模型输入
    ├─ mono_view: [B, 3, 518, 518]
    ├─ sat_view: [B, 3, 518, 518]
    ├─ prompt_point: [B, 2] 像素坐标
    ├─ prompt_bbox: [B, 4] 像素坐标 [x, y, w, h]
    ├─ prompt_mask: [B, 1, 518, 518]
    ├─ target_bbox: [B, 4] 归一化 [cx, cy, w, h]
    ├─ target_position: [B, 2] 归一化 [x, y]
    └─ yaw_radians: [B] 始终为mono相对于sat的旋转角度
```

---

## 详细处理步骤

### Step 1: 图像加载与 Resize

#### Mono 图像处理
```python
# 加载
mono_img = Image.open(mono_path).convert('RGB')  # 原始尺寸

# Resize 到 518×518
mono_img = mono_img.resize((518, 518), Image.BILINEAR)

# 坐标缩放
scale_x = 518 / original_width
scale_y = 518 / original_height
mono_point = mono_point * [scale_x, scale_y]
mono_bbox = mono_bbox * [scale_x, scale_y, scale_x, scale_y]
```

**关键点**：
- Mono 图使用 **Resize**（不是 crop），保持完整视野
- 所有坐标按比例缩放
- Mask 使用 `INTER_NEAREST` 保持二值性

#### Sat 图像处理
```python
# 加载
sat_img = Image.open(sat_path).convert('RGB')  # 1280×1280

# Crop 到 518×518（训练时随机，测试时中心）
if random_crop:
    # 确保 bbox 和 camera_position 都在 crop 区域内
    left, top = compute_valid_crop_range(sat_bbox, camera_position)
else:
    # 中心 crop
    left = (1280 - 518) // 2
    top = (1280 - 518) // 2

sat_img = sat_img.crop((left, top, left+518, top+518))

# 坐标调整
crop_offset = [left, top]
sat_bbox = sat_bbox - [left, top, 0, 0]
camera_position = camera_position - [left, top]
```

**关键点**：
- Sat 图使用 **Crop**（不是 resize），保持分辨率
- Crop 时确保目标物体和相机位置都在视野内
- 训练时随机 crop 增强泛化能力

### Step 2: Mask 解码

#### Mono Mask (RLE 格式)
```python
from pycocotools import mask as mask_utils

# RLE 格式示例
mono_segmentation = {
    "size": [518, 518],
    "counts": "compressed_string..."
}

# 解码
mono_mask = mask_utils.decode(mono_segmentation)  # [518, 518] uint8
```

#### Sat Mask (Polygon 格式)
```python
# Polygon 格式示例
sate_segmentation = [[x1, y1, x2, y2, x3, y3, ...]]

# 解码（通过 pycocotools）
rle = mask_utils.frPyObjects(sate_segmentation, height, width)
sat_mask = mask_utils.decode(rle)  # [1280, 1280] uint8

# Crop mask（与图像 crop 对应）
sat_mask = sat_mask[top:top+518, left:left+518]
```

### Step 3: 转换为 Tensor

```python
import torchvision.transforms.functional as TF

# 图像转 tensor（自动归一化到 [0, 1]）
mono_tensor = TF.to_tensor(mono_img)  # [3, 518, 518]
sat_tensor = TF.to_tensor(sat_img)    # [3, 518, 518]

# Mask 转 tensor
mono_mask_tensor = torch.from_numpy(mono_mask).unsqueeze(0).float()  # [1, 518, 518]
sat_mask_tensor = torch.from_numpy(sat_mask).unsqueeze(0).float()    # [1, 518, 518]
```

### Step 4: BBox 归一化（仅 Target）

```python
def _normalize_bbox(bbox, img_w, img_h):
    """
    输入: [x, y, w, h] 像素坐标（左上角 + 宽高）
    输出: [cx, cy, w, h] 归一化到 [0, 1]（中心点 + 宽高）
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [cx, cy, w_norm, h_norm]

# 归一化 target bbox
mono_bbox_norm = _normalize_bbox(mono_bbox, 518, 518)
sat_bbox_norm = _normalize_bbox(sat_bbox, 518, 518)

# 归一化 camera position
camera_position_norm = camera_position / [518, 518]
```

**为什么 Prompt 不归一化？**
- SAM 的 `PositionEmbeddingRandom` 需要像素坐标计算位置编码
- `pe_layer.forward_with_coords(coords, input_image_size)` 期望像素值
- 位置编码公式：`PE = sin/cos(coord / temperature)`，需要绝对位置信息

**为什么 Target 要归一化？**
- DETR 风格模型通常输出归一化坐标
- 归一化后 L1 loss 更稳定（不同尺度的 w/h 权重相同）
- GIoU loss 在归一化空间计算是尺度不变的

---




---

## 双向定位机制

### 方向说明

系统支持三种定位方向：

1. **`mono_to_sat`**: 在 mono 图上给 prompt，在 sat 图上定位目标
2. **`sat_to_mono`**: 在 sat 图上给 prompt，在 mono 图上定位目标
3. **`both`**: 训练时随机选择方向

### Prompt 和 Target 分配逻辑

```python
if direction == 'mono_to_sat':
    # Prompt 来自 mono 图
    prompt_point = mono_point      # 像素坐标
    prompt_bbox = mono_bbox        # 像素坐标 [x, y, w, h]
    prompt_mask = mono_mask        # [1, 518, 518]
    
    # Target 在 sat 图
    target_bbox = sat_bbox_norm    # 归一化 [cx, cy, w, h]
    
elif direction == 'sat_to_mono':
    # Prompt 来自 sat 图
    prompt_point = sat_point       # 像素坐标（已调整 crop offset）
    prompt_bbox = sat_bbox         # 像素坐标 [x, y, w, h]
    prompt_mask = sat_mask         # [1, 518, 518]（已 crop）
    
    # Target 在 mono 图
    target_bbox = mono_bbox_norm   # 归一化 [cx, cy, w, h]

# Camera position 始终在 sat 图上（卫星图范围更广）
target_position = camera_position_norm  # 归一化 [x, y]
```

### 为什么 camera_position 始终在 sat 图？

- 卫星图覆盖范围更广，可以看到相机的拍摄位置
- Mono 图是从相机视角拍摄的，无法标注自己的位置
- 定位任务的目标是找到"相机在哪里拍的"，这个位置在卫星图上

---

## 代码示例

### 完整的数据加载示例

```python
from torch.utils.data import DataLoader
from data.dataset import CrossViewDataset, collate_fn

# 创建数据集
dataset = CrossViewDataset(
    json_path='/data/xhj/location/data/results_filter.json',
    data_root='/data/GoogleEarth',
    mono_size=518,
    sat_size=1280,
    crop_sat=True,
    crop_size=518,
    random_crop=True,      # 训练时随机 crop
    direction='both',       # 双向训练
    test_mode=False,
)

# 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

# 获取一个 batch
batch = next(iter(loader))

# 查看数据形状
print(f"Mono view: {batch['mono_view'].shape}")          # [8, 3, 518, 518]
print(f"Sat view: {batch['sat_view'].shape}")            # [8, 3, 518, 518]
print(f"Prompt point: {batch['prompt_point'].shape}")    # [8, 2]
print(f"Prompt bbox: {batch['prompt_bbox'].shape}")      # [8, 4]
print(f"Prompt mask: {batch['prompt_mask'].shape}")      # [8, 1, 518, 518]
print(f"Target bbox: {batch['target_bbox'].shape}")      # [8, 4]
print(f"Target position: {batch['target_position'].shape}")  # [8, 2]
print(f"Directions: {batch['directions']}")              # ['mono_to_sat', 'sat_to_mono', ...]
```

### 模型输入准备

```python
# 准备 prompt（支持任意组合）
points = (batch['prompt_point'], torch.ones(B, 1))  # (coords, labels)
boxes = batch['prompt_bbox']                         # [B, 4]
masks = batch['prompt_mask']                         # [B, 1, 518, 518]

# 模型前向传播
outputs = model(
    mono_view=batch['mono_view'],
    sat_view=batch['sat_view'],
    points=points,
    boxes=boxes,
    masks=masks,
    prompt_views=batch['prompt_views'],
)

# 计算 loss
targets = {
    'target_bbox': batch['target_bbox'],
    'camera_position': batch['target_position'],
    'yaw_radians': batch['yaw_radians'],
}
losses = criterion(outputs, targets)
```

---

## 数据增强

### 当前支持的增强

1. **Sat 图随机 Crop**
   - 训练时：随机位置 crop，确保目标在视野内
   - 测试时：中心 crop

2. **双向定位随机采样**
   - `direction='both'` 时，每个 epoch 随机选择方向
   - 增强模型的双向定位能力





---

## 参考资料

- [SAM2 Prompt Encoder](https://github.com/facebookresearch/segment-anything-2)
- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [PyTorch Data Loading](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
