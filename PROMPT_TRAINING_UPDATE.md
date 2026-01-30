# Prompt训练更新总结

## 问题和解决方案

### 问题1：代码规范 - Import位置 ✅

**问题**：`dataset.py` 中 `import cv2` 在函数内部多次重复导入，不符合工程规范。

**解决**：
- 将 `import cv2` 移到文件顶部
- 删除所有函数内部的重复 `import cv2`

**修改文件**：`/data/xhj/location/data/dataset.py`

```python
# 文件顶部
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
...
```

---

### 问题2：sat_bbox的缩放逻辑 ✅

**问题**：用户质疑crop后为何还要缩放sat_bbox。

**分析结果**：**当前逻辑是正确的！**

**原因**：
```python
# 当前代码
adj_bbox[:2] = (adj_bbox[:2] - offset) * scale  # 左上角 (x, y)
adj_bbox[2:] = adj_bbox[2:] * scale              # 宽高 (w, h)
```

- bbox格式是 `[x, y, w, h]`
- **左上角坐标**：需要先减去crop的offset（平移），再乘以scale（缩放）
- **宽高**：只需要乘以scale（缩放），不需要减offset

**场景说明**：
1. 原始卫星图：1280x1280
2. Crop到：518x518 (cs = 518)
3. 如果 `cs != crop_size`，需要resize到crop_size
4. scale = crop_size / cs（通常为1.0，因为cs已经等于crop_size）

**结论**：无需修改，逻辑正确。

---

### 问题3：训练时应该随机选择一种prompt输入 ✅

**问题**：真实场景下，用户只会给**一种**输入（point/bbox/mask三选一），但训练脚本同时给了多种输入。

**解决方案**：

#### 1. 创建 `utils/prompt_utils.py`

提供三个函数：

```python
def prepare_random_prompt(batch, device, prompt_types=['point', 'bbox', 'mask']):
    """随机选择一种prompt类型（模拟真实场景）"""
    prompt_type = random.choice(prompt_types)
    
    if prompt_type == 'point':
        # 返回 (points, None, None)
    elif prompt_type == 'bbox':
        # 返回 (None, boxes, None)
    elif prompt_type == 'mask':
        # 返回 (None, None, masks)

def prepare_all_prompts(batch, device):
    """准备所有prompt（用于调试）"""
    # 返回 (points, boxes, masks)

def prepare_point_prompt(batch, device):
    """只准备point prompt（向后兼容）"""
    # 返回 (points, None, None)
```

**关键点**：
- `mono_bbox` 格式是 `[x, y, w, h]`，需要转换为 `[x1, y1, x2, y2]` 给模型
- 转换公式：
  ```python
  x1 = x
  y1 = y
  x2 = x + w
  y2 = y + h
  ```

#### 2. 更新训练脚本

**train.py**：
```python
from utils.prompt_utils import prepare_random_prompt

# 训练时：随机选择一种prompt
points, boxes, masks = prepare_random_prompt(batch, device)
outputs = model(
    front_view=front_view,
    satellite_view=sat_view,
    points=points,
    boxes=boxes,
    masks=masks,
)

# 验证时：使用point prompt（最常见）
point_coords = mono_point.unsqueeze(1)
point_labels = torch.ones(B, 1, device=device)
outputs = model(
    front_view=front_view,
    satellite_view=sat_view,
    points=(point_coords, point_labels),
    boxes=None,
    masks=None,
)
```

**train_accelerate.py**：同样的更新逻辑。

---

## PromptEncoder确认

### 支持的输入组合

从 `models/prompt_encoder.py` 和 `models/cross_view_localizer_v2.py` 确认：

**PromptEncoder可以接受任意组合**：
```python
def forward(
    self,
    points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    boxes: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 可以只给一个，也可以给多个
    # 但真实场景应该只给一个
```

**处理逻辑**：
- `points` → sparse_embeddings
- `boxes` → sparse_embeddings
- `masks` → dense_embeddings

**训练策略**：
- ✅ **训练时**：随机选择一种（point/bbox/mask三选一）
- ✅ **验证时**：使用point prompt（最常见的用户输入）
- ✅ **测试时**：可以测试所有三种输入的性能

---

## 更新的文件清单

### 新增文件
1. `/data/xhj/location/utils/prompt_utils.py` - Prompt准备工具函数

### 修改文件
1. `/data/xhj/location/data/dataset.py` - 整理import
2. `/data/xhj/location/train.py` - 随机prompt选择
3. `/data/xhj/location/train_accelerate.py` - 随机prompt选择

### 文档文件
1. `/data/xhj/location/PROMPT_TRAINING_UPDATE.md` - 本文档

---

## 使用方法

### 训练
```bash
# 标准训练（随机prompt）
python train.py --config configs/default.yaml

# Accelerate训练（随机prompt）
python train_accelerate.py --config configs/cesium.yaml
```

### 验证不同prompt类型

如果想测试特定prompt类型的性能，可以修改 `prepare_random_prompt` 的 `prompt_types` 参数：

```python
# 只测试point
points, boxes, masks = prepare_random_prompt(batch, device, prompt_types=['point'])

# 只测试bbox
points, boxes, masks = prepare_random_prompt(batch, device, prompt_types=['bbox'])

# 只测试mask
points, boxes, masks = prepare_random_prompt(batch, device, prompt_types=['mask'])

# 随机选择（默认）
points, boxes, masks = prepare_random_prompt(batch, device, prompt_types=['point', 'bbox', 'mask'])
```

---

## Bbox格式说明

### Dataset中的格式
- `mono_bbox`: `[x, y, w, h]` - 左上角坐标 + 宽高
- `sat_bbox`: `[x, y, w, h]` - 左上角坐标 + 宽高

### 模型输入格式
- `boxes`: `[B, N, 4]` in `[x1, y1, x2, y2]` - 两个角点坐标

### 转换
```python
# [x, y, w, h] → [x1, y1, x2, y2]
x1 = x
y1 = y
x2 = x + w
y2 = y + h
```

---

## 训练流程

```
Dataset
  ↓
加载 mono_point, mono_bbox, mono_mask
  ↓
随机选择一种prompt类型
  ↓
  ├─ point: (coords, labels)
  ├─ bbox: [B, 1, 4] in [x1,y1,x2,y2]
  └─ mask: [B, 1, H, W]
  ↓
Model(front_view, satellite_view, points/boxes/masks)
  ↓
PromptEncoder
  ↓
  ├─ sparse_embeddings (point/bbox)
  └─ dense_embeddings (mask)
  ↓
MultiTaskHead
  ↓
预测 bbox/mask/yaw/position
```

---

## 总结

✅ **所有问题已解决**：

1. **代码规范**：import已整理到文件顶部
2. **sat_bbox缩放**：逻辑正确，无需修改
3. **Prompt训练**：
   - ✅ 训练时随机选择一种prompt（模拟真实场景）
   - ✅ 验证时使用point prompt
   - ✅ 支持point/bbox/mask三种输入
   - ✅ Bbox格式正确转换

现在训练流程完全符合真实使用场景，模型将学会从**任意一种**prompt输入进行预测。
