# Cesium Drone Dataset

## 概述

这是从 `/data/cesium/output` 目录下合并的无人机前视图与卫星图匹配数据集。

- **数据集文件**: `/data/xhj/location/data/cesium_drone_dataset.json`
- **总样本数**: 119,314
- **图像根目录**: `/data/cesium/output`

## 城市分布

| 城市 | 样本数 |
|------|--------|
| Berlin | 28,510 |
| Chicago | 16,924 |
| London | 25,166 |
| Newyork | 29,298 |
| Paris | 13,568 |
| Tokyo | 5,848 |

## 数据格式

每个样本包含以下字段：

```json
{
  "city": "Paris",
  "mono_filename": "drone_paris_grid_0_3_real.png",
  "mono_point": [370.0, 564.5],
  "mono_bbox": [228, 452, 284, 225],
  "mono_segmentation": {
    "size": [1024, 1024],
    "counts": "..."
  },
  "sat_filename": "sate_paris_grid_0_real.png",
  "sate_bbox": [562, 713, 83, 85],
  "sate_segmentation": [[603.5, 713, 562, 755.5, 603.5, 798, 645, 755.5]],
  "rotation": 169,
  "camera_position": [640.0, 640.0]
}
```

### 字段说明

- **city**: 城市名称
- **mono_filename**: 无人机前视图文件名
- **mono_point**: 目标在前视图中的中心点坐标 [x, y]
- **mono_bbox**: 目标在前视图中的边界框 [x, y, w, h]
- **mono_segmentation**: 目标在前视图中的分割掩码（RLE格式）
- **sat_filename**: 卫星图文件名
- **sate_bbox**: 目标在卫星图中的边界框 [x, y, w, h]
- **sate_segmentation**: 目标在卫星图中的分割掩码（polygon格式）
- **rotation**: 相机朝向角度（度）
- **camera_position**: 相机在卫星图中的位置 [x, y]

## 目录结构

```
/data/cesium/output/
├── Berlin/
│   ├── drone/          # 无人机前视图
│   ├── sate/           # 卫星图
│   └── cleaned_final_dataset.json
├── Chicago/
│   ├── drone/
│   ├── sate/
│   └── cleaned_final_dataset.json
├── London/
│   ├── drone/
│   ├── sate/
│   └── cleaned_final_dataset.json
├── Newyork/
│   ├── drone/
│   ├── sate/
│   └── cleaned_final_dataset.json
├── Paris/
│   ├── drone/
│   ├── sate/
│   └── cleaned_final_dataset.json
└── Tokyo/
    ├── drone/
    ├── sate/
    └── cleaned_final_dataset.json
```

## 使用方法

### 1. 使用 CesiumDroneDataset 类

```python
from data.cesium_dataset import CesiumDroneDataset
from data.dataset import collate_fn
from torch.utils.data import DataLoader

# 创建数据集
dataset = CesiumDroneDataset(
    json_path='/data/xhj/location/data/cesium_drone_dataset.json',
    data_root='/data/cesium/output',
    mono_size=1024,
    sat_size=1024,
    crop_sat=True,
    crop_size=518,
    random_crop=True,  # 训练时设为True，测试时设为False
)

# 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

# 迭代数据
for batch in loader:
    front_view = batch['front_view']        # [B, 3, 1024, 1024]
    satellite_view = batch['satellite_view'] # [B, 3, 518, 518]
    mono_bbox = batch['mono_bbox']          # [B, 4]
    sat_bbox = batch['sat_bbox']            # [B, 4]
    camera_position = batch['camera_position'] # [B, 2]
    yaw_radians = batch['yaw_radians']      # [B]
    # ... 训练代码
```

### 2. 与原有 GoogleEarth 数据集一起使用

```python
from data.dataset import CrossViewDataset
from data.cesium_dataset import CesiumDroneDataset
from torch.utils.data import ConcatDataset, DataLoader

# GoogleEarth 数据集（无人车前视图）
googleearth_dataset = CrossViewDataset(
    json_path='/data/xhj/location/data/test_samples.json',
    data_root='/data/GoogleEarth',
)

# Cesium 数据集（无人机前视图）
cesium_dataset = CesiumDroneDataset(
    json_path='/data/xhj/location/data/cesium_drone_dataset.json',
    data_root='/data/cesium/output',
)

# 合并数据集
combined_dataset = ConcatDataset([googleearth_dataset, cesium_dataset])

# 创建 DataLoader
loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
```

## 数据转换说明

原始 cesium 数据格式与目标格式的主要差异：

1. **bbox格式转换**: 
   - 原始: `[x1, y1, x2, y2]` (两个角点)
   - 转换后: `[x, y, w, h]` (左上角+宽高)

2. **添加 mono_point**: 
   - 从 `mono_bbox` 计算中心点: `[x + w/2, y + h/2]`

3. **sate_segmentation格式转换**:
   - 原始: RLE格式 `{"size": [...], "counts": "..."}`
   - 转换后: Polygon格式 `[[x1, y1, x2, y2, ...]]`
   - 简化处理：从bbox生成4个中点的polygon

4. **添加 city 字段**:
   - 从文件所在目录名提取城市名称

5. **添加 camera_position**:
   - 默认值: `[640.0, 640.0]` (假设卫星图是1280x1280，相机在中心)

## 脚本说明

- **merge_cesium_datasets.py**: 合并所有城市的数据集
- **cesium_dataset.py**: CesiumDroneDataset 类定义
- **test_cesium_dataset.py**: 测试数据集加载

## 注意事项

1. 图像尺寸为 1024x1024（与 GoogleEarth 的 518x518 不同）
2. 目录结构使用 `drone/` 而非 `mono/`
3. 数据量较大（119K样本），建议使用多进程加载
4. 训练时建议使用 `crop_sat=True` 来裁剪卫星图到 518x518
