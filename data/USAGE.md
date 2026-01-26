# 数据加载使用说明

## 1. 添加相机位置

首先运行脚本添加 `camera_position` 字段到数据中：

```bash
cd /data/xhj/location/data
python add_camera_position.py
```

这会将相机位置设置为卫星图中心 `[640, 640]`（卫星图尺寸为1280x1280）。


## 3. Crop增强说明

### 为什么需要crop？

- 卫星图原始尺寸为 **1280x1280**，太大
- 模型输入需要 **518x518**
- Crop可以作为数据增强，增加样本多样性

### Crop策略

1. **以相机位置为中心crop**
   - 相机位置默认在卫星图中心 `[640, 640]`
   - 训练时：在相机位置附近随机偏移crop
   - 测试时：以相机位置为中心crop

2. **自动调整标注**
   - `sat_bbox`: 自动减去crop偏移量
   - `camera_position`: 自动减去crop偏移量
   - `crop_offset`: 记录crop的偏移量 `[left, top]`


## 4. 数据格式

### 输入数据格式 (JSON)

```json
{
  "city": "London",
  "mono_filename": "...",
  "mono_point": [x, y],
  "mono_bbox": [x, y, w, h],
  "sat_filename": "...",
  "sate_bbox": [x, y, w, h],
  "rotation": yaw_degrees,
  "camera_position": [x, y]
}
```

### 输出batch格式

```python
{
  'front_view': Tensor[B, 3, 518, 518],      # RGB, [0, 1]
  'satellite_view': Tensor[B, 3, 518, 518],  # RGB, [0, 1], cropped
  'mono_point': Tensor[B, 2],                # 像素坐标
  'mono_bbox': Tensor[B, 4],                 # (cx, cy, w, h) normalized [0, 1]
  'sat_bbox': Tensor[B, 4],                  # (cx, cy, w, h) normalized [0, 1]
  'camera_position': Tensor[B, 2],           # (x, y) normalized [0, 1]
  'yaw_radians': Tensor[B],                  # 弧度
  'yaw_degrees': Tensor[B],                  # 角度
  'crop_offset': Tensor[B, 2],               # crop偏移量 [left, top]
  'cities': List[str],
  'mono_filenames': List[str],
  'sat_filenames': List[str],
}
```

## 5. 训练配置示例

```python
# 训练集
train_dataset = CrossViewDataset(
    json_path='data/train.json',
    crop_sat=True,
    random_crop=True,  # 随机crop增强
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

```

## 6. 与模型集成

```python
from models import CrossViewLocalizer

model = CrossViewLocalizer(
    enable_bbox=True,
    enable_camera=True,
    enable_position=True,
)

for batch in train_loader:
    # 准备输入
    front_view = batch['front_view'].cuda()
    sat_view = batch['satellite_view'].cuda()
    
    # 使用mono_point作为prompt
    mono_points = batch['mono_point'].cuda()
    point_labels = torch.ones(mono_points.shape[0], 1).cuda()  # 正点
    
    # 前向传播
    outputs = model(
        front_view=front_view,
        satellite_view=sat_view,
        points=(mono_points.unsqueeze(1), point_labels),
    )
    
    # 计算loss
    # BBox loss
    bbox_loss = F.l1_loss(outputs['pred_boxes'], batch['sat_bbox'].cuda())
    
    # Yaw loss
    yaw_loss = F.mse_loss(outputs['yaw_radians'], batch['yaw_radians'].cuda())
    
    # Position loss
    position_loss = F.mse_loss(outputs['position'], batch['camera_position'].cuda())
    
    total_loss = bbox_loss + yaw_loss + position_loss
```

## 7. 注意事项

1. **坐标系统**
   - 所有坐标都是图像坐标系（左上角为原点）
   - BBox格式：`[cx, cy, w, h]` 归一化到 [0, 1]
   - Camera position：`[x, y]` 归一化到 [0, 1]

2. **Crop影响**
   - Crop会改变卫星图的有效区域
   - `crop_offset` 可用于将预测结果映射回原图
   - 例如：`original_x = predicted_x * crop_size + crop_offset[0]`

3. **数据增强**
   - 当前只实现了crop增强
