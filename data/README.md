# Location Dataset

基于前视图（Front View）在卫星图像上定位物体的数据集。

## 数据概览

- **总数据量**: 124,412 条
- **城市数量**: 5 个（London, Moscow, Newyork, Paris, Tokyo）
- **数据来源**: GoogleEarth 街景和卫星图像

### 城市分布

| 城市 | 数据条目数 |
|------|-----------|
| London | 24,130 |
| Moscow | 27,293 |
| Newyork | 35,457 |
| Paris | 24,179 |
| Tokyo | 13,353 |

## 数据结构

### 文件说明

- `results_filter.json` - 完整数据集（124,412 条）
- `test_samples.json` - 测试样本（10 条，随机种子=42）

### 数据格式

每条数据包含以下字段：

```json
{
  "city": "London",
  "mono_filename": "r-170_51.46010613,-0.05627133_2025-04_QEa0iGpFoGkJVe3Z64hSdA_d141_z3.jpg",
  "mono_point": [370.5, 227.0],
  "mono_bbox": [304.0, 172.0, 133.0, 110.0],
  "mono_segmentation": {
    "size": [518, 518],
    "counts": "VPj44d?c0I6J3M20O01O..."
  },
  "sat_filename": "51.46010613,-0.05627133_2025-04_QEa0iGpFoGkJVe3Z64hSdA_d141_z3.jpg",
  "sate_bbox": [123.64, 554.17, 148, 233],
  "sate_segmentation": [[228.64, 554.17, 123.64, 765.17, ...]],
  "rotation": -131.0,
  "camera_position": [640.0, 640.0]
}
```

### 字段说明

#### 基本信息
- **city** (str): 城市名称
- **rotation** (float): 相机 yaw 角（度）

#### 单目图像（Mono View）
- **mono_filename** (str): 单目图像文件名
  - 格式: `r{rotation}_{lat},{lon}_{date}_{pano_id}_d{direction}_z{zoom}.jpg`
  - 路径: `/data/GoogleEarth/{city}/mono/{mono_filename}`
  
- **mono_point** (list[float, float]): 物体中心点坐标 [x, y]
  - 图像坐标系，单位：像素
  
- **mono_bbox** (list[float, float, float, float]): 物体边界框 [x, y, width, height]
  - x, y: 左上角坐标
  - width, height: 宽度和高度
  - 图像坐标系，单位：像素
  
- **mono_segmentation** (dict): 物体分割掩码（COCO RLE 格式）
  - `size`: [height, width] - 图像尺寸（通常为 518×518）
  - `counts`: 压缩的游程编码字符串

#### 卫星图像（Satellite View）
- **sat_filename** (str): 卫星图像文件名
  - 格式: `{lat},{lon}_{date}_{pano_id}_d{direction}_z{zoom}.jpg`
  - 路径: `/data/GoogleEarth/{city}/sate/{sat_filename}`
  
- **sate_bbox** (list[float, float, float, float]): 物体边界框 [x, y, width, height]
  - x, y: 左上角坐标
  - width, height: 宽度和高度
  - 图像坐标系，单位：像素
  
- **sate_segmentation** (list[list[float]]): 物体分割掩码（COCO Polygon 格式）
  - 多边形顶点坐标列表 [[x1, y1, x2, y2, x3, y3, ...]]
  - 可能包含多个多边形（外轮廓和内部孔洞）

- **camera_position** (list[float, float]): 相机在卫星图中的位置 [x, y]
  - 默认值: [640.0, 640.0]（卫星图中心）
  - 图像坐标系，单位：像素
  - 用于crop数据增强时的参考点




## 坐标系统说明

### 图像坐标系
- 原点：图像左上角
- X 轴：向右为正
- Y 轴：向下为正
- 单位：像素

### 相机旋转
- **rotation**: 相机 yaw 角（偏航角）
- 范围：-180° 到 180°
- 正值：顺时针旋转
- 负值：逆时针旋转
- 默认相机位置在卫星图的中心


## 图像尺寸
- mono是518*518
- sate是1280*1280


## 注意事项

1. **图像路径**: 图像文件位于 `/data/GoogleEarth/{city}/{mono|sate}/` 目录
2. **分割格式**: 单目图使用 RLE，卫星图使用 Polygon
3. **坐标系**: 所有坐标均为图像坐标系（左上角为原点）
4. **相机位置**: 默认相机位置在卫星图中心
5. **依赖库**: 需要安装 `pycocotools` 处理分割掩码

