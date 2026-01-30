#!/usr/bin/env python3
"""
合并GoogleEarth和Cesium数据集为混合数据集
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def load_dataset(json_path: str) -> List[Dict]:
    """加载数据集"""
    with open(json_path, 'r') as f:
        return json.load(f)


def split_dataset(data: List[Dict], train_ratio: float = 0.9) -> tuple:
    """分割数据集为训练和验证集"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def merge_datasets():
    """合并数据集"""
    # 设置路径
    googleearth_json = "/data/xhj/location/data/test_samples.json"
    cesium_json = "/data/xhj/location/data/cesium_drone_dataset.json"
    output_dir = Path("/data/xhj/location/data")
    
    # 加载数据集
    print("加载数据集...")
    googleearth_data = load_dataset(googleearth_json)
    cesium_data = load_dataset(cesium_json)
    
    print(f"GoogleEarth数据: {len(googleearth_data)} 样本")
    print(f"Cesium数据: {len(cesium_data)} 样本")
    
    # 合并数据
    mixed_data = googleearth_data + cesium_data
    print(f"混合数据集: {len(mixed_data)} 样本")
    
    # 分割训练和验证集
    train_data, val_data = split_dataset(mixed_data, train_ratio=0.9)
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    # 统计信息
    train_cities = {}
    val_cities = {}
    for item in train_data:
        city = item['city']
        train_cities[city] = train_cities.get(city, 0) + 1
    for item in val_data:
        city = item['city']
        val_cities[city] = val_cities.get(city, 0) + 1
    
    print("\n训练集城市分布:")
    for city, count in sorted(train_cities.items()):
        print(f"  {city}: {count}")
    
    print("\n验证集城市分布:")
    for city, count in sorted(val_cities.items()):
        print(f"  {city}: {count}")
    
    # 保存数据集
    train_output = output_dir / "mixed_train_dataset.json"
    val_output = output_dir / "mixed_val_dataset.json"
    
    print(f"\n保存训练集到: {train_output}")
    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    print(f"保存验证集到: {val_output}")
    with open(val_output, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print("\n✅ 混合数据集创建完成!")
    
    # 创建使用说明
    usage = """
# 使用混合数据集

## 训练
python train_accelerate.py --config configs/mixed_dataset.yaml

## 测试加载
from data.dataset import CrossViewDataset

# 训练集
train_dataset = CrossViewDataset(
    json_path='/data/xhj/location/data/mixed_train_dataset.json',
    data_root='/data',  # 根目录包含GoogleEarth/和cesium/output/
    mono_size=518,
    sat_size=1280,  # 使用较大的尺寸以适应两种数据
    crop_sat=True,
    crop_size=518,
)

# 验证集
val_dataset = CrossViewDataset(
    json_path='/data/xhj/location/data/mixed_val_dataset.json',
    data_root='/data',
    mono_size=518,
    sat_size=1280,
    crop_sat=True,
    crop_size=518,
    random_crop=False,
)
"""
    
    print(usage)


if __name__ == '__main__':
    # 设置随机种子
    random.seed(42)
    
    merge_datasets()
