#!/usr/bin/env python3
"""
测试cesium drone数据集是否能正确加载
"""

import sys
sys.path.insert(0, '/data/xhj/location')

from data.dataset import CrossViewDataset, collate_fn
from torch.utils.data import DataLoader


def test_dataset():
    """测试数据集加载"""
    
    print("=" * 60)
    print("测试 Cesium Drone 数据集")
    print("=" * 60)
    
    # 创建数据集
    # 注意：cesium数据的目录结构是 drone/ 和 sate/，不是 mono/ 和 sate/
    # 需要修改data_root指向cesium output目录
    dataset = CrossViewDataset(
        json_path='/data/xhj/location/data/cesium_drone_dataset.json',
        data_root='/data/cesium/output',  # cesium数据根目录
        mono_size=1024,  # cesium数据是1024x1024
        sat_size=1024,   # cesium数据是1024x1024
        crop_sat=True,
        crop_size=518,
        random_crop=False,
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试加载第一个样本
    print("\n测试加载样本 0...")
    try:
        sample = dataset[0]
        print("✓ 成功加载样本 0")
        print(f"  城市: {sample['city']}")
        print(f"  前视图: {sample['front_view'].shape}")
        print(f"  卫星图: {sample['satellite_view'].shape}")
        print(f"  Mono bbox: {sample['mono_bbox']}")
        print(f"  Sat bbox: {sample['sat_bbox']}")
        print(f"  Camera position: {sample['camera_position']}")
        print(f"  Yaw (degrees): {sample['yaw_degrees']:.1f}")
        print(f"  Mono filename: {sample['mono_filename']}")
        print(f"  Sat filename: {sample['sat_filename']}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试DataLoader
    print("\n测试 DataLoader...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        batch = next(iter(loader))
        print("✓ 成功创建 batch")
        print(f"  Front views: {batch['front_view'].shape}")
        print(f"  Satellite views: {batch['satellite_view'].shape}")
        print(f"  Camera positions: {batch['camera_position'].shape}")
        print(f"  Yaw radians: {batch['yaw_radians'].shape}")
        print(f"  Cities: {batch['cities']}")
    except Exception as e:
        print(f"✗ DataLoader失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_dataset()
    sys.exit(0 if success else 1)
