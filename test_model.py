"""
测试脚本：验证模型和数据集能否正常运行
"""

import torch
import sys
sys.path.insert(0, '/data/xhj/location')

from data import CrossViewDataset, collate_fn
from models import build_cross_view_localizer_pi3


def test_dataset():
    """测试数据集"""
    print("=" * 50)
    print("Testing Dataset...")
    print("=" * 50)
    
    dataset = CrossViewDataset(
        json_path='/data/xhj/location/data/single.json',
        data_root='/data/GoogleEarth',
        crop_sat=True,
        random_crop=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试单向定位
    sample = dataset[0]
    print(f"\nUnidirectional localization (front -> satellite):")
    print(f"  Front view: {sample['front_view'].shape}")
    print(f"  Satellite view: {sample['satellite_view'].shape}")
    print(f"  Mono point: {sample['mono_point'].shape}")
    print(f"  Mono bbox: {sample['mono_bbox']}")
    print(f"  Mono mask: {sample['mono_mask'].shape}")
    print(f"  Sat bbox: {sample['sat_bbox']}")
    print(f"  Camera position: {sample['camera_position']}")
    print(f"  Yaw (radians): {sample['yaw']:.4f}")
    
    print("\n✓ Dataset test passed!")
    return dataset


def test_model():
    """测试模型前向传播"""
    print("\n" + "=" * 50)
    print("Testing Model...")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建模型（不加载预训练权重，只测试结构）
    model = build_cross_view_localizer_pi3(
        pretrained_pi3=None,
        freeze_backbone=False,
        img_size=518,
        decoder_size='large',
        num_heads=16,
        num_decoder_layers=6,
        num_object_queries=10,
        num_location_queries=16,
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model created successfully")
    print(f"  Output dim: {model.output_dim}")
    print(f"  Num patches per side: {model.num_patches_per_side}")
    
    # 测试前向传播
    B = 2
    front_view = torch.randn(B, 3, 518, 518, device=device)
    satellite_view = torch.randn(B, 3, 518, 518, device=device)
    
    # Point prompt (on front view)
    point_coords = torch.rand(B, 1, 2, device=device) * 518
    point_labels = torch.ones(B, 1, device=device)
    points = (point_coords, point_labels)
    
    # 测试单向定位 (front -> satellite)
    print("\nTesting unidirectional localization (front -> satellite)...")
    with torch.no_grad():
        outputs = model(
            front_view=front_view,
            satellite_view=satellite_view,
            points=points,
        )
    
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"  bbox_scores: {outputs['bbox_scores'].shape}")
    print(f"  heatmap: {outputs['heatmap'].shape}")
    print(f"  position: {outputs['position'].shape}")
    print(f"  rotation_matrix: {outputs['rotation_matrix'].shape}")
    print(f"  yaw: {outputs['yaw'].shape}")
    print(f"  pitch: {outputs['pitch'].shape}")
    print(f"  roll: {outputs['roll'].shape}")
    
    print("\n✓ Model test passed!")
    return model


def test_full_pipeline():
    """测试完整的数据加载 + 模型前向传播"""
    print("\n" + "=" * 50)
    print("Testing Full Pipeline...")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集
    dataset = CrossViewDataset(
        json_path='/data/xhj/location/data/single.json',
        data_root='/data/GoogleEarth',
        crop_sat=True,
        random_crop=False,
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # 创建模型
    model = build_cross_view_localizer_pi3(
        pretrained_pi3=None,
        freeze_backbone=False,
        img_size=518,
        decoder_size='large',
    )
    model = model.to(device)
    model.eval()
    
    # 获取一个 batch
    batch = next(iter(loader))
    
    front_view = batch['front_view'].to(device)
    satellite_view = batch['satellite_view'].to(device)
    mono_point = batch['mono_point'].to(device)
    
    # 准备 point prompt
    B = front_view.shape[0]
    point_coords = mono_point.unsqueeze(1)  # [B, 1, 2]
    point_labels = torch.ones(B, 1, device=device)
    points = (point_coords, point_labels)
    
    print(f"Batch loaded:")
    print(f"  front_view: {front_view.shape}")
    print(f"  satellite_view: {satellite_view.shape}")
    print(f"  mono_point: {mono_point.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            front_view=front_view,
            satellite_view=satellite_view,
            points=points,
        )
    
    print(f"\nOutputs:")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"  heatmap: {outputs['heatmap'].shape}")
    print(f"  position: {outputs['position']}")
    print(f"  yaw: {outputs['yaw']}")
    
    print("\n✓ Full pipeline test passed!")


if __name__ == '__main__':
    try:
        test_dataset()
        test_model()
        test_full_pipeline()
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
