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
        direction='both',
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试两个方向
    for direction in ['mono_to_sat', 'sat_to_mono']:
        dataset.direction = direction
        sample = dataset[0]
        print(f"\nDirection: {direction}")
        print(f"  Mono view: {sample['mono_view'].shape}")
        print(f"  Sat view: {sample['sat_view'].shape}")
        print(f"  Prompt point: {sample['prompt_point'].shape}")
        print(f"  Prompt bbox: {sample['prompt_bbox'].shape}")
        print(f"  Prompt mask: {sample['prompt_mask'].shape}")
        print(f"  Target bbox: {sample['target_bbox']}")
        print(f"  Target position: {sample['target_position']}")
    
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
    mono_view = torch.randn(B, 3, 518, 518, device=device)
    sat_view = torch.randn(B, 3, 518, 518, device=device)
    
    # Point prompt
    point_coords = torch.rand(B, 1, 2, device=device) * 518
    point_labels = torch.ones(B, 1, device=device)
    points = (point_coords, point_labels)
    
    # 测试 mono_to_sat 方向
    print("\nTesting mono_to_sat direction...")
    with torch.no_grad():
        outputs = model(
            mono_view=mono_view,
            sat_view=sat_view,
            points=points,
            prompt_views=['mono', 'mono'],
        )
    
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"  bbox_scores: {outputs['bbox_scores'].shape}")
    print(f"  heatmap: {outputs['heatmap'].shape}")
    print(f"  position: {outputs['position'].shape}")
    print(f"  yaw_radians: {outputs['yaw_radians'].shape}")
    
    # 测试 sat_to_mono 方向
    print("\nTesting sat_to_mono direction...")
    with torch.no_grad():
        outputs = model(
            mono_view=mono_view,
            sat_view=sat_view,
            points=points,
            prompt_views=['sat', 'sat'],
        )
    
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"  bbox_scores: {outputs['bbox_scores'].shape}")
    print(f"  heatmap: {outputs['heatmap'].shape}")
    print(f"  position: {outputs['position'].shape}")
    
    # 测试混合 batch
    print("\nTesting mixed batch...")
    with torch.no_grad():
        outputs = model(
            mono_view=mono_view,
            sat_view=sat_view,
            points=points,
            prompt_views=['mono', 'sat'],
        )
    
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"  position: {outputs['position'].shape}")
    
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
        direction='mono_to_sat',
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
    
    mono_view = batch['mono_view'].to(device)
    sat_view = batch['sat_view'].to(device)
    prompt_point = batch['prompt_point'].to(device)
    prompt_views = batch['prompt_views']
    
    # 准备 point prompt
    B = mono_view.shape[0]
    point_coords = prompt_point.unsqueeze(1)  # [B, 1, 2]
    point_labels = torch.ones(B, 1, device=device)
    points = (point_coords, point_labels)
    
    print(f"Batch loaded:")
    print(f"  mono_view: {mono_view.shape}")
    print(f"  sat_view: {sat_view.shape}")
    print(f"  prompt_views: {prompt_views}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            mono_view=mono_view,
            sat_view=sat_view,
            points=points,
            prompt_views=prompt_views,
        )
    
    print(f"\nOutputs:")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"  heatmap: {outputs['heatmap'].shape}")
    print(f"  position: {outputs['position']}")
    print(f"  yaw_degrees: {outputs['yaw_degrees']}")
    
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
