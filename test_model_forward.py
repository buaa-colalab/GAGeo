"""Quick test script to verify model forward pass works."""

import torch
import sys

def test_model():
    print("=" * 50)
    print("Testing CrossViewLocalizerDETR Forward Pass")
    print("=" * 50)
    
    # Import model
    print("\n1. Importing model...")
    from models import CrossViewLocalizerDETR
    print("   ✓ Import successful")
    
    # Create model with smaller config for testing
    print("\n2. Creating model...")
    model = CrossViewLocalizerDETR(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        vggt_depth=24,
        num_heads=16,
        num_decoder_layers=6,
        num_object_queries=10,       # Reduced for single-object detection
        num_location_queries=16,     # Reduced for efficiency
        heatmap_size=32,             # Output heatmap size
        freeze_vggt=False,
        use_prompt_fusion=True,
    )
    print("   ✓ Model created")
    
    # Move to GPU
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    print(f"   ✓ Model moved to {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    
    # Create dummy inputs
    print("\n3. Creating dummy inputs...")
    B = 2
    front_view = torch.randn(B, 3, 518, 518, device=device)
    satellite_view = torch.randn(B, 3, 518, 518, device=device)
    
    # Point prompt
    point_coords = torch.rand(B, 1, 2, device=device) * 518  # [B, N, 2]
    point_labels = torch.ones(B, 1, device=device)  # [B, N]
    points = (point_coords, point_labels)
    print(f"   ✓ Inputs created: front_view={front_view.shape}, satellite_view={satellite_view.shape}")
    print(f"   ✓ Points: coords={point_coords.shape}, labels={point_labels.shape}")
    
    # Forward pass
    print("\n4. Running forward pass...")
    with torch.no_grad():
        outputs = model(
            front_view=front_view,
            satellite_view=satellite_view,
            points=points,
        )
    print("   ✓ Forward pass successful!")
    
    # Check outputs
    print("\n5. Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Verify key outputs
    print("\n6. Verifying outputs...")
    assert outputs['pred_boxes'].shape == (B, 10, 4), f"pred_boxes shape mismatch: {outputs['pred_boxes'].shape}"
    assert outputs['bbox_scores'].shape == (B, 10), f"bbox_scores shape mismatch: {outputs['bbox_scores'].shape}"
    assert outputs['heatmap'].shape == (B, 518, 518), f"heatmap shape mismatch: {outputs['heatmap'].shape}"
    assert outputs['position'].shape == (B, 2), f"position shape mismatch: {outputs['position'].shape}"
    assert outputs['yaw_radians'].shape == (B,), f"yaw_radians shape mismatch: {outputs['yaw_radians'].shape}"
    print("   ✓ All output shapes correct!")
    
    # Check value ranges
    print("\n7. Checking value ranges...")
    print(f"   pred_boxes: min={outputs['pred_boxes'].min():.4f}, max={outputs['pred_boxes'].max():.4f} (expected [0,1])")
    print(f"   bbox_scores: min={outputs['bbox_scores'].min():.4f}, max={outputs['bbox_scores'].max():.4f} (expected [0,1])")
    print(f"   heatmap sum: {outputs['heatmap'].sum(dim=[1,2])} (expected ~1.0 per sample)")
    print(f"   position: {outputs['position']} (expected [0,1])")
    print(f"   yaw_radians: {outputs['yaw_radians']} (expected [-π, π])")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
    
    return True

if __name__ == '__main__':
    try:
        test_model()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
