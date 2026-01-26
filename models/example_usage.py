"""
Example usage of CrossViewLocalizer

This demonstrates the complete data flow:
1. Input: front_view + satellite_view images
2. User prompts: points, boxes, or masks on front_view
3. Output: bounding boxes in satellite_view
"""

import torch
from models import CrossViewLocalizer, build_cross_view_localizer


def example_basic_usage():
    """Basic usage with point prompts."""
    
    # Build model
    model = CrossViewLocalizer(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        vggt_depth=24,
        num_decoder_layers=6,
        use_multi_query=False,
        freeze_vggt=False,
    )
    model.eval()
    
    # Prepare inputs
    B = 2  # batch size
    front_view = torch.randn(B, 3, 518, 518)      # Front view images
    satellite_view = torch.randn(B, 3, 518, 518)  # Satellite view images
    
    # User clicks on front view (normalized pixel coordinates)
    point_coords = torch.tensor([
        [[100.0, 200.0], [150.0, 250.0]],  # Batch 1: 2 points
        [[300.0, 400.0], [0.0, 0.0]],      # Batch 2: 1 point + padding
    ])
    point_labels = torch.tensor([
        [1, 1],   # Both positive points
        [1, -1],  # 1 positive, 1 padding
    ])
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            front_view=front_view,
            satellite_view=satellite_view,
            points=(point_coords, point_labels),
        )
    
    print("=== Basic Usage (Point Prompts) ===")
    print(f"Predicted boxes shape: {outputs['pred_boxes'].shape}")  # [B, N, 4]
    print(f"Scores shape: {outputs['scores'].shape}")                # [B, N]
    print(f"Front features shape: {outputs['front_features'].shape}")
    print(f"Sat features shape: {outputs['sat_features'].shape}")


def example_box_prompts():
    """Usage with box prompts."""
    
    model = CrossViewLocalizer(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        vggt_depth=24,
    )
    model.eval()
    
    B = 2
    front_view = torch.randn(B, 3, 518, 518)
    satellite_view = torch.randn(B, 3, 518, 518)
    
    # User draws boxes on front view (x1, y1, x2, y2)
    boxes = torch.tensor([
        [[50.0, 50.0, 150.0, 150.0]],   # Batch 1: 1 box
        [[100.0, 100.0, 200.0, 200.0]], # Batch 2: 1 box
    ])
    
    with torch.no_grad():
        outputs = model(
            front_view=front_view,
            satellite_view=satellite_view,
            boxes=boxes,
        )
    
    print("\n=== Box Prompts ===")
    print(f"Predicted boxes: {outputs['pred_boxes'].shape}")
    print(f"Scores: {outputs['scores'].shape}")


def example_mask_prompts():
    """Usage with mask prompts."""
    
    model = CrossViewLocalizer(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        vggt_depth=24,
    )
    model.eval()
    
    B = 2
    front_view = torch.randn(B, 3, 518, 518)
    satellite_view = torch.randn(B, 3, 518, 518)
    
    # User draws mask on front view
    # Mask size should be 4x the embedding size for proper downsampling
    mask_size = 4 * 37  # 148
    masks = torch.zeros(B, 1, mask_size, mask_size)
    masks[:, :, 50:100, 50:100] = 1.0  # Mark a region
    
    # Also provide a point for sparse embedding
    point_coords = torch.tensor([
        [[75.0, 75.0]],
        [[75.0, 75.0]],
    ])
    point_labels = torch.ones(B, 1)
    
    with torch.no_grad():
        outputs = model(
            front_view=front_view,
            satellite_view=satellite_view,
            points=(point_coords, point_labels),
            masks=masks,
        )
    
    print("\n=== Mask Prompts ===")
    print(f"Predicted boxes: {outputs['pred_boxes'].shape}")
    print(f"Scores: {outputs['scores'].shape}")


def example_multi_query():
    """Usage with DETR-style learnable queries."""
    
    model = CrossViewLocalizer(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        vggt_depth=24,
        use_multi_query=True,
        num_queries=100,
    )
    model.eval()
    
    B = 2
    front_view = torch.randn(B, 3, 518, 518)
    satellite_view = torch.randn(B, 3, 518, 518)
    
    point_coords = torch.tensor([[[200.0, 200.0]], [[300.0, 300.0]]])
    point_labels = torch.ones(B, 1)
    
    with torch.no_grad():
        outputs = model(
            front_view=front_view,
            satellite_view=satellite_view,
            points=(point_coords, point_labels),
        )
    
    print("\n=== Multi-Query (DETR-style) ===")
    print(f"Predicted boxes: {outputs['pred_boxes'].shape}")  # [B, 100, 4]
    print(f"Scores: {outputs['scores'].shape}")                # [B, 100]


def example_with_pretrained():
    """Load pretrained VGGT weights."""
    
    # Build with pretrained VGGT (if available)
    # model = build_cross_view_localizer(
    #     pretrained_vggt="/path/to/vggt_checkpoint.pth",
    #     freeze_vggt=True,
    #     img_size=518,
    #     patch_size=14,
    #     embed_dim=1024,
    # )
    
    print("\n=== Pretrained Loading ===")
    print("Use build_cross_view_localizer() with pretrained_vggt path")


def print_model_info():
    """Print model architecture info."""
    
    model = CrossViewLocalizer(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        vggt_depth=24,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== Model Info ===")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Image size: {model.img_size}")
    print(f"Patch size: {model.patch_size}")
    print(f"Patches per side: {model.num_patches_per_side}")
    print(f"Total patches: {model.num_patches}")
    print(f"Embedding dim: {model.embed_dim}")
    print(f"Output dim (2*C): {model.output_dim}")


if __name__ == "__main__":
    print_model_info()
    example_basic_usage()
    example_box_prompts()
    example_mask_prompts()
    example_multi_query()
    example_with_pretrained()
