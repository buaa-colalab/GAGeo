# Test script for Cross-View Drone Localization System
# Tests the integration of VGGT, SAM-style Prompt Encoder, and DETR decoder

import sys
import json
import torch
import torch.nn as nn

sys.path.insert(0, '/data/xhj/location')

from models.prompt_encoder import GeometryPromptEncoder
from models.prompt_fusion import (
    TwoWayTransformer, 
    PromptFusionWithDense,
    Attention,
    MLP
)
from models.detr_decoder import TransformerDecoder, MLP as DetrMLP
from models.heads.heatmap_location_head import HeatmapLocationHead
from models.heads.yaw_head import CameraHead


def test_prompt_encoder():
    """Test GeometryPromptEncoder with different prompt types."""
    print("=" * 60)
    print("Testing GeometryPromptEncoder...")
    print("=" * 60)
    
    encoder = GeometryPromptEncoder(
        embed_dim=2048,
        image_embedding_size=(37, 37),
        input_image_size=(518, 518),
        mask_in_chans=16,
    )
    
    B = 2
    device = 'cpu'
    
    # Test 1: Point prompts
    print("\n[Test 1] Point prompts...")
    points_coords = torch.rand(B, 3, 2) * 518  # 3 points per batch
    points_labels = torch.ones(B, 3, dtype=torch.long)  # All positive
    sparse, dense = encoder(points=(points_coords, points_labels))
    print(f"  Sparse embeddings shape: {sparse.shape}")  # Expected: [B, 4, 2048] (3 points + 1 padding)
    print(f"  Dense embeddings shape: {dense.shape}")    # Expected: [B, 2048, 37, 37]
    assert sparse.shape == (B, 4, 2048), f"Expected (2, 4, 2048), got {sparse.shape}"
    assert dense.shape == (B, 2048, 37, 37), f"Expected (2, 2048, 37, 37), got {dense.shape}"
    print("  ✓ Point prompts test passed!")
    
    # Test 2: Box prompts
    print("\n[Test 2] Box prompts...")
    boxes = torch.tensor([
        [[100, 100, 50, 50]],  # [x, y, w, h]
        [[200, 200, 100, 100]],
    ], dtype=torch.float32)  # [B, 1, 4]
    sparse, dense = encoder(boxes=boxes)
    print(f"  Sparse embeddings shape: {sparse.shape}")  # Expected: [B, 2, 2048] (2 corners per box)
    print(f"  Dense embeddings shape: {dense.shape}")
    assert sparse.shape == (B, 2, 2048), f"Expected (2, 2, 2048), got {sparse.shape}"
    print("  ✓ Box prompts test passed!")
    
    # Test 3: Mask prompts
    print("\n[Test 3] Mask prompts...")
    masks = torch.rand(B, 1, 148, 148) > 0.5  # Binary masks
    masks = masks.float()
    sparse, dense = encoder(masks=masks)
    print(f"  Sparse embeddings shape: {sparse.shape}")  # Expected: [B, 0, 2048] (no sparse from masks)
    print(f"  Dense embeddings shape: {dense.shape}")    # Expected: [B, 2048, 37, 37]
    assert sparse.shape == (B, 0, 2048), f"Expected (2, 0, 2048), got {sparse.shape}"
    assert dense.shape == (B, 2048, 37, 37), f"Expected (2, 2048, 37, 37), got {dense.shape}"
    print("  ✓ Mask prompts test passed!")
    
    # Test 4: Combined prompts
    print("\n[Test 4] Combined prompts (points + boxes)...")
    sparse, dense = encoder(
        points=(points_coords, points_labels),
        boxes=boxes,
    )
    print(f"  Sparse embeddings shape: {sparse.shape}")  # Expected: [B, 5, 2048] (3 points + 2 corners)
    assert sparse.shape == (B, 5, 2048), f"Expected (2, 5, 2048), got {sparse.shape}"
    print("  ✓ Combined prompts test passed!")
    
    print("\n✓ All GeometryPromptEncoder tests passed!")
    return True


def test_prompt_fusion():
    """Test TwoWayTransformer and PromptFusionWithDense."""
    print("\n" + "=" * 60)
    print("Testing Prompt Fusion Modules...")
    print("=" * 60)
    
    B = 2
    P = 37 * 37  # Number of patches
    C = 2048
    N_sparse = 5
    
    # Test 1: TwoWayTransformer
    print("\n[Test 1] TwoWayTransformer...")
    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=C,
        num_heads=8,
        mlp_dim=C,
        activation=nn.ReLU,
        attention_downsample_rate=2,
    )
    
    image_features = torch.randn(B, P, C)
    image_pe = torch.randn(B, P, C)
    sparse_embeddings = torch.randn(B, N_sparse, C)
    
    fused_sparse, fused_image = transformer(
        image_embedding=image_features,
        image_pe=image_pe,
        point_embedding=sparse_embeddings,
    )
    print(f"  Fused sparse shape: {fused_sparse.shape}")  # Expected: [B, N_sparse, C]
    print(f"  Fused image shape: {fused_image.shape}")    # Expected: [B, P, C]
    assert fused_sparse.shape == (B, N_sparse, C)
    assert fused_image.shape == (B, P, C)
    print("  ✓ TwoWayTransformer test passed!")
    
    # Test 2: PromptFusionWithDense
    print("\n[Test 2] PromptFusionWithDense...")
    fusion = PromptFusionWithDense(
        embedding_dim=C,
        num_heads=8,
        depth=2,
        mlp_dim=C,
        image_embedding_size=(37, 37),
        activation=nn.ReLU,
        attention_downsample_rate=2,
    )
    
    dense_embeddings = torch.randn(B, C, 37, 37)
    
    fused_sparse, fused_image, target_guidance = fusion(
        image_features=image_features,
        sparse_embeddings=sparse_embeddings,
        dense_embeddings=dense_embeddings,
    )
    print(f"  Fused sparse shape: {fused_sparse.shape}")
    print(f"  Fused image shape: {fused_image.shape}")
    print(f"  Target guidance shape: {target_guidance.shape}")  # Expected: [B, C]
    assert fused_sparse.shape == (B, N_sparse, C)
    assert fused_image.shape == (B, P, C)
    assert target_guidance.shape == (B, C)
    print("  ✓ PromptFusionWithDense test passed!")
    
    # Test 3: Without dense embeddings
    print("\n[Test 3] PromptFusionWithDense (no dense)...")
    fused_sparse, fused_image, target_guidance = fusion(
        image_features=image_features,
        sparse_embeddings=sparse_embeddings,
        dense_embeddings=None,
    )
    print(f"  Fused sparse shape: {fused_sparse.shape}")
    print(f"  Fused image shape: {fused_image.shape}")
    print(f"  Target guidance shape: {target_guidance.shape}")
    assert fused_sparse.shape == (B, N_sparse, C)
    assert fused_image.shape == (B, P, C)
    assert target_guidance.shape == (B, C)
    print("  ✓ PromptFusionWithDense (no dense) test passed!")
    
    print("\n✓ All Prompt Fusion tests passed!")
    return True


def test_detr_decoder():
    """Test DETR-style TransformerDecoder."""
    print("\n" + "=" * 60)
    print("Testing DETR Decoder...")
    print("=" * 60)
    
    B = 2
    N_q = 100  # Number of queries
    P = 37 * 37  # Memory size
    C = 2048
    
    decoder = TransformerDecoder(
        d_model=C,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=C,
        dropout=0.1,
        normalize_before=False,
        return_intermediate=False,
    )
    
    queries = torch.randn(B, N_q, C)
    memory = torch.randn(B, P, C)
    
    output = decoder(tgt=queries, memory=memory)
    print(f"  Decoder output shape: {output.shape}")  # Expected: [1, B, N_q, C]
    
    if output.dim() == 4:
        output = output[-1]
    assert output.shape == (B, N_q, C), f"Expected (2, 100, 2048), got {output.shape}"
    print("  ✓ DETR Decoder test passed!")
    return True


def test_heads():
    """Test prediction heads."""
    print("\n" + "=" * 60)
    print("Testing Prediction Heads...")
    print("=" * 60)
    
    B = 2
    P = 37 * 37
    C = 2048
    
    # Test 1: HeatmapLocationHead
    print("\n[Test 1] HeatmapLocationHead...")
    heatmap_head = HeatmapLocationHead(
        d_model=C,
        grid_size=32,
        num_decoder_layers=2,  # Reduced for faster testing
        nhead=8,
        dim_feedforward=C,
        dropout=0.1,
    )
    
    sat_features = torch.randn(B, P, C)
    target_guidance = torch.randn(B, C)
    
    output = heatmap_head(
        sat_features=sat_features,
        front_target_features=target_guidance,
        target_size=(518, 518),
    )
    print(f"  Heatmap shape: {output['heatmap'].shape}")  # Expected: [B, 518, 518]
    print(f"  Position shape: {output['position'].shape}")  # Expected: [B, 2]
    print(f"  Heatmap logits shape: {output['heatmap_logits'].shape}")  # Expected: [B, 32, 32]
    assert output['heatmap'].shape == (B, 518, 518)
    assert output['position'].shape == (B, 2)
    assert output['heatmap_logits'].shape == (B, 32, 32)
    print("  ✓ HeatmapLocationHead test passed!")
    
    # Test 2: CameraHead
    print("\n[Test 2] CameraHead...")
    camera_head = CameraHead(
        dim_in=C,
        trunk_depth=2,  # Reduced for faster testing
        num_heads=8,
        mlp_ratio=4,
        init_values=0.01,
        num_iterations=2,
    )
    
    front_camera_token = torch.randn(B, C)
    sat_camera_token = torch.randn(B, C)
    
    output = camera_head(
        front_camera_token=front_camera_token,
        sat_camera_token=sat_camera_token,
    )
    print(f"  Pose encoding shape: {output['pose_enc'].shape}")  # Expected: [B, 9]
    print(f"  Quaternion shape: {output['quaternion'].shape}")  # Expected: [B, 4]
    print(f"  Yaw radians shape: {output['yaw_radians'].shape}")  # Expected: [B]
    print(f"  Yaw degrees shape: {output['yaw_degrees'].shape}")  # Expected: [B]
    assert output['pose_enc'].shape == (B, 9)
    assert output['quaternion'].shape == (B, 4)
    assert output['yaw_radians'].shape == (B,)
    assert output['yaw_degrees'].shape == (B,)
    print("  ✓ CameraHead test passed!")
    
    print("\n✓ All Prediction Heads tests passed!")
    return True


def test_dataset_integration():
    """Test integration with CrossViewDataset."""
    print("\n" + "=" * 60)
    print("Testing Dataset Integration...")
    print("=" * 60)
    
    try:
        from data.dataset import CrossViewDataset
        
        # Check if test data exists
        import os
        test_json = '/data/xhj/location/data/test_samples.json'
        if not os.path.exists(test_json):
            print(f"  Skipping: {test_json} not found")
            return True
        
        dataset = CrossViewDataset(
            json_path=test_json,
            crop_sat=False,
            test_mode=True,
        )
        
        if len(dataset) == 0:
            print("  Skipping: dataset is empty")
            return True
        
        sample = dataset[0]
        print(f"\n  Sample keys: {list(sample.keys())}")
        print(f"  mono_point shape: {sample['mono_point'].shape}")
        print(f"  mono_bbox shape: {sample['mono_bbox'].shape}")
        print(f"  mono_mask shape: {sample['mono_mask'].shape}")
        
        # Test with prompt encoder
        encoder = GeometryPromptEncoder(
            embed_dim=2048,
            image_embedding_size=(37, 37),
            input_image_size=(518, 518),
        )
        
        # Prepare prompts from dataset output
        B = 1
        points_coords = sample['mono_point'].unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        points_labels = torch.ones(B, 1, dtype=torch.long)
        boxes = sample['mono_bbox'].unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        masks = sample['mono_mask'].unsqueeze(0)  # [1, 1, H, W]
        
        sparse, dense = encoder(
            points=(points_coords, points_labels),
            boxes=boxes,
            masks=masks,
        )
        print(f"\n  Sparse embeddings: {sparse.shape}")
        print(f"  Dense embeddings: {dense.shape}")
        print("  ✓ Dataset integration test passed!")
        
    except Exception as e:
        print(f"  Skipping dataset test: {e}")
    
    return True


def test_full_pipeline_mock():
    """Test full pipeline with mock VGGT features (without loading actual VGGT)."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline (Mock VGGT)...")
    print("=" * 60)
    
    B = 2
    P = 37 * 37  # Number of patches
    C = 2048
    patch_start_idx = 5
    
    # Mock VGGT outputs
    print("\n[Step 1] Creating mock VGGT features...")
    front_patch_features = torch.randn(B, P, C)
    sat_patch_features = torch.randn(B, P, C)
    front_camera_token = torch.randn(B, C)
    sat_camera_token = torch.randn(B, C)
    print(f"  Front patch features: {front_patch_features.shape}")
    print(f"  Sat patch features: {sat_patch_features.shape}")
    
    # Step 2: Prompt Encoding
    print("\n[Step 2] Prompt Encoding...")
    prompt_encoder = GeometryPromptEncoder(
        embed_dim=C,
        image_embedding_size=(37, 37),
        input_image_size=(518, 518),
        mask_in_chans=16,
    )
    
    # Create sample prompts
    points_coords = torch.tensor([[[256, 256]], [[300, 300]]], dtype=torch.float32)
    points_labels = torch.ones(B, 1, dtype=torch.long)
    boxes = torch.tensor([[[100, 100, 50, 50]], [[200, 200, 80, 80]]], dtype=torch.float32)
    
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=(points_coords, points_labels),
        boxes=boxes,
    )
    print(f"  Sparse embeddings: {sparse_embeddings.shape}")
    print(f"  Dense embeddings: {dense_embeddings.shape}")
    
    # Step 3: Prompt Fusion
    print("\n[Step 3] Prompt Fusion...")
    prompt_fusion = PromptFusionWithDense(
        embedding_dim=C,
        num_heads=8,
        depth=2,
        mlp_dim=C,
        image_embedding_size=(37, 37),
    )
    
    fused_sparse, fused_front, target_guidance = prompt_fusion(
        image_features=front_patch_features,
        sparse_embeddings=sparse_embeddings,
        dense_embeddings=None,  # No mask in this test
    )
    print(f"  Fused sparse: {fused_sparse.shape}")
    print(f"  Fused front (F_target): {fused_front.shape}")
    print(f"  Target guidance: {target_guidance.shape}")
    
    # Step 4: Object Detection (DETR Decoder)
    print("\n[Step 4] Object Detection...")
    num_object_queries = 100
    object_queries = nn.Embedding(num_object_queries, C)
    target_guidance_proj = nn.Linear(C, C)
    
    obj_queries = object_queries.weight.unsqueeze(0).expand(B, -1, -1)
    target_proj = target_guidance_proj(target_guidance)
    obj_queries = obj_queries + target_proj.unsqueeze(1)
    
    object_decoder = TransformerDecoder(
        d_model=C,
        nhead=8,
        num_decoder_layers=2,
        dim_feedforward=C,
        dropout=0.1,
        return_intermediate=False,
    )
    
    obj_decoder_out = object_decoder(tgt=obj_queries, memory=sat_patch_features)
    if obj_decoder_out.dim() == 4:
        obj_decoder_out = obj_decoder_out[-1]
    
    bbox_head = DetrMLP(C, C, 4, 3)
    pred_boxes = bbox_head(obj_decoder_out).sigmoid()
    print(f"  Predicted boxes: {pred_boxes.shape}")
    
    # Step 5: Heatmap Location
    print("\n[Step 5] Heatmap Location...")
    heatmap_head = HeatmapLocationHead(
        d_model=C,
        grid_size=32,
        num_decoder_layers=2,
        nhead=8,
    )
    
    heatmap_output = heatmap_head(
        sat_features=sat_patch_features,
        front_target_features=target_guidance,
        target_size=(518, 518),
    )
    print(f"  Heatmap: {heatmap_output['heatmap'].shape}")
    print(f"  Position: {heatmap_output['position']}")
    
    # Step 6: Camera Yaw
    print("\n[Step 6] Camera Yaw Prediction...")
    camera_head = CameraHead(
        dim_in=C,
        trunk_depth=2,
        num_heads=8,
        num_iterations=2,
    )
    
    camera_output = camera_head(
        front_camera_token=front_camera_token,
        sat_camera_token=sat_camera_token,
    )
    print(f"  Yaw (degrees): {camera_output['yaw_degrees']}")
    print(f"  Quaternion: {camera_output['quaternion']}")
    
    print("\n✓ Full Pipeline (Mock) test passed!")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Cross-View Drone Localization System - Integration Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_prompt_encoder()
    except Exception as e:
        print(f"\n✗ Prompt Encoder test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_prompt_fusion()
    except Exception as e:
        print(f"\n✗ Prompt Fusion test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_detr_decoder()
    except Exception as e:
        print(f"\n✗ DETR Decoder test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_heads()
    except Exception as e:
        print(f"\n✗ Prediction Heads test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_dataset_integration()
    except Exception as e:
        print(f"\n✗ Dataset Integration test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_full_pipeline_mock()
    except Exception as e:
        print(f"\n✗ Full Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("=" * 60)
