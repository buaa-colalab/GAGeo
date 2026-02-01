"""
Cross-View Localization Test Script for DETR-style Model
测试模型在验证集/测试集上的性能
"""

import argparse
import json
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CrossViewLocalizerDETR
from data import CrossViewDataset, collate_fn
from utils import (
    DETRCriterion,
    compute_iou,
    box_cxcywh_to_xyxy,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Cross-View Localizer DETR')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory (Accelerate format)')
    parser.add_argument('--data_json', type=str, default=None,
                        help='Override test data json path')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./output/test_results')
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, cfg, device):
    """Load model from Accelerate checkpoint."""
    model = CrossViewLocalizerDETR(
        img_size=cfg['data']['img_size'],
        embed_dim=cfg['model']['embed_dim'],
        vggt_depth=cfg['model']['vggt_depth'],
        num_heads=cfg['model']['num_heads'],
        num_decoder_layers=cfg['model'].get('num_decoder_layers', 6),
        num_object_queries=cfg['model'].get('num_object_queries', 100),
        location_grid_size=cfg['model'].get('location_grid_size', 32),
        freeze_vggt=False,
        use_prompt_fusion=cfg['model'].get('use_prompt_fusion', True),
    )
    
    ckpt_path = Path(checkpoint_path)
    
    # Try Accelerate format first
    model_file = ckpt_path / 'pytorch_model.bin'
    if not model_file.exists():
        model_file = ckpt_path / 'model.safetensors'
    if not model_file.exists():
        # Try direct checkpoint file
        model_file = ckpt_path
    
    if model_file.exists() and model_file.is_file():
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Cannot find model checkpoint at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model performance."""
    model.eval()
    
    total_losses = {}
    all_pred_boxes = []
    all_target_boxes = []
    all_pred_positions = []
    all_target_positions = []
    all_yaw_errors = []
    all_bbox_scores = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        mono_point = batch['mono_point'].to(device)
        
        B = front_view.shape[0]
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(B, 1, device=device)
        
        outputs = model(
            front_view=front_view,
            satellite_view=sat_view,
            points=(point_coords, point_labels),
        )
        
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'yaw_radians': batch['yaw_radians'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        losses = criterion(outputs, targets)
        
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # Collect predictions
        if 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes']  # [B, N, 4]
            bbox_scores = outputs['bbox_scores']  # [B, N]
            
            # Select best box by score
            best_idx = bbox_scores.argmax(dim=1)  # [B]
            best_boxes = pred_boxes[torch.arange(B, device=device), best_idx]  # [B, 4]
            
            all_pred_boxes.append(best_boxes.cpu())
            all_target_boxes.append(batch['sat_bbox'])
            all_bbox_scores.append(bbox_scores.max(dim=1).values.cpu())
        
        if 'position' in outputs:
            all_pred_positions.append(outputs['position'].cpu())
            all_target_positions.append(batch['camera_position'])
        
        if 'yaw_radians' in outputs:
            pred_yaw = outputs['yaw_radians'].cpu()
            target_yaw = batch['yaw_radians']
            yaw_diff = torch.atan2(
                torch.sin(pred_yaw - target_yaw),
                torch.cos(pred_yaw - target_yaw)
            )
            all_yaw_errors.append(torch.abs(yaw_diff))
    
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Compute metrics
    metrics = {}
    
    # BBox metrics
    if all_pred_boxes:
        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        target_boxes = torch.cat(all_target_boxes, dim=0)
        
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        iou_matrix = compute_iou(pred_xyxy, target_xyxy)
        diag_iou = torch.diag(iou_matrix)
        
        metrics['mean_iou'] = diag_iou.mean().item()
        metrics['iou@0.5'] = (diag_iou >= 0.5).float().mean().item()
        metrics['iou@0.75'] = (diag_iou >= 0.75).float().mean().item()
        
        # BBox L1 error
        bbox_l1 = (pred_boxes - target_boxes).abs().mean().item()
        metrics['bbox_l1'] = bbox_l1
    
    # Position metrics
    if all_pred_positions:
        pred_pos = torch.cat(all_pred_positions, dim=0)
        target_pos = torch.cat(all_target_positions, dim=0)
        
        dist_error = torch.norm(pred_pos - target_pos, dim=1)
        metrics['mean_position_error'] = dist_error.mean().item()
        metrics['position_error_std'] = dist_error.std().item()
        metrics['position_error_pixels'] = dist_error.mean().item() * 518  # Assuming 518 img_size
    
    # Yaw metrics
    if all_yaw_errors:
        yaw_errors = torch.cat(all_yaw_errors, dim=0)
        metrics['mean_yaw_error_rad'] = yaw_errors.mean().item()
        metrics['mean_yaw_error_deg'] = torch.rad2deg(yaw_errors).mean().item()
    
    return avg_losses, metrics


def main():
    args = parse_args()
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    data_json = args.data_json or cfg['data']['val_json']
    dataset = CrossViewDataset(
        json_path=data_json,
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        random_crop=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg['data'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f'Test data: {len(dataset)} samples')
    
    # Model
    print(f'Loading model from {args.checkpoint}')
    model = load_model(args.checkpoint, cfg, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params/1e6:.1f}M')
    
    # Criterion
    criterion = DETRCriterion(
        weight_bbox=cfg['training'].get('weight_bbox', 5.0),
        weight_giou=cfg['training'].get('weight_giou', 2.0),
        weight_heatmap=cfg['training'].get('weight_heatmap', 1.0),
        weight_yaw=cfg['training'].get('weight_yaw', 1.0),
        img_size=cfg['data']['img_size'],
    )
    
    # Evaluate
    avg_losses, metrics = evaluate(model, dataloader, criterion, device)
    
    # Print results
    print('\n' + '='*50)
    print('Evaluation Results')
    print('='*50)
    
    print('\nLosses:')
    for k, v in avg_losses.items():
        print(f'  {k}: {v:.4f}')
    
    print('\nMetrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': args.checkpoint,
        'data_json': data_json,
        'num_samples': len(dataset),
        'losses': avg_losses,
        'metrics': metrics,
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
