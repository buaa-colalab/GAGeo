"""
Cross-View Localization Test Script

测试模型在验证集/测试集上的性能
"""

import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CrossViewLocalizer
from data.dataset import CrossViewDataset, collate_fn
from utils import (
    MultiTaskLoss, 
    load_vggt_weights, 
    load_dinov2_weights,
    compute_iou,
    compute_localization_accuracy,
    compute_distance_error,
    box_cxcywh_to_xyxy,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Cross-View Localizer')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_json', type=str, default=None,
                        help='Override test data json path')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./output/test_results')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization results')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型性能"""
    model.eval()
    
    total_loss = 0.0
    all_losses = {}
    
    all_pred_boxes = []
    all_target_boxes = []
    all_pred_positions = []
    all_target_positions = []
    all_yaw_errors = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        # Prepare inputs
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        mono_point = batch['mono_point'].to(device)
        
        B = front_view.shape[0]
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(B, 1, device=device)
        
        # Forward
        outputs = model(
            front_view=front_view,
            satellite_view=sat_view,
            points=(point_coords, point_labels)
        )
        
        # Prepare targets
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'yaw_radians': batch['yaw_radians'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        # Compute loss
        losses = criterion(outputs, targets)
        total_loss += losses['loss'].item()
        
        for k, v in losses.items():
            if k not in all_losses:
                all_losses[k] = 0.0
            all_losses[k] += v.item()
        
        # Collect predictions for metrics
        if 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes']
            if pred_boxes.dim() == 3:
                pred_boxes = pred_boxes[:, 0, :]  # [B, N, 4] -> [B, 4]
            all_pred_boxes.append(pred_boxes.cpu())
            all_target_boxes.append(batch['sat_bbox'])
        
        if 'position' in outputs:
            all_pred_positions.append(outputs['position'].cpu())
            all_target_positions.append(batch['camera_position'])
        
        if 'yaw_radians' in outputs:
            pred_yaw = outputs['yaw_radians'].cpu()
            target_yaw = batch['yaw_radians']
            # 计算角度误差 (考虑周期性)
            yaw_diff = torch.atan2(
                torch.sin(pred_yaw - target_yaw),
                torch.cos(pred_yaw - target_yaw)
            )
            all_yaw_errors.append(torch.abs(yaw_diff))
    
    num_batches = len(dataloader)
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in all_losses.items()}
    
    # Compute metrics
    metrics = {}
    
    # BBox metrics
    if all_pred_boxes:
        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        target_boxes = torch.cat(all_target_boxes, dim=0)
        
        # Convert to xyxy for IoU
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # IoU
        iou_matrix = compute_iou(pred_xyxy, target_xyxy)
        diag_iou = torch.diag(iou_matrix)
        metrics['mean_iou'] = diag_iou.mean().item()
        metrics['iou@0.5'] = (diag_iou >= 0.5).float().mean().item()
        metrics['iou@0.75'] = (diag_iou >= 0.75).float().mean().item()
        
        # Localization accuracy
        precision, recall, f1 = compute_localization_accuracy(pred_xyxy, target_xyxy, threshold=0.5)
        metrics['precision@0.5'] = precision
        metrics['recall@0.5'] = recall
        metrics['f1@0.5'] = f1
    
    # Position metrics
    if all_pred_positions:
        pred_pos = torch.cat(all_pred_positions, dim=0)
        target_pos = torch.cat(all_target_positions, dim=0)
        
        # 距离误差 (归一化坐标)
        dist_error = torch.norm(pred_pos - target_pos, dim=1)
        metrics['mean_position_error'] = dist_error.mean().item()
        metrics['position_error_std'] = dist_error.std().item()
    
    # Yaw metrics
    if all_yaw_errors:
        yaw_errors = torch.cat(all_yaw_errors, dim=0)
        metrics['mean_yaw_error_rad'] = yaw_errors.mean().item()
        metrics['mean_yaw_error_deg'] = torch.rad2deg(yaw_errors).mean().item()
    
    return avg_losses, metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    data_json = args.data_json or cfg['data']['val_json']
    dataset = CrossViewDataset(
        json_path=data_json,
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        random_crop=False,  # 测试时不随机crop
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
    model = CrossViewLocalizer(
        img_size=cfg['data']['crop_size'],
        embed_dim=cfg['model']['embed_dim'],
        vggt_depth=cfg['model']['vggt_depth'],
        num_heads=cfg['model']['num_heads'],
        num_decoder_layers=cfg['model']['num_decoder_layers'],
        enable_bbox=cfg['model']['enable_bbox'],
        enable_seg=cfg['model']['enable_seg'],
        enable_camera=cfg['model']['enable_camera'],
        enable_position=cfg['model']['enable_position'],
    ).to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f'Loaded model from epoch {epoch}')
    else:
        model.load_state_dict(checkpoint)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params/1e6:.1f}M')
    
    # Loss (for evaluation)
    criterion = MultiTaskLoss(
        weight_bbox=cfg['training']['weight_bbox'],
        weight_giou=cfg['training']['weight_giou'],
        weight_yaw=cfg['training']['weight_yaw'],
        weight_position=cfg['training']['weight_position'],
        weight_mask=cfg['training']['weight_mask'],
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
    
    import json
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
