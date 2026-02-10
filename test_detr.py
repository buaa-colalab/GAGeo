"""
Cross-View Localization Test Script for DETR-style Model
测试模型在验证集/测试集上的性能
"""

import argparse
import json
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CrossViewLocalizerPi3, build_cross_view_localizer_pi3
from data import CrossViewDataset, collate_fn
from utils import (
    DETRCriterion,
    compute_iou,
    box_cxcywh_to_xyxy,
)
from utils.prompt_utils import prepare_single_prompt
from utils.visualize import visualize_batch_sample


def parse_args():
    parser = argparse.ArgumentParser(description='Test Cross-View Localizer DETR')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory (Accelerate format)')
    parser.add_argument('--data_json', type=str, default=None,
                        help='Override test data json path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./output/test_results')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--prompt_type', type=str, default='all', choices=['point', 'bbox', 'mask', 'all'],
                        help='Prompt type to test (default: all)')
    parser.add_argument('--vis_samples', type=int, default=20,
                        help='Number of samples to visualize per prompt type (0 to disable)')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, cfg, device):
    """Load model from checkpoint."""
    model = build_cross_view_localizer_pi3(
        pretrained_pi3=None,
        freeze_backbone=False,
        img_size=cfg['data']['img_size'],
        decoder_size=cfg['model'].get('decoder_size', 'large'),
        num_heads=cfg['model']['num_heads'],
        num_decoder_layers=cfg['model'].get('num_decoder_layers', 6),
        num_object_queries=cfg['model'].get('num_object_queries', 10),
        num_location_queries=cfg['model'].get('num_location_queries', 16),
        num_intent_queries=cfg['model'].get('num_intent_queries', 32),
        prompt_fusion_layers=cfg['model'].get('prompt_fusion_layers', 3),
        dropout=cfg['model'].get('dropout', 0.1),
        contrastive=cfg['model'].get('contrastive', False),
        contrastive_proj_dim=cfg['model'].get('contrastive_proj_dim', 256),
        contrastive_queue_size=cfg['model'].get('contrastive_queue_size', 16384),
        contrastive_momentum=cfg['model'].get('contrastive_momentum', 0.999),
        contrastive_temperature=cfg['model'].get('contrastive_temperature', 0.07),
    )
    
    ckpt_path = Path(checkpoint_path)
    
    # Search for checkpoint file in multiple locations
    candidates = [
        ckpt_path / 'converted_fp32' / 'pytorch_model.bin',  # DeepSpeed converted
        ckpt_path / 'pytorch_model.bin',                      # Accelerate format
        ckpt_path / 'model.safetensors',                      # safetensors format
        ckpt_path,                                             # direct file path
    ]
    
    model_file = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            model_file = candidate
            break
    
    if model_file is None:
        raise FileNotFoundError(f"Cannot find model checkpoint at {checkpoint_path}")
    
    print(f'Loading weights from {model_file}')
    state_dict = torch.load(model_file, map_location=device, weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Strip 'module.' prefix (from Accelerate/DeepSpeed wrapping)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
        cleaned_state_dict[new_key] = v
    
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f'  Missing keys ({len(missing)}):')
        for k in missing:
            print(f'    - {k}')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}):')
        for k in unexpected:
            print(f'    - {k}')
    
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, prompt_type='point',
            vis_dir=None, vis_samples=0, img_size=518):
    """Evaluate model performance with specific prompt type."""
    model.eval()
    
    total_losses = {}
    all_pred_boxes = []
    all_target_boxes = []
    all_pred_positions = []
    all_target_positions = []
    all_rotation_errors = []
    all_bbox_scores = []
    vis_count = 0
    
    for batch in tqdm(dataloader, desc=f'Evaluating ({prompt_type})'):
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        
        # Use specified prompt type
        points, boxes, masks = prepare_single_prompt(batch, device, prompt_type=prompt_type)
        
        outputs = model(
            front_view=front_view,
            satellite_view=sat_view,
            points=points,
            boxes=boxes,
            masks=masks,
        )
        
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'rotation_matrix': batch['rotation_matrix'].to(device),
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
            B = pred_boxes.shape[0]
            best_idx = bbox_scores.argmax(dim=1)  # [B]
            best_boxes = pred_boxes[torch.arange(B, device=device), best_idx]  # [B, 4]
            
            all_pred_boxes.append(best_boxes.cpu())
            all_target_boxes.append(batch['sat_bbox'])
            all_bbox_scores.append(bbox_scores.max(dim=1).values.cpu())
        
        if 'position' in outputs:
            all_pred_positions.append(outputs['position'].cpu())
            all_target_positions.append(batch['camera_position'])
        
        if 'rotation_error_deg' in losses:
            all_rotation_errors.append(torch.tensor([losses['rotation_error_deg']]))
        
        # Visualize samples
        if vis_dir is not None and vis_count < vis_samples:
            B = front_view.shape[0]
            for i in range(B):
                if vis_count >= vis_samples:
                    break
                save_path = vis_dir / f'{prompt_type}_{vis_count:04d}.png'
                visualize_batch_sample(batch, outputs, i, img_size, save_path, prompt_type=prompt_type)
                vis_count += 1
    
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
        metrics['iou@0.25'] = (diag_iou >= 0.25).float().mean().item()
        metrics['iou@0.5'] = (diag_iou >= 0.5).float().mean().item()
        
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
    
    # Rotation metrics
    if all_rotation_errors:
        metrics['mean_rotation_error_deg'] = torch.cat(all_rotation_errors).mean().item()
    
    return avg_losses, metrics


def main():
    args = parse_args()
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    cfg = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f'Using {num_gpus} GPU(s): {args.gpu}')
    
    # Data
    data_json = args.data_json or cfg['data']['val_json']
    dataset = CrossViewDataset(
        json_path=data_json,
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        crop_sat=False,  # val/test 数据已经是 crop 好的
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
    
    # Multi-GPU DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f'Using DataParallel on {num_gpus} GPUs')
    
    # Criterion
    criterion = DETRCriterion(
        weight_bbox=cfg['training'].get('weight_bbox', 5.0),
        weight_giou=cfg['training'].get('weight_giou', 2.0),
        weight_heatmap=cfg['training'].get('weight_heatmap', 1.0),
        weight_rotation=cfg['training'].get('weight_rotation', 1.0),
        weight_contrastive=cfg['training'].get('weight_contrastive', 0.1),
        img_size=cfg['data']['img_size'],
    )
    
    # Evaluate with all three prompt types
    prompt_types = ['point', 'bbox', 'mask']
    all_results = {}
    
    for prompt_type in prompt_types:
        print(f'\n{"="*50}')
        print(f'Evaluating with {prompt_type.upper()} prompt')
        print('='*50)
        
        # Setup visualization directory
        vis_dir = None
        if args.vis_samples > 0:
            vis_dir = Path(args.output_dir) / 'vis' / prompt_type
            vis_dir.mkdir(parents=True, exist_ok=True)
        
        avg_losses, metrics = evaluate(
            model, dataloader, criterion, device, prompt_type=prompt_type,
            vis_dir=vis_dir, vis_samples=args.vis_samples, img_size=cfg['data']['img_size'],
        )
        
        print('\nLosses:')
        for k, v in avg_losses.items():
            print(f'  {k}: {v:.4f}')
        
        print('\nMetrics:')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')
        
        all_results[prompt_type] = {
            'losses': avg_losses,
            'metrics': metrics,
        }
    
    # Print summary comparison
    print(f'\n{"="*50}')
    print('Summary Comparison')
    print('='*50)
    print(f'\n{"Metric":<30} {"Point":<12} {"BBox":<12} {"Mask":<12}')
    print('-' * 66)
    
    # Compare key metrics
    key_metrics = ['mean_iou', 'iou@0.25', 'iou@0.5', 'mean_position_error', 'mean_rotation_error_deg']
    for metric in key_metrics:
        if metric in all_results['point']['metrics']:
            values = [all_results[pt]['metrics'].get(metric, 0) for pt in prompt_types]
            print(f'{metric:<30} {values[0]:<12.4f} {values[1]:<12.4f} {values[2]:<12.4f}')
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': args.checkpoint,
        'data_json': data_json,
        'num_samples': len(dataset),
        'results_by_prompt': all_results,
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
