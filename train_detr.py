"""
Cross-View Localization Training Script for DETR-style Model
Supports the unified decoder architecture with object queries and location queries.
"""

import argparse
import math
import yaml
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_scheduler

from models import CrossViewLocalizerDETR
from data import CrossViewDataset, collate_fn
from utils import (
    load_vggt_weights,
    freeze_backbone,
    get_param_groups,
    prepare_random_prompt,
    visualize_validation_samples,
    box_cxcywh_to_xyxy, 
    generalized_box_iou,
    DETRCriterion,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cross-View Localizer DETR')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file (YAML)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
    epoch: int,
    cfg: dict,
):
    """Train for one epoch."""
    model.train()
    
    total_losses = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not accelerator.is_main_process)
    
    for batch_idx, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            front_view = batch['front_view']
            sat_view = batch['satellite_view']
            
            # Prepare targets
            targets = {
                'sat_bbox': batch['sat_bbox'],
                'yaw_radians': batch['yaw_radians'],
                'camera_position': batch['camera_position'],
            }
            
            # Random prompt selection
            points, boxes, masks = prepare_random_prompt(batch, accelerator.device)
            
            # Forward
            with accelerator.autocast():
                outputs = model(
                    front_view=front_view,
                    satellite_view=sat_view,
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )
                losses = criterion(outputs, targets)
                loss = losses['loss']
            
            # Backward
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # Update progress bar
        if accelerator.is_main_process:
            pbar.set_postfix({
                'loss': f'{losses["loss"].item():.4f}',
                'bbox': f'{losses.get("loss_bbox", 0):.4f}' if isinstance(losses.get("loss_bbox", 0), float) else f'{losses.get("loss_bbox", torch.tensor(0)).item():.4f}',
            })
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    # Log
    accelerator.log({f"train/{k}": v for k, v in avg_losses.items()}, step=epoch)
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator,
    cfg: dict,
    epoch: int = 0,
):
    """Validate the model."""
    model.eval()
    
    total_losses = {}
    all_pos_errors = []
    all_yaw_errors = []
    
    for batch in tqdm(dataloader, desc='Validation', disable=not accelerator.is_main_process):
        front_view = batch['front_view']
        sat_view = batch['satellite_view']
        
        targets = {
            'sat_bbox': batch['sat_bbox'],
            'yaw_radians': batch['yaw_radians'],
            'camera_position': batch['camera_position'],
        }
        
        # Use point prompt for validation
        B = front_view.shape[0]
        mono_point = batch['mono_point']
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(B, 1, device=front_view.device)
        
        with accelerator.autocast():
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            losses = criterion(outputs, targets)
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # Compute metrics
        if 'position' in outputs:
            pos_error = (outputs['position'] - targets['camera_position']).norm(dim=-1)
            all_pos_errors.append(pos_error)
        
        if 'yaw_radians' in outputs:
            yaw_diff = outputs['yaw_radians'] - targets['yaw_radians']
            yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            all_yaw_errors.append(yaw_diff.abs())
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    # Gather metrics
    if all_pos_errors:
        all_pos_errors = accelerator.gather_for_metrics(torch.cat(all_pos_errors))
        avg_losses['pos_mae'] = all_pos_errors.mean().item()
        avg_losses['pos_mae_pixels'] = avg_losses['pos_mae'] * cfg['data']['img_size']
    
    if all_yaw_errors:
        all_yaw_errors = accelerator.gather_for_metrics(torch.cat(all_yaw_errors))
        avg_losses['yaw_mae'] = all_yaw_errors.mean().item()
        avg_losses['yaw_mae_deg'] = math.degrees(avg_losses['yaw_mae'])
    
    # Log
    accelerator.log({f"val/{k}": v for k, v in avg_losses.items()}, step=epoch)
    
    return avg_losses


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    if args.resume:
        cfg['checkpoint']['resume'] = args.resume
    
    # Initialize Accelerator
    gradient_accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    mixed_precision = cfg['training'].get('mixed_precision', 'bf16') if cfg['training'].get('use_amp', True) else "no"
    
    output_dir = Path(cfg['checkpoint']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard" if cfg['logging'].get('use_tensorboard', True) else None,
        project_dir=str(output_dir),
    )
    
    set_seed(42)
    
    # Save config
    if accelerator.is_main_process:
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)
        accelerator.print(f"Output directory: {output_dir}")
    
    # Initialize tracker
    if accelerator.is_main_process and cfg['logging'].get('use_tensorboard', True):
        accelerator.init_trackers(
            project_name="cross_view_detr",
            config={
                "batch_size": cfg['training']['batch_size'],
                "num_epochs": cfg['training']['num_epochs'],
                "lr_backbone": cfg['training']['lr_backbone'],
                "lr_heads": cfg['training']['lr_heads'],
            },
        )
    
    # Create datasets
    train_dataset = CrossViewDataset(
        json_path=cfg['data']['train_json'],
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        random_crop=True,
    )
    
    val_dataset = CrossViewDataset(
        json_path=cfg['data']['val_json'],
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        random_crop=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    accelerator.print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')
    
    # Create model
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
    
    # Load pretrained weights
    if cfg['model'].get('vggt_weights'):
        load_vggt_weights(model, cfg['model']['vggt_weights'], load_heads=False)
        accelerator.print(f'Loaded VGGT weights from {cfg["model"]["vggt_weights"]}')
    
    # Freeze backbone
    if cfg['model'].get('freeze_patch_embed', True):
        freeze_backbone(model, freeze_patch_embed=True, freeze_aggregator=cfg['model'].get('freeze_aggregator', False))
        accelerator.print('Froze backbone')
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
    # Create criterion
    criterion = DETRCriterion(
        weight_bbox=cfg['training'].get('weight_bbox', 5.0),
        weight_giou=cfg['training'].get('weight_giou', 2.0),
        weight_heatmap=cfg['training'].get('weight_heatmap', 1.0),
        weight_yaw=cfg['training'].get('weight_yaw', 1.0),
        img_size=cfg['data']['img_size'],
    )
    
    # Create optimizer
    param_groups = get_param_groups(
        model,
        lr_backbone=cfg['training']['lr_backbone'],
        lr_heads=cfg['training']['lr_heads'],
        weight_decay=cfg['training']['weight_decay'],
    )
    optimizer = AdamW(param_groups)
    
    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Create scheduler
    num_epochs = cfg['training']['num_epochs']
    warmup_epochs = cfg['training']['warmup_epochs']
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * warmup_epochs
    
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    scheduler = accelerator.prepare(scheduler)
    
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if cfg['checkpoint'].get('resume'):
        accelerator.print(f'Resuming from {cfg["checkpoint"]["resume"]}')
        accelerator.load_state(cfg['checkpoint']['resume'])
        ckpt_path = Path(cfg['checkpoint']['resume'])
        if (ckpt_path / 'training_state.pt').exists():
            training_state = torch.load(ckpt_path / 'training_state.pt', map_location='cpu')
            start_epoch = training_state.get('epoch', 0) + 1
            best_loss = training_state.get('best_loss', float('inf'))
        accelerator.print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        accelerator.print(f'\n{"="*50}')
        accelerator.print(f'Epoch {epoch}/{num_epochs}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, accelerator, epoch, cfg
        )
        accelerator.print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        
        # Validate
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, accelerator, cfg, epoch)
            accelerator.print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
            
            # Save best
            if val_losses['loss'] < best_loss:
                best_loss = val_losses['loss']
                accelerator.save_state(output_dir / 'best')
                if accelerator.is_main_process:
                    torch.save({
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'val_losses': val_losses,
                    }, output_dir / 'best' / 'training_state.pt')
                    accelerator.print(f'Saved best model (loss: {best_loss:.4f})')
        
        # Save checkpoint
        if (epoch + 1) % cfg['checkpoint']['save_freq'] == 0:
            save_dir = output_dir / f'epoch_{epoch}'
            accelerator.save_state(save_dir)
            if accelerator.is_main_process:
                torch.save({
                    'epoch': epoch,
                    'best_loss': best_loss,
                }, save_dir / 'training_state.pt')
    
    accelerator.end_training()
    accelerator.print(f'\nTraining completed! Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
