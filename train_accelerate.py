"""
Cross-View Localization Training Script with Hugging Face Accelerate
Supports DeepSpeed ZeRO for memory-efficient training of large models.
"""

import argparse
import math
import yaml
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from models import CrossViewLocalizer
from data.dataset import CrossViewDataset, collate_fn
from utils import (MultiTaskLoss, load_vggt_weights, load_dinov2_weights, 
                   freeze_backbone, get_param_groups, TensorBoardLogger)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cross-View Localizer with Accelerate + DeepSpeed')
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
    tb_logger: TensorBoardLogger = None,
):
    """Train for one epoch with Accelerate."""
    model.train()
    
    total_losses = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not accelerator.is_main_process)
    
    for batch_idx, batch in enumerate(pbar):
        # Data is already on device via Accelerate
        front_view = batch['front_view']
        sat_view = batch['satellite_view']
        mono_point = batch['mono_point']
        
        # 准备targets
        targets = {
            'sat_bbox': batch['sat_bbox'],
            'yaw_radians': batch['yaw_radians'],
            'camera_position': batch['camera_position'],
        }
        
        # 准备point prompt
        B = front_view.shape[0]
        point_coords = mono_point.unsqueeze(1)  # [B, 1, 2]
        point_labels = torch.ones(B, 1, device=front_view.device)  # 正点
        
        # Forward with automatic mixed precision via Accelerate
        with accelerator.autocast():
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=(point_coords, point_labels),
            )
            losses = criterion(outputs, targets)
            loss = losses['loss']
        
        # Check for NaN/Inf in loss before backward
        if not torch.isfinite(loss):
            accelerator.print(f"WARNING: Non-finite loss detected at batch {batch_idx}: {loss.item()}")
            accelerator.print(f"Losses: {', '.join([f'{k}={v.item():.4f}' for k, v in losses.items()])}")
            optimizer.zero_grad()
            continue
        
        # Backward with gradient accumulation handled by Accelerate
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            # Clip gradients and check for anomalies
            total_norm = accelerator.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            
            # Check gradient norm (handles both Tensor and float return types)
            norm_value = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
            if not math.isfinite(norm_value):
                accelerator.print(f"WARNING: Non-finite gradient norm at batch {batch_idx}, skipping update")
                optimizer.zero_grad()
                continue
        
        optimizer.step()
        scheduler.step()  # Per-step LR update for smooth warmup
        optimizer.zero_grad()
        
        # Accumulate losses (gather across processes)
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            # Use accelerator.gather for proper averaging across GPUs
            gathered_v = accelerator.gather(v.detach()).mean()
            total_losses[k] += gathered_v.item()
        
        # Update progress bar
        if accelerator.is_main_process:
            pbar.set_postfix({k: f'{v.item():.4f}' for k, v in losses.items()})
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    # Log to TensorBoard
    if tb_logger and accelerator.is_main_process:
        tb_logger.log_dict("train", avg_losses, epoch)
        tb_logger.log_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator,
    cfg: dict,
    epoch: int = 0,
    tb_logger: TensorBoardLogger = None,
):
    """Validate the model."""
    model.eval()
    
    total_losses = {}
    total_bbox_error = 0.0
    total_yaw_error = 0.0
    total_pos_error = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc='Validation', disable=not accelerator.is_main_process):
        front_view = batch['front_view']
        sat_view = batch['satellite_view']
        mono_point = batch['mono_point']
        
        targets = {
            'sat_bbox': batch['sat_bbox'],
            'yaw_radians': batch['yaw_radians'],
            'camera_position': batch['camera_position'],
        }
        
        B = front_view.shape[0]
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(B, 1, device=front_view.device)
        
        with accelerator.autocast():
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=(point_coords, point_labels),
            )
            losses = criterion(outputs, targets)
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            gathered_v = accelerator.gather(v.detach()).mean()
            total_losses[k] += gathered_v.item()
        
        # Compute metrics
        if 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes'][:, 0, :] if outputs['pred_boxes'].dim() == 3 else outputs['pred_boxes']
            bbox_error = (pred_boxes - targets['sat_bbox']).abs().mean()
            total_bbox_error += accelerator.gather(bbox_error).mean().item() * B
        
        if 'yaw_radians' in outputs:
            yaw_diff = outputs['yaw_radians'] - targets['yaw_radians']
            yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            yaw_error = yaw_diff.abs().mean()
            total_yaw_error += accelerator.gather(yaw_error).mean().item() * B
        
        if 'position' in outputs:
            pos_error = (outputs['position'] - targets['camera_position']).norm(dim=-1).mean()
            total_pos_error += accelerator.gather(pos_error).mean().item() * B
        
        num_samples += B * accelerator.num_processes
    
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    avg_losses['bbox_mae'] = total_bbox_error / num_samples if num_samples > 0 else 0
    avg_losses['yaw_mae'] = total_yaw_error / num_samples if num_samples > 0 else 0
    avg_losses['pos_error'] = total_pos_error / num_samples if num_samples > 0 else 0
    
    # Log to TensorBoard
    if tb_logger and accelerator.is_main_process:
        tb_logger.log_dict("val", avg_losses, epoch)
    
    return avg_losses


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Override resume path if provided via command line
    if args.resume:
        cfg['checkpoint']['resume'] = args.resume
    
    # Get gradient accumulation steps from config
    gradient_accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    
    # Initialize Accelerator
    # mixed_precision: "no", "fp16", "bf16"
    # bf16 is recommended for Ampere+ GPUs (RTX 3090/4090, A100, H100)
    if cfg['training'].get('use_amp', False):
        mixed_precision = cfg['training'].get('mixed_precision', 'bf16')
    else:
        mixed_precision = "no"
    
    # Setup output directory first for logging
    output_dir = Path(cfg['checkpoint']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard" if cfg['logging'].get('use_tensorboard', True) else None,
        project_dir=str(output_dir),  # Required for tensorboard logging
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Save config (output_dir already created above)
    if accelerator.is_main_process:
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)
        accelerator.print(f"Output directory: {output_dir}")
    
    # Initialize TensorBoard logger (only on main process)
    tb_logger = TensorBoardLogger(
        log_dir=str(log_dir),
        enabled=cfg['logging'].get('use_tensorboard', True),
        rank=0 if accelerator.is_main_process else 1,  # Only main process logs
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
    model = CrossViewLocalizer(
        img_size=cfg['data']['img_size'],
        embed_dim=cfg['model']['embed_dim'],
        vggt_depth=cfg['model']['vggt_depth'],
        num_heads=cfg['model']['num_heads'],
        num_decoder_layers=cfg['model']['num_decoder_layers'],
        enable_bbox=cfg['model']['enable_bbox'],
        enable_seg=cfg['model']['enable_seg'],
        enable_camera=cfg['model']['enable_camera'],
        enable_position=cfg['model']['enable_position'],
    )
    
    # Load pretrained weights (before wrapping with Accelerate)
    if cfg['model'].get('vggt_weights'):
        load_vggt_weights(model, cfg['model']['vggt_weights'], load_heads=False)
        accelerator.print(f'Loaded VGGT weights from {cfg["model"]["vggt_weights"]}')
    elif cfg['model'].get('dinov2_weights'):
        load_dinov2_weights(model, dinov2_path=cfg['model']['dinov2_weights'])
        accelerator.print(f'Loaded DINOv2 weights from {cfg["model"]["dinov2_weights"]}')
    
    # Freeze backbone (patch_embed only, NOT aggregator for better training)
    freeze_patch_embed = cfg['model'].get('freeze_patch_embed', True)
    freeze_aggregator = cfg['model'].get('freeze_aggregator', False)
    
    if freeze_patch_embed:
        freeze_backbone(model, freeze_patch_embed=True, freeze_aggregator=freeze_aggregator)
        accelerator.print(f'Froze backbone (patch_embed={freeze_patch_embed}, aggregator={freeze_aggregator})')
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
    # Create loss
    criterion = MultiTaskLoss(
        weight_bbox=cfg['training']['weight_bbox'],
        weight_giou=cfg['training']['weight_giou'],
        weight_yaw=cfg['training']['weight_yaw'],
        weight_position=cfg['training']['weight_position'],
        weight_mask=cfg['training']['weight_mask'],
    )
    
    # Create optimizer with different LR for backbone and heads
    param_groups = get_param_groups(
        model,
        lr_backbone=cfg['training']['lr_backbone'],
        lr_heads=cfg['training']['lr_heads'],
        weight_decay=cfg['training']['weight_decay'],
    )
    optimizer = AdamW(param_groups)
    
    # Note: Scheduler will be created after prepare() to get correct number of steps per epoch
    num_epochs = cfg['training']['num_epochs']
    warmup_epochs = cfg['training']['warmup_epochs']
    
    # Prepare with Accelerator (handles DDP/DeepSpeed wrapping)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Create scheduler with per-step warmup (after prepare to get correct steps per epoch)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * warmup_epochs
    
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler = accelerator.prepare(scheduler)
    
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if cfg['checkpoint'].get('resume'):
        accelerator.print(f'Resuming from {cfg["checkpoint"]["resume"]}')
        accelerator.load_state(cfg['checkpoint']['resume'])
        # Try to load epoch info
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
        with accelerator.accumulate(model):
            train_losses = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, accelerator, epoch, cfg, tb_logger
            )
        accelerator.print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        accelerator.print(f'LR - backbone: {optimizer.param_groups[0]["lr"]:.2e}, heads: {optimizer.param_groups[1]["lr"]:.2e}')
        
        # Validate
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, accelerator, cfg, epoch, tb_logger)
            accelerator.print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
            
            # Save best (all ranks must participate in save_state for NCCL sync)
            if val_losses['loss'] < best_loss:
                best_loss = val_losses['loss']
                accelerator.save_state(output_dir / 'best')
                # Save additional training state (only main process)
                if accelerator.is_main_process:
                    torch.save({
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'val_losses': val_losses,
                    }, output_dir / 'best' / 'training_state.pt')
                    accelerator.print(f'Saved best model (loss: {best_loss:.4f})')
        
        # Save checkpoint (all ranks must participate in save_state for NCCL sync)
        if (epoch + 1) % cfg['checkpoint']['save_freq'] == 0:
            save_dir = output_dir / f'epoch_{epoch}'
            accelerator.save_state(save_dir)
            # Save additional training state (only main process)
            if accelerator.is_main_process:
                torch.save({
                    'epoch': epoch,
                    'best_loss': best_loss,
                }, save_dir / 'training_state.pt')
    
    # Close TensorBoard logger
    if tb_logger:
        tb_logger.close()
    
    accelerator.print(f'\nTraining completed! Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
