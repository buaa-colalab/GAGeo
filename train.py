"""
Cross-View Localization Training Script
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import CrossViewLocalizer
from data.dataset import CrossViewDataset, collate_fn
from utils import MultiTaskLoss, load_vggt_weights, load_dinov2_weights, freeze_backbone, get_param_groups


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cross-View Localizer')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    # 允许命令行覆盖配置
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: dict,
):
    """Train for one epoch."""
    model.train()
    
    total_losses = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        mono_point = batch['mono_point'].to(device)
        
        # 准备targets
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'yaw_radians': batch['yaw_radians'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        # 准备point prompt
        B = front_view.shape[0]
        point_coords = mono_point.unsqueeze(1)  # [B, 1, 2]
        point_labels = torch.ones(B, 1, device=device)  # 正点
        
        optimizer.zero_grad()
        
        # Forward with AMP
        with autocast(enabled=cfg['training']['use_amp']):
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=(point_coords, point_labels),
            )
            losses = criterion(outputs, targets)
            loss = losses['loss']
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        
        # Update progress bar
        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in losses.items()})
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
):
    """Validate the model."""
    model.eval()
    
    total_losses = {}
    total_bbox_error = 0.0
    total_yaw_error = 0.0
    total_pos_error = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc='Validation'):
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        mono_point = batch['mono_point'].to(device)
        
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'yaw_radians': batch['yaw_radians'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        B = front_view.shape[0]
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(B, 1, device=device)
        
        with autocast(enabled=cfg['training']['use_amp']):
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
            total_losses[k] += v.item()
        
        # Compute metrics
        if 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes'][:, 0, :] if outputs['pred_boxes'].dim() == 3 else outputs['pred_boxes']
            bbox_error = (pred_boxes - targets['sat_bbox']).abs().mean()
            total_bbox_error += bbox_error.item() * B
        
        if 'yaw_radians' in outputs:
            yaw_diff = outputs['yaw_radians'] - targets['yaw_radians']
            yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
            yaw_error = yaw_diff.abs().mean()
            total_yaw_error += yaw_error.item() * B
        
        if 'position' in outputs:
            pos_error = (outputs['position'] - targets['camera_position']).norm(dim=-1).mean()
            total_pos_error += pos_error.item() * B
        
        num_samples += B
    
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    avg_losses['bbox_mae'] = total_bbox_error / num_samples if num_samples > 0 else 0
    avg_losses['yaw_mae'] = total_yaw_error / num_samples if num_samples > 0 else 0
    avg_losses['pos_error'] = total_pos_error / num_samples if num_samples > 0 else 0
    
    return avg_losses


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Override with command line args
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.lr:
        cfg['training']['lr_heads'] = args.lr
    if args.epochs:
        cfg['training']['num_epochs'] = args.epochs
    if args.output_dir:
        cfg['checkpoint']['output_dir'] = args.output_dir
    if args.resume:
        cfg['checkpoint']['resume'] = args.resume
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    output_dir = Path(cfg['checkpoint']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)
    
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
    
    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')
    
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
    ).to(device)
    
    # Load pretrained weights (优先VGGT，其次DINOv2)
    if cfg['model'].get('vggt_weights'):
        load_vggt_weights(model, cfg['model']['vggt_weights'], load_heads=False)
        print(f'Loaded VGGT weights from {cfg["model"]["vggt_weights"]}')
    elif cfg['model'].get('dinov2_weights'):
        load_dinov2_weights(model, dinov2_path=cfg['model']['dinov2_weights'])
        print(f'Loaded DINOv2 weights from {cfg["model"]["dinov2_weights"]}')
    
    # Freeze backbone
    if cfg['model']['freeze_patch_embed']:
        freeze_backbone(model, freeze_patch_embed=True, freeze_aggregator=cfg['model']['freeze_aggregator'])
        print('Froze backbone')
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
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
    
    # Create scheduler with warmup
    num_epochs = cfg['training']['num_epochs']
    warmup_epochs = cfg['training']['warmup_epochs']
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=cfg['training']['min_lr'])
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    # AMP scaler
    scaler = GradScaler(enabled=cfg['training']['use_amp'])
    
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if cfg['checkpoint']['resume']:
        ckpt = torch.load(cfg['checkpoint']['resume'], map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f'\n{"="*50}')
        print(f'Epoch {epoch}/{num_epochs}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg
        )
        print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        
        # Validate
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, device, cfg)
            print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
            
            # Save best
            if val_losses['loss'] < best_loss:
                best_loss = val_losses['loss']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'val_losses': val_losses,
                    'best_loss': best_loss,
                }, output_dir / 'best.pth')
                print(f'Saved best model (loss: {best_loss:.4f})')
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % cfg['checkpoint']['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
            }, output_dir / f'epoch_{epoch}.pth')
    
    print(f'\nTraining completed! Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
