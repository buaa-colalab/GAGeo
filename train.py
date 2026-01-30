"""
Cross-View Localization Training Script
"""

import argparse
import yaml
from pathlib import Path
import os
# Limit OpenBLAS threads to avoid resource exhaustion
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from tqdm import tqdm

from data import CrossViewDataset, collate_fn
from models import CrossViewLocalizer
from utils import (
    MultiTaskLoss, 
    load_vggt_weights, 
    load_dinov2_weights, 
    freeze_backbone, 
    get_param_groups, 
    TensorBoardLogger
)
from utils.prompt_utils import prepare_random_prompt


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cross-View Localizer')
    parser.add_argument('--config', type=str, default='configs/test.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device(s) to use (e.g., "0" or "0,1,2,3")')
    # DDP相关参数
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
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
    tb_logger: TensorBoardLogger = None,
):
    """Train for one epoch."""
    model.train()
    
    total_losses = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        
        # 准备targets
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'yaw_radians': batch['yaw_radians'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        # 随机选择一种prompt输入（模拟真实场景：point/bbox/mask三选一）
        points, boxes, masks = prepare_random_prompt(batch, device)
        
        optimizer.zero_grad()
        
        # Forward with AMP
        with autocast('cuda', enabled=cfg['training']['use_amp']):
            #print('use amp')
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
    
    # Log to TensorBoard
    if tb_logger:
        tb_logger.log_dict("train", avg_losses, epoch)
        tb_logger.log_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
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
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation')):
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'yaw_radians': batch['yaw_radians'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        # 验证时测试所有三种prompt类型（轮流使用）
        prompt_type = batch_idx % 3  # 0: point, 1: bbox, 2: mask
        
        if prompt_type == 0:
            # Point prompt
            B = front_view.shape[0]
            mono_point = batch['mono_point'].to(device)
            point_coords = mono_point.unsqueeze(1)
            point_labels = torch.ones(B, 1, device=device)
            points, boxes, masks = (point_coords, point_labels), None, None
        elif prompt_type == 1:
            # Bbox prompt
            from utils.prompt_utils import prepare_random_prompt
            points, boxes, masks = prepare_random_prompt(batch, device, prompt_types=['bbox'])
        else:
            # Mask prompt
            from utils.prompt_utils import prepare_random_prompt
            points, boxes, masks = prepare_random_prompt(batch, device, prompt_types=['mask'])
        
        with autocast('cuda', enabled=cfg['training']['use_amp']):
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=points,
                boxes=boxes,
                masks=masks,
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
    
    # Log to TensorBoard
    if tb_logger:
        tb_logger.log_dict("val", avg_losses, epoch)
    
    return avg_losses


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
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
    if args.gpu:
        cfg['training']['gpu'] = args.gpu
    
    # Setup GPU
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process:
            print(f'Distributed training on {world_size} GPUs')
            print(f'Main process using device: {device} (GPU: {torch.cuda.get_device_name(local_rank)})')
    else:
        gpu_ids = cfg['training'].get('gpu', None)
        if gpu_ids is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
            if is_main_process:
                print(f'Set CUDA_VISIBLE_DEVICES={gpu_ids}')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() and is_main_process:
            print(f'Using device: {device} (GPU: {torch.cuda.get_device_name(0)})')
            print(f'Available GPUs: {torch.cuda.device_count()}')
        elif is_main_process:
            print(f'Using device: {device}')
    
    output_dir = Path(cfg['checkpoint']['output_dir'])
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)
    
    # Initialize TensorBoard logger
    log_dir = output_dir / 'logs'
    tb_logger = TensorBoardLogger(
        log_dir=str(log_dir),
        enabled=cfg['logging'].get('use_tensorboard', True),
        rank=rank,
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
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    if is_main_process:
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
        if is_main_process:
            print(f'Loaded VGGT weights from {cfg["model"]["vggt_weights"]}')
    elif cfg['model'].get('dinov2_weights'):
        load_dinov2_weights(model, dinov2_path=cfg['model']['dinov2_weights'])
        if is_main_process:
            print(f'Loaded DINOv2 weights from {cfg["model"]["dinov2_weights"]}')
    
    # Freeze backbone
    if cfg['model']['freeze_patch_embed']:
        freeze_backbone(model, freeze_patch_embed=True, freeze_aggregator=cfg['model']['freeze_aggregator'])
        if is_main_process:
            print('Froze backbone')
    
    # Wrap with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process:
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
    model_without_ddp = model.module if is_distributed else model
    param_groups = get_param_groups(
        model_without_ddp,
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
    scaler = GradScaler('cuda', enabled=cfg['training']['use_amp'])
    
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    if cfg['checkpoint']['resume']:
        ckpt = torch.load(cfg['checkpoint']['resume'], map_location=device)
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        if is_main_process:
            print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if is_main_process:
            print(f'\n{"="*50}')
            print(f'Epoch {epoch}/{num_epochs}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Set epoch for distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg, tb_logger
        )
        if is_main_process:
            print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        
        # Validate
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, device, cfg, epoch, tb_logger)
            if is_main_process:
                print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
                
                # Save best
                if val_losses['loss'] < best_loss:
                    best_loss = val_losses['loss']
                    torch.save({
                        'epoch': epoch,
                        'model': model_without_ddp.state_dict(),
                        'val_losses': val_losses,
                        'best_loss': best_loss,
                    }, output_dir / 'best.pth')
                    print(f'Saved best model (loss: {best_loss:.4f})')
        
        scheduler.step()
        
        # Save checkpoint
        if is_main_process and (epoch + 1) % cfg['checkpoint']['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
            }, output_dir / f'epoch_{epoch}.pth')
    
    # Close TensorBoard logger
    tb_logger.close()
    
    if is_main_process:
        print(f'\nTraining completed! Best loss: {best_loss:.4f}')
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
