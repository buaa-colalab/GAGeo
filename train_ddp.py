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
from models import CrossViewLocalizerPi3, build_cross_view_localizer_pi3
from utils.prompt_utils import prepare_random_prompt, prepare_single_prompt
from utils import (
    get_param_groups,
    box_cxcywh_to_xyxy, 
    generalized_box_iou,
    DETRCriterion,
)
from utils.visualize_ddp import visualize_validation_samples_ddp
from transformers import get_scheduler
import math
from torch.utils.tensorboard import SummaryWriter


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
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: dict,
    is_main_process: bool = True,
    tb_writer: SummaryWriter = None,
):
    """Train for one epoch."""
    model.train()
    
    total_losses = {}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not is_main_process)
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        
        # 准备targets
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'rotation_matrix': batch['rotation_matrix'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        # 随机选择一种prompt输入（模拟真实场景：point/bbox/mask三选一）
        points, boxes, masks = prepare_random_prompt(batch, device)
        
        optimizer.zero_grad()
        
        # Forward with AMP
        with autocast('cuda', enabled=cfg['training']['use_amp']):
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=points,
                boxes=boxes,
                masks=masks,
                mono_mask=batch['mono_mask'].to(device),
                sat_mask=batch['sat_mask'].to(device),
            )
            losses = criterion(outputs, targets)
            loss = losses['loss']
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        
        # Update progress bar with all losses
        if is_main_process:
            postfix = {}
            for k, v in losses.items():
                if k.startswith('loss') or k == 'rotation_error_deg':
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    postfix[k.replace('loss_', '')] = f'{val:.4f}'
            pbar.set_postfix(postfix)
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    # Log to TensorBoard
    if tb_writer and is_main_process:
        for k, v in avg_losses.items():
            tb_writer.add_scalar(f'train/{k}', v, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    epoch: int = 0,
    is_main_process: bool = True,
    tb_writer: SummaryWriter = None,
):
    """Validate the model."""
    model.eval()
    
    total_losses = {}
    all_pos_errors = []
    all_rotation_errors = []
    
    for batch in tqdm(dataloader, desc='Validation', disable=not is_main_process):
        front_view = batch['front_view'].to(device)
        sat_view = batch['satellite_view'].to(device)
        
        targets = {
            'sat_bbox': batch['sat_bbox'].to(device),
            'rotation_matrix': batch['rotation_matrix'].to(device),
            'camera_position': batch['camera_position'].to(device),
        }
        
        # Use point prompt for validation
        points, boxes, masks = prepare_single_prompt(batch, device, prompt_type='point')
        
        with autocast('cuda', enabled=cfg['training']['use_amp']):
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=points,
                boxes=boxes,
                masks=masks,
                mono_mask=batch['mono_mask'].to(device),
                sat_mask=batch['sat_mask'].to(device),
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
        
        if 'rotation_error_deg' in losses:
            all_rotation_errors.append(torch.tensor([losses['rotation_error_deg']]))
    
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    # Gather metrics across all processes in DDP
    if all_pos_errors:
        all_pos_errors = torch.cat(all_pos_errors)
        # Gather from all processes if distributed
        if dist.is_initialized():
            gathered_errors = [torch.zeros_like(all_pos_errors) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_errors, all_pos_errors)
            all_pos_errors = torch.cat(gathered_errors)
        avg_losses['pos_mae'] = all_pos_errors.mean().item()
        avg_losses['pos_mae_pixels'] = avg_losses['pos_mae'] * cfg['data']['img_size']
    
    if all_rotation_errors:
        avg_losses['rotation_mae_deg'] = torch.cat(all_rotation_errors).mean().item()
    
    # Log to TensorBoard
    if tb_writer and is_main_process:
        for k, v in avg_losses.items():
            tb_writer.add_scalar(f'val/{k}', v, epoch)
    
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
    
    # Initialize TensorBoard writer
    tb_writer = None
    if is_main_process and cfg['logging'].get('use_tensorboard', True):
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(log_dir))
        print(f'TensorBoard logs will be saved to: {log_dir}')
    
    # Create datasets
    train_dataset = CrossViewDataset(
        json_path=cfg['data']['train_json'],
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        crop_sat=True,  # 训练模式：随机crop
    )
    
    val_dataset = CrossViewDataset(
        json_path=cfg['data']['val_json'],
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        crop_sat=False,  # val/test数据已经是crop好的，无需再crop
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
    
    # Create model with Pi3 backbone
    model = build_cross_view_localizer_pi3(
        pretrained_pi3=cfg['model'].get('pi3_weights'),
        freeze_backbone=False,  # We'll freeze selectively below
        img_size=cfg['data']['img_size'],
        decoder_size=cfg['model'].get('decoder_size', 'large'),
        num_intent_queries=cfg['model'].get('num_intent_queries', 32),
        num_object_queries=cfg['model'].get('num_object_queries', 10),
        num_location_queries=cfg['model'].get('num_location_queries', 16),
        num_heads=cfg['model'].get('num_heads', 8),
        prompt_fusion_layers=cfg['model'].get('prompt_fusion_layers', 3),
        num_decoder_layers=cfg['model'].get('num_decoder_layers', 6),
        dropout=cfg['model'].get('dropout', 0.1),
        contrastive=cfg['model'].get('contrastive', True),
        contrastive_proj_dim=cfg['model'].get('contrastive_proj_dim', 256),
        contrastive_queue_size=cfg['model'].get('contrastive_queue_size', 16384),
        contrastive_momentum=cfg['model'].get('contrastive_momentum', 0.999),
        contrastive_temperature=cfg['model'].get('contrastive_temperature', 0.07),
    ).to(device)
    
    if cfg['model'].get('pi3_weights'):
        if is_main_process:
            print(f'Loaded Pi3 weights from {cfg["model"]["pi3_weights"]}')
    
    # Freeze DINOv2 encoder (keep decoder trainable)
    if cfg['model'].get('freeze_dinov2', True):
        for param in model.backbone.encoder.parameters():
            param.requires_grad = False
        if is_main_process:
            print('Froze DINOv2 encoder')
    
    # Optionally freeze Pi3 decoder
    if cfg['model'].get('freeze_decoder', False):
        for param in model.backbone.decoder.parameters():
            param.requires_grad = False
        if is_main_process:
            print('Froze Pi3 decoder')
    
    # Wrap with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
    # Create criterion
    criterion = DETRCriterion(
        weight_bbox=cfg['training']['weight_bbox'],
        weight_giou=cfg['training']['weight_giou'],
        weight_heatmap=cfg['training'].get('weight_heatmap', 1.0),
        weight_rotation=cfg['training'].get('weight_rotation', 1.0),
        weight_contrastive=cfg['training'].get('weight_contrastive', 0.1),
        img_size=cfg['data']['img_size'],
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
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, cfg, is_main_process, tb_writer
        )
        if is_main_process:
            print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        
        # Validate
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, device, cfg, epoch, is_main_process, tb_writer)
            if is_main_process:
                print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
                
                # Save best
                if val_losses['loss'] < best_loss:
                    best_loss = val_losses['loss']
                    torch.save({
                        'epoch': epoch,
                        'model': model_without_ddp.state_dict(),
                        'val_losses': val_losses,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_loss': best_loss,
                    }, output_dir / 'best.pth')
                
                # Visualize validation samples with different prompt types
                if cfg['logging'].get('visualize', True):
                    vis_prompt_types = cfg['logging'].get('vis_prompt_types', ['point'])
                    for prompt_type in vis_prompt_types:
                        visualize_validation_samples_ddp(
                            model, val_loader, device, cfg, epoch, is_main_process, 
                            num_samples=cfg['logging'].get('vis_samples', 10),
                            prompt_type=prompt_type
                        )
        
        # Scheduler is stepped inside train_one_epoch
        
        # Save checkpoint
        if is_main_process and (epoch + 1) % cfg['checkpoint']['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
            }, output_dir / f'epoch_{epoch}.pth')
    
    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()
    
    if is_main_process:
        print(f'\nTraining completed! Best loss: {best_loss:.4f}')
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
