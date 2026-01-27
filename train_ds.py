"""
Cross-View Localization Training Script with DeepSpeed and TensorBoard
Refactored version - DeepSpeed config from YAML
"""

import argparse
import yaml
import json
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

from models import CrossViewLocalizer
from data.dataset import CrossViewDataset, collate_fn
from utils import MultiTaskLoss, load_vggt_weights, load_dinov2_weights, freeze_backbone, get_param_groups


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cross-View Localizer')
    parser.add_argument('--config', type=str, default='configs/test.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    
    if DEEPSPEED_AVAILABLE:
        parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_deepspeed_config(cfg: dict) -> dict:
    """Build DeepSpeed config from YAML config."""
    ds_cfg = cfg.get('deepspeed', {})
    train_cfg = cfg['training']
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    config = {
        "train_batch_size": train_cfg['batch_size'] * world_size,
        "train_micro_batch_size_per_gpu": train_cfg['batch_size'],
        "gradient_accumulation_steps": ds_cfg.get('gradient_accumulation_steps', 1),
        "gradient_clipping": train_cfg['grad_clip'],
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": train_cfg['lr_heads'],
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": train_cfg['weight_decay']
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": train_cfg['min_lr'],
                "warmup_max_lr": train_cfg['lr_heads'],
                "warmup_num_steps": train_cfg['warmup_epochs'] * 100,
                "total_num_steps": train_cfg['num_epochs'] * 100
            }
        },
        
        "fp16": {
            "enabled": train_cfg.get('use_amp', False),
            "loss_scale": 0,
            "initial_scale_power": 16,
        },
        
        "zero_optimization": {
            "stage": ds_cfg.get('zero_stage', 2),
            "offload_optimizer": {
                "device": "cpu" if ds_cfg.get('offload_optimizer', False) else "none"
            },
            "offload_param": {
                "device": "cpu" if ds_cfg.get('offload_param', False) else "none"
            },
        },
        
        "activation_checkpointing": {
            "partition_activations": ds_cfg.get('activation_checkpointing', False),
            "cpu_checkpointing": False
        },
        
        "tensorboard": {
            "enabled": cfg['logging'].get('use_tensorboard', True),
            "output_path": str(Path(cfg['checkpoint']['output_dir']) / "tensorboard"),
        }
    }
    
    return config


def get_rank():
    if DEEPSPEED_AVAILABLE and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_rank()
    elif torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, cfg, tb_writer, global_step, use_ds):
    model.train()
    total_losses = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(not is_main_process()))
    
    for batch_idx, batch in enumerate(pbar):
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
        
        outputs = model(front_view=front_view, satellite_view=sat_view, points=(point_coords, point_labels))
        losses = criterion(outputs, targets)
        loss = losses['loss']
        
        if use_ds:
            model.backward(loss)
            model.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            optimizer.step()
        
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        
        if batch_idx % cfg['logging'].get('log_freq', 50) == 0 and tb_writer:
            for k, v in losses.items():
                tb_writer.add_scalar(f"train_step/{k}", v.item(), global_step)
            lr = model.get_lr()[0] if use_ds else optimizer.param_groups[0]['lr']
            tb_writer.add_scalar("train/lr", lr, global_step)
        
        global_step += 1
        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in losses.items()})
    
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    if tb_writer:
        for k, v in avg_losses.items():
            tb_writer.add_scalar(f"train_epoch/{k}", v, epoch)
    
    return avg_losses, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, cfg, tb_writer):
    model.eval()
    total_losses = {}
    total_bbox_error = total_yaw_error = total_pos_error = num_samples = 0
    
    for batch in tqdm(dataloader, desc='Validation', disable=(not is_main_process())):
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
        
        outputs = model(front_view=front_view, satellite_view=sat_view, points=(point_coords, point_labels))
        losses = criterion(outputs, targets)
        
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        
        if 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes'][:, 0, :] if outputs['pred_boxes'].dim() == 3 else outputs['pred_boxes']
            total_bbox_error += (pred_boxes - targets['sat_bbox']).abs().mean().item() * B
        
        if 'yaw_radians' in outputs:
            yaw_diff = torch.atan2(torch.sin(outputs['yaw_radians'] - targets['yaw_radians']),
                                   torch.cos(outputs['yaw_radians'] - targets['yaw_radians']))
            total_yaw_error += yaw_diff.abs().mean().item() * B
        
        if 'position' in outputs:
            total_pos_error += (outputs['position'] - targets['camera_position']).norm(dim=-1).mean().item() * B
        
        num_samples += B
    
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    avg_losses['bbox_mae'] = total_bbox_error / num_samples if num_samples > 0 else 0
    avg_losses['yaw_mae'] = total_yaw_error / num_samples if num_samples > 0 else 0
    avg_losses['pos_error'] = total_pos_error / num_samples if num_samples > 0 else 0
    
    if tb_writer:
        for k, v in avg_losses.items():
            tb_writer.add_scalar(f"val/{k}", v, epoch)
    
    return avg_losses


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Setup distributed
    use_deepspeed = cfg.get('deepspeed', {}).get('enabled', False) and DEEPSPEED_AVAILABLE
    
    if use_deepspeed:
        deepspeed.init_distributed()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'RANK' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0
        if cfg['training'].get('gpu'):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['training']['gpu'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process():
        print(f"{'='*60}")
        print(f"Training Mode: {'DeepSpeed' if use_deepspeed else 'DDP/Single GPU'}")
        if use_deepspeed:
            print(f"ZeRO Stage: {cfg['deepspeed'].get('zero_stage', 2)}")
        print(f"Device: {device}")
        print(f"{'='*60}")
    
    # Setup output
    output_dir = Path(cfg['checkpoint']['output_dir'])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)
    
    # TensorBoard
    tb_writer = None
    if is_main_process() and cfg['logging'].get('use_tensorboard', True):
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    
    # Data
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
    
    is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'],
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg['data']['num_workers'], collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'],
        shuffle=False, sampler=val_sampler,
        num_workers=cfg['data']['num_workers'], collate_fn=collate_fn,
        pin_memory=True,
    )
    
    if is_main_process():
        print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')
    
    # Model
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
    
    # Load pretrained weights
    if cfg['model'].get('vggt_weights'):
        load_vggt_weights(model, cfg['model']['vggt_weights'], load_heads=False)
        if is_main_process():
            print(f'Loaded VGGT weights')
    elif cfg['model'].get('dinov2_weights'):
        load_dinov2_weights(model, dinov2_path=cfg['model']['dinov2_weights'])
        if is_main_process():
            print(f'Loaded DINOv2 weights')
    
    if cfg['model']['freeze_patch_embed']:
        freeze_backbone(model, freeze_patch_embed=True, freeze_aggregator=cfg['model']['freeze_aggregator'])
        if is_main_process():
            print('Froze backbone')
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
    criterion = MultiTaskLoss(
        weight_bbox=cfg['training']['weight_bbox'],
        weight_giou=cfg['training']['weight_giou'],
        weight_yaw=cfg['training']['weight_yaw'],
        weight_position=cfg['training']['weight_position'],
        weight_mask=cfg['training']['weight_mask'],
    )
    
    # Initialize training
    if use_deepspeed:
        ds_config = build_deepspeed_config(cfg)
        if is_main_process():
            with open(output_dir / 'deepspeed_config.json', 'w') as f:
                json.dump(ds_config, f, indent=2)
        
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=ds_config
        )
        model = model_engine
        device = model_engine.device
    else:
        model = model.to(device)
        if is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        model_without_ddp = model.module if is_distributed else model
        param_groups = get_param_groups(
            model_without_ddp,
            lr_backbone=cfg['training']['lr_backbone'],
            lr_heads=cfg['training']['lr_heads'],
            weight_decay=cfg['training']['weight_decay'],
        )
        optimizer = AdamW(param_groups)
        
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=cfg['training']['warmup_epochs'])
        main_scheduler = CosineAnnealingLR(optimizer, T_max=cfg['training']['num_epochs'] - cfg['training']['warmup_epochs'], 
                                          eta_min=cfg['training']['min_lr'])
        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[cfg['training']['warmup_epochs']])
    
    # Training loop
    start_epoch = 0
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(start_epoch, cfg['training']['num_epochs']):
        if is_main_process():
            lr = model.get_lr()[0] if use_deepspeed else optimizer.param_groups[0]['lr']
            print(f'\n{"="*50}')
            print(f'Epoch {epoch}/{cfg["training"]["num_epochs"]}, LR: {lr:.2e}')
        
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_losses, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg, tb_writer, global_step, use_deepspeed
        )
        
        if is_main_process():
            print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, device, epoch, cfg, tb_writer)
            
            if is_main_process():
                print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
                
                if val_losses['loss'] < best_loss:
                    best_loss = val_losses['loss']
                    if use_deepspeed:
                        model.save_checkpoint(str(output_dir / 'best'), client_state={'epoch': epoch, 'best_loss': best_loss})
                    else:
                        model_without_ddp = model.module if is_distributed else model
                        torch.save({'epoch': epoch, 'model': model_without_ddp.state_dict(), 'best_loss': best_loss},
                                  output_dir / 'best.pth')
                    print(f'Saved best model (loss: {best_loss:.4f})')
        
        if not use_deepspeed:
            scheduler.step()
        
        if is_main_process() and (epoch + 1) % cfg['checkpoint']['save_freq'] == 0:
            if use_deepspeed:
                model.save_checkpoint(str(output_dir / f'epoch_{epoch}'))
            else:
                model_without_ddp = model.module if is_distributed else model
                torch.save({'epoch': epoch, 'model': model_without_ddp.state_dict()}, output_dir / f'epoch_{epoch}.pth')
    
    if is_main_process():
        print(f'\nTraining completed! Best loss: {best_loss:.4f}')
        if tb_writer:
            tb_writer.close()


if __name__ == '__main__':
    main()
