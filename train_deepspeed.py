"""
Cross-View Localization Training Script with DeepSpeed and TensorBoard
"""

import argparse
import yaml
from pathlib import Path
import os
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not available, falling back to DDP")

from torch.utils.tensorboard import SummaryWriter

from models import CrossViewLocalizer
from data.dataset import CrossViewDataset, collate_fn
from utils import MultiTaskLoss, load_vggt_weights, load_dinov2_weights, freeze_backbone, get_param_groups


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cross-View Localizer with DeepSpeed')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device(s) to use (e.g., "0" or "0,1,2,3")')
    
    # DeepSpeed参数
    parser.add_argument('--deepspeed', action='store_true',
                        help='Enable DeepSpeed training')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                        help='DeepSpeed config file (optional, will auto-generate if not provided)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--zero_stage', type=int, default=2, choices=[0, 1, 2, 3],
                        help='DeepSpeed ZeRO optimization stage')
    
    # 允许命令行覆盖配置
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    
    # DeepSpeed会自动添加一些参数
    if DEEPSPEED_AVAILABLE:
        parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_deepspeed_config(cfg: dict, args) -> dict:
    """Generate DeepSpeed config from training config."""
    ds_config = {
        "train_batch_size": cfg['training']['batch_size'] * torch.cuda.device_count(),
        "train_micro_batch_size_per_gpu": cfg['training']['batch_size'],
        "gradient_accumulation_steps": cfg['training'].get('gradient_accumulation_steps', 1),
        "gradient_clipping": cfg['training']['grad_clip'],
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg['training']['lr_heads'],
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": cfg['training']['weight_decay']
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": cfg['training']['min_lr'],
                "warmup_max_lr": cfg['training']['lr_heads'],
                "warmup_num_steps": cfg['training']['warmup_epochs'] * 100,  # 估算
                "total_num_steps": cfg['training']['num_epochs'] * 100
            }
        },
        
        "fp16": {
            "enabled": cfg['training']['use_amp'],
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "bf16": {
            "enabled": False
        },
        
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_optimizer": {
                "device": "none"
            },
            "offload_param": {
                "device": "none"
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5
        },
        
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False
        },
        
        "wall_clock_breakdown": False,
        "tensorboard": {
            "enabled": True,
            "output_path": str(Path(cfg['checkpoint']['output_dir']) / "tensorboard"),
            "job_name": "cross_view_localization"
        }
    }
    
    return ds_config


class TensorBoardLogger:
    """TensorBoard logger with distributed training support."""
    
    def __init__(self, log_dir: str, rank: int = 0):
        self.rank = rank
        self.writer = None
        if rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to: {log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_dict(self, prefix: str, metrics: dict, step: int):
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)
    
    def log_lr(self, lr: float, step: int):
        if self.writer:
            self.writer.add_scalar("train/learning_rate", lr, step)
    
    def log_image(self, tag: str, img_tensor, step: int):
        if self.writer:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_images(self, tag: str, img_tensor, step: int):
        if self.writer:
            self.writer.add_images(tag, img_tensor, step)
    
    def log_histogram(self, tag: str, values, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def flush(self):
        if self.writer:
            self.writer.flush()
    
    def close(self):
        if self.writer:
            self.writer.close()


def train_one_epoch(
    model,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epoch: int,
    cfg: dict,
    tb_logger: TensorBoardLogger,
    global_step: int,
    use_deepspeed: bool = False,
    model_engine=None,
):
    """Train for one epoch."""
    model.train()
    
    total_losses = {}
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(not is_main_process()))
    
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
        
        # Forward
        if use_deepspeed and model_engine is not None:
            outputs = model_engine(
                front_view=front_view,
                satellite_view=sat_view,
                points=(point_coords, point_labels),
            )
            losses = criterion(outputs, targets)
            loss = losses['loss']
            
            # Backward with DeepSpeed
            model_engine.backward(loss)
            model_engine.step()
        else:
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                outputs = model(
                    front_view=front_view,
                    satellite_view=sat_view,
                    points=(point_coords, point_labels),
                )
                losses = criterion(outputs, targets)
                loss = losses['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            optimizer.step()
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        
        # Log to TensorBoard
        if batch_idx % cfg['logging'].get('log_freq', 50) == 0:
            tb_logger.log_dict("train_step", {k: v.item() for k, v in losses.items()}, global_step)
            if use_deepspeed and model_engine is not None:
                lr = model_engine.get_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']
            tb_logger.log_lr(lr, global_step)
        
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({k: f'{v.item():.4f}' for k, v in losses.items()})
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Log epoch averages
    tb_logger.log_dict("train_epoch", avg_losses, epoch)
    
    return avg_losses, global_step


@torch.no_grad()
def validate(
    model,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    cfg: dict,
    tb_logger: TensorBoardLogger,
):
    """Validate the model."""
    model.eval()
    
    total_losses = {}
    total_bbox_error = 0.0
    total_yaw_error = 0.0
    total_pos_error = 0.0
    num_samples = 0
    
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
        
        with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
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
    
    # Log to TensorBoard
    tb_logger.log_dict("val", avg_losses, epoch)
    
    return avg_losses


def is_main_process():
    """Check if current process is the main process."""
    if DEEPSPEED_AVAILABLE and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_rank() == 0
    elif torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def get_rank():
    """Get current process rank."""
    if DEEPSPEED_AVAILABLE and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_rank()
    elif torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size():
    """Get world size."""
    if DEEPSPEED_AVAILABLE and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_world_size()
    elif torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def setup_distributed(args):
    """Initialize distributed training."""
    if args.deepspeed and DEEPSPEED_AVAILABLE:
        deepspeed.init_distributed()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return local_rank
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        return 0


def main():
    args = parse_args()
    
    # Setup distributed
    local_rank = setup_distributed(args)
    use_deepspeed = args.deepspeed and DEEPSPEED_AVAILABLE
    
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
    
    # Setup GPU
    if not use_deepspeed and args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        print(f"{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"DeepSpeed: {use_deepspeed}")
        if use_deepspeed:
            print(f"ZeRO Stage: {args.zero_stage}")
        print(f"World Size: {get_world_size()}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"{'='*60}")
    
    output_dir = Path(cfg['checkpoint']['output_dir'])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)
    
    # Initialize TensorBoard
    tb_log_dir = output_dir / "tensorboard"
    tb_logger = TensorBoardLogger(str(tb_log_dir), rank=get_rank())
    
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
    is_distributed = get_world_size() > 1
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
    
    if is_main_process():
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
    )
    
    # Load pretrained weights
    if cfg['model'].get('vggt_weights'):
        load_vggt_weights(model, cfg['model']['vggt_weights'], load_heads=False)
        if is_main_process():
            print(f'Loaded VGGT weights from {cfg["model"]["vggt_weights"]}')
    elif cfg['model'].get('dinov2_weights'):
        load_dinov2_weights(model, dinov2_path=cfg['model']['dinov2_weights'])
        if is_main_process():
            print(f'Loaded DINOv2 weights from {cfg["model"]["dinov2_weights"]}')
    
    # Freeze backbone
    if cfg['model']['freeze_patch_embed']:
        freeze_backbone(model, freeze_patch_embed=True, freeze_aggregator=cfg['model']['freeze_aggregator'])
        if is_main_process():
            print('Froze backbone')
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
    # Create loss
    criterion = MultiTaskLoss(
        weight_bbox=cfg['training']['weight_bbox'],
        weight_giou=cfg['training']['weight_giou'],
        weight_yaw=cfg['training']['weight_yaw'],
        weight_position=cfg['training']['weight_position'],
        weight_mask=cfg['training']['weight_mask'],
    )
    
    # Initialize DeepSpeed or standard training
    model_engine = None
    optimizer = None
    scheduler = None
    
    if use_deepspeed:
        # Generate DeepSpeed config
        ds_config = get_deepspeed_config(cfg, args)
        
        # Save DeepSpeed config
        if is_main_process():
            ds_config_path = output_dir / 'deepspeed_config.json'
            import json
            with open(ds_config_path, 'w') as f:
                json.dump(ds_config, f, indent=2)
            print(f"DeepSpeed config saved to {ds_config_path}")
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        model = model_engine
        device = model_engine.device
    else:
        model = model.to(device)
        
        # Wrap with DDP if distributed
        if is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        # Create optimizer
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
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=cfg['training']['min_lr'])
        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    global_step = 0
    
    if cfg['checkpoint']['resume']:
        if use_deepspeed:
            _, client_state = model_engine.load_checkpoint(cfg['checkpoint']['resume'])
            if client_state:
                start_epoch = client_state.get('epoch', 0) + 1
                best_loss = client_state.get('best_loss', float('inf'))
                global_step = client_state.get('global_step', 0)
        else:
            ckpt = torch.load(cfg['checkpoint']['resume'], map_location=device)
            model_without_ddp = model.module if is_distributed else model
            model_without_ddp.load_state_dict(ckpt['model'])
            if optimizer and 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler and 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            global_step = ckpt.get('global_step', 0)
        
        if is_main_process():
            print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    num_epochs = cfg['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        if is_main_process():
            print(f'\n{"="*50}')
            if use_deepspeed:
                lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else cfg['training']['lr_heads']
            else:
                lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}/{num_epochs}, LR: {lr:.2e}')
        
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_losses, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg,
            tb_logger, global_step, use_deepspeed, model_engine
        )
        
        if is_main_process():
            print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        
        # Validate
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, device, epoch, cfg, tb_logger)
            
            if is_main_process():
                print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
                
                # Save best
                if val_losses['loss'] < best_loss:
                    best_loss = val_losses['loss']
                    
                    if use_deepspeed:
                        client_state = {
                            'epoch': epoch,
                            'best_loss': best_loss,
                            'global_step': global_step,
                            'val_losses': val_losses,
                        }
                        model_engine.save_checkpoint(str(output_dir / 'best'), client_state=client_state)
                    else:
                        model_without_ddp = model.module if is_distributed else model
                        torch.save({
                            'epoch': epoch,
                            'model': model_without_ddp.state_dict(),
                            'val_losses': val_losses,
                            'best_loss': best_loss,
                            'global_step': global_step,
                        }, output_dir / 'best.pth')
                    
                    print(f'Saved best model (loss: {best_loss:.4f})')
        
        # Step scheduler (non-DeepSpeed)
        if scheduler is not None and not use_deepspeed:
            scheduler.step()
        
        # Save checkpoint
        if is_main_process() and (epoch + 1) % cfg['checkpoint']['save_freq'] == 0:
            if use_deepspeed:
                client_state = {
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'global_step': global_step,
                }
                model_engine.save_checkpoint(str(output_dir / f'epoch_{epoch}'), client_state=client_state)
            else:
                model_without_ddp = model.module if is_distributed else model
                torch.save({
                    'epoch': epoch,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'best_loss': best_loss,
                    'global_step': global_step,
                }, output_dir / f'epoch_{epoch}.pth')
        
        # Flush TensorBoard
        tb_logger.flush()
    
    if is_main_process():
        print(f'\nTraining completed! Best loss: {best_loss:.4f}')
    
    tb_logger.close()


if __name__ == '__main__':
    main()
