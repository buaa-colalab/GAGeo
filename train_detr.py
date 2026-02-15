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

from models import CrossViewLocalizerPi3, build_cross_view_localizer_pi3
from data import CrossViewDataset, collate_fn
from utils.prompt_utils import prepare_random_prompt, prepare_single_prompt
from utils import (
    get_param_groups,
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
    defaults = {
        "ROOT_DIR": os.environ.get("ROOT_DIR", "/data/home/scxi704/run/xhj"),
        "WORKSPACE_NAME": os.environ.get("WORKSPACE_NAME", "location_v3"),
    }
    defaults["WORKSPACE_DIR"] = f"{defaults['ROOT_DIR']}/{defaults['WORKSPACE_NAME']}"

    def _expand_str(s: str) -> str:
        s = os.path.expandvars(s)
        for k, v in defaults.items():
            s = s.replace(f"${{{k}}}", v)
        return s

    def _expand_env(obj):
        if isinstance(obj, dict):
            return {k: _expand_env(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand_env(v) for v in obj]
        if isinstance(obj, str):
            return _expand_str(obj)
        return obj

    with open(config_path, 'r') as f:
        return _expand_env(yaml.safe_load(f))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
    epoch: int,
    cfg: dict,
    global_step: int = 0,
):
    """Train for one epoch with step-level TensorBoard logging."""
    model.train()
    
    total_losses = {}
    log_freq = cfg['logging'].get('log_freq', 50)
    img_size = cfg['data']['img_size']
    
    # Running window for step-level TensorBoard logging
    running_losses = {}
    running_count = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not accelerator.is_main_process)
    
    for batch_idx, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            front_view = batch['front_view']
            sat_view = batch['satellite_view']
            
            # Prepare targets
            targets = {
                'sat_bbox': batch['sat_bbox'],
                'rotation_matrix': batch['rotation_matrix'],
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
                    mono_mask=batch['mono_mask'],
                    sat_mask=batch['sat_mask'],
                )
                losses = criterion(outputs, targets)
                loss = losses['loss']
            
            # Backward
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            
            optimizer.step()
            optimizer.zero_grad()
        
        # FIX: scheduler.step() must be called ONLY after actual optimizer updates,
        # not on every micro-batch. accelerator.accumulate() skips optimizer.step()
        # and zero_grad() on non-sync steps, but does NOT skip scheduler.step().
        if accelerator.sync_gradients:
            scheduler.step()
            global_step += 1
        
        # Accumulate losses for epoch-level average
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # --- Step-level TensorBoard logging ---
        running_count += 1
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            running_losses[k] = running_losses.get(k, 0.0) + val
        
        # Derived metrics: position error in pixels
        if 'pos_error' in losses:
            pe = losses['pos_error']
            pe_val = pe.item() if isinstance(pe, torch.Tensor) else pe
            running_losses['pos_error_px'] = running_losses.get('pos_error_px', 0.0) + pe_val * img_size
        
        # Log to TensorBoard every log_freq optimizer steps
        if accelerator.sync_gradients and global_step % log_freq == 0 and global_step > 0:
            log_dict = {}
            for k, v in running_losses.items():
                log_dict[f"train_step/{k}"] = v / max(running_count, 1)
            # Learning rates
            log_dict["train_step/lr_backbone"] = optimizer.param_groups[0]['lr']
            if len(optimizer.param_groups) > 1:
                log_dict["train_step/lr_heads"] = optimizer.param_groups[-1]['lr']
            log_dict["train_step/epoch"] = epoch
            accelerator.log(log_dict, step=global_step)
            running_losses = {}
            running_count = 0
        
        # Update progress bar with all losses
        if accelerator.is_main_process:
            postfix = {}
            for k, v in losses.items():
                if k.startswith('loss') or k in ('rotation_error_deg', 'bbox_iou', 'pos_error'):
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    postfix[k.replace('loss_', '')] = f'{val:.4f}'
            pbar.set_postfix(postfix)
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    # Log epoch-level summary
    accelerator.log({f"train/{k}": v for k, v in avg_losses.items()}, step=epoch)
    
    return avg_losses, global_step


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
    all_rotation_errors = []
    
    for batch in tqdm(dataloader, desc='Validation', disable=not accelerator.is_main_process):
        front_view = batch['front_view']
        sat_view = batch['satellite_view']
        
        targets = {
            'sat_bbox': batch['sat_bbox'],
            'rotation_matrix': batch['rotation_matrix'],
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
                mono_mask=batch['mono_mask'],
                sat_mask=batch['sat_mask'],
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
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    # Gather metrics
    if all_pos_errors:
        all_pos_errors = accelerator.gather_for_metrics(torch.cat(all_pos_errors))
        avg_losses['pos_mae'] = all_pos_errors.mean().item()
        avg_losses['pos_mae_pixels'] = avg_losses['pos_mae'] * cfg['data']['img_size']
    
    if all_rotation_errors:
        avg_losses['rotation_mae_deg'] = torch.cat(all_rotation_errors).mean().item()
    
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
        crop_sat=True,  # 训练模式：随机crop
    )
    
    val_dataset = CrossViewDataset(
        json_path=cfg['data']['val_json'],
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        crop_sat=False,  # val/test数据已经是crop好的，无需再crop
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
    
    # Create model with Pi3 backbone
    model = build_cross_view_localizer_pi3(
        pretrained_pi3=cfg['model'].get('pi3_weights'),
        freeze_backbone=False,  # We'll freeze selectively below
        freeze_prompt_encoder=cfg['model'].get('freeze_prompt_encoder', True),
        load_camera_head_weights=cfg['model'].get('load_camera_head_weights', True),
        sam_weights=cfg['model'].get('sam_weights'),
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
        sam_embed_dim=cfg['model'].get('sam_embed_dim'),
    )
    
    if cfg['model'].get('pi3_weights'):
        accelerator.print(f'Loaded Pi3 weights from {cfg["model"]["pi3_weights"]}')
    
    # Freeze DINOv2 encoder (keep decoder trainable)
    if cfg['model'].get('freeze_dinov2', True):
        for param in model.backbone.encoder.parameters():
            param.requires_grad = False
        accelerator.print('Froze DINOv2 encoder')
    
    # Log prompt encoder freeze status
    if cfg['model'].get('freeze_prompt_encoder', True) and accelerator.is_main_process:
        print('Froze SAM Prompt Encoder')
    
    # Optionally freeze Pi3 decoder
    if cfg['model'].get('freeze_decoder', False):
        for param in model.backbone.decoder.parameters():
            param.requires_grad = False
        accelerator.print('Froze Pi3 decoder')
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
    # Create criterion
    criterion = DETRCriterion(
        weight_bbox=cfg['training']['weight_bbox'],
        weight_giou=cfg['training']['weight_giou'],
        weight_heatmap=cfg['training']['weight_heatmap'],
        weight_rotation=cfg['training'].get('weight_rotation', 1.0),
        weight_contrastive=cfg['training'].get('weight_contrastive', 0.05),
        img_size=cfg['data']['img_size'],
        matcher_cost_class=cfg['training'].get('matcher_cost_class', 1.0),
        matcher_cost_bbox=cfg['training'].get('matcher_cost_bbox', 5.0),
        matcher_cost_giou=cfg['training'].get('matcher_cost_giou', 2.0),
        heatmap_sigma=cfg['training'].get('heatmap_sigma', 0.05),
        heatmap_label_smooth=cfg['training'].get('heatmap_label_smooth', 0.01),
        weight_class=cfg['training'].get('weight_class', 2.0),
        smooth_rotation=cfg['training'].get('smooth_rotation', True),
    )
    
    # Create optimizer
    param_groups = get_param_groups(
        model,
        lr_backbone=cfg['training']['lr_backbone'],
        lr_heads=cfg['training']['lr_heads'],
        weight_decay=cfg['training']['weight_decay'],
    )
    optimizer = AdamW(param_groups)

    if accelerator.is_main_process:
        accelerator.print(f"Optimizer param groups (before prepare): {len(optimizer.param_groups)}")
        for i, pg in enumerate(optimizer.param_groups):
            accelerator.print(
                f"  group[{i}] lr={pg['lr']:.2e}, weight_decay={pg.get('weight_decay', 0.0):.2e}, "
                f"num_tensors={len(pg['params'])}"
            )
    
    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Create scheduler
    num_epochs = cfg['training']['num_epochs']
    warmup_epochs = cfg['training']['warmup_epochs']
    grad_accum_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    
    # FIX: scheduler steps = actual optimizer steps, not micro-batch steps
    # With gradient accumulation, optimizer updates once every grad_accum_steps batches
    steps_per_epoch = len(train_loader) // grad_accum_steps
    num_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps = steps_per_epoch * warmup_epochs
    
    accelerator.print(f'Scheduler: {num_training_steps} total steps, {num_warmup_steps} warmup steps '
                       f'({steps_per_epoch} steps/epoch, accumulation={grad_accum_steps})')
    
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    # Note: HF warmup scheduler sets lr to 0 at step=0, then ramps to initial_lr.
    if accelerator.is_main_process:
        accelerator.print(f"Optimizer param groups (after scheduler init): {len(optimizer.param_groups)}")
        for i, pg in enumerate(optimizer.param_groups):
            accelerator.print(
                f"  group[{i}] lr={pg['lr']:.2e}, initial_lr={pg.get('initial_lr', pg['lr']):.2e}"
            )
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    resume_global_step = 0
    if cfg['checkpoint'].get('resume'):
        accelerator.print(f'Resuming from {cfg["checkpoint"]["resume"]}')
        accelerator.load_state(cfg['checkpoint']['resume'])
        ckpt_path = Path(cfg['checkpoint']['resume'])
        if (ckpt_path / 'training_state.pt').exists():
            training_state = torch.load(ckpt_path / 'training_state.pt', map_location='cpu')
            start_epoch = training_state.get('epoch', 0) + 1
            best_loss = training_state.get('best_loss', float('inf'))
            resume_global_step = training_state.get('global_step', 0)
        accelerator.print(f'Resumed from epoch {start_epoch}, global_step {resume_global_step}')
    
    # Training loop
    global_step = resume_global_step
    for epoch in range(start_epoch, num_epochs):
        accelerator.print(f'\n{"="*50}')
        accelerator.print(f'Epoch {epoch}/{num_epochs}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Log current LR for all param groups
        if accelerator.is_main_process:
            for i, pg in enumerate(optimizer.param_groups):
                accelerator.print(f'  param_group[{i}] lr={pg["lr"]:.2e}')
        
        # Train
        train_losses, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, accelerator, epoch, cfg,
            global_step=global_step,
        )
        accelerator.print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))
        
        # Validate
        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, accelerator, cfg, epoch)
            accelerator.print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))
            
            # Visualize validation samples with different prompt types
            if cfg['logging'].get('visualize', True):
                vis_prompt_types = cfg['logging'].get('vis_prompt_types', ['point'])
                num_vis = cfg['logging'].get('vis_samples', 10)
                for prompt_type in vis_prompt_types:
                    visualize_validation_samples(model, val_loader, accelerator, cfg, epoch, 
                                                num_samples=num_vis, prompt_type=prompt_type)
            
            # Save best
            if val_losses['loss'] < best_loss:
                best_loss = val_losses['loss']
                accelerator.save_state(output_dir / 'best')
                if accelerator.is_main_process:
                    torch.save({
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'val_losses': val_losses,
                        'global_step': global_step,
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
                    'global_step': global_step,
                }, save_dir / 'training_state.pt')
    
    accelerator.end_training()
    accelerator.print(f'\nTraining completed! Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
