"""
GAGeo Training Script
Unified backbone architecture with token injection, attention masks, deep supervision.
"""
import os
os.environ["TMPDIR"] = "/tmp"
import argparse
import math
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_scheduler

from models import build_cross_view_localizer
from datasets import CrossViewDataset, collate_fn
from utils.prompt_utils import prepare_random_prompt, prepare_single_prompt
from utils import (
    get_param_groups,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    DETRCriterion,
)

try:
    import wandb  # noqa: F401
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train GAGeo')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file (YAML)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override checkpoint.output_dir for this run')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    workspace_dir = Path(os.environ.get("WORKSPACE_DIR", Path.cwd())).resolve()
    root_dir = Path(os.environ.get("ROOT_DIR", workspace_dir.parent)).resolve()
    defaults = {
        "ROOT_DIR": str(root_dir),
        "WORKSPACE_NAME": os.environ.get("WORKSPACE_NAME", workspace_dir.name),
        "CHECKPOINT_DIR": os.environ.get("CHECKPOINT_DIR", str(workspace_dir / "checkpoints_offline")),
    }
    defaults["WORKSPACE_DIR"] = str(workspace_dir)
    defaults["DATA_ROOT"] = os.environ.get("DATA_ROOT", str(workspace_dir / "data" / "urban"))
    defaults["JSON_ROOT"] = os.environ.get("JSON_ROOT", str(workspace_dir / "data" / "json"))
    defaults["OUTPUT_ROOT"] = os.environ.get("OUTPUT_ROOT", f"{defaults['WORKSPACE_DIR']}/outputs")

    def _expand_str(s: str) -> str:
        # First pass: normal env expansion
        s = os.path.expandvars(s)
        # Fallback expansion for unexported shell vars
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


def resolve_resume_path(resume_value, output_dir=None, prefer_directory=False):
    """Resolve a resume hint to an existing checkpoint artifact.

    Supported inputs:
    - `/path/to/exp/best` for Accelerate checkpoints
    - `/path/to/exp/best.pth` for DDP checkpoints
    - `/path/to/exp` to auto-resolve under the experiment output dir
    """
    if not resume_value:
        return None

    raw = Path(str(resume_value)).expanduser()
    candidates = []

    def add_candidate(path_like):
        path = Path(path_like)
        if path not in candidates:
            candidates.append(path)

    add_candidate(raw)
    if raw.is_dir():
        add_candidate(raw / 'best')
        add_candidate(raw / 'best.pth')
    elif raw.suffix == '.pth':
        add_candidate(raw.with_suffix(''))
    else:
        add_candidate(raw.with_suffix('.pth'))
        add_candidate(raw / 'best')
        add_candidate(raw / 'best.pth')

    if output_dir:
        out = Path(output_dir)
        if raw == out:
            add_candidate(out / 'best')
            add_candidate(out / 'best.pth')

    existing = [path for path in candidates if path.exists()]
    if not existing:
        return raw

    if prefer_directory:
        for path in existing:
            if path.is_dir():
                return path
    else:
        for path in existing:
            if path.is_file():
                return path
    return existing[0]


def _assert_bbox_prompt_for_forward(boxes: torch.Tensor, front_view: torch.Tensor) -> None:
    """Guard: training-time bbox prompt must be pixel-space [x, y, w, h]."""
    if boxes is None:
        return
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes shape [B, N, 4], got {tuple(boxes.shape)}")

    H = front_view.shape[-2]
    W = front_view.shape[-1]
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    if torch.any(w <= 0) or torch.any(h <= 0):
        raise ValueError("Invalid bbox prompt: width/height must be > 0 (pixel xywh)")
    if torch.any(x < -1e-4) or torch.any(y < -1e-4):
        raise ValueError("Invalid bbox prompt: x/y must be non-negative (pixel xywh)")
    if torch.any(x > (W - 1 + 1e-4)) or torch.any(y > (H - 1 + 1e-4)):
        raise ValueError("Invalid bbox prompt: x/y exceed image boundary (pixel xywh expected)")
    if torch.any(w > (W + 1e-4)) or torch.any(h > (H + 1e-4)):
        raise ValueError("Invalid bbox prompt: w/h exceed image size (pixel xywh expected)")


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
    supervision_layers = cfg.get('model', {}).get('supervision_layers', [4, 11, 17])
    decoder_size = cfg.get('model', {}).get('decoder_size', 'large')
    num_stage_layers = 18 if decoder_size == 'large' else 12
    final_stage_idx = num_stage_layers - 1
    skip_final_inter_curve = final_stage_idx in supervision_layers
    supervision_layers_set = set(supervision_layers)

    running_losses = {}
    running_count = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not accelerator.is_main_process)

    for batch_idx, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            front_view = batch['front_view']
            sat_view = batch['satellite_view']

            targets = {
                'sat_bbox': batch['sat_bbox'],
                'rotation_matrix': batch['rotation_matrix'],
                'camera_position': batch['camera_position'],
            }
            #: Pass sat_mask as target for mask loss
            if 'sat_mask' in batch:
                targets['sat_mask'] = batch['sat_mask']

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
                    mono_mask=batch.get('mono_mask'),
                    sat_mask=batch.get('sat_mask'),
                )
                losses = criterion(outputs, targets)
                loss = losses['loss']

            # Backward
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            scheduler.step()
            global_step += 1

        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

        # Running window
        running_count += 1
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v

            # Only keep valid intermediate keys in running window
            if k.startswith('inter_'):
                parts = k.split('_', 2)  # inter_{idx}_{metric}
                if len(parts) != 3:
                    continue
                try:
                    layer_idx = int(parts[1])
                except ValueError:
                    continue
                # Ignore unexpected layers and optionally skip final stage curve
                if layer_idx not in supervision_layers_set:
                    continue
                if skip_final_inter_curve and layer_idx == final_stage_idx:
                    continue

            running_losses[k] = running_losses.get(k, 0.0) + val

        if 'pos_error' in losses:
            pe = losses['pos_error']
            pe_val = pe.item() if isinstance(pe, torch.Tensor) else pe
            running_losses['pos_error_px'] = running_losses.get('pos_error_px', 0.0) + pe_val * img_size

        # Log to TensorBoard
        if accelerator.sync_gradients and global_step % log_freq == 0 and global_step > 0:
            log_dict = {}
            for k, v in running_losses.items():
                # Intermediate supervision curves are logged under trian_step_idx/*
                # (and final stage is skipped if it is already represented by train_step/*)
                if k.startswith('inter_'):
                    continue
                log_dict[f"train_step/{k}"] = v / max(running_count, 1)

            # Per-layer intermediate curves (requested name: trian_step_idx)
            for k, v in running_losses.items():
                if not k.startswith('inter_'):
                    continue
                parts = k.split('_', 2)  # inter_{idx}_{metric}
                if len(parts) != 3:
                    continue
                try:
                    layer_idx = int(parts[1])
                except ValueError:
                    continue
                if skip_final_inter_curve and layer_idx == final_stage_idx:
                    continue
                metric_name = parts[2]
                log_dict[f"trian_step_{layer_idx}/{metric_name}"] = v / max(running_count, 1)

            log_dict["train_step/lr_backbone"] = optimizer.param_groups[0]['lr']
            if len(optimizer.param_groups) > 1:
                log_dict["train_step/lr_new_tokens"] = optimizer.param_groups[1]['lr']
            if len(optimizer.param_groups) > 2:
                log_dict["train_step/lr_heads"] = optimizer.param_groups[2]['lr']
            log_dict["train_step/epoch"] = epoch
            accelerator.log(log_dict, step=global_step)
            running_losses = {}
            running_count = 0

        # Progress bar
        if accelerator.is_main_process:
            postfix = {}
            for k, v in losses.items():
                if k.startswith('loss') or k in ('rotation_error_deg', 'bbox_iou', 'pos_error', 'mask_iou', 'heatmap_center_prob'):
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    postfix[k.replace('loss_', '')] = f'{val:.4f}'
            pbar.set_postfix(postfix)

    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
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
    all_mask_ious = []

    for batch in tqdm(dataloader, desc='Validation', disable=not accelerator.is_main_process):
        front_view = batch['front_view']
        sat_view = batch['satellite_view']

        targets = {
            'sat_bbox': batch['sat_bbox'],
            'rotation_matrix': batch['rotation_matrix'],
            'camera_position': batch['camera_position'],
        }
        if 'sat_mask' in batch:
            targets['sat_mask'] = batch['sat_mask']

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
                mono_mask=batch.get('mono_mask'),
                sat_mask=batch.get('sat_mask'),
            )
            losses = criterion(outputs, targets)

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

        if 'position' in outputs:
            pos_error = (outputs['position'] - targets['camera_position']).norm(dim=-1)
            all_pos_errors.append(pos_error)

        if 'rotation_error_deg' in losses:
            all_rotation_errors.append(torch.tensor([losses['rotation_error_deg']]))

        if 'mask_iou' in losses:
            all_mask_ious.append(torch.tensor([losses['mask_iou']]))

    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}

    if all_pos_errors:
        all_pos_errors = accelerator.gather_for_metrics(torch.cat(all_pos_errors))
        avg_losses['pos_mae'] = all_pos_errors.mean().item()
        avg_losses['pos_mae_pixels'] = avg_losses['pos_mae'] * cfg['data']['img_size']

    if all_rotation_errors:
        avg_losses['rotation_mae_deg'] = torch.cat(all_rotation_errors).mean().item()

    if all_mask_ious:
        avg_losses['mask_iou_avg'] = torch.cat(all_mask_ious).mean().item()

    accelerator.log({f"val/{k}": v for k, v in avg_losses.items()}, step=epoch)

    return avg_losses


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.resume:
        cfg['checkpoint']['resume'] = args.resume
    if args.output_dir:
        cfg.setdefault('checkpoint', {})['output_dir'] = args.output_dir

    use_deep_supervision = cfg.get('model', {}).get('use_deep_supervision', True)
    use_contrastive_loss = cfg.get('model', {}).get('use_contrastive_loss', cfg.get('model', {}).get('contrastive', True))
    use_rot_pos_supervision = cfg.get('training', {}).get('use_rot_pos_supervision', True)
    use_heatmap_loss = cfg.get('training', {}).get('use_heatmap_loss', use_rot_pos_supervision)
    num_bbox_mask_queries = int(cfg.get('model', {}).get('num_bbox_mask_queries', 1))
    mask_inject_mode = cfg.get('model', {}).get('mask_inject_mode', 'global_kv')
    use_global_attn_mask = bool(cfg.get('model', {}).get('use_global_attn_mask', True))
    contrastive_queue_size = int(cfg.get('model', {}).get('contrastive_queue_size', 16384))

    if not use_deep_supervision:
        cfg['model']['supervision_layers'] = []
        cfg['model']['supervision_weights'] = []

    cfg['model']['contrastive'] = use_contrastive_loss
    train_subset = cfg.get('data', {}).get('train_subset', cfg.get('data', {}).get('train_direction', 'all'))
    val_subset = cfg.get('data', {}).get('val_subset', cfg.get('data', {}).get('val_direction', 'all'))
    cfg.setdefault('data', {})['train_subset'] = train_subset
    cfg.setdefault('data', {})['val_subset'] = val_subset

    gradient_accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    mixed_precision = cfg['training'].get('mixed_precision', 'bf16') if cfg['training'].get('use_amp', True) else "no"

    # ============ RTX 5090 Performance Knobs ============
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    output_dir = Path(cfg['checkpoint']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = cfg['logging'].get('use_wandb', cfg['logging'].get('use_tensorboard', True))
    if use_wandb and not WANDB_AVAILABLE:
        use_wandb = False
        print("[warn] wandb is not installed; W&B logging is disabled.")
    wandb_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if use_wandb and not wandb_api_key and wandb_mode not in {"offline", "dryrun", "disabled"}:
        use_wandb = False
        print("[warn] WANDB_API_KEY is not configured. W&B logging is disabled.")
    if use_wandb and wandb_api_key:
        try:
            import wandb
            wandb.login(key=wandb_api_key, relogin=True)
        except Exception as exc:
            use_wandb = False
            print(f"[warn] wandb login failed: {exc}. W&B logging is disabled.")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if use_wandb else None,
        project_dir=str(output_dir),
    )

    set_seed(42)

    if accelerator.is_main_process:
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)
        accelerator.print(f"Output directory: {output_dir}")
        accelerator.print(
            f"Training settings -> deep_supervision={use_deep_supervision}, "
            f"contrastive={use_contrastive_loss}, rot_pos_supervision={use_rot_pos_supervision}, "
            f"heatmap_loss={use_heatmap_loss}, num_bbox_mask_queries={num_bbox_mask_queries}, "
            f"mask_inject_mode={mask_inject_mode}, use_global_attn_mask={use_global_attn_mask}, "
            f"contrastive_queue_size={contrastive_queue_size}"
        )
        accelerator.print(f"Dataset subset -> train_subset={train_subset}, val_subset={val_subset}")

    if accelerator.is_main_process and use_wandb:
        experiment_name = os.environ.get("WANDB_NAME", os.environ.get("EXPRIMENT_NAME", "default"))
        wandb_project = cfg['logging'].get('wandb_project', os.environ.get("WANDB_PROJECT", "gageo"))
        wandb_dir = os.environ.get("WANDB_DIR", str(output_dir / "wandb"))
        try:
            accelerator.init_trackers(
                project_name=wandb_project,
                config={
                    "batch_size": cfg['training']['batch_size'],
                    "num_epochs": cfg['training']['num_epochs'],
                    "lr_backbone": cfg['training']['lr_backbone'],
                    "lr_new_tokens": cfg['training'].get('lr_new_tokens', 5e-4),
                    "lr_heads": cfg['training']['lr_heads'],
                },
                init_kwargs={
                    "wandb": {
                        "name": experiment_name,
                        "dir": wandb_dir,
                        **({"mode": wandb_mode} if wandb_mode in {"offline", "dryrun", "disabled"} else {}),
                    }
                },
            )
        except Exception as exc:
            print(f"[warn] wandb tracker init failed: {exc}. Continue without W&B.")

    # Create datasets
    train_dataset = CrossViewDataset(
        json_path=cfg['data']['train_json'],
        data_root=cfg['data']['data_root'],
        mono_size=cfg['data']['img_size'],
        crop_size=cfg['data']['crop_size'],
        crop_sat=True,
        view_subset=train_subset,
    )

    val_dataset = CrossViewDataset(
        json_path=cfg['data']['val_json'],
        data_root=cfg['data']['data_root'],
        mono_size=cfg['data']['img_size'],
        crop_size=cfg['data']['crop_size'],
        crop_sat=False,
        view_subset=val_subset,
    )

    num_workers = int(cfg['data']['num_workers'])
    persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    accelerator.print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # ============ Create Model ============
    model = build_cross_view_localizer(
        pretrained_pi3=cfg['model'].get('pi3_weights'),
        freeze_backbone=False,  # Freeze selectively below
        freeze_prompt_encoder=cfg['model'].get('freeze_prompt_encoder', True),
        load_camera_head_weights=cfg['model'].get('load_camera_head_weights', True),
        sam_weights=cfg['model'].get('sam_weights'),
        img_size=cfg['data']['img_size'],
        patch_size=cfg['model'].get('patch_size', 14),
        decoder_size=cfg['model'].get('decoder_size', 'large'),
        num_learnable_tokens=cfg['model'].get('num_learnable_tokens', 2),
        num_bbox_mask_queries=num_bbox_mask_queries,
        num_heatmap_queries=cfg['model'].get('num_heatmap_queries', 1),
        supervision_layers=cfg['model'].get('supervision_layers', [4, 11, 17]),
        supervision_weights=cfg['model'].get('supervision_weights', [0.1, 0.3, 0.6]),
        mask_inject_mode=mask_inject_mode,
        use_global_attn_mask=use_global_attn_mask,
        dropout=cfg['model'].get('dropout', 0.1),
        contrastive=use_contrastive_loss,
        contrastive_proj_dim=cfg['model'].get('contrastive_proj_dim', 256),
        contrastive_queue_size=contrastive_queue_size,
        contrastive_momentum=cfg['model'].get('contrastive_momentum', 0.999),
        contrastive_temperature=cfg['model'].get('contrastive_temperature', 0.07),
        sam_embed_dim=cfg['model'].get('sam_embed_dim'),
    )

    if cfg['model'].get('pi3_weights'):
        accelerator.print(f'Loaded Pi3 weights from {cfg["model"]["pi3_weights"]}')

    # Verify pair-layer indexing logic: one stage = local+global
    if accelerator.is_main_process:
        num_stage_layers = model.backbone.num_stage_layers
        final_stage_idx = num_stage_layers - 1
        supervision_layers = cfg['model'].get('supervision_layers', [4, 11, 17])
        accelerator.print(
            f"Supervision stage indexing (0-based, local+global as one layer): "
            f"num_stage_layers={num_stage_layers}, final_stage_idx={final_stage_idx}, "
            f"configured={supervision_layers}, includes_final={final_stage_idx in supervision_layers}"
        )

    # Freeze the visual encoder when requested.
    freeze_encoder = cfg['model'].get('freeze_encoder', True)
    if freeze_encoder:
        for param in model.backbone.encoder.parameters():
            param.requires_grad = False
        accelerator.print('Froze visual encoder')

    # torch.compile can speed up single-GPU frozen encoder, but it is unstable on some
    # multi-GPU + DeepSpeed setups due to inductor/triton cache races.
    compile_encoder = cfg.get('training', {}).get('compile_encoder', False)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if compile_encoder and hasattr(torch, 'compile'):
        if world_size > 1:
            accelerator.print(
                f"[WARN] compile_encoder=True but WORLD_SIZE={world_size}; "
                "skip torch.compile to avoid distributed inductor cache failures."
            )
        else:
            try:
                model.backbone.encoder = torch.compile(
                    model.backbone.encoder, mode='reduce-overhead', fullgraph=False,
                )
                accelerator.print('Compiled visual encoder with torch.compile (reduce-overhead)')
            except Exception as e:
                accelerator.print(f'[WARN] torch.compile failed, skipping: {e}')

    # Freeze prompt encoder (keep projection layers)
    if cfg['model'].get('freeze_prompt_encoder', True):
        model._freeze_prompt_encoder()
        accelerator.print('Froze SAM Prompt Encoder (projections trainable)')

    # Freeze SAM mask downscaling conv layers
    if cfg['model'].get('freeze_mask_conv', True):
        for name, param in model.prompt_encoder.named_parameters():
            if 'mask_downscaling' in name:
                param.requires_grad = False
        accelerator.print('Froze SAM mask downscaling conv layers')

    # Optionally freeze Pi3 decoder
    if cfg['model'].get('freeze_decoder', False):
        for param in model.backbone.decoder.parameters():
            param.requires_grad = False
        accelerator.print('Froze Pi3 decoder')

    # Gradient checkpointing (trade compute for memory → enables larger batch)
    if cfg.get('training', {}).get('gradient_checkpointing', False):
        if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'gradient_checkpointing_enable'):
            model.backbone.encoder.gradient_checkpointing_enable()
            accelerator.print('Enabled gradient checkpointing on visual encoder')
        # For decoder blocks: enable torch gradient checkpointing
        for blk in model.backbone.decoder:
            blk.requires_grad_(True)  # ensure checkpointing works
        accelerator.print('Gradient checkpointing enabled (use with larger batch_size for speedup)')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    accelerator.print(f'Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')

    # ============ Create Criterion ============
    criterion = DETRCriterion(
        weight_bbox=cfg['training']['weight_bbox'],
        weight_giou=cfg['training']['weight_giou'],
        weight_mask_bce=cfg['training'].get('weight_mask_bce', 2.0),
        weight_mask_dice=cfg['training'].get('weight_mask_dice', 5.0),
        weight_heatmap=cfg['training']['weight_heatmap'],
        weight_rotation=cfg['training'].get('weight_rotation', 0.05),
        weight_contrastive=cfg['training'].get('weight_contrastive', 0.05),
        weight_class=cfg['training'].get('weight_class', 2.0),
        img_size=cfg['data']['img_size'],
        matcher_cost_class=cfg['training'].get('matcher_cost_class', 1.0),
        matcher_cost_bbox=cfg['training'].get('matcher_cost_bbox', 5.0),
        matcher_cost_giou=cfg['training'].get('matcher_cost_giou', 2.0),
        smooth_rotation=cfg['training'].get('smooth_rotation', True),
        supervision_layers=cfg['model'].get('supervision_layers', [4, 11, 17]),
        supervision_weights=cfg['model'].get('supervision_weights', [0.1, 0.3, 0.6]),
        use_deep_supervision=use_deep_supervision,
        use_contrastive_loss=use_contrastive_loss,
        use_rot_pos_supervision=use_rot_pos_supervision,
        use_heatmap_loss=use_heatmap_loss,
    )

    # ============ Create Optimizer with 3-Group LR ============
    param_groups = get_param_groups(
        model,
        lr_backbone=cfg['training']['lr_backbone'],
        lr_heads=cfg['training']['lr_heads'],
        weight_decay=cfg['training']['weight_decay'],
        lr_new_tokens=cfg['training'].get('lr_new_tokens'),
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

    # Scheduler
    num_epochs = cfg['training']['num_epochs']
    warmup_epochs = cfg['training']['warmup_epochs']
    grad_accum_steps = cfg['training'].get('gradient_accumulation_steps', 1)

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
    # NOTE:
    # Do not register scheduler as a custom checkpoint object here.
    # Historical checkpoints in this project were saved without custom objects,
    # and registering now causes load_state() to fail with a mismatch error.
    # We recover scheduler position from `resume_global_step` below.

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
    resume_path = resolve_resume_path(
        cfg['checkpoint'].get('resume'),
        output_dir=output_dir,
        prefer_directory=True,
    )
    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.exists():
            accelerator.print(
                f"[WARN] Resume checkpoint not found: {ckpt_path}. "
                "Will start training from scratch."
            )
            cfg['checkpoint']['resume'] = None
        else:
            accelerator.print(f'Resuming from {resume_path}')
            accelerator.load_state(str(ckpt_path))
            if (ckpt_path / 'training_state.pt').exists():
                training_state = torch.load(ckpt_path / 'training_state.pt', map_location='cpu')
                start_epoch = training_state.get('epoch', 0) + 1
                best_loss = training_state.get('best_loss', float('inf'))
                resume_global_step = training_state.get('global_step', 0)

            # Backward compatibility: old checkpoints may not contain scheduler state.
            # In that case, recover LR schedule position from saved global_step.
            sched_step = int(getattr(scheduler, 'last_epoch', -1))
            if resume_global_step > 0 and sched_step < resume_global_step:
                accelerator.print(
                    f"[WARN] Scheduler state seems stale (last_epoch={sched_step}, "
                    f"global_step={resume_global_step}). Fast-forwarding scheduler..."
                )
                for _ in range(resume_global_step - sched_step):
                    scheduler.step()
            accelerator.print(f'Resumed from epoch {start_epoch}, global_step {resume_global_step}')

    # ============ Training Loop ============
    global_step = resume_global_step
    for epoch in range(start_epoch, num_epochs):
        accelerator.print(f'\n{"="*50}')
        accelerator.print(f'Epoch {epoch}/{num_epochs}')

        if accelerator.is_main_process:
            for i, pg in enumerate(optimizer.param_groups):
                group_name = ['backbone', 'new_tokens', 'heads'][i] if i < 3 else f'group_{i}'
                accelerator.print(f'  {group_name} lr={pg["lr"]:.2e}')

        train_losses, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, accelerator, epoch, cfg,
            global_step=global_step,
        )
        accelerator.print(f'Train - ' + ', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()]))

        if (epoch + 1) % cfg['logging']['val_freq'] == 0:
            val_losses = validate(model, val_loader, criterion, accelerator, cfg, epoch)
            accelerator.print(f'Val   - ' + ', '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()]))

            # IMPORTANT: use globally reduced val loss for checkpoint decision.
            # If each rank uses its local val loss, branches may diverge and trigger
            # distributed deadlock / timeout at epoch boundaries.
            val_loss_global = accelerator.reduce(
                torch.tensor(val_losses['loss'], device=accelerator.device, dtype=torch.float32),
                reduction='mean'
            ).item()
            val_losses['loss_global'] = val_loss_global

            if val_loss_global < best_loss:
                best_loss = val_loss_global
                accelerator.save_state(output_dir / 'best')
                if accelerator.is_main_process:
                    torch.save({
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'val_losses': val_losses,
                        'global_step': global_step,
                    }, output_dir / 'best' / 'training_state.pt')
                    accelerator.print(f'Saved best model (loss: {best_loss:.4f})')

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
