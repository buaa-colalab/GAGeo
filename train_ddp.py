"""
GAGeo training with native PyTorch DDP.

This entrypoint mirrors train.py as closely as possible:
- same config parsing
- same model / criterion / optimizer construction
- same prompt sampling
- same logging tags

Only the distributed/runtime layer changes:
- torchrun + DistributedDataParallel instead of Accelerate/DeepSpeed
"""
import os
os.environ["TMPDIR"] = "/tmp"

import argparse
import contextlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_scheduler

from datasets import CrossViewDataset, collate_fn
from models import build_cross_view_localizer
from utils import DETRCriterion, get_param_groups
from utils.prompt_utils import prepare_random_prompt, prepare_single_prompt

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAGeo with native DDP")

    parser.add_argument("--config", type=str, required=True, help="Path to training config file (YAML)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override checkpoint.output_dir for this run")
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

    with open(config_path, "r", encoding="utf-8") as f:
        return _expand_env(yaml.safe_load(f))


def resolve_resume_path(resume_value, output_dir=None, prefer_file=True):
    """Resolve a resume hint to an existing checkpoint file or directory."""
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
        add_candidate(raw / "best.pth")
        add_candidate(raw / "best")
    elif raw.suffix == ".pth":
        add_candidate(raw.with_suffix(""))
    else:
        add_candidate(raw.with_suffix(".pth"))
        add_candidate(raw / "best.pth")
        add_candidate(raw / "best")

    if output_dir:
        out = Path(output_dir)
        if raw == out:
            add_candidate(out / "best.pth")
            add_candidate(out / "best")

    existing = [path for path in candidates if path.exists()]
    if not existing:
        return raw

    if prefer_file:
        for path in existing:
            if path.is_file():
                return path
    else:
        for path in existing:
            if path.is_dir():
                return path
    return existing[0]


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device, is_distributed


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    rank = int(os.environ.get("RANK", "0"))
    seed = int(seed) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def get_amp_dtype(cfg: dict):
    if not cfg.get("training", {}).get("use_amp", True):
        return None
    mp = str(cfg.get("training", {}).get("mixed_precision", "bf16")).lower()
    if mp in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mp in {"fp16", "float16", "16"}:
        return torch.float16
    return None


def reduce_mean_scalar(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def maybe_init_wandb(cfg: dict, output_dir: Path, experiment_name: str, is_main_process: bool):
    """Create a W&B run on rank0 only."""
    enabled = cfg["logging"].get("use_wandb", cfg["logging"].get("use_tensorboard", True))
    if not enabled or not is_main_process:
        return None
    if wandb is None:
        print("[warn] wandb is not installed; W&B logging is disabled.")
        return None

    wandb_dir = Path(os.environ.get("WANDB_DIR", str(output_dir / "wandb")))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_project = cfg["logging"].get("wandb_project", os.environ.get("WANDB_PROJECT", "gageo"))
    wandb_name = os.environ.get("WANDB_NAME", experiment_name)
    wandb_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    api_key = os.environ.get("WANDB_API_KEY", "").strip()

    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception as exc:
            print(f"[warn] wandb login failed: {exc}. W&B logging is disabled.")
            return None
    elif wandb_mode not in {"offline", "dryrun", "disabled"}:
        print("[warn] WANDB_API_KEY is not configured. W&B logging is disabled.")
        return None

    try:
        init_kwargs = dict(project=wandb_project, name=wandb_name, dir=str(wandb_dir), config=cfg)
        if wandb_mode in {"offline", "dryrun", "disabled"}:
            init_kwargs["mode"] = wandb_mode
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[warn] wandb init failed: {exc}. W&B logging is disabled.")
        return None


def wandb_log(run, metrics: dict[str, float], step: int | None = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, scaler, epoch: int, best_loss: float, global_step: int, val_losses: dict | None = None):
    model_to_save = model.module if isinstance(model, DDP) else model
    state = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "global_step": global_step,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if val_losses is not None:
        state["val_losses"] = val_losses
    torch.save(state, path)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epoch: int,
    cfg: dict,
    global_step: int,
    writer,
    is_main_process: bool,
    world_size: int,
    is_distributed: bool,
):
    model.train()

    total_losses = {}
    running_losses = {}
    running_count = 0
    log_freq = cfg["logging"].get("log_freq", 50)
    img_size = cfg["data"]["img_size"]
    supervision_layers = cfg.get("model", {}).get("supervision_layers", [4, 11, 17])
    decoder_size = cfg.get("model", {}).get("decoder_size", "large")
    num_stage_layers = 18 if decoder_size == "large" else 12
    final_stage_idx = num_stage_layers - 1
    skip_final_inter_curve = final_stage_idx in supervision_layers
    supervision_layers_set = set(supervision_layers)
    grad_accum_steps = int(cfg["training"].get("gradient_accumulation_steps", 1))
    amp_dtype = get_amp_dtype(cfg)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process)

    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)
        front_view = batch["front_view"]
        sat_view = batch["satellite_view"]
        targets = {
            "sat_bbox": batch["sat_bbox"],
            "rotation_matrix": batch["rotation_matrix"],
            "camera_position": batch["camera_position"],
        }
        if "sat_mask" in batch:
            targets["sat_mask"] = batch["sat_mask"]

        points, boxes, masks = prepare_random_prompt(batch, device)
        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
        sync_context = contextlib.nullcontext()
        if is_distributed and not should_step:
            sync_context = model.no_sync()

        with sync_context:
            with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
                outputs = model(
                    front_view=front_view,
                    satellite_view=sat_view,
                    points=points,
                    boxes=boxes,
                    masks=masks,
                    mono_mask=batch.get("mono_mask"),
                    sat_mask=batch.get("sat_mask"),
                )
                losses = criterion(outputs, targets)
                loss = losses["loss"] / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        batch_losses = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                batch_losses[key] = value.detach().float().mean().item()
            else:
                batch_losses[key] = float(value)
        for key, value in batch_losses.items():
            total_losses[key] = total_losses.get(key, 0.0) + value

        running_count += 1
        for key, value in batch_losses.items():
            if key.startswith("inter_"):
                parts = key.split("_", 2)
                if len(parts) != 3:
                    continue
                try:
                    layer_idx = int(parts[1])
                except ValueError:
                    continue
                if layer_idx not in supervision_layers_set:
                    continue
                if skip_final_inter_curve and layer_idx == final_stage_idx:
                    continue
            running_losses[key] = running_losses.get(key, 0.0) + value

        if "pos_error" in batch_losses:
            running_losses["pos_error_px"] = running_losses.get("pos_error_px", 0.0) + batch_losses["pos_error"] * img_size

        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if writer is not None and global_step % log_freq == 0:
                log_dict = {}
                for key, value in running_losses.items():
                    if key.startswith("inter_"):
                        continue
                    log_dict[f"train_step/{key}"] = value / max(running_count, 1)
                for key, value in running_losses.items():
                    if not key.startswith("inter_"):
                        continue
                    parts = key.split("_", 2)
                    if len(parts) != 3:
                        continue
                    try:
                        layer_idx = int(parts[1])
                    except ValueError:
                        continue
                    if skip_final_inter_curve and layer_idx == final_stage_idx:
                        continue
                    metric_name = parts[2]
                    log_dict[f"trian_step_{layer_idx}/{metric_name}"] = value / max(running_count, 1)

                log_dict["train_step/lr_backbone"] = optimizer.param_groups[0]["lr"]
                if len(optimizer.param_groups) > 1:
                    log_dict["train_step/lr_new_tokens"] = optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 2:
                    log_dict["train_step/lr_heads"] = optimizer.param_groups[2]["lr"]
                log_dict["train_step/epoch"] = epoch
                wandb_log(writer, log_dict, step=global_step)
                running_losses = {}
                running_count = 0

        if is_main_process:
            postfix = {}
            for key, value in batch_losses.items():
                if key.startswith("loss") or key in ("rotation_error_deg", "bbox_iou", "pos_error", "mask_iou", "heatmap_center_prob"):
                    postfix[key.replace("loss_", "")] = f"{value:.4f}"
            pbar.set_postfix(postfix)

    avg_losses = {}
    denom = float(len(dataloader))
    for key, value in total_losses.items():
        avg = value / denom
        avg_losses[key] = reduce_mean_scalar(avg, device, world_size)
    if writer is not None:
        wandb_log(writer, {f"train/{key}": value for key, value in avg_losses.items()}, step=global_step)
    return avg_losses, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    epoch: int,
    writer,
    is_main_process: bool,
    world_size: int,
    log_step: int | None = None,
):
    model.eval()

    total_losses = {}
    pos_error_sum = 0.0
    pos_error_count = 0
    rot_error_sum = 0.0
    rot_error_count = 0
    mask_iou_sum = 0.0
    mask_iou_count = 0
    amp_dtype = get_amp_dtype(cfg)

    for batch in tqdm(dataloader, desc="Validation", disable=not is_main_process):
        batch = move_batch_to_device(batch, device)
        front_view = batch["front_view"]
        sat_view = batch["satellite_view"]
        targets = {
            "sat_bbox": batch["sat_bbox"],
            "rotation_matrix": batch["rotation_matrix"],
            "camera_position": batch["camera_position"],
        }
        if "sat_mask" in batch:
            targets["sat_mask"] = batch["sat_mask"]

        points, boxes, masks = prepare_single_prompt(batch, device, prompt_type="point")

        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            outputs = model(
                front_view=front_view,
                satellite_view=sat_view,
                points=points,
                boxes=boxes,
                masks=masks,
                mono_mask=batch.get("mono_mask"),
                sat_mask=batch.get("sat_mask"),
            )
            losses = criterion(outputs, targets)

        batch_losses = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                batch_losses[key] = value.detach().float().mean().item()
            else:
                batch_losses[key] = float(value)
        for key, value in batch_losses.items():
            total_losses[key] = total_losses.get(key, 0.0) + value

        if "position" in outputs:
            pos_error = (outputs["position"] - targets["camera_position"]).norm(dim=-1)
            pos_error_sum += pos_error.detach().float().sum().item()
            pos_error_count += int(pos_error.numel())
        if "rotation_error_deg" in losses:
            rot_error_sum += float(losses["rotation_error_deg"])
            rot_error_count += 1
        if "mask_iou" in losses:
            mask_iou_sum += float(losses["mask_iou"])
            mask_iou_count += 1

    avg_losses = {}
    denom = float(len(dataloader))
    for key, value in total_losses.items():
        avg_losses[key] = reduce_mean_scalar(value / denom, device, world_size)

    metrics = {
        "pos_error_sum": pos_error_sum,
        "pos_error_count": pos_error_count,
        "rot_error_sum": rot_error_sum,
        "rot_error_count": rot_error_count,
        "mask_iou_sum": mask_iou_sum,
        "mask_iou_count": mask_iou_count,
    }
    metrics_tensor = torch.tensor(
        [
            metrics["pos_error_sum"],
            float(metrics["pos_error_count"]),
            metrics["rot_error_sum"],
            float(metrics["rot_error_count"]),
            metrics["mask_iou_sum"],
            float(metrics["mask_iou_count"]),
        ],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    if metrics_tensor[1].item() > 0:
        avg_losses["pos_mae"] = (metrics_tensor[0] / metrics_tensor[1]).item()
        avg_losses["pos_mae_pixels"] = avg_losses["pos_mae"] * cfg["data"]["img_size"]
    if metrics_tensor[3].item() > 0:
        avg_losses["rotation_mae_deg"] = (metrics_tensor[2] / metrics_tensor[3]).item()
    if metrics_tensor[5].item() > 0:
        avg_losses["mask_iou_avg"] = (metrics_tensor[4] / metrics_tensor[5]).item()

    if writer is not None:
        wandb_log(writer, {f"val/{key}": value for key, value in avg_losses.items()}, step=log_step if log_step is not None else epoch)
    return avg_losses


def main():
    args = parse_args()
    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    is_main_process = rank == 0
    cfg = load_config(args.config)

    if args.resume:
        cfg["checkpoint"]["resume"] = args.resume
    if args.output_dir:
        cfg.setdefault("checkpoint", {})["output_dir"] = args.output_dir

    use_deep_supervision = cfg.get("model", {}).get("use_deep_supervision", True)
    use_contrastive_loss = cfg.get("model", {}).get("use_contrastive_loss", cfg.get("model", {}).get("contrastive", True))
    use_rot_pos_supervision = cfg.get("training", {}).get("use_rot_pos_supervision", True)
    use_heatmap_loss = cfg.get("training", {}).get("use_heatmap_loss", use_rot_pos_supervision)
    num_bbox_mask_queries = int(cfg.get("model", {}).get("num_bbox_mask_queries", 1))
    mask_inject_mode = cfg.get("model", {}).get("mask_inject_mode", "global_kv")
    use_global_attn_mask = bool(cfg.get("model", {}).get("use_global_attn_mask", True))
    contrastive_queue_size = int(cfg.get("model", {}).get("contrastive_queue_size", 16384))

    if not use_deep_supervision:
        cfg["model"]["supervision_layers"] = []
        cfg["model"]["supervision_weights"] = []

    cfg["model"]["contrastive"] = use_contrastive_loss
    train_subset = cfg.get("data", {}).get("train_subset", cfg.get("data", {}).get("train_direction", "all"))
    val_subset = cfg.get("data", {}).get("val_subset", cfg.get("data", {}).get("val_direction", "all"))
    cfg.setdefault("data", {})["train_subset"] = train_subset
    cfg.setdefault("data", {})["val_subset"] = val_subset

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    seed_everything(42)
    output_dir = Path(cfg["checkpoint"]["output_dir"])
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        print(f"Output directory: {output_dir}")
        print(
            f"Training settings -> deep_supervision={use_deep_supervision}, "
            f"contrastive={use_contrastive_loss}, rot_pos_supervision={use_rot_pos_supervision}, "
            f"heatmap_loss={use_heatmap_loss}, num_bbox_mask_queries={num_bbox_mask_queries}, "
            f"mask_inject_mode={mask_inject_mode}, use_global_attn_mask={use_global_attn_mask}, "
            f"contrastive_queue_size={contrastive_queue_size}"
        )
        print(f"Dataset subset -> train_subset={train_subset}, val_subset={val_subset}")

    experiment_name = os.environ.get("WANDB_NAME", os.environ.get("EXPRIMENT_NAME", "default"))
    writer = maybe_init_wandb(cfg, output_dir, experiment_name, is_main_process)

    train_dataset = CrossViewDataset(
        json_path=cfg["data"]["train_json"],
        data_root=cfg["data"]["data_root"],
        mono_size=cfg["data"]["img_size"],
        crop_size=cfg["data"]["crop_size"],
        crop_sat=True,
        view_subset=train_subset,
    )
    val_dataset = CrossViewDataset(
        json_path=cfg["data"]["val_json"],
        data_root=cfg["data"]["data_root"],
        mono_size=cfg["data"]["img_size"],
        crop_size=cfg["data"]["crop_size"],
        crop_sat=False,
        view_subset=val_subset,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    num_workers = int(cfg["data"]["num_workers"])
    persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    if is_main_process:
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    model = build_cross_view_localizer(
        pretrained_pi3=cfg["model"].get("pi3_weights"),
        freeze_backbone=False,
        freeze_prompt_encoder=cfg["model"].get("freeze_prompt_encoder", True),
        load_camera_head_weights=cfg["model"].get("load_camera_head_weights", True),
        sam_weights=cfg["model"].get("sam_weights"),
        img_size=cfg["data"]["img_size"],
        patch_size=cfg["model"].get("patch_size", 14),
        decoder_size=cfg["model"].get("decoder_size", "large"),
        num_learnable_tokens=cfg["model"].get("num_learnable_tokens", 2),
        num_bbox_mask_queries=num_bbox_mask_queries,
        num_heatmap_queries=cfg["model"].get("num_heatmap_queries", 1),
        supervision_layers=cfg["model"].get("supervision_layers", [4, 11, 17]),
        supervision_weights=cfg["model"].get("supervision_weights", [0.1, 0.3, 0.6]),
        mask_inject_mode=mask_inject_mode,
        use_global_attn_mask=use_global_attn_mask,
        dropout=cfg["model"].get("dropout", 0.1),
        contrastive=use_contrastive_loss,
        contrastive_proj_dim=cfg["model"].get("contrastive_proj_dim", 256),
        contrastive_queue_size=contrastive_queue_size,
        contrastive_momentum=cfg["model"].get("contrastive_momentum", 0.999),
        contrastive_temperature=cfg["model"].get("contrastive_temperature", 0.07),
        sam_embed_dim=cfg["model"].get("sam_embed_dim"),
    ).to(device)

    if cfg["model"].get("pi3_weights") and is_main_process:
        print(f'Loaded Pi3 weights from {cfg["model"]["pi3_weights"]}')

    if is_main_process:
        num_stage_layers = model.backbone.num_stage_layers
        final_stage_idx = num_stage_layers - 1
        supervision_layers = cfg["model"].get("supervision_layers", [4, 11, 17])
        print(
            f"Supervision stage indexing (0-based, local+global as one layer): "
            f"num_stage_layers={num_stage_layers}, final_stage_idx={final_stage_idx}, "
            f"configured={supervision_layers}, includes_final={final_stage_idx in supervision_layers}"
        )

    freeze_encoder = cfg["model"].get("freeze_encoder", True)
    if freeze_encoder:
        for param in model.backbone.encoder.parameters():
            param.requires_grad = False
        if is_main_process:
            print("Froze visual encoder")

    compile_encoder = cfg.get("training", {}).get("compile_encoder", False)
    if compile_encoder and hasattr(torch, "compile"):
        if world_size > 1:
            if is_main_process:
                print("[WARN] compile_encoder=True but WORLD_SIZE>1; skip torch.compile in DDP mode.")
        else:
            try:
                model.backbone.encoder = torch.compile(model.backbone.encoder, mode="reduce-overhead", fullgraph=False)
                if is_main_process:
                    print("Compiled visual encoder with torch.compile")
            except Exception as exc:
                if is_main_process:
                    print(f"[WARN] torch.compile failed, skipping: {exc}")

    if cfg["model"].get("freeze_prompt_encoder", True):
        model._freeze_prompt_encoder()
        if is_main_process:
            print("Froze SAM Prompt Encoder (projections trainable)")

    if cfg["model"].get("freeze_mask_conv", True):
        for name, param in model.prompt_encoder.named_parameters():
            if "mask_downscaling" in name:
                param.requires_grad = False
        if is_main_process:
            print("Froze SAM mask downscaling conv layers")

    if cfg["model"].get("freeze_decoder", False):
        for param in model.backbone.decoder.parameters():
            param.requires_grad = False
        if is_main_process:
            print("Froze Pi3 decoder")

    if cfg.get("training", {}).get("gradient_checkpointing", False):
        if hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "gradient_checkpointing_enable"):
            model.backbone.encoder.gradient_checkpointing_enable()
            if is_main_process:
                print("Enabled gradient checkpointing on visual encoder")
        for blk in model.backbone.decoder:
            blk.requires_grad_(True)
        if is_main_process:
            print("Gradient checkpointing enabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    criterion = DETRCriterion(
        weight_bbox=cfg["training"]["weight_bbox"],
        weight_giou=cfg["training"]["weight_giou"],
        weight_mask_bce=cfg["training"].get("weight_mask_bce", 2.0),
        weight_mask_dice=cfg["training"].get("weight_mask_dice", 5.0),
        weight_heatmap=cfg["training"]["weight_heatmap"],
        weight_rotation=cfg["training"].get("weight_rotation", 0.05),
        weight_contrastive=cfg["training"].get("weight_contrastive", 0.05),
        weight_class=cfg["training"].get("weight_class", 2.0),
        img_size=cfg["data"]["img_size"],
        matcher_cost_class=cfg["training"].get("matcher_cost_class", 1.0),
        matcher_cost_bbox=cfg["training"].get("matcher_cost_bbox", 5.0),
        matcher_cost_giou=cfg["training"].get("matcher_cost_giou", 2.0),
        smooth_rotation=cfg["training"].get("smooth_rotation", True),
        supervision_layers=cfg["model"].get("supervision_layers", [4, 11, 17]),
        supervision_weights=cfg["model"].get("supervision_weights", [0.1, 0.3, 0.6]),
        use_deep_supervision=use_deep_supervision,
        use_contrastive_loss=use_contrastive_loss,
        use_rot_pos_supervision=use_rot_pos_supervision,
        use_heatmap_loss=use_heatmap_loss,
    )

    param_groups = get_param_groups(
        model,
        lr_backbone=cfg["training"]["lr_backbone"],
        lr_heads=cfg["training"]["lr_heads"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_new_tokens=cfg["training"].get("lr_new_tokens"),
    )
    optimizer = AdamW(param_groups)

    if is_main_process:
        print(f"Optimizer param groups (before DDP wrap): {len(optimizer.param_groups)}")
        for i, pg in enumerate(optimizer.param_groups):
            print(
                f"  group[{i}] lr={pg['lr']:.2e}, weight_decay={pg.get('weight_decay', 0.0):.2e}, "
                f"num_tensors={len(pg['params'])}"
            )

    if is_distributed:
        # The contrastive head owns a MoCo queue buffer that is intentionally
        # updated locally on each rank. Do not broadcast rank0 buffers every
        # forward, otherwise local queues would be overwritten and DDP would add
        # unnecessary synchronization around non-parameter state.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    num_epochs = cfg["training"]["num_epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]
    grad_accum_steps = int(cfg["training"].get("gradient_accumulation_steps", 1))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    num_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps = steps_per_epoch * warmup_epochs
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    if is_main_process:
        print(
            f"Scheduler: {num_training_steps} total steps, {num_warmup_steps} warmup steps "
            f"({steps_per_epoch} steps/epoch, accumulation={grad_accum_steps})"
        )
        for i, pg in enumerate(optimizer.param_groups):
            print(f"  group[{i}] lr={pg['lr']:.2e}, initial_lr={pg.get('initial_lr', pg['lr']):.2e}")

    amp_dtype = get_amp_dtype(cfg)
    scaler = None
    if amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    start_epoch = 0
    best_loss = float("inf")
    global_step = 0
    resume_path = resolve_resume_path(
        cfg["checkpoint"].get("resume"),
        output_dir=output_dir,
        prefer_file=True,
    )
    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.exists():
            if is_main_process:
                print(f"[WARN] Resume checkpoint not found: {ckpt_path}. Start from scratch.")
        else:
            map_location = {"cuda:0": f"cuda:{local_rank}"}
            ckpt = torch.load(ckpt_path, map_location=map_location)
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_loss = float(ckpt.get("best_loss", float("inf")))
            global_step = int(ckpt.get("global_step", 0))
            if is_main_process:
                print(f"Resumed from {ckpt_path} at epoch {start_epoch}, global_step {global_step}")

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process:
            print(f'\n{"=" * 50}')
            print(f"Epoch {epoch}/{num_epochs}")
            for i, pg in enumerate(optimizer.param_groups):
                group_name = ["backbone", "new_tokens", "heads"][i] if i < 3 else f"group_{i}"
                print(f"  {group_name} lr={pg['lr']:.2e}")

        train_losses, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            cfg=cfg,
            global_step=global_step,
            writer=writer,
            is_main_process=is_main_process,
            world_size=world_size,
            is_distributed=is_distributed,
        )
        if is_main_process:
            print("Train - " + ", ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()]))

        if (epoch + 1) % cfg["logging"]["val_freq"] == 0:
            val_losses = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                cfg=cfg,
                epoch=epoch,
                writer=writer,
                is_main_process=is_main_process,
                world_size=world_size,
                log_step=global_step,
            )
            if is_main_process:
                print("Val   - " + ", ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()]))
                if val_losses["loss"] < best_loss:
                    best_loss = val_losses["loss"]
                    save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, scaler, epoch, best_loss, global_step, val_losses)
                    print(f"Saved best model (loss: {best_loss:.4f})")

        if is_main_process and (epoch + 1) % cfg["checkpoint"]["save_freq"] == 0:
            save_checkpoint(output_dir / f"epoch_{epoch}.pth", model, optimizer, scheduler, scaler, epoch, best_loss, global_step)

        if is_distributed:
            dist.barrier()

    if writer is not None:
        writer.finish()
    if is_main_process:
        print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
