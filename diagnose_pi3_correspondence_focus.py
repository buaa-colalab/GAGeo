#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline diagnostic for Pi3 cross-view correspondence preservation/focus.

This script is intentionally standalone:
- it does not modify training code or config files
- it reuses existing V3 dataset/model loading paths
- it only supports backbone_type=pi3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent / ".mplconfig").resolve()),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import CrossViewDataset, collate_fn
from models import Pi3BackboneV2, build_cross_view_localizer_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Pi3 cross-view correspondence preservation/focus.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrained_pi3_ckpt", type=str, required=True)
    parser.add_argument("--finetuned_gageo_ckpt", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diagnostics/pi3_correspondence_focus",
    )
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_type", type=str, default="point")
    parser.add_argument("--view_subset", type=str, default="all")
    parser.add_argument("--view_pair", type=str, default=None)
    parser.add_argument("--num_register_tokens", type=int, default=5)
    parser.add_argument(
        "--query_mask_source",
        type=str,
        default="auto",
        choices=["auto", "mono_mask", "point_radius"],
    )
    parser.add_argument("--point_radius_tokens", type=int, default=2)
    parser.add_argument("--num_query_tokens_for_corr", type=int, default=256)
    parser.add_argument("--num_ref_tokens_for_corr", type=int, default=512)
    parser.add_argument("--use_shuffle_baseline", action="store_true")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--num_vis", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_cfg_with_env(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if config_path.endswith(".json") else yaml.safe_load(f)

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return _expand(cfg)


def normalize_view_subset(view_subset: Optional[str], view_pair: Optional[str]) -> str:
    raw = view_subset if view_subset not in (None, "") else view_pair
    raw = "all" if raw in (None, "") else str(raw)
    key = raw.strip().lower().replace("-", "_")
    alias = {
        "all": "all",
        "both": "all",
        "d2s": "drone_to_satellite",
        "drone": "drone_to_satellite",
        "drone_to_satellite": "drone_to_satellite",
        "g2s": "ground_to_satellite",
        "ground": "ground_to_satellite",
        "ground_to_satellite": "ground_to_satellite",
    }
    if key not in alias:
        raise ValueError(
            f"Unsupported subset={raw!r}. "
            "Use one of: all, d2s/drone_to_satellite, g2s/ground_to_satellite."
        )
    return alias[key]


def resolve_checkpoint(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_file():
        return path
    candidates = [
        path / "best.pth",
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
        path / "pytorch_model.bin",
        path / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve checkpoint file from: {path}")


def extract_state_dict(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ["module", "model", "state_dict", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                if sd:
                    first_key = next(iter(sd.keys()))
                    if first_key.startswith("module."):
                        sd = {k[len("module."):]: v for k, v in sd.items()}
                return sd
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format")


def remap_legacy_mask_head_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    needle = ".output_hypernetworks_mlps.0."
    replacement = ".output_hypernetwork_mlp."
    for key, value in state_dict.items():
        if needle in key:
            remapped[key.replace(needle, replacement)] = value
        else:
            remapped[key] = value
    return remapped


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_finetuned_model(
    cfg: Dict[str, Any],
    device: torch.device,
    finetuned_ckpt: str,
) -> torch.nn.Module:
    mc = cfg["model"]
    dc = cfg["data"]
    if str(mc.get("backbone_type", "pi3")).strip().lower() != "pi3":
        raise ValueError(
            f"This diagnostic only supports backbone_type=pi3, got {mc.get('backbone_type')!r}"
        )

    model = build_cross_view_localizer_v2(
        pretrained_pi3=None,
        freeze_backbone=False,
        freeze_prompt_encoder=False,
        load_camera_head_weights=False,
        sam_weights=None,
        img_size=dc.get("img_size", 518),
        patch_size=mc.get("patch_size", 14),
        decoder_size=mc.get("decoder_size", "large"),
        num_learnable_tokens=mc.get("num_learnable_tokens", 2),
        num_bbox_mask_queries=mc.get("num_bbox_mask_queries"),
        num_heatmap_queries=mc.get("num_heatmap_queries", 1),
        supervision_layers=mc.get("supervision_layers", [4, 11, 17]),
        supervision_weights=mc.get("supervision_weights", [0.1, 0.3, 0.6]),
        mask_inject_mode=mc.get("mask_inject_mode", "global_kv"),
        use_global_attn_mask=mc.get("use_global_attn_mask", True),
        dropout=mc.get("dropout", 0.1),
        contrastive=mc.get("contrastive", True),
        contrastive_proj_dim=mc.get("contrastive_proj_dim", 256),
        contrastive_queue_size=mc.get("contrastive_queue_size", 16384),
        contrastive_momentum=mc.get("contrastive_momentum", 0.999),
        contrastive_temperature=mc.get("contrastive_temperature", 0.07),
        sam_embed_dim=mc.get("sam_embed_dim", 256),
        num_mask_tokens=mc.get("num_mask_tokens", 1),
        backbone_type=mc.get("backbone_type", "pi3"),
        encoder_name=mc.get("encoder_name", "vit_b16"),
        encoder_pretrained=False,
        encoder_weights=mc.get("encoder_weights", "LVD142M"),
        joint_vit_variant=mc.get("joint_vit_variant"),
        joint_vit_weights=mc.get("joint_vit_weights"),
        adapter_dim=mc.get("adapter_dim", 1024),
        adapter_depth=mc.get("adapter_depth", 36),
        adapter_num_heads=mc.get("adapter_num_heads", 16),
        use_frame_pos_embed=mc.get("use_frame_pos_embed", False),
        use_spatial_bbox_head=mc.get("use_spatial_bbox_head", False),
    )

    resolved_ckpt = resolve_checkpoint(finetuned_ckpt)
    ckpt_obj = torch.load(resolved_ckpt, map_location="cpu")
    state_dict = remap_legacy_mask_head_keys(extract_state_dict(ckpt_obj))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[fine-tuned] loaded checkpoint: {resolved_ckpt}")
    print(f"[fine-tuned] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_pi3_weights_into_v2(backbone: Pi3BackboneV2, checkpoint_path: str) -> None:
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path))
    else:
        obj = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = extract_state_dict(obj) if isinstance(obj, dict) else obj

    filtered = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.") or key.startswith("decoder.") or key.startswith("register_token"):
            filtered[key] = value
        if key.startswith("rope.") or key.startswith("position_getter."):
            filtered[key] = value
    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    print(f"[pretrained] loaded π3 weights from {ckpt_path}")
    print(f"[pretrained] loaded={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")


def build_pretrained_backbone(
    cfg: Dict[str, Any],
    device: torch.device,
    pretrained_pi3_ckpt: str,
) -> Pi3BackboneV2:
    mc = cfg["model"]
    dc = cfg["data"]
    backbone = Pi3BackboneV2(
        pos_type="rope100",
        decoder_size=mc.get("decoder_size", "large"),
        img_size=dc.get("img_size", 518),
        patch_size=mc.get("patch_size", 14),
        num_learnable_tokens=0,
        supervision_layers=[],
        mask_inject_mode=mc.get("mask_inject_mode", "global_kv"),
        use_global_attn_mask=mc.get("use_global_attn_mask", True),
        use_frame_pos_embed=False,
    )
    load_pi3_weights_into_v2(backbone, pretrained_pi3_ckpt)
    backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return arr


def mask_to_contours(mask: np.ndarray) -> List[np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_mask_contours(ax, mask: np.ndarray, color: Tuple[float, float, float]) -> None:
    for contour in mask_to_contours(mask):
        if contour.shape[0] < 2:
            continue
        pts = contour[:, 0, :]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)


def draw_bbox(ax, bbox_norm: np.ndarray, image_hw: Tuple[int, int], color: Tuple[float, float, float]) -> None:
    h, w = image_hw
    cx, cy, bw, bh = bbox_norm.astype(np.float32)
    x = (cx - bw / 2.0) * w
    y = (cy - bh / 2.0) * h
    rect = plt.Rectangle((x, y), bw * w, bh * h, fill=False, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)


def point_to_token_mask(
    point_xy: torch.Tensor,
    image_hw: Tuple[int, int],
    token_hw: Tuple[int, int],
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    img_h, img_w = image_hw
    token_h, token_w = token_hw
    px = float(point_xy[0].item())
    py = float(point_xy[1].item())
    tx = int(np.clip(round((px / max(img_w, 1)) * token_w), 0, token_w - 1))
    ty = int(np.clip(round((py / max(img_h, 1)) * token_h), 0, token_h - 1))
    ys = torch.arange(token_h, device=device)
    xs = torch.arange(token_w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    mask = ((grid_x - tx) ** 2 + (grid_y - ty) ** 2) <= (radius ** 2)
    return mask.reshape(-1)


def downsample_mask_to_tokens(mask: torch.Tensor, token_hw: Tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        mask[None].float(),
        size=token_hw,
        mode="nearest",
    )[0, 0]
    return resized > 0.5


def choose_query_mask(
    mono_mask: torch.Tensor,
    mono_point: torch.Tensor,
    token_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    source: str,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    if source in {"auto", "mono_mask"}:
        mask = downsample_mask_to_tokens(mono_mask, token_hw).to(device).reshape(-1)
        if source == "mono_mask":
            return mask
        if mask.any():
            return mask
    return point_to_token_mask(mono_point, image_hw, token_hw, radius, device)


def normalize_features(q_feat: torch.Tensor, r_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return F.normalize(q_feat.float(), dim=-1), F.normalize(r_feat.float(), dim=-1)


@torch.no_grad()
def extract_tokens_from_finetuned(
    model: torch.nn.Module,
    front_view: torch.Tensor,
    sat_view: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    out = model.backbone(
        front_view=front_view,
        satellite_view=sat_view,
        sparse_embeddings=None,
        dense_embeddings=None,
        prompt_coords=None,
    )
    q_feat, r_feat = normalize_features(out["front_features"], out["sate_features"])
    n_tokens = r_feat.shape[1]
    side = int(math.sqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(f"Non-square token grid: N={n_tokens}")
    return q_feat, r_feat, (side, side)


@torch.no_grad()
def extract_tokens_from_pretrained(
    backbone: Pi3BackboneV2,
    front_view: torch.Tensor,
    sat_view: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    out = backbone(
        front_view=front_view,
        satellite_view=sat_view,
        sparse_embeddings=None,
        dense_embeddings=None,
        prompt_coords=None,
    )
    q_feat, r_feat = normalize_features(out["front_features"], out["sate_features"])
    n_tokens = r_feat.shape[1]
    side = int(math.sqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(f"Non-square token grid: N={n_tokens}")
    return q_feat, r_feat, (side, side)


def sample_indices_from_mask(
    mask: torch.Tensor,
    max_num: int,
    generator: torch.Generator,
) -> torch.Tensor:
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        idx = torch.arange(mask.numel(), device=mask.device)
    if idx.numel() <= max_num:
        return idx
    perm = torch.randperm(idx.numel(), generator=generator)[:max_num].to(idx.device)
    return idx[perm]


def sample_uniform_indices(
    total: int,
    max_num: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if total <= max_num:
        return torch.arange(total, device=device)
    perm = torch.randperm(total, generator=generator)[:max_num]
    return perm.to(device)


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        return float("nan")
    corr = torch.sum(x * y) / denom
    return float(corr.item())


def compute_similarity_map(
    q_feat: torch.Tensor,
    r_feat: torch.Tensor,
    query_mask_token: torch.Tensor,
    token_hw: Tuple[int, int],
) -> torch.Tensor:
    q_obj = q_feat[query_mask_token]
    if q_obj.numel() == 0:
        q_obj = q_feat
    sim = q_obj @ r_feat.transpose(0, 1)
    sim_map = sim.mean(dim=0).reshape(token_hw)
    return sim_map


def similarity_mass_inside_gt(sim_map: torch.Tensor, sat_mask_token: torch.Tensor, eps: float = 1e-6) -> float:
    score = sim_map.flatten().float()
    score = score - score.min()
    denom = score.sum()
    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        score = torch.ones_like(score) / max(score.numel(), 1)
    else:
        score = score / denom
    return float(score[sat_mask_token.flatten()].sum().item())


def upsample_heatmap(sim_map: torch.Tensor, image_hw: Tuple[int, int]) -> np.ndarray:
    up = F.interpolate(
        sim_map[None, None].float(),
        size=image_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return up.detach().cpu().numpy()


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def finite_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_heatmap_visualization(
    query_img: torch.Tensor,
    sat_img: torch.Tensor,
    sat_mask: torch.Tensor,
    sim_map_pre: torch.Tensor,
    sim_map_ft: torch.Tensor,
    output_path: Path,
    sim_map_shuffle: Optional[torch.Tensor] = None,
    query_mask: Optional[torch.Tensor] = None,
    mono_point: Optional[torch.Tensor] = None,
    sat_bbox: Optional[torch.Tensor] = None,
) -> None:
    query_np = tensor_to_numpy_image(query_img)
    sat_np = tensor_to_numpy_image(sat_img)
    sat_mask_np = sat_mask.detach().cpu().numpy()
    image_hw = (sat_np.shape[0], sat_np.shape[1])

    heatmaps_raw = [upsample_heatmap(sim_map_pre, image_hw), upsample_heatmap(sim_map_ft, image_hw)]
    titles = ["Pretrained π3", "Fine-tuned π3"]
    if sim_map_shuffle is not None:
        heatmaps_raw.append(upsample_heatmap(sim_map_shuffle, image_hw))
        titles.append("Shuffle")

    global_min = min(float(np.min(h)) for h in heatmaps_raw)
    global_max = max(float(np.max(h)) for h in heatmaps_raw)
    denom = max(global_max - global_min, 1e-6)
    heatmaps_norm = [(h - global_min) / denom for h in heatmaps_raw]

    ncols = 2 + len(heatmaps_norm)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(query_np)
    axes[0].set_title("Query")
    if query_mask is not None:
        draw_mask_contours(axes[0], query_mask.detach().cpu().numpy(), color=(1.0, 0.2, 0.2))
    if mono_point is not None:
        px, py = float(mono_point[0].item()), float(mono_point[1].item())
        axes[0].scatter([px], [py], c="yellow", s=50, edgecolors="black", linewidths=0.75)
    axes[0].axis("off")

    axes[1].imshow(sat_np)
    axes[1].set_title("Satellite + GT")
    if np.any(sat_mask_np > 0):
        draw_mask_contours(axes[1], sat_mask_np, color=(0.0, 1.0, 0.0))
    elif sat_bbox is not None:
        draw_bbox(axes[1], sat_bbox.detach().cpu().numpy(), image_hw, color=(0.0, 1.0, 0.0))
    axes[1].axis("off")

    for ax, title, heatmap in zip(axes[2:], titles, heatmaps_norm):
        ax.imshow(sat_np)
        ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        if np.any(sat_mask_np > 0):
            draw_mask_contours(ax, sat_mask_np, color=(0.0, 1.0, 0.0))
        elif sat_bbox is not None:
            draw_bbox(ax, sat_bbox.detach().cpu().numpy(), image_hw, color=(0.0, 1.0, 0.0))
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def maybe_relpath(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def to_serializable_float(value: float) -> Optional[float]:
    return None if not np.isfinite(value) else float(value)


def build_dataset_and_loader(cfg: Dict[str, Any], cli_args: argparse.Namespace) -> Tuple[Subset, DataLoader]:
    dc = dict(cfg.get("data", {}))
    view_subset = normalize_view_subset(cli_args.view_subset, cli_args.view_pair)
    dataset = CrossViewDataset(
        json_path=cli_args.split_json or dc.get("val_json"),
        data_root=cli_args.data_root or dc.get("data_root"),
        mono_size=dc.get("img_size", 518),
        sat_size=dc.get("sat_size", 1280),
        crop_sat=False,
        crop_size=dc.get("crop_size", 518),
        view_subset=view_subset,
        transform=None,
    )
    num_samples = min(int(cli_args.num_samples), len(dataset))
    rng = random.Random(cli_args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    subset = Subset(dataset, indices[:num_samples])
    loader = DataLoader(
        subset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=int(cli_args.num_workers),
        pin_memory=(cli_args.device.startswith("cuda") and torch.cuda.is_available()),
        collate_fn=collate_fn,
    )
    return subset, loader


def summarize_metrics(rows: Sequence[Dict[str, Any]], use_shuffle_baseline: bool) -> Dict[str, Any]:
    corr = [row["cross_view_sim_corr"] for row in rows]
    mass_pre = [row["mass_in_gt_pre"] for row in rows]
    mass_ft = [row["mass_in_gt_ft"] for row in rows]
    mass_delta = [row["mass_in_gt_delta"] for row in rows]

    summary = {
        "num_samples": len(rows),
        "cross_view_similarity_preservation": {
            "corr_mean": finite_mean(corr),
            "corr_median": finite_median(corr),
        },
        "similarity_mass_inside_gt_mask": {
            "mass_in_gt_pre_mean": finite_mean(mass_pre),
            "mass_in_gt_ft_mean": finite_mean(mass_ft),
            "mass_in_gt_delta_mean": finite_mean(mass_delta),
        },
    }

    if use_shuffle_baseline:
        corr_shuffle = [row["cross_view_sim_corr_shuffle"] for row in rows]
        mass_shuffle = [row["mass_in_gt_shuffle"] for row in rows]
        summary["cross_view_similarity_preservation"]["corr_shuffle_mean"] = finite_mean(corr_shuffle)
        summary["similarity_mass_inside_gt_mask"]["mass_in_gt_shuffle_mean"] = finite_mean(mass_shuffle)
    return summary


def process_batch(
    batch: Dict[str, Any],
    pretrained_backbone: Pi3BackboneV2,
    finetuned_model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    vis_state: Dict[str, int],
    output_dir: Path,
    rows: List[Dict[str, Any]],
) -> None:
    front_view = batch["front_view"].to(device, non_blocking=True)
    sat_view = batch["satellite_view"].to(device, non_blocking=True)

    pre_q, pre_r, token_hw = extract_tokens_from_pretrained(pretrained_backbone, front_view, sat_view)
    ft_q, ft_r, token_hw_ft = extract_tokens_from_finetuned(finetuned_model, front_view, sat_view)
    if token_hw != token_hw_ft:
        raise ValueError(f"Token grid mismatch: pretrained={token_hw}, fine-tuned={token_hw_ft}")

    print_once_key = "_printed_view_order"
    if not hasattr(process_batch, print_once_key):
        print("[token extraction] view order = ['satellite', 'front']")
        setattr(process_batch, print_once_key, True)

    generator = torch.Generator()
    generator.manual_seed(args.seed + len(rows))

    batch_size = front_view.shape[0]
    img_hw = (front_view.shape[-2], front_view.shape[-1])
    for i in range(batch_size):
        query_mask_token = choose_query_mask(
            mono_mask=batch["mono_mask"][i].to(device),
            mono_point=batch["mono_point"][i].to(device),
            token_hw=token_hw,
            image_hw=img_hw,
            source=args.query_mask_source,
            radius=args.point_radius_tokens,
            device=device,
        )
        sat_mask_token = downsample_mask_to_tokens(batch["sat_mask"][i], token_hw).to(device).reshape(-1)
        if not sat_mask_token.any():
            bbox = batch["sat_bbox"][i].detach().cpu().numpy().astype(np.float32)
            token_h, token_w = token_hw
            cx, cy, bw, bh = bbox
            x1 = max(0, int(math.floor((cx - bw / 2.0) * token_w)))
            x2 = min(token_w, int(math.ceil((cx + bw / 2.0) * token_w)))
            y1 = max(0, int(math.floor((cy - bh / 2.0) * token_h)))
            y2 = min(token_h, int(math.ceil((cy + bh / 2.0) * token_h)))
            fallback = torch.zeros(token_h, token_w, dtype=torch.bool, device=device)
            fallback[y1:y2, x1:x2] = True
            sat_mask_token = fallback.reshape(-1)

        q_idx = sample_indices_from_mask(
            query_mask_token,
            max_num=args.num_query_tokens_for_corr,
            generator=generator,
        )
        r_idx = sample_uniform_indices(
            total=pre_r.shape[1],
            max_num=args.num_ref_tokens_for_corr,
            device=device,
            generator=generator,
        )

        pre_q_s = pre_q[i, q_idx]
        pre_r_s = pre_r[i, r_idx]
        ft_q_s = ft_q[i, q_idx]
        ft_r_s = ft_r[i, r_idx]

        s_pre = pre_q_s @ pre_r_s.transpose(0, 1)
        s_ft = ft_q_s @ ft_r_s.transpose(0, 1)
        corr = pearson_corr(s_pre, s_ft)

        corr_shuffle = float("nan")
        sim_map_shuffle = None
        mass_shuffle = float("nan")
        if args.use_shuffle_baseline:
            perm = torch.randperm(pre_r_s.shape[0], generator=generator).to(device)
            pre_r_shuffle = pre_r_s[perm]
            s_shuffle = pre_q_s @ pre_r_shuffle.transpose(0, 1)
            corr_shuffle = pearson_corr(s_pre, s_shuffle)

            full_perm = torch.randperm(pre_r.shape[1], generator=generator).to(device)
            sim_map_shuffle = compute_similarity_map(
                pre_q[i],
                pre_r[i, full_perm],
                query_mask_token,
                token_hw,
            )
            mass_shuffle = similarity_mass_inside_gt(sim_map_shuffle, sat_mask_token)

        sim_map_pre = compute_similarity_map(pre_q[i], pre_r[i], query_mask_token, token_hw)
        sim_map_ft = compute_similarity_map(ft_q[i], ft_r[i], query_mask_token, token_hw)
        mass_pre = similarity_mass_inside_gt(sim_map_pre, sat_mask_token)
        mass_ft = similarity_mass_inside_gt(sim_map_ft, sat_mask_token)

        row = {
            "sample_index": len(rows),
            "city": batch["cities"][i],
            "mono_filename": batch["mono_filenames"][i],
            "sat_filename": batch["sat_filenames"][i],
            "cross_view_sim_corr": to_serializable_float(corr),
            "cross_view_sim_corr_shuffle": to_serializable_float(corr_shuffle),
            "mass_in_gt_pre": to_serializable_float(mass_pre),
            "mass_in_gt_ft": to_serializable_float(mass_ft),
            "mass_in_gt_delta": to_serializable_float(mass_ft - mass_pre),
            "mass_in_gt_shuffle": to_serializable_float(mass_shuffle),
        }
        rows.append(row)

        if args.save_vis and vis_state["saved"] < args.num_vis:
            save_heatmap_visualization(
                query_img=batch["front_view"][i],
                sat_img=batch["satellite_view"][i],
                sat_mask=batch["sat_mask"][i, 0],
                sim_map_pre=sim_map_pre.detach().cpu(),
                sim_map_ft=sim_map_ft.detach().cpu(),
                output_path=output_dir / "vis" / f"sample_{vis_state['saved']:06d}.png",
                sim_map_shuffle=sim_map_shuffle.detach().cpu() if sim_map_shuffle is not None else None,
                query_mask=batch["mono_mask"][i, 0],
                mono_point=batch["mono_point"][i],
                sat_bbox=batch["sat_bbox"][i],
            )
            vis_state["saved"] += 1


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    cfg = load_cfg_with_env(args.config)
    cfg.setdefault("data", {})
    cfg["data"]["val_json"] = args.split_json
    cfg["data"]["data_root"] = args.data_root

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] {maybe_relpath(args.config)}")
    print(f"[data] split_json={maybe_relpath(args.split_json)}")
    print(f"[data] data_root={maybe_relpath(args.data_root)}")
    print(f"[subset] {normalize_view_subset(args.view_subset, args.view_pair)}")
    print(f"[device] {device}")

    _, loader = build_dataset_and_loader(cfg, args)
    pretrained_backbone = build_pretrained_backbone(cfg, device, args.pretrained_pi3_ckpt)
    finetuned_model = build_finetuned_model(cfg, device, args.finetuned_gageo_ckpt)

    rows: List[Dict[str, Any]] = []
    vis_state = {"saved": 0}
    progress = tqdm(loader, desc="Diagnosing", leave=True)
    with torch.no_grad():
        for batch in progress:
            process_batch(
                batch=batch,
                pretrained_backbone=pretrained_backbone,
                finetuned_model=finetuned_model,
                args=args,
                device=device,
                vis_state=vis_state,
                output_dir=output_dir,
                rows=rows,
            )
            progress.set_postfix(
                samples=len(rows),
                corr=f"{finite_mean([r['cross_view_sim_corr'] for r in rows]):.4f}" if rows else "nan",
            )

    summary = summarize_metrics(rows, use_shuffle_baseline=args.use_shuffle_baseline)
    save_json(output_dir / "metrics_summary.json", summary)
    save_jsonl(output_dir / "metrics_per_sample.jsonl", rows)
    summary_row = {"num_samples": summary["num_samples"]}
    summary_row.update(summary["cross_view_similarity_preservation"])
    summary_row.update(summary["similarity_mass_inside_gt_mask"])
    save_csv(output_dir / "metrics_summary.csv", [summary_row])

    print(f"[done] samples={len(rows)} vis_saved={vis_state['saved']}")
    print(f"[done] summary -> {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
