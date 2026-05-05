#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative visualization: camera position (heatmap) and rotation (arrow)
across three prompt types.

For each selected sample, generates ONE row with 11 panels:
  [Front+Pt | Front+BBox | Front+Mask |
   Sat+Heat(Pt) | Sat+Heat(BBox) | Sat+Heat(Mask) | Sat+Heat(GT) |
   Sat+Arrow(Pt) | Sat+Arrow(BBox) | Sat+Arrow(Mask) | Sat+Arrow(GT)]

Heatmap: jet-coloured continuous distribution overlaid on satellite.
Arrow:   placed at the argmax position on satellite, yaw → direction.

Usage:
    python visualize_prompt_camera.py \
        --config  output_v3/ablation_4_all_on/config.yaml \
        --checkpoint output_v3/ablation_4_all_on/best \
        --split unseen_test \
        --num_samples 10 \
        --output_dir vis_results/camera_prompt_compare
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers (duplicated from visualize_prompt_segmentation for standalone use)
# ---------------------------------------------------------------------------

def get_workspace_dir() -> Path:
    root_dir = os.environ.get("ROOT_DIR", "")
    workspace_name = os.environ.get("WORKSPACE_NAME", "")
    if root_dir and workspace_name:
        return Path(root_dir) / workspace_name
    return Path(__file__).resolve().parent


def load_cfg_with_env(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f) if config_path.endswith(".json") else __import__("yaml").safe_load(f)

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj
    return _expand(cfg)


def decode_segmentation(segmentation, h: int, w: int) -> np.ndarray:
    if segmentation is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in segmentation:
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        return mask
    if isinstance(segmentation, dict) and "counts" in segmentation:
        from pycocotools import mask as mask_utils
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, h, w)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def _is_drone_item(item: Dict[str, Any]) -> bool:
    task_type = str(item.get("task_type", "")).lower()
    if task_type in {"drone", "ground"}:
        return task_type == "drone"
    return "drone" in str(item.get("mono_filename", "")).lower()


def filter_subset(data_list: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    key = str(subset).strip().lower().replace("-", "_")
    if key in {"all", "both"}:
        return data_list
    if key in {"drone_to_satellite", "d2s", "drone"}:
        return [x for x in data_list if _is_drone_item(x)]
    if key in {"ground_to_satellite", "g2s", "ground"}:
        return [x for x in data_list if not _is_drone_item(x)]
    raise ValueError(
        f"Unsupported subset={subset!r}. "
        f"Use one of: all, drone_to_satellite(d2s), ground_to_satellite(g2s), both."
    )


def euler_to_rotation_matrix_np(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def build_gt_arrow_yaw(item: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Build GT yaw angle for arrow visualization in image coordinates.

    relative_yaw semantic (dataset):
    - angle between camera direction and image -y axis
    - clockwise is positive

    Empirically, subsets use different conventions:
    - ground_to_satellite: use theta = radians(relative_yaw - 90)
    - drone_to_satellite: keep legacy mapping theta = radians(relative_yaw)
      (matches prior visualization behaviour for drone samples)
    """
    if "relative_yaw" in item and item.get("relative_yaw") is not None:
        angle_deg = float(item.get("relative_yaw", 0.0))
        if _is_drone_item(item):
            angle_rad = np.radians(angle_deg)
        else:
            angle_rad = np.radians(angle_deg - 90.0)
        return float(angle_rad), True

    # Fallback for legacy fields if relative_yaw is unavailable.
    if "rotation" in item and item.get("rotation") is not None:
        angle_deg = float(item["rotation"])
        if _is_drone_item(item):
            angle_rad = np.radians(angle_deg)
        else:
            angle_rad = np.radians(angle_deg - 90.0)
        return float(angle_rad), True

    return 0.0, False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    from models import build_cross_view_localizer_v2
    mc = cfg["model"]
    dc = cfg["data"]
    model = build_cross_view_localizer_v2(
        pretrained_pi3=None, freeze_backbone=False, freeze_prompt_encoder=False,
        load_camera_head_weights=False, sam_weights=None,
        img_size=dc.get("img_size", 518),
        decoder_size=mc.get("decoder_size", "large"),
        num_learnable_tokens=mc.get("num_learnable_tokens", 2),
        num_bbox_mask_queries=mc.get("num_bbox_mask_queries"),
        num_heatmap_queries=mc.get("num_heatmap_queries", 1),
        supervision_layers=mc.get("supervision_layers", [4, 11, 17]),
        supervision_weights=mc.get("supervision_weights", [0.1, 0.3, 0.6]),
        dropout=mc.get("dropout", 0.1),
        contrastive=mc.get("contrastive", True),
        contrastive_proj_dim=mc.get("contrastive_proj_dim", 256),
        contrastive_queue_size=mc.get("contrastive_queue_size", 16384),
        contrastive_momentum=mc.get("contrastive_momentum", 0.999),
        contrastive_temperature=mc.get("contrastive_temperature", 0.07),
        sam_embed_dim=mc.get("sam_embed_dim", 256),
        num_mask_tokens=mc.get("num_mask_tokens", 1),
        mask_inject_mode=mc.get("mask_inject_mode", "global_kv"),
        use_global_attn_mask=mc.get("use_global_attn_mask", True),
        use_frame_pos_embed=mc.get("use_frame_pos_embed", False),
    )
    model.to(device); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    for c in [path / "pytorch_model" / "mp_rank_00_model_states.pt",
              path / "mp_rank_00_model_states.pt",
              path / "pytorch_model.bin", path / "model.safetensors"]:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve checkpoint from: {path}")


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("module", "model_state_dict", "state_dict", "model"):
            if k in obj:
                return extract_state_dict(obj[k]) if k == "module" else obj[k]
    return obj


def load_model(cfg, checkpoint_path, device):
    model = build_model(cfg, device)
    ckpt = resolve_checkpoint(Path(checkpoint_path).resolve())
    obj = torch.load(str(ckpt), map_location="cpu")
    sd = extract_state_dict(obj)
    remapped = {}
    for k, v in sd.items():
        remapped[k.replace(".output_hypernetworks_mlps.0.", ".output_hypernetwork_mlp.")] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"Loaded {len(remapped)} keys. Missing={len(missing)}, Unexpected={len(unexpected)}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_model(model, front_t, sat_t, prompt_type, mono_point, mono_bbox,
              mono_mask_np, img_size, device):
    """Return dict with heatmap [H,W], position [2], rotation_matrix [3,3], yaw."""
    front = front_t.unsqueeze(0).to(device)
    sat = sat_t.unsqueeze(0).to(device)
    points, boxes, masks = None, None, None
    if prompt_type == "point":
        pc = mono_point.unsqueeze(0).unsqueeze(1).to(device)
        pl = torch.ones(1, 1, device=device)
        points = (pc, pl)
    elif prompt_type == "bbox":
        b = mono_bbox.clone().unsqueeze(0).unsqueeze(1).to(device)
        b[:, :, 0] = b[:, :, 0].clamp(0, img_size - 1)
        b[:, :, 1] = b[:, :, 1].clamp(0, img_size - 1)
        b[:, :, 2] = b[:, :, 2].clamp(min=1.0, max=img_size)
        b[:, :, 3] = b[:, :, 3].clamp(min=1.0, max=img_size)
        boxes = b
    elif prompt_type == "mask":
        m = torch.from_numpy((mono_mask_np > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        masks = m
    out = model(front_view=front, satellite_view=sat, points=points, boxes=boxes, masks=masks)
    heatmap = out["heatmap"][0].cpu().numpy()        # [S, S] probability
    heatmap_logits = out["heatmap_logits"][0].cpu().numpy()  # [37, 37]
    position = out["position"][0].cpu().numpy()      # [2] normalised
    R = out["rotation_matrix"][0].cpu().numpy()      # [3, 3]
    yaw = float(out["yaw"][0].cpu())
    return {"heatmap": heatmap, "heatmap_logits": heatmap_logits,
            "position": position, "R": R, "yaw": yaw}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_heatmap_overlay(sat_rgb: np.ndarray, heatmap: np.ndarray,
                         alpha: float = 0.55, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay a continuous heatmap on satellite image.
    heatmap: [H, W] in [0, 1].
    Result: red where prob is high, blue where low.
    """
    S = sat_rgb.shape[0]
    # Smooth upsample to target size
    if heatmap.shape[0] != S or heatmap.shape[1] != S:
        heatmap = cv2.resize(heatmap, (S, S), interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 255]
    hm_min, hm_max = heatmap.min(), heatmap.max()
    if hm_max - hm_min > 1e-8:
        hm_norm = ((heatmap - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
    else:
        hm_norm = np.zeros_like(heatmap, dtype=np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, colormap)  # BGR
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    blended = (sat_rgb.astype(float) * (1 - alpha) + hm_color.astype(float) * alpha).astype(np.uint8)
    return blended


def make_gt_heatmap(sat_rgb: np.ndarray, gt_pos_norm: np.ndarray,
                    sigma: float = 0.05, alpha: float = 0.55) -> np.ndarray:
    """Generate Gaussian GT heatmap from normalised position and overlay."""
    S = sat_rgb.shape[0]
    y_coords = np.linspace(0, 1, S, dtype=np.float32)
    x_coords = np.linspace(0, 1, S, dtype=np.float32)
    gx, gy = np.meshgrid(x_coords, y_coords)
    tx, ty = float(gt_pos_norm[0]), float(gt_pos_norm[1])
    dist_sq = (gx - tx) ** 2 + (gy - ty) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma * sigma)).astype(np.float32)
    return make_heatmap_overlay(sat_rgb, heatmap, alpha)


def draw_arrow_on_image(img_rgb: np.ndarray, cx: float, cy: float,
                        yaw_rad: float,
                        arrow_length: int = 50,
                        arrow_color: Tuple[int, int, int] = (255, 0, 0),
                        arrow_thickness: int = 3,
                        tip_length: float = 0.35) -> np.ndarray:
    """Draw a direction arrow at (cx, cy) pointing in `yaw_rad` direction."""
    out = img_rgb.copy()
    dx = arrow_length * math.cos(yaw_rad)
    dy = arrow_length * math.sin(yaw_rad)
    pt1 = (int(round(cx)), int(round(cy)))
    pt2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.arrowedLine(out, pt1, pt2, arrow_color, arrow_thickness, tipLength=tip_length)
    # Draw a small filled circle at the center
    cv2.circle(out, pt1, 5, arrow_color, -1)
    return out


def draw_point_on_image(img_rgb: np.ndarray, pt: np.ndarray,
                        color: Tuple[int, int, int], radius: int = 8,
                        thickness: int = 2) -> np.ndarray:
    out = img_rgb.copy()
    cx, cy = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(out, (cx, cy), radius, color, thickness)
    cv2.circle(out, (cx, cy), radius // 3, color, -1)
    return out


def draw_bbox_on_image(img_rgb: np.ndarray, bbox_xywh: np.ndarray,
                       color: Tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    out = img_rgb.copy()
    x, y, w, h = bbox_xywh.astype(int)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_mask_prompt_on_image(img_rgb: np.ndarray, mask_np: np.ndarray,
                              color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = img_rgb.copy()
    overlay = np.full_like(out, color, dtype=np.uint8)
    mask3 = np.stack([(mask_np > 0)] * 3, axis=-1)
    out = np.where(mask3, (out * (1 - alpha) + overlay * alpha).astype(np.uint8), out)
    contours, _ = cv2.findContours((mask_np > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def concat_images_horizontal(images: List[np.ndarray], gap: int = 4,
                             gap_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    strips = []
    for i, im in enumerate(images):
        if im.shape[0] < h:
            pad = np.full((h - im.shape[0], im.shape[1], 3), gap_color, dtype=np.uint8)
            im = np.vstack([im, pad])
        if i > 0:
            strips.append(np.full((h, gap, 3), gap_color, dtype=np.uint8))
        strips.append(im)
    return np.hstack(strips)


def concat_images_vertical(images: List[np.ndarray], gap: int = 4,
                           gap_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    w = max(im.shape[1] for im in images)
    strips = []
    for i, im in enumerate(images):
        if im.shape[1] < w:
            pad = np.full((im.shape[0], w - im.shape[1], 3), gap_color, dtype=np.uint8)
            im = np.hstack([im, pad])
        if i > 0:
            strips.append(np.full((gap, w, 3), gap_color, dtype=np.uint8))
        strips.append(im)
    return np.vstack(strips)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ws = get_workspace_dir()
    p = argparse.ArgumentParser("Visualize prompt-dependent camera predictions")
    p.add_argument("--config", type=str, default=str(ws / "output_v3" / "config.yaml"))
    p.add_argument("--checkpoint", type=str, default=str(ws / "output_v3" / "best"))
    p.add_argument("--split", type=str, default="unseen_test",
                   choices=["val", "test", "unseen_test"])
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--output_dir", type=str,
                   default=str(ws / "vis_results" / "camera_prompt_compare"))
    p.add_argument("--view_subsets", nargs="+", default=["drone_to_satellite", "ground_to_satellite"],
                   help="Save visualizations by subset. e.g. drone_to_satellite ground_to_satellite")
    p.add_argument("--gpu", type=str, default="0")
    # ---- Visual config ----
    p.add_argument("--img_gap", type=int, default=4)
    p.add_argument("--gap_color", type=str, default="255,255,255")
    p.add_argument("--heatmap_alpha", type=float, default=0.55,
                   help="Heatmap overlay opacity")
    p.add_argument("--gt_sigma", type=float, default=0.05,
                   help="Gaussian σ for GT heatmap")
    p.add_argument("--color_point", type=str, default="0,255,0")
    p.add_argument("--color_bbox", type=str, default="0,128,255")
    p.add_argument("--color_mask", type=str, default="255,0,128")
    p.add_argument("--color_gt", type=str, default="255,255,0")
    p.add_argument("--point_radius", type=int, default=8)
    p.add_argument("--bbox_thickness", type=int, default=3)
    # ---- Arrow config ----
    p.add_argument("--arrow_length", type=int, default=55,
                   help="Arrow length in pixels")
    p.add_argument("--arrow_thickness", type=int, default=3)
    p.add_argument("--arrow_tip_length", type=float, default=0.35)
    p.add_argument("--arrow_color_point", type=str, default="0,255,0")
    p.add_argument("--arrow_color_bbox", type=str, default="0,128,255")
    p.add_argument("--arrow_color_mask", type=str, default="255,0,128")
    p.add_argument("--arrow_color_gt", type=str, default="255,255,0")
    # ---- Sample selection ----
    p.add_argument("--ranking_metric", type=str, default="sequential",
                   choices=["random", "sequential"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_rgb(s: str) -> Tuple[int, int, int]:
    return tuple(int(x.strip()) for x in s.split(","))  # type: ignore


def main():
    args = parse_args()
    ws = get_workspace_dir()
    os.environ.setdefault("ROOT_DIR", str(ws.parent))
    os.environ.setdefault("WORKSPACE_NAME", ws.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg_with_env(args.config)
    img_size = int(cfg["data"].get("img_size", 518))
    data_root = cfg["data"]["data_root"]

    # Colours
    C_pt = parse_rgb(args.color_point)
    C_bb = parse_rgb(args.color_bbox)
    C_mk = parse_rgb(args.color_mask)
    C_gt = parse_rgb(args.color_gt)
    gap_c = parse_rgb(args.gap_color)
    AC_pt = parse_rgb(args.arrow_color_point)
    AC_bb = parse_rgb(args.arrow_color_bbox)
    AC_mk = parse_rgb(args.arrow_color_mask)
    AC_gt = parse_rgb(args.arrow_color_gt)

    # Load model
    print("[1/4] Loading model ...")
    model = load_model(cfg, args.checkpoint, device)

    # Load data
    print("[2/4] Loading data ...")
    split_to_json = {"val": "val_all.json", "test": "test_all.json", "unseen_test": "unseen_test.json"}
    json_path = ws / "data" / split_to_json[args.split]
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    # Only keep samples with camera pose GT
    data_list = [it for it in data_list
                 if (it.get("camera_position") is not None and len(it.get("camera_position", [])) >= 2)
                 and ("relative_yaw" in it or "rotation" in it)]
    print(f"  {args.split}: {len(data_list)} samples with camera GT")

    if args.ranking_metric == "random":
        rng = np.random.default_rng(args.seed)
        rng.shuffle(data_list)

    # Inference + visualise by subset
    print("[3/4] Running inference by subset ...")
    os.makedirs(args.output_dir, exist_ok=True)
    S = img_size
    subsets = [str(x) for x in args.view_subsets]

    for subset in subsets:
        subset_data = filter_subset(data_list, subset)
        N = min(args.num_samples, len(subset_data))
        subset_dir = os.path.join(args.output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        print(f"  subset={subset}: {len(subset_data)} samples, saving {N} to {subset_dir}")

        for rank, item in enumerate(tqdm(subset_data[:N], desc=f"Visualising[{subset}]")):
            city = item.get("city", "")
            mono_path = os.path.join(data_root, city, "mono", item["mono_filename"])
            sat_path = os.path.join(data_root, city, "crop_sate",
                                    item.get("sat_filename") or item.get("sate_filename"))

            mono_rgb = cv2.cvtColor(cv2.imread(mono_path), cv2.COLOR_BGR2RGB)
            sat_rgb = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
            h_m, w_m = mono_rgb.shape[:2]
            h_s, w_s = sat_rgb.shape[:2]

            mono_point = np.array(item["mono_point"][:2], dtype=np.float32)
            mono_bbox = np.array(item.get("mono_bbox", [0, 0, 0, 0])[:4], dtype=np.float32)
            mono_mask = decode_segmentation(item.get("mono_segmentation"), h_m, w_m)

            # GT yaw for arrow visualization
            gt_yaw, has_rot = build_gt_arrow_yaw(item)
            gt_pos = np.array(item["camera_position"][:2], dtype=np.float32)
            gt_pos_is_pixel = True

            if (h_m, w_m) != (S, S):
                sx, sy = S / w_m, S / h_m
                mono_rgb = cv2.resize(mono_rgb, (S, S))
                mono_point = np.array([mono_point[0] * sx, mono_point[1] * sy], dtype=np.float32)
                mono_bbox = np.array(
                    [mono_bbox[0] * sx, mono_bbox[1] * sy, mono_bbox[2] * sx, mono_bbox[3] * sy],
                    dtype=np.float32,
                )
                mono_mask = cv2.resize(mono_mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
            if (h_s, w_s) != (S, S):
                sx_s, sy_s = S / w_s, S / h_s
                sat_rgb = cv2.resize(sat_rgb, (S, S))
                if gt_pos_is_pixel:
                    gt_pos = np.array([gt_pos[0] * sx_s, gt_pos[1] * sy_s], dtype=np.float32)

            # Normalise GT position to [0,1]
            gt_pos_norm = np.clip(gt_pos / S, 0.0, 1.0).astype(np.float32)
            gt_yaw = gt_yaw if has_rot else 0.0

            front_t = to_tensor(Image.fromarray(mono_rgb))
            sat_t = to_tensor(Image.fromarray(sat_rgb))
            pt_point = torch.from_numpy(mono_point)
            pt_bbox = torch.from_numpy(mono_bbox)

            # Run model with 3 prompts
            results = {}
            for pt in ("point", "bbox", "mask"):
                results[pt] = run_model(model, front_t, sat_t, pt, pt_point, pt_bbox, mono_mask, img_size, device)

            # ==== Position (heatmap) ====
            front_pt = draw_point_on_image(mono_rgb, mono_point, C_pt, args.point_radius)
            front_bb = draw_bbox_on_image(mono_rgb, mono_bbox, C_bb, args.bbox_thickness)
            front_mk = draw_mask_prompt_on_image(mono_rgb, mono_mask, C_mk, 0.35)

            heat_pt = make_heatmap_overlay(sat_rgb, results["point"]["heatmap"], args.heatmap_alpha)
            heat_bb = make_heatmap_overlay(sat_rgb, results["bbox"]["heatmap"], args.heatmap_alpha)
            heat_mk = make_heatmap_overlay(sat_rgb, results["mask"]["heatmap"], args.heatmap_alpha)
            heat_gt = make_gt_heatmap(sat_rgb, gt_pos_norm, args.gt_sigma, args.heatmap_alpha)

            # ==== Rotation (arrow) ====

            def pos_to_px(pos_norm):
                return float(pos_norm[0]) * (S - 1), float(pos_norm[1]) * (S - 1)

            cx_pt, cy_pt = pos_to_px(results["point"]["position"])
            cx_bb, cy_bb = pos_to_px(results["bbox"]["position"])
            cx_mk, cy_mk = pos_to_px(results["mask"]["position"])
            cx_gt, cy_gt = float(gt_pos_norm[0]) * (S - 1), float(gt_pos_norm[1]) * (S - 1)

            arrow_pt = draw_arrow_on_image(
                sat_rgb, cx_pt, cy_pt, results["point"]["yaw"],
                args.arrow_length, AC_pt, args.arrow_thickness, args.arrow_tip_length,
            )
            arrow_bb = draw_arrow_on_image(
                sat_rgb, cx_bb, cy_bb, results["bbox"]["yaw"],
                args.arrow_length, AC_bb, args.arrow_thickness, args.arrow_tip_length,
            )
            arrow_mk = draw_arrow_on_image(
                sat_rgb, cx_mk, cy_mk, results["mask"]["yaw"],
                args.arrow_length, AC_mk, args.arrow_thickness, args.arrow_tip_length,
            )
            arrow_gt = draw_arrow_on_image(
                sat_rgb, cx_gt, cy_gt, gt_yaw,
                args.arrow_length, AC_gt, args.arrow_thickness, args.arrow_tip_length,
            )

            combined = concat_images_horizontal(
                [front_pt, front_bb, front_mk, heat_pt, heat_bb, heat_mk, heat_gt,
                 arrow_pt, arrow_bb, arrow_mk, arrow_gt],
                gap=args.img_gap, gap_color=gap_c,
            )

            out_name = f"rank{rank:03d}_sample{rank:05d}.png"
            cv2.imwrite(os.path.join(subset_dir, out_name), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"\n[4/4] Saved visualizations to: {args.output_dir} (grouped by subset)")


if __name__ == "__main__":
    main()
