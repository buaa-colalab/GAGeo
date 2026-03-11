#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative visualization: segmentation results across three prompt types.

For each selected sample, generates ONE row of 7 panels:
  [Front+Point | Front+BBox | Front+Mask | Sat+SegPt | Sat+SegBBox | Sat+SegMask | Sat+GT]

Colours of the prompt annotation match the corresponding mask overlay on the
satellite image.  A thin configurable gap separates the panels.

Usage:
    python visualize_prompt_segmentation.py \
        --config  output_v3/ablation_4_all_on/config.yaml \
        --checkpoint output_v3/ablation_4_all_on/best \
        --split unseen_test \
        --num_samples 10 \
        --output_dir vis_results/seg_prompt_compare
"""
from __future__ import annotations

import argparse
import json
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
# helpers borrowed from evaluate_custom_v2
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


# ---------------------------------------------------------------------------
# Model loading (mirrors evaluate_custom_v2.build_model_from_cfg)
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    from models import build_cross_view_localizer_v2
    mc = cfg["model"]
    dc = cfg["data"]
    model = build_cross_view_localizer_v2(
        pretrained_pi3=None,
        freeze_backbone=False,
        freeze_prompt_encoder=False,
        load_camera_head_weights=False,
        sam_weights=None,
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
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
        path / "pytorch_model.bin",
        path / "model.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot resolve checkpoint from: {path}")


def extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "module" in obj:
            return extract_state_dict(obj["module"])
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model" in obj:
            return obj["model"]
    return obj


def load_model(cfg, checkpoint_path: str, device):
    model = build_model(cfg, device)
    ckpt_file = resolve_checkpoint(Path(checkpoint_path).resolve())
    obj = torch.load(str(ckpt_file), map_location="cpu")
    sd = extract_state_dict(obj)
    # Legacy mask head remap
    remapped: Dict[str, torch.Tensor] = {}
    needle = ".output_hypernetworks_mlps.0."
    repl = ".output_hypernetwork_mlp."
    for k, v in sd.items():
        remapped[k.replace(needle, repl)] = v if needle in k else v
        if needle not in k:
            remapped[k] = v
    sd = remapped
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded state_dict ({len(sd)} keys). Missing={len(missing)}, Unexpected={len(unexpected)}")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_model(model, front_t, sat_t, prompt_type, mono_point, mono_bbox, mono_mask_np, img_size, device):
    """Run model with a single prompt type, return predicted mask [H,W] float."""
    front = front_t.unsqueeze(0).to(device)
    sat = sat_t.unsqueeze(0).to(device)
    points, boxes, masks = None, None, None
    if prompt_type == "point":
        pc = mono_point.unsqueeze(0).unsqueeze(1).to(device)  # [1,1,2]
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
    outputs = model(front_view=front, satellite_view=sat, points=points, boxes=boxes, masks=masks)
    pred_masks = outputs["mask_pred"]  # [1, Q, H, W]
    if pred_masks.shape[1] > 1 and "bbox_scores" in outputs:
        best = outputs["bbox_scores"].argmax(dim=1)
        pred_mask = pred_masks[0, best[0]].cpu().numpy()
    else:
        pred_mask = pred_masks[0, 0].cpu().numpy()
    return pred_mask


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def blend_mask_on_image(img_rgb: np.ndarray, mask_binary: np.ndarray,
                        color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """Overlay a coloured semi-transparent mask on an RGB image."""
    out = img_rgb.copy()
    overlay = np.full_like(out, color, dtype=np.uint8)
    mask3 = np.stack([mask_binary] * 3, axis=-1).astype(bool)
    out = np.where(mask3, (out * (1 - alpha) + overlay * alpha).astype(np.uint8), out)
    # Draw contour for crispness
    contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_point_on_image(img_rgb: np.ndarray, pt: np.ndarray,
                        color: Tuple[int, int, int], radius: int = 8, thickness: int = 2) -> np.ndarray:
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
    return blend_mask_on_image(img_rgb, (mask_np > 0).astype(np.uint8), color, alpha)


def concat_images_horizontal(images: List[np.ndarray], gap: int = 4,
                             gap_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Concatenate images horizontally with a uniform gap."""
    h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] < h:
            pad = np.full((h - im.shape[0], im.shape[1], 3), gap_color, dtype=np.uint8)
            im = np.vstack([im, pad])
        padded.append(im)
    strips = []
    for i, im in enumerate(padded):
        if i > 0:
            strips.append(np.full((h, gap, 3), gap_color, dtype=np.uint8))
        strips.append(im)
    return np.hstack(strips)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ws = get_workspace_dir()
    p = argparse.ArgumentParser("Visualize prompt-dependent segmentation")
    p.add_argument("--config", type=str, default=str(ws / "output_v3" / "config.yaml"))
    p.add_argument("--checkpoint", type=str, default=str(ws / "output_v3" / "best"))
    p.add_argument("--split", type=str, default="unseen_test",
                   choices=["val", "test", "unseen_test"])
    p.add_argument("--num_samples", type=int, default=10,
                   help="Number of samples to visualize")
    p.add_argument("--output_dir", type=str, default=str(ws / "vis_results" / "seg_prompt_compare"))
    p.add_argument("--view_subsets", nargs="+", default=["drone_to_satellite", "ground_to_satellite"],
                   help="Save visualizations by subset. e.g. drone_to_satellite ground_to_satellite")
    p.add_argument("--gpu", type=str, default="0")
    # ---------- Visual config ----------
    p.add_argument("--img_gap", type=int, default=4,
                   help="Pixel gap between panels")
    p.add_argument("--gap_color", type=str, default="255,255,255",
                   help="RGB colour for gap (default white)")
    p.add_argument("--mask_alpha", type=float, default=0.45,
                   help="Mask overlay opacity")
    p.add_argument("--color_point", type=str, default="0,255,0",
                   help="RGB for point prompt / mask overlay (green)")
    p.add_argument("--color_bbox", type=str, default="0,128,255",
                   help="RGB for bbox prompt / mask overlay (blue)")
    p.add_argument("--color_mask", type=str, default="255,0,128",
                   help="RGB for mask prompt / mask overlay (magenta-red)")
    p.add_argument("--color_gt", type=str, default="255,255,0",
                   help="RGB for GT mask overlay (yellow)")
    p.add_argument("--point_radius", type=int, default=8)
    p.add_argument("--bbox_thickness", type=int, default=3)
    # ---------- Sample selection ----------
    p.add_argument("--ranking_metric", type=str, default="mask_gap",
                   choices=["mask_gap", "random", "sequential"],
                   help="How to choose samples (mask_gap: pick where mask>bbox>point gap is large)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_rgb(s: str) -> Tuple[int, int, int]:
    return tuple(int(x.strip()) for x in s.split(","))  # type: ignore


def compute_iou(p: np.ndarray, g: np.ndarray) -> float:
    p, g = p.astype(bool), g.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


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

    # Load model
    print("[1/4] Loading model ...")
    model = load_model(cfg, args.checkpoint, device)

    # Load data
    print("[2/4] Loading data ...")
    split_to_json = {"val": "val_all.json", "test": "test_all.json", "unseen_test": "unseen_test.json"}
    json_path = ws / "data" / split_to_json[args.split]
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print(f"  {args.split}: {len(data_list)} samples")

    # Colours
    C_pt = parse_rgb(args.color_point)
    C_bb = parse_rgb(args.color_bbox)
    C_mk = parse_rgb(args.color_mask)
    C_gt = parse_rgb(args.color_gt)
    gap_c = parse_rgb(args.gap_color)

    # Prepare + save per subset
    print("[3/4] Running inference and saving by subset (3 prompts each) ...")
    subsets = [str(x) for x in args.view_subsets]
    S = img_size
    os.makedirs(args.output_dir, exist_ok=True)

    for subset in subsets:
        subset_data = filter_subset(data_list, subset)
        print(f"  subset={subset}: {len(subset_data)} samples")

        records: List[Dict[str, Any]] = []
        for idx, item in enumerate(tqdm(subset_data, desc=f"Inference[{subset}]")):
            city = item.get("city", "")
            mono_name = item["mono_filename"]
            sat_name = item.get("sat_filename") or item.get("sate_filename")
            mono_path = os.path.join(data_root, city, "mono", mono_name)
            sat_path = os.path.join(data_root, city, "crop_sate", sat_name)

            mono_rgb = cv2.cvtColor(cv2.imread(mono_path), cv2.COLOR_BGR2RGB)
            sat_rgb = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
            h_m, w_m = mono_rgb.shape[:2]
            h_s, w_s = sat_rgb.shape[:2]

            mono_point = np.array(item["mono_point"][:2], dtype=np.float32)
            mono_bbox = np.array(item.get("mono_bbox", [0, 0, 0, 0])[:4], dtype=np.float32)
            mono_mask = decode_segmentation(item.get("mono_segmentation"), h_m, w_m)
            gt_mask = decode_segmentation(item.get("sate_segmentation"), h_s, w_s)

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
                sat_rgb = cv2.resize(sat_rgb, (S, S))
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)

            front_t = to_tensor(Image.fromarray(mono_rgb))
            sat_t = to_tensor(Image.fromarray(sat_rgb))
            pt_point = torch.from_numpy(mono_point)
            pt_bbox = torch.from_numpy(mono_bbox)

            pred_masks: Dict[str, np.ndarray] = {}
            ious: Dict[str, float] = {}
            for pt in ("point", "bbox", "mask"):
                pm = run_model(model, front_t, sat_t, pt, pt_point, pt_bbox, mono_mask, img_size, device)
                if pm.shape[0] != S or pm.shape[1] != S:
                    pm = cv2.resize(pm, (S, S), interpolation=cv2.INTER_LINEAR)
                pred_masks[pt] = pm
                ious[pt] = compute_iou((pm > 0.5).astype(np.uint8), (gt_mask > 0).astype(np.uint8))

            records.append({
                "idx": idx,
                "mono_rgb": mono_rgb,
                "sat_rgb": sat_rgb,
                "mono_point": mono_point,
                "mono_bbox": mono_bbox,
                "mono_mask": mono_mask,
                "gt_mask": gt_mask,
                "pred_masks": pred_masks,
                "ious": ious,
            })

        if args.ranking_metric == "mask_gap":
            for r in records:
                gap = (r["ious"]["mask"] - r["ious"]["point"])
                gap2 = min(r["ious"]["mask"] - r["ious"]["bbox"], r["ious"]["bbox"] - r["ious"]["point"])
                r["score"] = gap + 0.3 * max(gap2, 0)
            records.sort(key=lambda x: x["score"], reverse=True)
        elif args.ranking_metric == "random":
            rng = np.random.default_rng(args.seed)
            rng.shuffle(records)

        subset_dir = os.path.join(args.output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        N = min(args.num_samples, len(records))
        print(f"  Saving subset={subset}, num_samples={N}, out={subset_dir}")

        for rank, rec in enumerate(records[:N]):
            idx = rec["idx"]
            mono = rec["mono_rgb"]
            sat = rec["sat_rgb"]

            front_pt = draw_point_on_image(mono, rec["mono_point"], C_pt, args.point_radius)
            front_bb = draw_bbox_on_image(mono, rec["mono_bbox"], C_bb, args.bbox_thickness)
            front_mk = draw_mask_prompt_on_image(mono, rec["mono_mask"], C_mk, 0.35)

            pm_pt_bin = (rec["pred_masks"]["point"] > 0.5).astype(np.uint8)
            pm_bb_bin = (rec["pred_masks"]["bbox"] > 0.5).astype(np.uint8)
            pm_mk_bin = (rec["pred_masks"]["mask"] > 0.5).astype(np.uint8)

            sat_seg_pt = blend_mask_on_image(sat, pm_pt_bin, C_pt, args.mask_alpha)
            sat_seg_bb = blend_mask_on_image(sat, pm_bb_bin, C_bb, args.mask_alpha)
            sat_seg_mk = blend_mask_on_image(sat, pm_mk_bin, C_mk, args.mask_alpha)
            sat_gt = blend_mask_on_image(sat, (rec["gt_mask"] > 0).astype(np.uint8), C_gt, args.mask_alpha)

            row = concat_images_horizontal(
                [front_pt, front_bb, front_mk, sat_seg_pt, sat_seg_bb, sat_seg_mk, sat_gt],
                gap=args.img_gap, gap_color=gap_c,
            )
            iou_info = f"pt={rec['ious']['point']:.3f}_bb={rec['ious']['bbox']:.3f}_mk={rec['ious']['mask']:.3f}"
            out_name = f"rank{rank:03d}_sample{idx:05d}_{iou_info}.png"
            cv2.imwrite(os.path.join(subset_dir, out_name), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    print(f"\nAll visualizations saved to: {args.output_dir} (grouped by subset)")


if __name__ == "__main__":
    main()
