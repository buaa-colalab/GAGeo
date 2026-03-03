#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize worst-K zero-shot ground->drone triplets by detection IoU.

For each selected sample:
- Left: ground image with point prompt
- Right: drone image with GT bbox (green) and Pred bbox (red)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_zero_shot_ground_to_drone import (
    GroundDroneTripletDataset,
    bbox_cxcywh_norm_to_xyxy_abs,
    bbox_iou_np,
    build_model_from_cfg,
    clip_bbox_xyxy,
    collate_fn,
    extract_state_dict,
    get_workspace_dir,
    load_cfg_with_env,
    remap_legacy_mask_head_keys,
    resolve_checkpoint,
)


def parse_args():
    ws_dir = get_workspace_dir()
    root_dir = os.environ.get("ROOT_DIR", str(ws_dir.parent))
    output_dir = ws_dir / "output_v3"
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    p = argparse.ArgumentParser(description="Visualize worst-K zero-shot ground->drone samples")
    p.add_argument("--triplet_json", type=str, default=str(Path(root_dir) / "University-Release" / "verified_triplets_sam2_masks.json"))
    p.add_argument("--root_dir", type=str, default=str(Path(root_dir) / "University-Release"))
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--worst_k", type=int, default=50)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(output_dir / f"vis_zero_shot_ground_to_drone_worst50_{now}"),
    )
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


@torch.no_grad()
def infer_and_collect(model, loader: DataLoader, img_size: int, device: torch.device) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc="Infer and collect IoU"):
        front = batch["front_view"].to(device, non_blocking=True)
        sat = batch["sat_view"].to(device, non_blocking=True)
        mono_point = batch["mono_point"].to(device, non_blocking=True)
        gt_bbox_xyxy = batch["gt_bbox_xyxy"].numpy().astype(np.float32)

        bsz = front.shape[0]
        point_coords = mono_point.unsqueeze(1)  # [B,1,2]
        point_labels = torch.ones(bsz, 1, device=device)

        outputs = model(
            front_view=front,
            satellite_view=sat,
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
            mono_mask=None,
            sat_mask=None,
        )
        pred_boxes_all = outputs["pred_boxes"]  # [B, Q, 4]

        if pred_boxes_all.shape[1] > 1 and "bbox_scores" in outputs:
            best_idx = outputs["bbox_scores"].argmax(dim=1)  # [B]
            pred_bbox_norm = pred_boxes_all[torch.arange(bsz), best_idx]
        else:
            pred_bbox_norm = pred_boxes_all[:, 0]

        for i in range(bsz):
            pb = clip_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_bbox_norm[i], img_size), img_size)
            gb = clip_bbox_xyxy(gt_bbox_xyxy[i], img_size)
            iou = bbox_iou_np(pb, gb)
            point_xy = batch["mono_point"][i].numpy().astype(np.float32)

            records.append(
                {
                    "sample_id": int(batch["sample_id"][i]),
                    "ground_image": batch["ground_image"][i],
                    "drone_image": batch["drone_image"][i],
                    "iou": float(iou),
                    "point_xy": [float(point_xy[0]), float(point_xy[1])],
                    "pred_bbox_xyxy": [float(x) for x in pb.tolist()],
                    "gt_bbox_xyxy": [float(x) for x in gb.tolist()],
                }
            )
    return records


def _draw_bbox(img: np.ndarray, box_xyxy: List[float], color: tuple, text: str) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def visualize_one(root_dir: Path, rec: Dict[str, Any], img_size: int) -> np.ndarray:
    gpath = root_dir / rec["ground_image"]
    dpath = root_dir / rec["drone_image"]
    g = cv2.imread(str(gpath))
    d = cv2.imread(str(dpath))
    if g is None:
        raise FileNotFoundError(f"Ground image not found: {gpath}")
    if d is None:
        raise FileNotFoundError(f"Drone image not found: {dpath}")

    g = cv2.resize(g, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    d = cv2.resize(d, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    px, py = [int(round(v)) for v in rec["point_xy"]]
    cv2.circle(g, (px, py), 6, (0, 255, 255), -1)
    cv2.circle(g, (px, py), 9, (0, 0, 0), 2)
    cv2.putText(g, "Point Prompt", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    _draw_bbox(d, rec["gt_bbox_xyxy"], (0, 255, 0), "GT")
    _draw_bbox(d, rec["pred_bbox_xyxy"], (0, 0, 255), "Pred")

    top_bar = np.zeros((48, img_size * 2, 3), dtype=np.uint8)
    info = f"sample_id={rec['sample_id']}  IoU={rec['iou']:.4f}"
    cv2.putText(top_bar, info, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    pair = np.concatenate([g, d], axis=1)
    canvas = np.concatenate([top_bar, pair], axis=0)
    return canvas


def main():
    args = parse_args()
    ws_dir = get_workspace_dir()
    os.environ.setdefault("ROOT_DIR", str(ws_dir.parent))
    os.environ.setdefault("WORKSPACE_NAME", ws_dir.name)

    preset_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not preset_cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        preset_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    else:
        print(f"Using preset CUDA_VISIBLE_DEVICES={preset_cvd}; ignore --gpu={args.gpu}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA_VISIBLE_DEVICES={preset_cvd}")

    cfg = load_cfg_with_env(args.config)
    cfg.setdefault("data", {})["img_size"] = int(args.img_size)

    print("\n[1/4] Loading model ...")
    model = build_model_from_cfg(cfg, device)
    ckpt_file = resolve_checkpoint(Path(args.checkpoint).resolve())
    print(f"Checkpoint file: {ckpt_file}")
    obj = torch.load(str(ckpt_file), map_location="cpu")
    sd = extract_state_dict(obj)
    sd, renamed = remap_legacy_mask_head_keys(sd)
    if renamed > 0:
        print(f"Applied legacy mask-head key remap: {renamed} tensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded state_dict keys: {len(sd)}")
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    print("\n[2/4] Building dataset ...")
    ds = GroundDroneTripletDataset(
        triplet_json=args.triplet_json,
        root_dir=args.root_dir,
        input_size=int(args.img_size),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("\n[3/4] Running inference and ranking worst samples ...")
    records = infer_and_collect(model, loader, int(args.img_size), device)
    records = sorted(records, key=lambda x: x["iou"])
    k = min(max(1, int(args.worst_k)), len(records))
    worst = records[:k]
    print(f"Total samples: {len(records)} | Visualizing worst: {k}")

    print("\n[4/4] Saving visualizations ...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(args.root_dir)

    for rank, rec in enumerate(worst, start=1):
        canvas = visualize_one(root_dir, rec, int(args.img_size))
        out_name = f"rank_{rank:03d}_iou_{rec['iou']:.4f}_sid_{rec['sample_id']:06d}.png"
        cv2.imwrite(str(out_dir / out_name), canvas)

    summary_path = out_dir / "worst_samples_summary.json"
    payload = {
        "total_count": len(records),
        "worst_k": k,
        "triplet_json": args.triplet_json,
        "root_dir": args.root_dir,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "img_size": int(args.img_size),
        "records": worst,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved json: {args.save_json}")

    print(f"Done. Visualization dir: {out_dir}")


if __name__ == "__main__":
    main()

