#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot ground->drone evaluation for Cross-View Localizer V2.

Task:
- Input: ground image + point prompt (on ground image) + drone image
- Model output: bbox on drone image
- Metric: mean IoU · ACC@25 · ACC@50

Triplet JSON format (per item):
{
  "drone_image": "train/drone/xxxx/image-01.jpeg",
  "ground_image": "train/street/xxxx/1.jpg",
  "drone_image_bbox": [x, y, w, h],
  "ground_image_point": {"x": ..., "y": ...}
}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from models import build_cross_view_localizer_v2


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


def bbox_xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b.astype(np.float32)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_cxcywh_norm_to_xyxy_abs(b: torch.Tensor, img_size: int) -> np.ndarray:
    cx, cy, w, h = b.detach().cpu().numpy().astype(np.float32)
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def clip_bbox_xyxy(b: np.ndarray, size: int) -> np.ndarray:
    b = b.copy()
    b[0::2] = np.clip(b[0::2], 0, size - 1)
    b[1::2] = np.clip(b[1::2], 0, size - 1)
    return b


def bbox_iou_np(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(b1[2] - b1[0])) * max(0.0, float(b1[3] - b1[1]))
    a2 = max(0.0, float(b2[2] - b2[0])) * max(0.0, float(b2[3] - b2[1]))
    return inter / (a1 + a2 - inter + 1e-16)


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

    raise FileNotFoundError(f"Cannot resolve checkpoint file from: {path}")


def extract_state_dict(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ["module", "model", "state_dict", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                if len(sd) > 0:
                    first_k = next(iter(sd.keys()))
                    if first_k.startswith("module."):
                        sd = {kk[len("module."):]: vv for kk, vv in sd.items()}
                return sd
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format")


def build_model_from_cfg(cfg: Dict[str, Any], device: torch.device):
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
    )

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class GroundDroneTripletDataset(Dataset):
    def __init__(self, triplet_json: str, root_dir: str, input_size: int = 518):
        self.root_dir = Path(root_dir)
        self.input_size = input_size

        with open(triplet_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: List[Dict[str, Any]] = []
        for x in raw:
            if not isinstance(x, dict):
                continue
            if not x.get("drone_image") or not x.get("ground_image"):
                continue
            gp = x.get("ground_image_point", None)
            db = x.get("drone_image_bbox", None)
            if not isinstance(gp, dict) or ("x" not in gp or "y" not in gp):
                continue
            if not isinstance(db, (list, tuple)) or len(db) < 4:
                continue
            self.samples.append(x)

        if len(self.samples) == 0:
            raise ValueError("No valid triplets found in JSON")

        print(f"Loaded valid triplets: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        ground_path = self.root_dir / item["ground_image"]
        drone_path = self.root_dir / item["drone_image"]

        ground = self._load_rgb(ground_path)
        drone = self._load_rgb(drone_path)

        Hg, Wg = ground.shape[:2]
        Hd, Wd = drone.shape[:2]

        point = np.array([
            float(item["ground_image_point"]["x"]),
            float(item["ground_image_point"]["y"]),
        ], dtype=np.float32)

        gt_bbox_xywh = np.array(item["drone_image_bbox"][:4], dtype=np.float32)

        # Model expects patch-multiple size (V2 uses 518, divisible by patch size 14).
        # Raw triplet images may be 512x512; we always remap image/point/bbox to S.
        S = self.input_size

        if (Hg, Wg) != (S, S):
            sxg, syg = S / Wg, S / Hg
            ground = cv2.resize(ground, (S, S), interpolation=cv2.INTER_LINEAR)
            point = np.array([point[0] * sxg, point[1] * syg], dtype=np.float32)

        if (Hd, Wd) != (S, S):
            sxd, syd = S / Wd, S / Hd
            drone = cv2.resize(drone, (S, S), interpolation=cv2.INTER_LINEAR)
            gt_bbox_xywh = np.array(
                [
                    gt_bbox_xywh[0] * sxd,
                    gt_bbox_xywh[1] * syd,
                    gt_bbox_xywh[2] * sxd,
                    gt_bbox_xywh[3] * syd,
                ],
                dtype=np.float32,
            )

        gt_bbox_xyxy = bbox_xywh_to_xyxy(gt_bbox_xywh)

        return {
            "front_view": to_tensor(Image.fromarray(ground)),
            "sat_view": to_tensor(Image.fromarray(drone)),
            "mono_point": torch.from_numpy(point),
            "gt_bbox_xyxy": torch.from_numpy(gt_bbox_xyxy),
            "sample_id": idx,
            "ground_image": item["ground_image"],
            "drone_image": item["drone_image"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "front_view": torch.stack([x["front_view"] for x in batch], dim=0),
        "sat_view": torch.stack([x["sat_view"] for x in batch], dim=0),
        "mono_point": torch.stack([x["mono_point"] for x in batch], dim=0),
        "gt_bbox_xyxy": torch.stack([x["gt_bbox_xyxy"] for x in batch], dim=0),
        "sample_id": [x["sample_id"] for x in batch],
        "ground_image": [x["ground_image"] for x in batch],
        "drone_image": [x["drone_image"] for x in batch],
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, img_size: int, device: torch.device):
    ious: List[float] = []
    per_sample: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="Evaluating ground->drone zero-shot"):
        front = batch["front_view"].to(device, non_blocking=True)
        sat = batch["sat_view"].to(device, non_blocking=True)
        mono_point = batch["mono_point"].to(device, non_blocking=True)

        B = front.shape[0]
        point_coords = mono_point.unsqueeze(1)  # [B,1,2]
        point_labels = torch.ones(B, 1, device=device)

        outputs = model(
            front_view=front,
            satellite_view=sat,
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
            mono_mask=None,
            sat_mask=None,
        )

        pred_bbox_norm = outputs["pred_boxes"][:, 0]  # [B,4] normalized cxcywh
        gt_bbox_xyxy = batch["gt_bbox_xyxy"].numpy().astype(np.float32)

        for i in range(B):
            pb = clip_bbox_xyxy(bbox_cxcywh_norm_to_xyxy_abs(pred_bbox_norm[i], img_size), img_size)
            gb = clip_bbox_xyxy(gt_bbox_xyxy[i], img_size)
            iou = bbox_iou_np(pb, gb)
            ious.append(iou)

            per_sample.append(
                {
                    "sample_id": int(batch["sample_id"][i]),
                    "ground_image": batch["ground_image"][i],
                    "drone_image": batch["drone_image"][i],
                    "iou": float(iou),
                    "pred_bbox_xyxy": [float(x) for x in pb.tolist()],
                    "gt_bbox_xyxy": [float(x) for x in gb.tolist()],
                }
            )

    mean_iou = float(np.mean(ious)) if ious else 0.0
    acc25 = float(np.mean([1.0 if x > 0.25 else 0.0 for x in ious])) if ious else 0.0
    acc50 = float(np.mean([1.0 if x > 0.50 else 0.0 for x in ious])) if ious else 0.0

    metrics = {
        "count": len(ious),
        "mean_iou": mean_iou,
        "acc25": acc25,
        "acc50": acc50,
    }
    return metrics, per_sample


def parse_args():
    ws_dir = get_workspace_dir()
    root_dir = os.environ.get("ROOT_DIR", str(ws_dir.parent))
    output_dir = ws_dir / "output_v2"
    p = argparse.ArgumentParser(description="Zero-shot ground->drone evaluation (point prompt only)")
    p.add_argument("--triplet_json", type=str, default=str(Path(root_dir) / "University-Release" / "verified_triplets.json"))
    p.add_argument("--root_dir", type=str, default=str(Path(root_dir) / "University-Release"))
    p.add_argument("--config", type=str, default=str(output_dir / "config.yaml"))
    p.add_argument("--checkpoint", type=str, default=str(output_dir / "best"))
    p.add_argument(
        "--img_size",
        type=int,
        default=518,
        help="Model input size (raw 512x512 will be resized to this size; point and bbox are scaled accordingly)",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


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

    # override input size by CLI for zero-shot test setup
    cfg.setdefault("data", {})["img_size"] = int(args.img_size)

    print("\n[1/3] Loading model ...")
    model = build_model_from_cfg(cfg, device)

    ckpt_file = resolve_checkpoint(Path(args.checkpoint).resolve())
    print(f"Checkpoint file: {ckpt_file}")
    obj = torch.load(str(ckpt_file), map_location="cpu")
    sd = extract_state_dict(obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded state_dict keys: {len(sd)}")
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    print("\n[2/3] Building dataset ...")
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

    print("\n[3/3] Evaluating ...")
    metrics, per_sample = evaluate(model, loader, int(args.img_size), device)

    print("\n" + "=" * 72)
    print("Zero-shot Ground->Drone (point prompt) Metrics")
    print("=" * 72)
    print(f"Count    : {metrics['count']}")
    print(f"mean IoU : {metrics['mean_iou']:.4f}")
    print(f"ACC@25   : {metrics['acc25']:.4f}")
    print(f"ACC@50   : {metrics['acc50']:.4f}")
    print("=" * 72)

    if args.save_json:
        out = {
            "metrics": metrics,
            "setting": {
                "triplet_json": args.triplet_json,
                "root_dir": args.root_dir,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "img_size": int(args.img_size),
                "task": "ground_to_drone_zero_shot",
                "prompt": "point",
                "bbox_format": "xywh(coco) -> converted to xyxy for IoU",
            },
            "per_sample": per_sample,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
