#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate v3 model on CVOGL-Seg with online panorama->mono conversion."""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from cvoglseg_online_data import CVOGLSegOnlineDataset
from evaluate_custom_v2 import (
    build_model_from_cfg,
    count_params,
    evaluate_split,
    extract_state_dict,
    get_workspace_dir,
    load_cfg_with_env,
    resolve_checkpoint,
    save_results_json,
)


class EvalCVOGLSegV3Dataset(Dataset):
    def __init__(
        self,
        cvogl_root: str,
        cvoglseg_root: str,
        img_size: int = 518,
        split_name: str = "test",
        subsets: List[str] | None = None,
    ):
        self.base = CVOGLSegOnlineDataset(
            cvogl_root=cvogl_root,
            cvoglseg_root=cvoglseg_root,
            split_name=split_name,
            img_size=img_size,
        )
        wanted = set(subsets or ["CVOGL_SVI", "CVOGL_DroneAerial"])
        self.indices = [
            i for i in range(len(self.base)) if self.base.records[i].dataset_name in wanted
        ]
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        item = self.base[self.indices[idx]]
        mono = item["query_img"]
        sat = item["sat_img"]
        point = item["point_xy"]
        bbox_xyxy = item["gt_bbox_xyxy"]
        mask = item["gt_mask"]

        mono_t = to_tensor(Image.fromarray(mono))
        sat_t = to_tensor(Image.fromarray(sat))
        gt_pos = np.array([0.0, 0.0], dtype=np.float32)
        gt_rot = np.eye(3, dtype=np.float32)

        return {
            "front_view": mono_t,
            "sat_view": sat_t,
            "mono_point": torch.from_numpy(point.astype(np.float32)),
            "mono_bbox": torch.zeros((4,), dtype=torch.float32),
            "mono_mask": torch.zeros((self.img_size, self.img_size), dtype=torch.uint8),
            "gt_bbox_xyxy": torch.from_numpy(bbox_xyxy.astype(np.float32)),
            "gt_mask": torch.from_numpy((mask > 0).astype(np.uint8)),
            "gt_rotation_matrix": torch.from_numpy(gt_rot),
            "gt_position": torch.from_numpy(gt_pos),
            "has_rotation": False,
            "has_pose": False,
            "sat_rgb": sat,
            "task_type": item["task_type"],
            "size_category": item["size_category"],
            "shape_category": item["shape_category"],
            "index": int(item["index"]),
            "subset": item["dataset_name"],
        }


def collate_eval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "front_view": torch.stack([x["front_view"] for x in batch], dim=0),
        "sat_view": torch.stack([x["sat_view"] for x in batch], dim=0),
        "mono_point": torch.stack([x["mono_point"] for x in batch], dim=0),
        "mono_bbox": torch.stack([x["mono_bbox"] for x in batch], dim=0),
        "mono_mask": torch.stack([x["mono_mask"] for x in batch], dim=0),
        "gt_bbox_xyxy": torch.stack([x["gt_bbox_xyxy"] for x in batch], dim=0),
        "gt_mask": torch.stack([x["gt_mask"] for x in batch], dim=0),
        "gt_rotation_matrix": torch.stack([x["gt_rotation_matrix"] for x in batch], dim=0),
        "gt_position": torch.stack([x["gt_position"] for x in batch], dim=0),
        "has_rotation": [x["has_rotation"] for x in batch],
        "has_pose": [x["has_pose"] for x in batch],
        "sat_rgb": [x["sat_rgb"] for x in batch],
        "task_type": [x["task_type"] for x in batch],
        "size_category": [x["size_category"] for x in batch],
        "shape_category": [x["shape_category"] for x in batch],
        "index": [x["index"] for x in batch],
        "subset": [x["subset"] for x in batch],
    }


def _fmt(v, w=8, d=4):
    if v is None:
        return "-".center(w)
    if isinstance(v, (int, np.integer)):
        return str(int(v)).rjust(w)
    return f"{float(v):.{d}f}".rjust(w)


def _to_baseline_style(results):
    out = OrderedDict()
    for name, r in results.items():
        if r is None:
            out[name] = None
            continue
        one = dict(r)
        # evaluate_custom_v2 fields -> baseline print fields
        if "patch_miou" not in one and "model_miou" in one:
            one["patch_miou"] = one.get("model_miou")
            one["patch_mdice"] = one.get("model_mdice")
            one["patch_aae"] = one.get("model_aae")
            one["patch_me"] = one.get("model_me")
        if "avg_sim" not in one:
            one["avg_sim"] = None
        out[name] = one
    return out


def print_results(results, split):
    print(f'\n{"=" * 155}')
    print(f"  Baseline Det/Seg Evaluation — {split}")
    print(f'{"=" * 155}')
    hdr = (
        f'{"Count":>7} {"Sim":>7} {"mIoU":>7} {"A@.5:.95":>8} {"@0.5":>6} {"@0.75":>6} {"Lat(ms)":>8}'
        f' │ {"P_IoU":>7} {"P_Dice":>7} {"P_AAE":>8} {"P_ME":>7}'
        f' │ {"S_IoU":>7} {"S_Dice":>7} {"S_AAE":>8} {"S_ME":>7}'
    )
    print(f'  {"Group":<30} {hdr}')
    print(f'  {"─" * 30} {"─" * 38} {"─" * 34} {"─" * 34}')

    styled = _to_baseline_style(results)
    for name, r in styled.items():
        if r is None:
            continue
        line = f"  {name:<30} "
        line += f'{_fmt(r["count"], 7, 0)} {_fmt(r.get("avg_sim"), 7, 4)} '
        line += f'{_fmt(r.get("det_miou"), 7)} {_fmt(r["det_avg_acc"], 8)} '
        line += f'{_fmt(r["det_acc50"], 6)} {_fmt(r["det_acc75"], 6)} {_fmt(r.get("latency_ms"), 8, 2)}'
        line += f' │ {_fmt(r.get("patch_miou"), 7)} {_fmt(r.get("patch_mdice"), 7)} '
        line += f'{_fmt(r.get("patch_aae"), 8, 1)} {_fmt(r.get("patch_me"), 7, 2)}'
        line += f' │ {_fmt(r.get("sam_miou"), 7)} {_fmt(r.get("sam_mdice"), 7)} '
        line += f'{_fmt(r.get("sam_aae"), 8, 1)} {_fmt(r.get("sam_me"), 7, 2)}'
        print(line)
    print()


def parse_args():
    ws_dir = get_workspace_dir()
    default_cfg = ws_dir / "output_v3" / "ablation_4_all_on" / "config.yaml"
    default_ckpt = ws_dir / "output_v3" / "ablation_4_all_on" / "best"
    p = argparse.ArgumentParser(description="Evaluate v3 on CVOGL-Seg (online)")
    p.add_argument("--config", type=str, default=str(default_cfg))
    p.add_argument("--checkpoint", type=str, default=str(default_ckpt))
    p.add_argument("--cvogl_root", type=str, default="/data/home/scxi704/run/baseline/CVOS-Code/dataset/CVOGL")
    p.add_argument("--cvoglseg_root", type=str, default="/data/home/scxi704/run/baseline/CVOS-Code/dataset/CVOGL-Seg")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--splits", nargs="+", default=["val", "test"], help="CVOGL split names")
    p.add_argument("--subsets", nargs="+", default=["CVOGL_SVI", "CVOGL_DroneAerial"])
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--sam_checkpoint", type=str, default="")
    p.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    ws_dir = get_workspace_dir()
    os.environ.setdefault("ROOT_DIR", str(ws_dir.parent))
    os.environ.setdefault("WORKSPACE_NAME", ws_dir.name)
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg_with_env(args.config)
    img_size = int(cfg["data"].get("img_size", 518))

    print("[1/3] Loading model ...")
    model = build_model_from_cfg(cfg, device)
    total, trainable = count_params(model)
    print(f"Model params: total={total:,} ({total/1e6:.3f}M), trainable={trainable:,} ({trainable/1e6:.3f}M)")

    ckpt = resolve_checkpoint(Path(args.checkpoint).resolve())
    obj = torch.load(str(ckpt), map_location="cpu")
    sd = extract_state_dict(obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint: {ckpt}")
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    print("[2/3] Loading SAM ...")
    sam_predictor = None
    if args.sam_checkpoint:
        _cvos_dir = str(ws_dir.parents[1] / "baseline" / "CVOS-Code")
        import sys

        if os.path.isdir(_cvos_dir):
            sys.path.insert(0, _cvos_dir)
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
        sam.to(device)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = False
        sam_predictor = SamPredictor(sam)

    print("[3/3] Evaluating ...")
    all_results: Dict[str, Dict[str, OrderedDict]] = OrderedDict()

    for split in args.splits:
        split_results: Dict[str, OrderedDict] = OrderedDict()
        for subset in args.subsets:
            ds = EvalCVOGLSegV3Dataset(
                cvogl_root=args.cvogl_root,
                cvoglseg_root=args.cvoglseg_root,
                img_size=img_size,
                split_name=split,
                subsets=[subset],
            )
            if len(ds) == 0:
                print(f"[warn] empty dataset: split={split}, subset={subset}, skip")
                continue
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_eval,
            )
            t0 = time.time()
            results = evaluate_split(
                model=model,
                sam_predictor=sam_predictor,
                loader=loader,
                img_size=img_size,
                device=device,
                use_sam=(sam_predictor is not None),
                prompt_type="point",
            )
            dt = time.time() - t0
            print_results(results, f"{subset}_{split}")
            print(f"Split {split} | subset {subset} done in {dt:.1f}s")
            split_results[subset] = OrderedDict(point=results)
        if split_results:
            all_results[split] = split_results

    if args.save_json:
        save_results_json(all_results, args.save_json)


if __name__ == "__main__":
    main()

