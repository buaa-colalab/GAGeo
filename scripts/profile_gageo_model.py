#!/usr/bin/env python3
"""Profile params, latency, and profiler FLOPs for GAGeo variants.

The script uses point prompts because rebuttal tables report point-prompt
numbers. It is intentionally checkpoint-optional: pass --checkpoint for final
reporting, or omit it for architecture/resource smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from models import build_cross_view_localizer_v2


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_state_dict(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model", "module", "state_dict", "model_state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            sd = obj[key]
            break
    else:
        sd = obj
    if sd and next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def resolve_checkpoint(path: str) -> Path:
    """Resolve Accelerate save_state directories and plain checkpoint files."""
    ckpt_path = Path(path)
    if ckpt_path.is_file():
        return ckpt_path
    for name in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        candidate = ckpt_path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve checkpoint from: {path}")


def build_from_cfg(cfg: Dict[str, Any], pretrained: bool) -> torch.nn.Module:
    mc = cfg["model"]
    dc = cfg["data"]
    return build_cross_view_localizer_v2(
        pretrained_pi3=mc.get("pi3_weights") if pretrained else None,
        freeze_backbone=False,
        freeze_prompt_encoder=False,
        load_camera_head_weights=bool(mc.get("load_camera_head_weights", False)) and pretrained,
        sam_weights=mc.get("sam_weights") if pretrained else None,
        img_size=dc.get("img_size", 518),
        patch_size=mc.get("patch_size", 14 if dc.get("img_size", 518) == 518 else 16),
        decoder_size=mc.get("decoder_size", "large"),
        num_learnable_tokens=mc.get("num_learnable_tokens", 2),
        num_bbox_mask_queries=mc.get("num_bbox_mask_queries"),
        num_heatmap_queries=mc.get("num_heatmap_queries", 1),
        supervision_layers=mc.get("supervision_layers", [4, 11, 17]),
        supervision_weights=mc.get("supervision_weights", [0.1, 0.3, 0.6]),
        mask_inject_mode=mc.get("mask_inject_mode", "global_kv"),
        use_global_attn_mask=mc.get("use_global_attn_mask", True),
        contrastive=mc.get("contrastive", True),
        contrastive_proj_dim=mc.get("contrastive_proj_dim", 256),
        contrastive_queue_size=mc.get("contrastive_queue_size", 16384),
        contrastive_momentum=mc.get("contrastive_momentum", 0.999),
        contrastive_temperature=mc.get("contrastive_temperature", 0.07),
        sam_embed_dim=mc.get("sam_embed_dim", 256),
        backbone_type=mc.get("backbone_type", "pi3"),
        encoder_name=mc.get("encoder_name", "vit_b16"),
        encoder_pretrained=bool(mc.get("encoder_pretrained", True)) and pretrained,
        encoder_weights=mc.get("encoder_weights", "LVD142M"),
        joint_vit_variant=mc.get("joint_vit_variant"),
        joint_vit_weights=mc.get("joint_vit_weights"),
        adapter_dim=mc.get("adapter_dim", 1024),
        adapter_depth=mc.get("adapter_depth", 36),
        adapter_num_heads=mc.get("adapter_num_heads", 16),
        use_frame_pos_embed=mc.get("use_frame_pos_embed", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--save_json", default="")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_from_cfg(cfg, pretrained=not args.no_pretrained and not args.checkpoint)

    if args.checkpoint:
        ckpt_path = resolve_checkpoint(args.checkpoint)
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(ckpt_path), device="cpu")
        else:
            obj = torch.load(str(ckpt_path), map_location="cpu")
            sd = extract_state_dict(obj)
        model.load_state_dict(sd, strict=False)

    model.to(device).eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    S = int(cfg["data"].get("img_size", 518))
    B = int(args.batch_size)
    front = torch.rand(B, 3, S, S, device=device)
    sat = torch.rand(B, 3, S, S, device=device)
    point_coords = torch.full((B, 1, 2), S / 2.0, device=device)
    point_labels = torch.ones(B, 1, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(front, sat, points=(point_coords, point_labels))
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device)
        else:
            peak_mem = 0

        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = model(front, sat, points=(point_coords, point_labels))
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0 / max(args.iters * B, 1)

        flops = None
        try:
            with torch.profiler.profile(with_flops=True) as prof:
                _ = model(front, sat, points=(point_coords, point_labels))
            flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
        except Exception as exc:
            print(f"[WARN] FLOPs profiling failed: {exc}")

    out = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "batch_size": B,
        "img_size": S,
        "total_params": total,
        "trainable_params": trainable,
        "latency_ms_per_sample": latency_ms,
        "peak_memory_bytes": peak_mem,
        "flops": flops,
    }
    print(json.dumps(out, indent=2))
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
