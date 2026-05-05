#!/usr/bin/env python3
"""Collect all pretrained checkpoints needed by rebuttal training scripts.

The training configs in this repo expect these filenames under CHECKPOINT_DIR:
  - pi3_model.safetensors
  - sam2.1_hiera_large.pt
  - vit_b_16_imagenet1k_v1.pth
  - vit_h_14_imagenet1k_swag_e2e_v1.pth

Run this once on a machine with network access, then copy the output directory
to any offline training machine and set CHECKPOINT_DIR to that directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

SAM2_HIERA_L_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)


def copy_or_download(
    name: str,
    dst: Path,
    source: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {name}: {dst}")
        return

    if source:
        src = Path(source).expanduser()
        if src.exists():
            print(f"[copy] {name}: {src} -> {dst}")
            shutil.copy2(src, dst)
            return

    if url:
        print(f"[download] {name}: {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)
        return

    raise FileNotFoundError(
        f"Cannot prepare {name}. Provide a valid local source or URL."
    )


def download_torchvision_vit_b(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] ViT-B/16: {dst}")
        return
    import torchvision.models as tv_models
    import torch

    print(f"[download] ViT-B/16 ImageNet-1K -> {dst}")
    state = tv_models.ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
    torch.save(state, dst)


def download_torchvision_vit_h(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] ViT-H/14: {dst}")
        return
    import torchvision.models as tv_models
    import torch

    print(f"[download] ViT-H/14 SWAG E2E -> {dst}")
    state = tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.get_state_dict(progress=True)
    torch.save(state, dst)


def maybe_download_pi3_from_hf(dst: Path, repo_id: str, filename: str) -> bool:
    if not repo_id:
        return False
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[warn] huggingface_hub is not installed; cannot download Pi3 from HF.")
        return False

    print(f"[download] Pi3 from HF {repo_id}/{filename} -> {dst}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy2(cached, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="/mnt/data/wrp/checkpoints_offline")
    parser.add_argument("--pi3_source", default="/mnt/data/wrp/GaGeo/ckpt/pi3/model.safetensors")
    parser.add_argument("--sam_source", default="/mnt/data/wrp/GaGeo/ckpt/sam2.1_hiera_large.pt")
    parser.add_argument("--pi3_hf_repo", default="", help="Optional HF repo id for Pi3 if pi3_source is unavailable")
    parser.add_argument("--pi3_hf_filename", default="model.safetensors")
    parser.add_argument("--skip_vit_h", action="store_true", help="Skip the large ViT-H download")
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    pi3_dst = out / "pi3_model.safetensors"
    try:
        copy_or_download("Pi3", pi3_dst, source=args.pi3_source)
    except FileNotFoundError:
        if not maybe_download_pi3_from_hf(pi3_dst, args.pi3_hf_repo, args.pi3_hf_filename):
            print(
                "[error] Pi3 checkpoint was not prepared. Re-run with --pi3_source "
                "or --pi3_hf_repo/--pi3_hf_filename.",
                file=sys.stderr,
            )
            return 2

    copy_or_download(
        "SAM2.1 Hiera-L",
        out / "sam2.1_hiera_large.pt",
        source=args.sam_source,
        url=SAM2_HIERA_L_URL,
    )
    download_torchvision_vit_b(out / "vit_b_16_imagenet1k_v1.pth")
    if not args.skip_vit_h:
        download_torchvision_vit_h(out / "vit_h_14_imagenet1k_swag_e2e_v1.pth")

    manifest = out / "offline_paths.yaml"
    manifest.write_text(
        "\n".join(
            [
                f"CHECKPOINT_DIR: {out}",
                f"pi3_weights: {pi3_dst}",
                f"sam_weights: {out / 'sam2.1_hiera_large.pt'}",
                f"vit_b_weights: {out / 'vit_b_16_imagenet1k_v1.pth'}",
                f"vit_h_weights: {out / 'vit_h_14_imagenet1k_swag_e2e_v1.pth'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[done] checkpoint directory: {out}")
    print(f"[done] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
