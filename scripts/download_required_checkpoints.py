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
DEFAULT_PI3_HF_REPO = "yyfz233/Pi3"
DEFAULT_PI3_HF_CANDIDATES = (
    "model.safetensors",
    "pi3_model.safetensors",
    "pytorch_model.safetensors",
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
    """Download Pi3 weights from Hugging Face Hub.

    The Pi3 repo layout may change over time, so we first try the explicitly
    requested filename, then fall back to common safetensors names, and finally
    inspect the repo file list to find a suitable `.safetensors` checkpoint.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("[warn] huggingface_hub is not installed; cannot download Pi3 from HF.")
        return False

    repo = str(repo_id or DEFAULT_PI3_HF_REPO).strip()
    candidates: list[str] = []
    if filename:
        candidates.append(str(filename).strip())
    for candidate in DEFAULT_PI3_HF_CANDIDATES:
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        try:
            print(f"[download] Pi3 from HF {repo}/{candidate} -> {dst}")
            cached = hf_hub_download(repo_id=repo, filename=candidate)
            shutil.copy2(cached, dst)
            return True
        except Exception as exc:
            print(f"[warn] Pi3 HF candidate not available: {candidate} ({exc})")

    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:
        print(f"[warn] Failed to inspect HF repo {repo}: {exc}")
        return False

    safetensor_files = [name for name in repo_files if name.endswith(".safetensors")]
    preferred = sorted(
        safetensor_files,
        key=lambda name: (
            0 if Path(name).name in DEFAULT_PI3_HF_CANDIDATES else 1,
            0 if "model" in Path(name).name.lower() else 1,
            len(name),
            name,
        ),
    )
    if not preferred:
        print(f"[warn] No .safetensors checkpoint found in HF repo {repo}.")
        return False

    chosen = preferred[0]
    print(f"[download] Pi3 from HF {repo}/{chosen} -> {dst}")
    cached = hf_hub_download(repo_id=repo, filename=chosen)
    shutil.copy2(cached, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="/mnt/data/wrp/checkpoints_offline")
    parser.add_argument("--pi3_source", default="/mnt/data/wrp/GaGeo/ckpt/pi3/model.safetensors")
    parser.add_argument("--sam_source", default="/mnt/data/wrp/GaGeo/ckpt/sam2.1_hiera_large.pt")
    parser.add_argument(
        "--pi3_hf_repo",
        default=DEFAULT_PI3_HF_REPO,
        help="HF repo id for Pi3 checkpoint fallback",
    )
    parser.add_argument(
        "--pi3_hf_filename",
        default="",
        help="Optional exact Pi3 filename inside the HF repo; auto-discovered when empty",
    )
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
