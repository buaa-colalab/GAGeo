#!/usr/bin/env python3
"""Prefetch pretrained weights needed by a GAGeo config.

Kept as a real script so distributed job wrappers do not rely on heredoc stdin.
"""

import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) != 2:
        print("Usage: prefetch_gageo_pretrained.py <config_path>", file=sys.stderr)
        return 1

    cfg_path = Path(sys.argv[1])
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    mc = cfg.get("model", {})
    if not mc.get("encoder_pretrained", True):
        return 0

    encoder_name = str(mc.get("encoder_name", "")).strip().lower()
    backbone_type = str(mc.get("backbone_type", "")).strip().lower()
    joint_vit_variant = str(mc.get("joint_vit_variant", encoder_name)).strip().lower()
    encoder_weights = str(mc.get("encoder_weights", "")).strip()
    joint_vit_weights = str(mc.get("joint_vit_weights", "") or "").strip()

    if backbone_type in {"dinov2_joint_vit", "joint_vit", "dinov2_vit", "gageo_dinov2_vit"}:
        if joint_vit_weights:
            return 0
        import torchvision.models as tv_models

        if joint_vit_variant in {"vit_h14", "vit-h14", "vit_h_14", "h14"}:
            weights = getattr(
                tv_models.ViT_H_14_Weights,
                encoder_weights or "IMAGENET1K_SWAG_E2E_V1",
                tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
            )
            weights.get_state_dict(progress=True)
        else:
            weights = getattr(
                tv_models.ViT_B_16_Weights,
                encoder_weights or "IMAGENET1K_V1",
                tv_models.ViT_B_16_Weights.IMAGENET1K_V1,
            )
            weights.get_state_dict(progress=True)
    elif encoder_name in {"vit_b16", "vit-b16", "vit_b_16", "imagenet_vit_b16"}:
        import torchvision.models as tv_models

        tv_models.ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
    elif encoder_name in {"dinov2_g14", "dinov2-g14", "dinov2_vitg14", "dinov2_vitg14_reg"}:
        import torch

        from models.dinov2.hub.utils import _DINOV2_BASE_URL

        model_base_name = "dinov2_vitg14"
        model_full_name = "dinov2_vitg14"
        url = _DINOV2_BASE_URL + "/{}/{}_pretrain.pth".format(model_base_name, model_full_name)
        torch.hub.load_state_dict_from_url(url, map_location="cpu")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
