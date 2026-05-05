# DINOv2 encoder + pretrained ViT fusion backbone for 2D-GAGeo ablations.
#
# This backbone keeps GAGeo's DINOv2 visual encoder and downstream task-token
# heads, but replaces Pi3's 3D alternating decoder with standard pretrained
# ViT encoder blocks that operate on concatenated cross-view feature tokens.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dinov2.hub.backbones import dinov2_vitl14_reg


class DINOv2JointViTBackbone(nn.Module):
    """
    GAGeo-compatible 2D joint ViT fusion backbone.

    Token flow:
      image pair -> DINOv2-L/14-reg patch tokens -> project to ViT dim
      -> concat [sat, front, task, prompt] tokens -> pretrained ViT blocks
      -> project sat/front/task tokens to the 2048-dim GAGeo head interface.
    """

    def __init__(
        self,
        joint_vit_variant: str = "vit_b16",
        encoder_pretrained: bool = True,
        encoder_weights: str = "IMAGENET1K_V1",
        joint_vit_weights: Optional[str] = None,
        img_size: int = 518,
        patch_size: int = 14,
        num_learnable_tokens: int = 2,
        supervision_layers: List[int] = None,
        output_dim: int = 2048,
        mask_inject_mode: str = "pre_backbone",
        **_: object,
    ):
        super().__init__()

        if int(patch_size) != 14:
            raise ValueError("DINOv2JointViTBackbone requires patch_size=14 for DINOv2-L/14 tokens.")

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.num_patches_per_side = self.img_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_learnable_tokens = int(num_learnable_tokens)
        self.supervision_layers = [] if supervision_layers is None else list(supervision_layers)
        self.mask_inject_mode = str(mask_inject_mode).strip().lower()
        self.output_dim = int(output_dim)

        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        self.dino_embed_dim = int(self.encoder.blocks[0].attn.qkv.in_features)

        vit = self._build_torchvision_vit(
            joint_vit_variant=joint_vit_variant,
            pretrained=bool(encoder_pretrained),
            weights_name=encoder_weights,
            weights_path=joint_vit_weights,
        )
        self.joint_vit_variant = joint_vit_variant
        self.dec_embed_dim = int(vit.hidden_dim)
        self.dec_depth = len(vit.encoder.layers)
        self.num_stage_layers = self.dec_depth

        for layer_idx in self.supervision_layers:
            if layer_idx < 0 or layer_idx >= self.num_stage_layers:
                raise ValueError(
                    f"supervision layer {layer_idx} out of range [0, {self.num_stage_layers - 1}] "
                    f"for joint_vit_variant={joint_vit_variant}"
                )

        # Keep the familiar `decoder` attribute so existing freeze/checkpointing
        # training code can address the fusion transformer without branching.
        self.decoder = vit.encoder.layers
        self.vit_norm = vit.encoder.ln
        self.vit_pos_embedding = nn.Parameter(vit.encoder.pos_embedding.detach().clone())

        self.dino_to_vit = nn.Linear(self.dino_embed_dim, self.dec_embed_dim)
        self.view_embedding = nn.Parameter(torch.zeros(1, 2, 1, self.dec_embed_dim))
        self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_tokens, self.dec_embed_dim))
        nn.init.normal_(self.learnable_queries, std=0.02)

        self.prompt_proj = None
        self.prompt_coord_mlp = nn.Sequential(
            nn.Linear(2, self.dec_embed_dim),
            nn.GELU(),
            nn.Linear(self.dec_embed_dim, self.dec_embed_dim),
        )

        self.intermediate_projs = nn.ModuleDict()
        for layer_idx in self.supervision_layers:
            self.intermediate_projs[str(layer_idx)] = nn.Linear(self.dec_embed_dim, self.output_dim)
        self.final_proj = nn.Linear(self.dec_embed_dim, self.output_dim)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @staticmethod
    @staticmethod
    def _extract_plain_state_dict(obj: object) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    return DINOv2JointViTBackbone._extract_plain_state_dict(obj[key])
            return {
                str(k).removeprefix("module."): v
                for k, v in obj.items()
                if isinstance(v, torch.Tensor)
            }
        raise TypeError(f"Unsupported ViT checkpoint object type: {type(obj)!r}")

    @staticmethod
    @staticmethod
    def _read_local_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
        ckpt_path = str(weights_path).strip()
        if not ckpt_path:
            raise ValueError("empty local ViT checkpoint path")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        return DINOv2JointViTBackbone._extract_plain_state_dict(state_dict)

    @staticmethod
    def _infer_local_vit_image_size(state_dict: Dict[str, torch.Tensor], patch_size: int, fallback: int) -> int:
        pos = state_dict.get("encoder.pos_embedding")
        if pos is None or pos.ndim != 3 or pos.shape[1] <= 1:
            return int(fallback)
        grid = int((int(pos.shape[1]) - 1) ** 0.5)
        if grid * grid + 1 != int(pos.shape[1]):
            return int(fallback)
        return int(grid * patch_size)

    @staticmethod
    def _load_local_torchvision_vit(
        vit: nn.Module,
        weights_path: str,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        ckpt_path = str(weights_path).strip()
        if state_dict is None:
            state_dict = DINOv2JointViTBackbone._read_local_state_dict(ckpt_path)
        missing, unexpected = vit.load_state_dict(state_dict, strict=False)
        # Torchvision checkpoints may contain classifier heads with shapes that
        # are unused by GAGeo, but the encoder/body must load cleanly.
        bad_missing = [k for k in missing if not k.startswith("heads.")]
        bad_unexpected = [k for k in unexpected if not k.startswith("heads.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"Failed to load local ViT weights from {ckpt_path}: "
                f"missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}"
            )
        print(f"Loaded local torchvision ViT weights from {ckpt_path}")

    @staticmethod
    def _build_torchvision_vit(
        joint_vit_variant: str,
        pretrained: bool,
        weights_name: str,
        weights_path: Optional[str] = None,
    ) -> nn.Module:
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError("torchvision is required for DINOv2JointViTBackbone.") from exc

        key = str(joint_vit_variant).strip().lower()
        weights_key = str(weights_name).strip()
        local_path = str(weights_path).strip() if weights_path else ""

        if key in {"vit_b16", "vit-b16", "vit_b_16", "b16"}:
            local_state = DINOv2JointViTBackbone._read_local_state_dict(local_path) if local_path else None
            image_size = (
                DINOv2JointViTBackbone._infer_local_vit_image_size(local_state, patch_size=16, fallback=224)
                if local_state is not None
                else 224
            )
            weights = None
            if pretrained and not local_path:
                enum = tv_models.ViT_B_16_Weights
                weights = getattr(enum, weights_key, enum.IMAGENET1K_V1)
            vit_kwargs = {"image_size": image_size} if weights is None else {}
            vit = tv_models.vit_b_16(weights=weights, **vit_kwargs)
            if pretrained and local_path:
                DINOv2JointViTBackbone._load_local_torchvision_vit(vit, local_path, local_state)
            return vit

        if key in {"vit_h14", "vit-h14", "vit_h_14", "h14"}:
            local_state = DINOv2JointViTBackbone._read_local_state_dict(local_path) if local_path else None
            image_size = (
                DINOv2JointViTBackbone._infer_local_vit_image_size(local_state, patch_size=14, fallback=518)
                if local_state is not None
                else 518
            )
            weights = None
            if pretrained and not local_path:
                enum = tv_models.ViT_H_14_Weights
                weights = getattr(enum, weights_key, enum.IMAGENET1K_SWAG_E2E_V1)
            vit_kwargs = {"image_size": image_size} if weights is None else {}
            vit = tv_models.vit_h_14(weights=weights, **vit_kwargs)
            if pretrained and local_path:
                DINOv2JointViTBackbone._load_local_torchvision_vit(vit, local_path, local_state)
            return vit

        raise ValueError(f"Unsupported joint_vit_variant={joint_vit_variant!r}")

    def load_dinov2_encoder_from_pi3(self, checkpoint_path: str) -> None:
        """Load only the DINOv2 encoder weights from a Pi3/GAGeo checkpoint."""
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        encoder_state = {
            key[len("encoder.") :]: value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        new_missing = [key for key in missing if key != "mask_token"]
        print(f"Loaded DINOv2 encoder weights from {checkpoint_path}")
        print(f"  Loaded encoder keys: {len(encoder_state)}")
        if new_missing:
            print(f"  Missing encoder keys: {len(new_missing)}")
            for key in new_missing[:10]:
                print(f"    - {key}")
        if unexpected:
            print(f"  Unexpected encoder keys: {len(unexpected)}")

    def _resize_encoder_input(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == (self.img_size, self.img_size):
            return image
        return F.interpolate(
            image,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _encode_dinov2(self, images_flat: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(images_flat, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        if hidden.shape[1] != self.num_patches:
            raise ValueError(
                f"DINOv2 produced {hidden.shape[1]} patches, expected {self.num_patches} "
                f"for img_size={self.img_size}, patch_size={self.patch_size}"
            )
        return hidden

    def _interpolated_vit_patch_pos(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pos = self.vit_pos_embedding.to(device=device, dtype=dtype)
        patch_pos = pos[:, 1:]
        old_hw = int(patch_pos.shape[1] ** 0.5)
        new_hw = self.num_patches_per_side
        if old_hw != new_hw:
            patch_pos = patch_pos.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_hw, new_hw), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, -1)
        return patch_pos

    def _prompt_tokens(
        self,
        sparse_embeddings: Optional[torch.Tensor],
        prompt_coords: Optional[torch.Tensor],
        B: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if sparse_embeddings is None or sparse_embeddings.shape[1] == 0:
            return None

        tokens = sparse_embeddings.to(device=device, dtype=dtype)
        if self.prompt_proj is not None:
            tokens = self.prompt_proj(tokens)

        # SAM sparse embeddings already contain prompt-type information. This
        # small learned coordinate term preserves explicit point/box geometry
        # after projection into the ViT fusion space.
        if prompt_coords is not None:
            coord = prompt_coords.to(device=device, dtype=dtype).clamp(0, 1)
            tokens = tokens + self.prompt_coord_mlp(coord)
        return tokens

    def forward(
        self,
        front_view: torch.Tensor,
        satellite_view: torch.Tensor,
        sparse_embeddings: Optional[torch.Tensor] = None,
        dense_embeddings: Optional[torch.Tensor] = None,
        prompt_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        satellite_view = self._resize_encoder_input(satellite_view)
        front_view = self._resize_encoder_input(front_view)

        images = torch.stack([satellite_view, front_view], dim=1)
        images = (images - self.image_mean) / self.image_std
        B, N, _, H, W = images.shape
        if N != 2:
            raise ValueError(f"Expected 2 views, got {N}")

        images_flat = images.reshape(B * N, 3, H, W).to(dtype=self.image_mean.dtype)
        dino_tokens = self._encode_dinov2(images_flat)
        dino_tokens = self.dino_to_vit(dino_tokens).reshape(B, N, self.num_patches, self.dec_embed_dim)

        if dense_embeddings is not None:
            dense_flat = dense_embeddings.flatten(2).transpose(1, 2).to(dtype=dino_tokens.dtype)
            if dense_flat.shape[1] != self.num_patches:
                raise ValueError(
                    f"dense prompt has {dense_flat.shape[1]} tokens, expected {self.num_patches}. "
                    "Check img_size/patch_size and prompt encoder image_embedding_size."
                )
            dino_tokens[:, 1] = dino_tokens[:, 1] + dense_flat

        patch_pos = self._interpolated_vit_patch_pos(dtype=dino_tokens.dtype, device=dino_tokens.device)
        view_pos = self.view_embedding.to(dtype=dino_tokens.dtype)
        dino_tokens = dino_tokens + patch_pos.unsqueeze(1) + view_pos

        sat_tokens = dino_tokens[:, 0]
        front_tokens = dino_tokens[:, 1]
        learnable_tokens = self.learnable_queries.expand(B, -1, -1).to(dtype=dino_tokens.dtype)
        prompt_tokens = self._prompt_tokens(sparse_embeddings, prompt_coords, B, dino_tokens.dtype, dino_tokens.device)

        seq_parts = [sat_tokens, front_tokens, learnable_tokens]
        if prompt_tokens is not None:
            seq_parts.append(prompt_tokens)
        hidden = torch.cat(seq_parts, dim=1)

        intermediate_outputs = {}
        final_stage = max(self.supervision_layers) if self.supervision_layers else None
        for layer_idx, block in enumerate(self.decoder):
            hidden = block(hidden)
            if layer_idx in self.supervision_layers:
                proj = self.intermediate_projs[str(layer_idx)]
                sat_end = self.num_patches
                front_end = sat_end + self.num_patches
                learn_end = front_end + self.num_learnable_tokens
                out_dict = {
                    "learnable": proj(hidden[:, front_end:learn_end]),
                    "sate_patches": proj(hidden[:, :sat_end]),
                }
                if layer_idx != final_stage:
                    out_dict["front_patches"] = proj(hidden[:, sat_end:front_end])
                intermediate_outputs[layer_idx] = out_dict

        hidden = self.vit_norm(hidden)
        sat_end = self.num_patches
        front_end = sat_end + self.num_patches
        learn_end = front_end + self.num_learnable_tokens

        sat_out = self.final_proj(hidden[:, :sat_end])
        front_out = self.final_proj(hidden[:, sat_end:front_end])
        learnable_out = self.final_proj(hidden[:, front_end:learn_end])

        return {
            "features": torch.stack([sat_out, front_out], dim=1),
            "sate_features": sat_out,
            "front_features": front_out,
            "sate_camera_token": sat_out.mean(dim=1),
            "front_camera_token": front_out.mean(dim=1),
            "learnable_out": learnable_out,
            "intermediate": intermediate_outputs,
        }
