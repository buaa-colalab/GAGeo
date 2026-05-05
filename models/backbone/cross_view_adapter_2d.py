# 2D-pretrained cross-view adapter backbone for GAGeo ablations.
#
# This module intentionally keeps GAGeo's cross-view token interaction and task
# token decoding while replacing the Pi3 3D/multi-view pretrained prior with a
# 2D visual encoder plus a randomly initialized cross-view adapter.

from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dinov2.hub.backbones import dinov2_vitg14
from ..dinov2.layers import Mlp
from ..layers.attention import FlashAttentionRope
from ..layers.block import BlockRope
from ..layers.pos_embed import PositionGetter, RoPE2D
from .pi3_backbone_v2 import BlockRopeWithMask


class TorchvisionViTFeatureExtractor(nn.Module):
    """Patch-token feature extractor for torchvision ViT models."""

    def __init__(self, model_name: str = "vit_b_16", img_size: int = 512, pretrained: bool = True):
        super().__init__()
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for ImageNet supervised ViT backbones. "
                "Install torchvision or choose a DINOv2 backbone."
            ) from exc

        if model_name != "vit_b_16":
            raise ValueError(f"Unsupported torchvision ViT model: {model_name}")

        self.model = tv_models.vit_b_16(weights=None, image_size=img_size)
        if pretrained:
            weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1
            state_dict = weights.get_state_dict(progress=True)
            state_dict = self._interpolate_state_dict_pos_embedding(state_dict)
            self.model.load_state_dict(state_dict, strict=True)
        self.embed_dim = self.model.hidden_dim
        self.patch_size = int(self.model.patch_size)

        # GAGeo uses task heads, not ImageNet logits.
        self.model.heads = nn.Identity()

    def _interpolate_state_dict_pos_embedding(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Resize ImageNet ViT position embeddings to this experiment image size."""
        key = "encoder.pos_embedding"
        pos = state_dict[key]
        target = self.model.encoder.pos_embedding
        if pos.shape == target.shape:
            return state_dict

        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        old_hw = int(patch_pos.shape[1] ** 0.5)
        new_hw = int((target.shape[1] - 1) ** 0.5)
        patch_pos = patch_pos.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=(new_hw, new_hw),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, -1)
        state_dict = dict(state_dict)
        state_dict[key] = torch.cat([cls_pos, patch_pos], dim=1)
        return state_dict

    def forward(self, x: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        n = x.shape[0]
        x = self.model._process_input(x)
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.model.encoder(x)
        return {"x_norm_patchtokens": x[:, 1:]}


def build_2d_encoder(
    encoder_name: str,
    img_size: int,
    pretrained: bool = True,
    weights: str = "LVD142M",
) -> nn.Module:
    key = str(encoder_name).strip().lower()
    if key in {"vit_b16", "vit-b16", "vit_b_16", "imagenet_vit_b16"}:
        return TorchvisionViTFeatureExtractor("vit_b_16", img_size=img_size, pretrained=pretrained)
    if key in {"dinov2_g14", "dinov2-g14", "dinov2_vitg14", "dinov2_vitg14_reg"}:
        # Use the non-register DINOv2-g encoder and keep patch tokens only. The
        # adapter owns its own learned registers for a fair GAGeo-style layout.
        return dinov2_vitg14(pretrained=pretrained, img_size=img_size, weights=weights)
    raise ValueError(f"Unsupported 2D encoder_name={encoder_name!r}")


class CrossViewAdapter2D(nn.Module):
    """
    GAGeo-compatible 2D-pretrained cross-view adapter.

    Output keys and tensor semantics match Pi3BackboneV2 so the existing
    CrossViewLocalizerV2 heads/losses/evaluators can be reused unchanged.
    """

    def __init__(
        self,
        encoder_name: str = "vit_b16",
        encoder_pretrained: bool = True,
        encoder_weights: str = "LVD142M",
        pos_type: str = "rope100",
        adapter_dim: int = 1024,
        adapter_depth: int = 36,
        adapter_num_heads: int = 16,
        img_size: int = 512,
        patch_size: int = 16,
        num_learnable_tokens: int = 2,
        supervision_layers: List[int] = None,
        mask_inject_mode: str = "global_kv",
        use_global_attn_mask: bool = True,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.num_patches_per_side = self.img_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_learnable_tokens = int(num_learnable_tokens)
        self.supervision_layers = [4, 11, 17] if supervision_layers is None else list(supervision_layers)
        self.mask_inject_mode = str(mask_inject_mode).strip().lower()
        if self.mask_inject_mode not in {"global_kv", "global_qkv", "pre_backbone"}:
            raise ValueError(
                f"Unsupported mask_inject_mode={mask_inject_mode!r}. "
                "Use one of: global_kv, global_qkv, pre_backbone."
            )
        self.use_global_attn_mask = bool(use_global_attn_mask)

        self.encoder = build_2d_encoder(
            encoder_name=encoder_name,
            img_size=self.img_size,
            pretrained=encoder_pretrained,
            weights=encoder_weights,
        )
        enc_embed_dim = int(getattr(self.encoder, "embed_dim", 0))
        if enc_embed_dim <= 0 and hasattr(self.encoder, "blocks"):
            enc_embed_dim = int(self.encoder.blocks[0].attn.qkv.in_features)
        if enc_embed_dim <= 0:
            raise ValueError(f"Cannot infer encoder dim for {encoder_name!r}")

        self.dec_embed_dim = int(adapter_dim)
        self.dec_depth = int(adapter_depth)
        if self.dec_depth % 2 != 0:
            raise ValueError(f"adapter_depth must be even, got {adapter_depth}")
        self.output_dim = 2 * self.dec_embed_dim
        self.num_stage_layers = self.dec_depth // 2

        for l in self.supervision_layers:
            if l < 0 or l >= self.num_stage_layers:
                raise ValueError(
                    f"supervision layer {l} out of range [0, {self.num_stage_layers - 1}] "
                    f"for adapter_depth={self.dec_depth}"
                )
        self.supervision_block_indices = {2 * l + 1: l for l in self.supervision_layers}

        self.encoder_proj = (
            nn.Identity() if enc_embed_dim == self.dec_embed_dim else nn.Linear(enc_embed_dim, self.dec_embed_dim)
        )

        self.pos_type = pos_type if pos_type is not None else "none"
        if not self.pos_type.startswith("rope"):
            raise NotImplementedError(f"Position type {pos_type} not supported")
        freq = float(self.pos_type[len("rope"):])
        self.rope = RoPE2D(freq=freq)
        self.position_getter = PositionGetter()

        self.decoder = nn.ModuleList([
            BlockRope(
                dim=self.dec_embed_dim,
                num_heads=adapter_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            ) for _ in range(self.dec_depth)
        ])
        self.masked_blocks = nn.ModuleList([BlockRopeWithMask(blk) for blk in self.decoder])

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_tokens, self.dec_embed_dim))
        nn.init.normal_(self.learnable_queries, std=0.02)
        self.prompt_proj = None

        self.intermediate_projs = nn.ModuleDict()
        for stage_idx in self.supervision_layers:
            self.intermediate_projs[str(stage_idx)] = nn.Linear(self.dec_embed_dim, self.output_dim)
        self.final_proj = nn.Linear(self.dec_embed_dim, self.output_dim)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def _resize_encoder_input(self, image: torch.Tensor) -> torch.Tensor:
        """Force both views onto the encoder's fixed patch grid."""
        if image.shape[-2:] == (self.img_size, self.img_size):
            return image
        return F.interpolate(
            image,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _build_global_attn_mask(
        self,
        N_sate: int,
        N_front: int,
        N_learn: int,
        N_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if N_prompt == 0:
            return None

        N_total = N_sate + N_front + N_learn + N_prompt
        mask = torch.ones(1, 1, N_total, N_total, device=device, dtype=torch.bool)

        sate_end = N_sate
        front_end = N_sate + N_front
        learn_end = front_end + N_learn
        prompt_start = learn_end

        mask[:, :, prompt_start:, :sate_end] = False
        mask[:, :, :sate_end, prompt_start:] = False

        mask[:, :, prompt_start:, prompt_start:] = False
        for i in range(N_prompt):
            mask[:, :, prompt_start + i, prompt_start + i] = True

        if N_learn > 0:
            mask[:, :, front_end:learn_end, prompt_start:] = False
            mask[:, :, prompt_start:, front_end:learn_end] = False

        return mask

    def _build_prompt_positions(
        self,
        sparse_embeddings: torch.Tensor,
        prompt_coords: Optional[torch.Tensor],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        K = sparse_embeddings.shape[1]
        if prompt_coords is not None:
            pos = prompt_coords.clone()
            pos[:, :, 0] = pos[:, :, 0] * self.num_patches_per_side
            pos[:, :, 1] = pos[:, :, 1] * self.num_patches_per_side
            return pos.to(device=device, dtype=dtype)
        return torch.zeros(B, K, 2, device=device, dtype=dtype)

    def _encode_images(self, images_flat: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(images_flat, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        hidden = self.encoder_proj(hidden)
        expected = self.num_patches
        if hidden.shape[1] != expected:
            raise ValueError(
                f"{self.encoder_name} produced {hidden.shape[1]} patches, expected {expected} "
                f"for img_size={self.img_size}, patch_size={self.patch_size}"
            )
        return hidden

    def decode_with_extra_tokens(
        self,
        hidden: torch.Tensor,
        N: int,
        H: int,
        W: int,
        sparse_embeddings: Optional[torch.Tensor] = None,
        dense_embeddings: Optional[torch.Tensor] = None,
        prompt_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        BN, hw, C = hidden.shape
        B = BN // N
        patch_h = patch_w = H // self.patch_size

        hidden = hidden.reshape(B, N, hw, C)
        sate_hidden = hidden[:, 0]
        front_hidden = hidden[:, 1]

        reg_token = self.register_token.to(hidden.device, dtype=hidden.dtype).repeat(B, 1, 1, 1)
        sate_hidden = torch.cat([reg_token[:, 0], sate_hidden], dim=1)
        front_hidden = torch.cat([reg_token[:, 0], front_hidden], dim=1)

        N_sate = sate_hidden.shape[1]
        N_front_base = front_hidden.shape[1]

        learnable_hidden = self.learnable_queries.expand(B, -1, -1).to(hidden.dtype)
        N_learn = self.num_learnable_tokens

        N_prompt = 0
        prompt_hidden = None
        prompt_pos = None
        if sparse_embeddings is not None and sparse_embeddings.shape[1] > 0:
            prompt_tokens = sparse_embeddings.to(hidden.dtype)
            if self.prompt_proj is not None:
                prompt_tokens = self.prompt_proj(prompt_tokens)
            N_prompt = prompt_tokens.shape[1]
            prompt_hidden = prompt_tokens

        base_pos = self.position_getter(B, patch_h, patch_w, hidden.device)
        if self.patch_start_idx > 0:
            base_pos = base_pos + 1
            pos_special = torch.zeros(B, self.patch_start_idx, 2, device=hidden.device, dtype=base_pos.dtype)
            base_pos_with_reg = torch.cat([pos_special, base_pos], dim=1)
        else:
            base_pos_with_reg = base_pos

        sate_pos = base_pos_with_reg
        front_pos = base_pos_with_reg
        learnable_pos = torch.zeros(B, N_learn, 2, device=hidden.device, dtype=base_pos.dtype)

        if N_prompt > 0:
            prompt_pos = self._build_prompt_positions(sparse_embeddings, prompt_coords, B, hidden.device, base_pos.dtype)

        dense_flat = None
        if dense_embeddings is not None:
            dense_flat = dense_embeddings.flatten(2).transpose(1, 2).to(hidden.dtype)
            if dense_flat.shape[1] != self.num_patches:
                raise ValueError(
                    f"dense prompt has {dense_flat.shape[1]} tokens, expected {self.num_patches}. "
                    "Check img_size/patch_size and prompt encoder image_embedding_size."
                )
            if self.mask_inject_mode == "pre_backbone":
                f_patch_start = self.patch_start_idx
                f_patch_end = f_patch_start + self.num_patches
                front_hidden[:, f_patch_start:f_patch_end] += dense_flat

        global_mask = None
        if self.use_global_attn_mask:
            global_mask = self._build_global_attn_mask(
                N_sate, N_front_base, N_learn, N_prompt, hidden.device, hidden.dtype
            )

        final_output = []
        intermediate_outputs = {}

        for i in range(len(self.decoder)):
            if i % 2 == 0:
                sate_local = torch.cat([sate_hidden, learnable_hidden], dim=1)
                sate_local_pos = torch.cat([sate_pos, learnable_pos], dim=1)
                sate_local = self.masked_blocks[i](sate_local, xpos=sate_local_pos)

                front_local = front_hidden
                front_local_pos = front_pos
                if prompt_hidden is not None:
                    front_local = torch.cat([front_local, prompt_hidden], dim=1)
                    front_local_pos = torch.cat([front_local_pos, prompt_pos], dim=1)
                front_local = self.masked_blocks[i](front_local, xpos=front_local_pos)

                sate_hidden = sate_local[:, :N_sate]
                learnable_hidden = sate_local[:, N_sate:]
                front_hidden = front_local[:, :N_front_base]
                if prompt_hidden is not None:
                    prompt_hidden = front_local[:, N_front_base:]
            else:
                global_qv_tokens = [sate_hidden, front_hidden, learnable_hidden]
                global_pos_tokens = [sate_pos, front_pos, learnable_pos]
                if prompt_hidden is not None:
                    global_qv_tokens.append(prompt_hidden)
                    global_pos_tokens.append(prompt_pos)
                global_qv = torch.cat(global_qv_tokens, dim=1)

                if dense_flat is not None and self.mask_inject_mode in ("global_kv", "global_qkv"):
                    g_front_patch_start = N_sate + self.patch_start_idx
                    g_front_patch_end = g_front_patch_start + self.num_patches
                    if self.mask_inject_mode == "global_qkv":
                        global_qv[:, g_front_patch_start:g_front_patch_end] += dense_flat
                        global_k = global_qv
                    else:
                        global_k = global_qv.clone()
                        global_k[:, g_front_patch_start:g_front_patch_end] += dense_flat
                else:
                    global_k = global_qv

                global_pos = torch.cat(global_pos_tokens, dim=1)
                global_hidden = self.masked_blocks[i].forward_qkv(
                    global_qv,
                    global_k,
                    qpos=global_pos,
                    kvpos=global_pos,
                    attn_mask=global_mask,
                )

                g_front_end = N_sate + N_front_base
                g_learn_end = g_front_end + N_learn
                sate_hidden = global_hidden[:, :N_sate]
                front_hidden = global_hidden[:, N_sate:g_front_end]
                learnable_hidden = global_hidden[:, g_front_end:g_learn_end]
                if prompt_hidden is not None:
                    prompt_hidden = global_hidden[:, g_learn_end:]

            if i in self.supervision_block_indices:
                stage_idx = self.supervision_block_indices[i]
                proj = self.intermediate_projs[str(stage_idx)]
                out_dict = {
                    "learnable": proj(learnable_hidden),
                    "sate_patches": proj(sate_hidden[:, self.patch_start_idx:]),
                }
                final_stage = max(self.supervision_layers)
                if stage_idx != final_stage:
                    out_dict["front_patches"] = proj(
                        front_hidden[:, self.patch_start_idx:self.patch_start_idx + self.num_patches]
                    )
                intermediate_outputs[stage_idx] = out_dict

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(torch.stack([sate_hidden, front_hidden[:, :N_front_base]], dim=1))

        features = torch.cat([final_output[0], final_output[1]], dim=-1)
        learnable_final = self.final_proj(learnable_hidden)

        return {
            "features": features,
            "sate_features": features[:, 0, self.patch_start_idx:, :],
            "front_features": features[:, 1, self.patch_start_idx:, :],
            "sate_camera_token": features[:, 0, 0, :],
            "front_camera_token": features[:, 1, 0, :],
            "learnable_out": learnable_final,
            "intermediate": intermediate_outputs,
        }

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
        images_flat = images.reshape(B * N, 3, H, W)
        target_dtype = self.image_mean.dtype
        if images_flat.dtype != target_dtype:
            images_flat = images_flat.to(target_dtype)

        hidden = self._encode_images(images_flat)
        return self.decode_with_extra_tokens(
            hidden,
            N,
            H,
            W,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            prompt_coords=prompt_coords,
        )
