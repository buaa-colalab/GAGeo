# Pi3 Backbone V2 for Cross-View Localization
# Supports injecting extra tokens (learnable queries + prompt tokens) into the decoder
# with custom attention masks for local/global attention layers.

import torch
import torch.nn as nn
from functools import partial
from typing import List, Tuple, Optional, Dict

from ..dinov2.layers import Mlp
from ..layers.pos_embed import RoPE2D, PositionGetter
from ..layers.block import BlockRope
from ..layers.attention import FlashAttentionRope
from ..dinov2.hub.backbones import dinov2_vitl14_reg


class MaskedFlashAttentionRope(nn.Module):
    """
    FlashAttentionRope wrapper that supports attention masks.
    
    Backend policy:
    - no mask: prefer FLASH_ATTENTION
    - with mask: prefer FLASH_ATTENTION, then EFFICIENT_ATTENTION, then MATH fallback
    
    Note: mask is expected as SDPA bool mask (True=keep, False=block),
    but additive float masks are also accepted for backward compatibility.
    """
    
    def __init__(self, base_attn: FlashAttentionRope):
        super().__init__()
        # Share all parameters with the base attention module
        self.base_attn = base_attn
    
    @property
    def qkv(self):
        return self.base_attn.qkv
    
    @property
    def proj(self):
        return self.base_attn.proj
    
    @property
    def proj_drop(self):
        return self.base_attn.proj_drop
    
    @property
    def q_norm(self):
        return self.base_attn.q_norm
    
    @property
    def k_norm(self):
        return self.base_attn.k_norm
    
    @property
    def rope(self):
        return self.base_attn.rope
    
    @property 
    def num_heads(self):
        return self.base_attn.num_heads
    
    def forward(self, x, attn_bias=None, xpos=None, attn_mask=None):
        """Forward with optional attention mask for SDPA."""
        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend
        
        B, N, C = x.shape
        qkv = self.base_attn.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        if attn_mask is not None:
            # Prefer bool keep-mask for best backend compatibility in SDPA.
            sdpa_mask = attn_mask
            if torch.is_floating_point(attn_mask):
                # Backward compatibility: additive mask (0 for keep, -inf for block)
                sdpa_mask = torch.isfinite(attn_mask) & (attn_mask >= 0)

            # Try Flash first (newer PyTorch may support masked Flash SDPA),
            # then Efficient Attention, then Math fallback.
            with nn.attention.sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]):
                x = scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        else:
            # No mask: use Flash Attention for maximum speed
            if q.dtype == torch.bfloat16:
                with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    x = scaled_dot_product_attention(q, k, v)
            else:
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    x = scaled_dot_product_attention(q, k, v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.base_attn.proj(x)
        x = self.base_attn.proj_drop(x)
        return x

    def forward_qkv(self, x_q, x_kv, qpos=None, kvpos=None, attn_mask=None):
        """
        Forward with decoupled Q and K/V inputs.
        Used by V3 global attention where K includes front+mask while Q/V use original tokens.
        """
        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend

        Bq, Nq, Cq = x_q.shape
        Bk, Nk, Ck = x_kv.shape
        if Bq != Bk or Cq != Ck:
            raise ValueError(f"Q and KV shape mismatch: q={x_q.shape}, kv={x_kv.shape}")

        # Compute Q from x_q
        q_all = self.base_attn.qkv(x_q)
        q_all = q_all.reshape(Bq, Nq, 3, self.num_heads, Cq // self.num_heads).transpose(1, 3)
        q = q_all[:, :, 0]

        # Compute K,V from x_kv
        kv_all = self.base_attn.qkv(x_kv)
        kv_all = kv_all.reshape(Bk, Nk, 3, self.num_heads, Ck // self.num_heads).transpose(1, 3)
        k = kv_all[:, :, 1]
        v = kv_all[:, :, 2]

        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kvpos)

        if attn_mask is not None:
            sdpa_mask = attn_mask
            if torch.is_floating_point(attn_mask):
                sdpa_mask = torch.isfinite(attn_mask) & (attn_mask >= 0)
            with nn.attention.sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]):
                x = scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        else:
            if q.dtype == torch.bfloat16:
                with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    x = scaled_dot_product_attention(q, k, v)
            else:
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(Bq, Nq, Cq)
        x = self.base_attn.proj(x)
        x = self.base_attn.proj_drop(x)
        return x


class BlockRopeWithMask(nn.Module):
    """BlockRope wrapper that passes attention masks through to attention."""
    
    def __init__(self, block: BlockRope):
        super().__init__()
        self.block = block
        # Wrap the attention with mask support
        self.masked_attn = MaskedFlashAttentionRope(block.attn)
    
    def forward(self, x, xpos=None, attn_mask=None):
        """Forward with optional attn_mask."""
        if attn_mask is None:
            # No mask: use original block for maximum compatibility
            return self.block(x, xpos=xpos)
        
        # With mask: use custom forward path
        def attn_residual_func(x_in):
            return self.block.ls1(self.masked_attn(self.block.norm1(x_in), xpos=xpos, attn_mask=attn_mask))
        
        def ffn_residual_func(x_in):
            return self.block.ls2(self.block.mlp(self.block.norm2(x_in)))
        
        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x

    def forward_qkv(self, x_q, x_kv, qpos=None, kvpos=None, attn_mask=None):
        """Forward with decoupled Q and K/V inputs."""
        def attn_residual_func(x_in):
            return self.block.ls1(
                self.masked_attn.forward_qkv(
                    self.block.norm1(x_in),
                    self.block.norm1(x_kv),
                    qpos=qpos,
                    kvpos=kvpos,
                    attn_mask=attn_mask,
                )
            )

        def ffn_residual_func(x_in):
            return self.block.ls2(self.block.mlp(self.block.norm2(x_in)))

        x = x_q + attn_residual_func(x_q)
        x = x + ffn_residual_func(x)
        return x


class Pi3BackboneV2(nn.Module):
    """
    Pi3 Backbone V2 with support for:
    1. Injecting extra tokens (learnable queries + prompt tokens) into the decoder
    2. Custom attention masks for local/global attention
    3. Intermediate layer output for deep supervision (layers 4, 11, 17)
    4. Mask prompt via element-wise addition to front view tokens
    
    Token layout:
    - Local stage (frame-wise):
      - Satellite stream: [register(5) | sate_patches(1369) | learnable(Q)]
      - Front stream: [register(5) | front_patches(1369) | prompt(K)]
    - Global stage:
      - [sate | front | learnable | prompt]
    
    Args:
        pos_type: Positional encoding type (default 'rope100')
        decoder_size: Decoder size ('small', 'base', 'large')
        img_size: Input image size (default 518)
        patch_size: Patch size (default 14)
        num_learnable_tokens: Number of learnable query tokens (default 2)
        supervision_layers: Pair-layer indices for deep supervision (0-indexed),
            where one layer = (local block + global block).
            For decoder depth 36, valid pair-layer indices are [0..17].
    """
    
    def __init__(
        self,
        pos_type: str = 'rope100',
        decoder_size: str = 'large',
        img_size: int = 518,
        patch_size: int = 14,
        num_learnable_tokens: int = 2,
        supervision_layers: List[int] = None,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size  # 37
        self.num_patches = self.num_patches_per_side ** 2   # 1369
        self.num_learnable_tokens = num_learnable_tokens
        self.supervision_layers = [4, 11, 17] if supervision_layers is None else list(supervision_layers)
        
        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        del self.encoder.mask_token
        
        # ----------------------
        #  Positional Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError(f"Position type {pos_type} not supported")
        
        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        
        if decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        else:
            raise NotImplementedError(f"Decoder size {decoder_size} not supported")
        
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.output_dim = 2 * dec_embed_dim  # Concatenate last two layers
        self.num_stage_layers = self.dec_depth // 2  # one stage = local+global

        # Validate supervision layers are stage indices (0-based pair layers)
        for l in self.supervision_layers:
            if l < 0 or l >= self.num_stage_layers:
                raise ValueError(
                    f"supervision layer {l} out of range [0, {self.num_stage_layers - 1}] "
                    f"for decoder_size={decoder_size} (dec_depth={self.dec_depth})"
                )
        # Map stage index -> decoder block index (after global block)
        self.supervision_block_indices = {2 * l + 1: l for l in self.supervision_layers}
        
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
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
                rope=self.rope
            ) for _ in range(dec_depth)
        ])
        
        # Wrap decoder blocks with mask support
        self.masked_blocks = nn.ModuleList([
            BlockRopeWithMask(blk) for blk in self.decoder
        ])
        
        # ----------------------
        #     Register tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)
        
        # ----------------------
        #   Learnable Query Tokens (new)
        # ----------------------
        self.learnable_queries = nn.Parameter(torch.randn(1, num_learnable_tokens, dec_embed_dim))
        nn.init.normal_(self.learnable_queries, std=0.02)
        
        # Projection for prompt tokens from SAM dim to decoder dim
        # (will be set externally if sam_embed_dim != dec_embed_dim)
        self.prompt_proj = None
        
        # Projection for intermediate supervision (stage-indexed)
        # Each supervised stage's single-layer features -> output_dim
        self.intermediate_projs = nn.ModuleDict()
        for stage_idx in self.supervision_layers:
            self.intermediate_projs[str(stage_idx)] = nn.Linear(dec_embed_dim, self.output_dim)

        # Final token projection (always available)
        self.final_proj = nn.Linear(dec_embed_dim, self.output_dim)
        
        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
    
    def _build_local_attn_mask(
        self,
        N_sate: int,
        N_front: int,
        N_learn: int,
        N_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        V3 Local stage uses a masked GLOBAL self-attention over all tokens:
          Q=K=V=(sate, front, learnable, prompt)

        Constraints:
          - sate and front are mutually invisible
          - prompt and sate are mutually invisible
          - prompt tokens are mutually invisible (except self)
          - learnable tokens can interact with both sate/front/prompt
        """
        N_total = N_sate + N_front + N_learn + N_prompt
        mask = torch.ones(1, 1, N_total, N_total, device=device, dtype=torch.bool)

        sate_end = N_sate
        front_end = N_sate + N_front
        learn_end = front_end + N_learn
        prompt_start = learn_end

        # sate <-> front: mutual block
        mask[:, :, :sate_end, sate_end:front_end] = False
        mask[:, :, sate_end:front_end, :sate_end] = False

        if N_prompt > 0:
            # prompt <-> sate: mutual block
            mask[:, :, prompt_start:, :sate_end] = False
            mask[:, :, :sate_end, prompt_start:] = False

            # prompt <-> prompt: mutual block except diagonal
            mask[:, :, prompt_start:, prompt_start:] = False
            for i in range(N_prompt):
                mask[:, :, prompt_start + i, prompt_start + i] = True

        return mask
    
    def _build_global_attn_mask(
        self,
        N_sate: int,
        N_front: int,
        N_learn: int,
        N_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Build attention mask for global (cross-view) attention.
        
        Layout: [sate(N_sate) | front(N_front) | learnable(N_learn) | prompt(N_prompt)]
        
        Rules:
        - Prompt tokens cannot see sate tokens (and vice versa)
        - Prompt tokens cannot see each other
        - Learnable queries and prompt tokens are mutually invisible
        - Everything else is allowed
        
        Returns:
            mask: [1, 1, N_total, N_total] bool keep mask for SDPA
        """
        if N_prompt == 0:
            return None
        
        N_total = N_sate + N_front + N_learn + N_prompt
        mask = torch.ones(1, 1, N_total, N_total, device=device, dtype=torch.bool)
        
        sate_end = N_sate
        front_end = N_sate + N_front
        learn_end = front_end + N_learn
        prompt_start = learn_end
        
        # Prompt <-> Sate: mutual block
        mask[:, :, prompt_start:, :sate_end] = False  # prompt cannot see sate
        mask[:, :, :sate_end, prompt_start:] = False  # sate cannot see prompt
        
        # Prompt <-> Prompt: mutual block
        mask[:, :, prompt_start:, prompt_start:] = False
        # Each prompt can see itself
        for i in range(N_prompt):
            mask[:, :, prompt_start + i, prompt_start + i] = True

        if N_learn > 0:
            # Learnable <-> Prompt: mutual block
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
        """
        Build RoPE positions for prompt tokens.
        
        For point/box prompts, use their normalized coordinates.
        For tokens without spatial meaning, use (0, 0).
        
        Args:
            sparse_embeddings: [B, K, C] prompt embeddings
            prompt_coords: [B, K, 2] optional normalized coordinates
        
        Returns:
            prompt_pos: [B, K, 2] positions for RoPE
        """
        K = sparse_embeddings.shape[1]
        if prompt_coords is not None:
            # Scale normalized [0,1] coords to patch grid coordinates
            # RoPE expects integer-like positions
            pos = prompt_coords.clone()
            pos[:, :, 0] = pos[:, :, 0] * self.num_patches_per_side
            pos[:, :, 1] = pos[:, :, 1] * self.num_patches_per_side
            return pos.to(dtype=dtype)
        else:
            # Default: position (0, 0) for all prompt tokens
            return torch.zeros(B, K, 2, device=device, dtype=dtype)
    
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
        """
        Apply decoder blocks with extra tokens and attention masks.
        
        Args:
            hidden: [B*N, hw, C] encoded features (from DINOv2)
            N: number of views (2: sate, front)
            H, W: image height and width
            sparse_embeddings: [B, K, C_prompt] sparse prompt embeddings (optional)
            dense_embeddings: [B, C, Hp, Wp] dense mask embeddings (optional)
            prompt_coords: [B, K, 2] normalized prompt coordinates (optional)
        
        Returns:
            Dict with:
                - 'features': [B, 2, tokens, 2*C] final features
                - 'learnable_out': [B, num_learnable, 2*C] learnable query outputs
                - 'intermediate': List of intermediate outputs at supervision layers
        """
        BN, hw, C = hidden.shape
        B = BN // N
        patch_h = patch_w = H // self.patch_size
        
        # Reshape to [B, N, hw, C]
        hidden = hidden.reshape(B, N, hw, C)
        sate_hidden = hidden[:, 0]   # [B, hw, C]
        front_hidden = hidden[:, 1]  # [B, hw, C]
        
        # Add register tokens
        reg_token = self.register_token.to(hidden.device, dtype=hidden.dtype)
        reg_token = reg_token.repeat(B, 1, 1, 1)  # [B, 1, 5, C]
        
        sate_hidden = torch.cat([reg_token[:, 0], sate_hidden], dim=1)    # [B, 5+hw, C]
        front_hidden = torch.cat([reg_token[:, 0], front_hidden], dim=1)  # [B, 5+hw, C]
        
        N_sate = sate_hidden.shape[1]  # 5 + 1369 = 1374
        N_front_base = front_hidden.shape[1]  # 1374
        
        # NOTE (V3): dense mask embedding is NOT directly added to front_hidden.
        # It is only injected into K during global attention.
        
        # Prepare learnable query tokens (for front local stream)
        learnable_hidden = self.learnable_queries.expand(B, -1, -1).to(hidden.dtype)  # [B, N_learn, C]
        N_learn = self.num_learnable_tokens

        # Prepare prompt tokens (for satellite local stream)
        N_prompt = 0
        prompt_hidden = None
        prompt_pos = None
        if sparse_embeddings is not None and sparse_embeddings.shape[1] > 0:
            prompt_tokens = sparse_embeddings.to(hidden.dtype)  # [B, K, C]
            # Project if dimensions don't match
            if self.prompt_proj is not None:
                prompt_tokens = self.prompt_proj(prompt_tokens)
            N_prompt = prompt_tokens.shape[1]
            prompt_hidden = prompt_tokens
        
        # Build RoPE positions
        base_pos = self.position_getter(B, patch_h, patch_w, hidden.device)  # [B, hw, 2]
        
        # Add register token positions (at origin)
        if self.patch_start_idx > 0:
            base_pos = base_pos + 1  # shift patch positions by 1
            pos_special = torch.zeros(B, self.patch_start_idx, 2, device=hidden.device, dtype=base_pos.dtype)
            base_pos_with_reg = torch.cat([pos_special, base_pos], dim=1)  # [B, 1374, 2]
        else:
            base_pos_with_reg = base_pos
        
        sate_pos = base_pos_with_reg  # [B, 1374, 2]
        
        # Front pos = base pos + learnable pos (0,0)
        learnable_pos = torch.zeros(B, N_learn, 2, device=hidden.device, dtype=base_pos.dtype)
        front_pos = base_pos_with_reg
        
        if N_prompt > 0:
            prompt_pos = self._build_prompt_positions(
                sparse_embeddings, prompt_coords, B, hidden.device, base_pos.dtype
            )
        
        # Pre-compute global attention mask
        global_mask = self._build_global_attn_mask(
            N_sate, N_front_base, N_learn, N_prompt, hidden.device, hidden.dtype
        )
        
        # Decoder loop
        final_output = []
        intermediate_outputs = {}
        
        dense_flat = None
        if dense_embeddings is not None:
            dense_flat = dense_embeddings.flatten(2).transpose(1, 2).to(hidden.dtype)  # [B,1369,C]
        
        for i in range(len(self.decoder)):
            if i % 2 == 0:
                # ---- Local stage (Pi3 frame attention style): self-attention per frame ----
                sate_local = torch.cat([sate_hidden, learnable_hidden], dim=1)
                sate_local_pos = torch.cat([sate_pos, learnable_pos], dim=1)
                sate_local = self.masked_blocks[i](sate_local, xpos=sate_local_pos)

                front_local = front_hidden
                front_local_pos = front_pos
                if prompt_hidden is not None:
                    front_local = torch.cat([front_local, prompt_hidden], dim=1)
                    front_local_pos = torch.cat([front_local_pos, prompt_pos], dim=1)
                front_local = self.masked_blocks[i](front_local, xpos=front_local_pos)

                # Split back to base/frame-specific token sets
                sate_hidden = sate_local[:, :N_sate]
                learnable_hidden = sate_local[:, N_sate:]

                front_hidden = front_local[:, :N_front_base]
                if prompt_hidden is not None:
                    prompt_hidden = front_local[:, N_front_base:]
            else:
                # ---- Global attention (V4):
                # Q,V,K token layout = (sate, front, learnable, prompt)
                global_qv_tokens = [sate_hidden, front_hidden, learnable_hidden]
                global_pos_tokens = [sate_pos, front_pos, learnable_pos]
                if prompt_hidden is not None:
                    global_qv_tokens.append(prompt_hidden)
                    global_pos_tokens.append(prompt_pos)
                global_qv = torch.cat(global_qv_tokens, dim=1)

                if dense_flat is not None:
                    # Only clone when we need to inject dense mask into K
                    global_k = global_qv.clone()
                    g_front_patch_start = N_sate + self.patch_start_idx
                    g_front_patch_end = g_front_patch_start + self.num_patches
                    global_k[:, g_front_patch_start:g_front_patch_end] += dense_flat
                else:
                    global_k = global_qv  # No mask prompt → K == QV, skip clone

                global_pos = torch.cat(global_pos_tokens, dim=1)
                global_hidden = self.masked_blocks[i].forward_qkv(
                    global_qv,
                    global_k,
                    qpos=global_pos,
                    kvpos=global_pos,
                    attn_mask=global_mask,
                )
                
                # Split back
                g_front_end = N_sate + N_front_base
                g_learn_end = g_front_end + N_learn
                sate_hidden = global_hidden[:, :N_sate]
                front_hidden = global_hidden[:, N_sate:g_front_end]
                learnable_hidden = global_hidden[:, g_front_end:g_learn_end]
                if prompt_hidden is not None:
                    prompt_hidden = global_hidden[:, g_learn_end:]
            
            # Collect intermediate outputs for deep supervision
            # One supervision layer = (local + global), so collect after global block.
            if i in self.supervision_block_indices:
                stage_idx = self.supervision_block_indices[i]

                # Extract learnable query outputs from this stage
                inter_learn = learnable_hidden  # [B, N_learn, C]
                inter_sate = sate_hidden[:, self.patch_start_idx:]   # [B, 1369, C]
                
                # Project from single-layer C to output_dim (2*C)
                proj = self.intermediate_projs[str(stage_idx)]
                out_dict = {
                    'learnable': proj(inter_learn),  # [B, 2, 2*C]
                    'sate_patches': proj(inter_sate),  # [B, 1369, 2*C]
                }
                # front_patches projection is expensive (1369 tokens); only store
                # if this layer actually needs it (extra supervision layers).
                final_stage = max(self.supervision_layers)
                if stage_idx != final_stage:
                    out_dict['front_patches'] = proj(
                        front_hidden[:, self.patch_start_idx:self.patch_start_idx + self.num_patches]
                    )
                intermediate_outputs[stage_idx] = out_dict
            
            # Collect last two layers for final output
            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                # Combine sate and front for final output
                combined_sate = sate_hidden
                combined_front = front_hidden[:, :N_front_base]  # Remove extra tokens for feature concat
                final_output.append(
                    torch.stack([combined_sate, combined_front], dim=1)  # [B, 2, tokens, C]
                )
        
        # Concatenate last two layers: [B, 2, tokens, 2*C]
        features = torch.cat([final_output[0], final_output[1]], dim=-1)
        
        # Extract final learnable query outputs
        learnable_final = self.final_proj(learnable_hidden)
        
        return {
            'features': features,           # [B, 2, 1374, 2*C]
            'sate_features': features[:, 0, self.patch_start_idx:, :],  # [B, 1369, 2*C]
            'front_features': features[:, 1, self.patch_start_idx:, :],  # [B, 1369, 2*C]
            'sate_camera_token': features[:, 0, 0, :],  # [B, 2*C]
            'front_camera_token': features[:, 1, 0, :],  # [B, 2*C]
            'learnable_out': learnable_final,  # [B, 2, 2*C]
            'intermediate': intermediate_outputs,
        }
    
    def forward(
        self,
        front_view: torch.Tensor,
        satellite_view: torch.Tensor,
        sparse_embeddings: Optional[torch.Tensor] = None,
        dense_embeddings: Optional[torch.Tensor] = None,
        prompt_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode images + decode with extra tokens.
        
        Args:
            front_view: [B, 3, H, W] front view image
            satellite_view: [B, 3, H, W] satellite view image
            sparse_embeddings: [B, K, C] sparse prompt embeddings (optional)
            dense_embeddings: [B, C, Hp, Wp] dense mask embeddings (optional)
            prompt_coords: [B, K, 2] normalized prompt coordinates (optional)
        
        Returns:
            Dict with all output features
        """
        # Normalize
        images = torch.stack([satellite_view, front_view], dim=1)  # [B, 2, 3, H, W]
        images = (images - self.image_mean) / self.image_std
        
        B, N, _, H, W = images.shape
        
        # Encode with DINOv2
        images_flat = images.reshape(B * N, 3, H, W)
        target_dtype = self.image_mean.dtype
        if images_flat.dtype != target_dtype:
            images_flat = images_flat.to(target_dtype)
        
        hidden = self.encoder(images_flat, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        
        # Decode with Pi3 decoder + extra tokens
        return self.decode_with_extra_tokens(
            hidden, N, H, W,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            prompt_coords=prompt_coords,
        )
