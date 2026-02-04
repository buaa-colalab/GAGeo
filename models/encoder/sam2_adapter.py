"""
SAM2 Prompt Encoder Adapter
将 SAM2 的 256 维 prompt encoder 适配到 2048 维
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SAM2PromptEncoderAdapter(nn.Module):
    """
    适配器：SAM2 (256-dim) → Your Model (2048-dim)
    
    加载 SAM2 预训练权重，然后通过投影层映射到目标维度
    """
    
    def __init__(
        self,
        sam2_prompt_encoder: nn.Module,
        target_embed_dim: int = 2048,
        target_image_embedding_size: Tuple[int, int] = (37, 37),
    ):
        super().__init__()
        
        # SAM2 的 prompt encoder (冻结)
        self.sam2_encoder = sam2_prompt_encoder
        for param in self.sam2_encoder.parameters():
            param.requires_grad = False
        
        sam2_dim = sam2_prompt_encoder.embed_dim  # 256
        self.target_embed_dim = target_embed_dim
        self.target_image_embedding_size = target_image_embedding_size
        
        # 投影层：256 → 2048
        self.sparse_proj = nn.Sequential(
            nn.Linear(sam2_dim, target_embed_dim // 2),
            nn.LayerNorm(target_embed_dim // 2),
            nn.GELU(),
            nn.Linear(target_embed_dim // 2, target_embed_dim),
        )
        
        # Dense embedding 投影：[B, 256, 64, 64] → [B, 2048, 37, 37]
        self.dense_proj = nn.Sequential(
            nn.Conv2d(sam2_dim, target_embed_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, target_embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(target_embed_dim // 2, target_embed_dim, kernel_size=1),
        )
        
        # 初始化投影层
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (coords [B, N, 2], labels [B, N])
            boxes: [B, M, 4]
            masks: [B, 1, H, W]
        
        Returns:
            sparse_embeddings: [B, N_sparse, 2048]
            dense_embeddings: [B, 2048, 37, 37]
        """
        # Step 1: SAM2 编码 (256-dim)
        with torch.no_grad():
            sparse_256, dense_256 = self.sam2_encoder(points, boxes, masks)
        
        # Step 2: 投影到 2048-dim
        sparse_2048 = self.sparse_proj(sparse_256)  # [B, N, 2048]
        
        # Step 3: Dense 投影 + Resize
        # dense_256: [B, 256, 64, 64] → [B, 2048, 64, 64] → [B, 2048, 37, 37]
        dense_2048 = self.dense_proj(dense_256)
        if dense_2048.shape[2:] != self.target_image_embedding_size:
            dense_2048 = torch.nn.functional.interpolate(
                dense_2048,
                size=self.target_image_embedding_size,
                mode='bilinear',
                align_corners=False
            )
        
        return sparse_2048, dense_2048
    
    @classmethod
    def from_sam2_checkpoint(
        cls,
        sam2_checkpoint_path: str,
        target_embed_dim: int = 2048,
        target_image_embedding_size: Tuple[int, int] = (37, 37),
    ):
        """
        从 SAM2 checkpoint 加载并创建 adapter
        
        Args:
            sam2_checkpoint_path: SAM2 模型权重路径
            target_embed_dim: 目标维度
            target_image_embedding_size: 目标空间尺寸
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # 加载 SAM2 模型
        sam2_model = build_sam2(
            config_file="sam2.1_hiera_l.yaml",  # 或其他配置
            ckpt_path=sam2_checkpoint_path,
        )
        
        # 提取 prompt encoder
        sam2_prompt_encoder = sam2_model.sam_prompt_encoder
        
        # 创建 adapter
        adapter = cls(
            sam2_prompt_encoder=sam2_prompt_encoder,
            target_embed_dim=target_embed_dim,
            target_image_embedding_size=target_image_embedding_size,
        )
        
        return adapter
