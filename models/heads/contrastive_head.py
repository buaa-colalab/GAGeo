"""
MoCo-style Cross-View Contrastive Learning Head

Satellite features as anchor (query), mono features as key.
Uses momentum key encoder + queue following MoCo (https://arxiv.org/abs/1911.05722).

把 satellite 当作 anchor，训练 ground-satellite / drone-satellite 后可泛化到 ground-drone。
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewContrastiveHead(nn.Module):
    """
    MoCo-style contrastive head for cross-view feature alignment.
    
    Pipeline:
    1. Masked Average Pooling: F [B, N, D] + mask -> [B, D]
    2. Query encoder (satellite) / Key encoder (mono, momentum-updated)
    3. InfoNCE loss with momentum queue
    
    Args:
        in_dim: Input feature dimension (e.g. 2048)
        proj_dim: Projection output dimension (default 256)
        queue_size: MoCo queue size (default 16384)
        momentum: EMA momentum for key encoder (default 0.999)
        temperature: InfoNCE temperature (default 0.07)
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        proj_dim: int = 256,
        queue_size: int = 16384,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Query encoder (satellite) — trained by gradient
        self.encoder_q = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )
        
        # Key encoder (mono) — momentum-updated, no gradient
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Queue (stores mono/key projections as negatives)
        self.register_buffer('queue', F.normalize(torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """EMA update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new key embeddings. keys: [B, proj_dim]"""
        # Keep the queue update local by default under DDP. Calling all_gather
        # inside every forward makes the training loop fragile: if any rank is
        # delayed by data loading or skips the contrastive path, all other ranks
        # block in this extra collective. Local queues are sufficient for the
        # MoCo loss and avoid an avoidable NCCL failure point.
        if os.environ.get("GAGEO_CONTRASTIVE_ALL_GATHER", "0").lower() in {"1", "true", "yes", "on"}:
            keys = _concat_all_gather(keys)
        
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Handle wrap-around
        if ptr + batch_size > self.queue_size:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    @staticmethod
    def _masked_avg_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked average pooling over patch tokens.
        
        Args:
            features: [B, N, D] patch features
            mask: [B, 1, H, W] binary mask at image resolution
        Returns:
            pooled: [B, D]
        """
        B, N, D = features.shape
        H_p = W_p = int(N ** 0.5)  # 37
        
        patch_mask = F.adaptive_avg_pool2d(mask.float(), (H_p, W_p))  # [B, 1, H_p, W_p]
        patch_mask = (patch_mask > 0.5).to(dtype=features.dtype).reshape(B, -1)  # [B, N]
        
        # Fallback: if mask is all-zero, use uniform weights
        mask_sum = patch_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        weights = patch_mask / mask_sum  # [B, N]
        
        return (features * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
    
    def forward(
        self,
        mono_features: torch.Tensor,
        sat_features: torch.Tensor,
        mono_mask: torch.Tensor,
        sat_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MoCo-style InfoNCE contrastive loss.
        Satellite = query (anchor), Mono = key (positive), Queue = negatives.
        
        Args:
            mono_features: [B, N, D] front-view patch features
            sat_features: [B, N, D] satellite patch features
            mono_mask: [B, 1, H, W] front-view object mask
            sat_mask: [B, 1, H, W] satellite object mask
        Returns:
            loss: scalar contrastive loss
        """
        # 1. Masked Average Pooling
        sat_pooled = self._masked_avg_pool(sat_features, sat_mask)    # [B, D]
        mono_pooled = self._masked_avg_pool(mono_features, mono_mask)  # [B, D]
        
        # 2. Ensure dtype matches encoder weights (handles DeepSpeed bf16)
        target_dtype = next(self.encoder_q.parameters()).dtype
        sat_pooled = sat_pooled.to(target_dtype)
        mono_pooled = mono_pooled.to(target_dtype)
        
        # 3. Query: satellite through encoder_q (gradient flows)
        q = F.normalize(self.encoder_q(sat_pooled), dim=-1)  # [B, proj_dim]
        
        # 4. Key: mono through encoder_k (no gradient, momentum-updated)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(mono_pooled), dim=-1)  # [B, proj_dim]
        
        # 4. InfoNCE logits
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)                  # [B, 1]
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())      # [B, K]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature           # [B, 1+K]
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        # 5. Update queue with mono keys
        self._dequeue_and_enqueue(k)
        
        return loss


@torch.no_grad()
def _concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all GPUs. No-op if not distributed."""
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)
