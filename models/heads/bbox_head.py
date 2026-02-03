# BBox Prediction Head for DETR-style detection

import torch
import torch.nn as nn
from typing import Dict

from ..layers.mlp import MLP


class BBoxHead(nn.Module):
    """
    DETR-style BBox prediction head.
    
    Takes object query outputs from decoder and predicts:
    - Bounding boxes (cx, cy, w, h) normalized to [0, 1]
    - Confidence scores
    
    Args:
        hidden_dim: Input feature dimension
        num_classes: Number of object classes (1 for single-object detection)
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_classes: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 3-layer MLP for bbox regression (DETR-style)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Classification head (confidence score)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, query_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_features: [B, N_queries, C] decoder output for object queries
            
        Returns:
            Dict with:
                - pred_boxes: [B, N, 4] normalized (cx, cy, w, h)
                - bbox_scores: [B, N] confidence scores
                - class_logits: [B, N, num_classes] raw logits
        """
        # Predict boxes
        pred_boxes = self.bbox_embed(query_features).sigmoid()  # [B, N, 4]
        
        # Predict class/confidence
        class_logits = self.class_embed(query_features)  # [B, N, num_classes]
        
        # For single-class detection, squeeze to get scores
        if self.num_classes == 1:
            bbox_scores = class_logits.squeeze(-1).sigmoid()  # [B, N]
        else:
            bbox_scores = class_logits.softmax(dim=-1).max(dim=-1)[0]  # [B, N]
        
        return {
            'pred_boxes': pred_boxes,
            'bbox_scores': bbox_scores,
            'class_logits': class_logits,
        }
