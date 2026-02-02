# Task-specific prediction heads for cross-view localization
from .pi3_camera_head import Pi3CameraHead
from .bbox_head import BBoxHead
from .heatmap_head import HeatmapHead

__all__ = [
    'Pi3CameraHead',
    'BBoxHead',
    'HeatmapHead',
]
