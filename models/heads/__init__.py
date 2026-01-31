# Task-specific prediction heads for cross-view localization
from .yaw_head import CameraHead, YawHead

__all__ = [
    'CameraHead',
    'YawHead',
]
