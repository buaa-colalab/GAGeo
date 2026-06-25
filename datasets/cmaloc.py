"""CMA-Loc dataset loader with satellite cropping and coordinate transforms."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CrossViewDataset(Dataset):
    """Cross-view localization dataset for CMA-Loc annotations."""
    
    def __init__(
        self,
        json_path: str,
        data_root: str = "/data/GoogleEarth",
        mono_size: int = 518,
        sat_size: int = 1280,
        crop_sat: bool = True,
        crop_size: int = 518,
        view_subset: str = "all",
        transform: Optional[callable] = None,
    ):
        """
        Args:
            json_path: Path to a CMA-Loc annotation JSON file.
            data_root: Root directory containing city image folders.
            mono_size: Front-view input size.
            sat_size: Raw satellite image size.
            crop_sat: True for random training crops from `sate/`; false for
                pre-cropped evaluation images from `crop_sate/`.
            crop_size: Satellite crop size after cropping/resizing.
            view_subset: all, drone_to_satellite, or ground_to_satellite.
            transform: Optional extra transform hook.
        """
        self.data_root = Path(data_root)
        self.mono_size = mono_size
        self.sat_size = sat_size
        self.crop_sat = crop_sat
        self.crop_size = crop_size
        self.view_subset = self._normalize_view_subset(view_subset)
        self.transform = transform
        
        # Load annotations.
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        total_before_filter = len(self.data)
        if self.view_subset != "all":
            self.data = [item for item in self.data if self._sample_matches_subset(item, self.view_subset)]

        if len(self.data) == 0:
            raise ValueError(
                f"No samples left after filtering with view_subset={self.view_subset!r} "
                f"from {json_path}. Please check dataset annotations and filter settings."
            )

        print(
            f"Loaded {len(self.data)} / {total_before_filter} samples from {json_path} "
            f"(view_subset={self.view_subset})"
        )
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def _normalize_view_subset(view_subset: str) -> str:
        key = str(view_subset).strip().lower().replace("-", "_")
        alias = {
            "all": "all",
            "both": "all",
            "mono_to_sat": "all",
            "drone": "drone_to_satellite",
            "d2s": "drone_to_satellite",
            "drone_to_sat": "drone_to_satellite",
            "drone_to_satellite": "drone_to_satellite",
            "ground": "ground_to_satellite",
            "g2s": "ground_to_satellite",
            "ground_to_sat": "ground_to_satellite",
            "ground_to_satellite": "ground_to_satellite",
        }
        if key not in alias:
            raise ValueError(
                f"Unsupported view_subset={view_subset!r}. "
                f"Use one of: all, drone_to_satellite (d2s), ground_to_satellite (g2s)."
            )
        return alias[key]

    @staticmethod
    def _is_drone_sample(item: Dict) -> bool:
        mono_filename = str(item.get('mono_filename', '')).lower()
        return 'drone' in mono_filename

    @classmethod
    def _sample_matches_subset(cls, item: Dict, view_subset: str) -> bool:
        if view_subset == "all":
            return True
        is_drone = cls._is_drone_sample(item)
        if view_subset == "drone_to_satellite":
            return is_drone
        if view_subset == "ground_to_satellite":
            return not is_drone
        return True
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        sat_filename = item['sat_filename']
        
        # Load images.
        mono_img = self._load_image(item['city'], 'mono', item['mono_filename'])
        if self.crop_sat:
            # Training mode: load raw satellite images.
            sat_img = self._load_image(item['city'], 'sate', sat_filename)
        else:
            # Evaluation mode: load pre-cropped satellite images.
            sat_img = self._load_image(item['city'], 'crop_sate', sat_filename)
        
        # Load annotations.
        mono_point = np.array(item['mono_point'], dtype=np.float32)
        mono_bbox = np.array(item['mono_bbox'], dtype=np.float32)  # [x, y, w, h]
        sat_point = np.array(item['sate_point'], dtype=np.float32)
        sat_bbox = np.array(item['sate_bbox'], dtype=np.float32)
        
        # Decode masks.
        mono_mask = self._decode_segmentation(item['mono_segmentation'], self.mono_size)
        if self.crop_sat:
            # Training masks match raw satellite images.
            sat_mask = self._decode_segmentation(item.get('sate_segmentation'), self.sat_size)
        else:
            # Evaluation masks are already cropped.
            sat_mask = self._decode_segmentation(item.get('sate_segmentation'), self.crop_size)
        
        # Camera position and relative pose in radians.
        camera_position = np.array(item.get('camera_position', [self.sat_size/2, self.sat_size/2]), dtype=np.float32)
        yaw = np.deg2rad(float(item['relative_yaw']))
        if 'drone' in item['mono_filename']:
            pitch = np.deg2rad(float(item.get('relative_pitch', 45.0)))
            roll = np.deg2rad(float(item.get('relative_roll', 0.0)))
        else:
            pitch = np.deg2rad(float(item.get('relative_pitch', 90.0)))
            roll = np.deg2rad(float(item.get('relative_roll', 0.0)))
        
        # Build target rotation matrix with ZYX convention.
        rotation_matrix = self._euler_to_rotation_matrix(yaw, pitch, roll)
        
        # Resize front-view image and annotations.
        mono_img, mono_point, mono_bbox, mono_mask = self._resize_mono(
            mono_img, mono_point, mono_bbox, mono_mask
        )
        mono_bbox = self._clip_bbox_xywh(mono_bbox, self.mono_size, self.mono_size)
        
        # Crop satellite image for training.
        if self.crop_sat:
            # Random crop from raw satellite image.
            sat_img, sat_bbox, camera_position, crop_offset = self._crop_satellite(
                sat_img, sat_bbox, camera_position
            )
            # Apply the same crop to the satellite mask.
            sat_mask = self._crop_sat_mask(sat_mask, crop_offset, self.crop_size)
        else:
            # Evaluation image is already cropped.
            crop_offset = np.array([0, 0], dtype=np.float32)
            sat_img, sat_bbox, camera_position, sat_mask = self._resize_satellite_eval(
                sat_img, sat_bbox, camera_position, sat_mask
            )
            sat_bbox = self._clip_bbox_xywh(sat_bbox, sat_img.size[0], sat_img.size[1])
        
        # Convert images to normalized tensors.
        mono_tensor = self._to_tensor(mono_img)
        sat_tensor = self._to_tensor(sat_img)
        
        # Normalize boxes to [0, 1] in cxcywh format.
        mono_bbox_norm = self._normalize_bbox(mono_bbox, self.mono_size, self.mono_size)
        sat_bbox_norm = self._normalize_bbox(sat_bbox, sat_tensor.shape[2], sat_tensor.shape[1])
        
        # Normalize camera position to [0, 1].
        camera_position_norm = camera_position / np.array([sat_tensor.shape[2], sat_tensor.shape[1]], dtype=np.float32)
        
        # Convert masks to tensors.
        mono_mask_tensor = torch.from_numpy(mono_mask).unsqueeze(0).float()  # [1, H, W]
        sat_mask_tensor = torch.from_numpy(sat_mask).unsqueeze(0).float()  # [1, H, W]
        
        return {
            'front_view': mono_tensor,
            'satellite_view': sat_tensor,
            'mono_point': torch.from_numpy(mono_point),
            'mono_bbox': torch.from_numpy(mono_bbox_norm),
            'mono_mask': mono_mask_tensor,
            'sat_mask': sat_mask_tensor,
            'sat_bbox': torch.from_numpy(sat_bbox_norm),
            'camera_position': torch.from_numpy(camera_position_norm),
            'rotation_matrix': torch.from_numpy(rotation_matrix),
            'yaw': torch.tensor(yaw, dtype=torch.float32),
            'pitch': torch.tensor(pitch, dtype=torch.float32),
            'roll': torch.tensor(roll, dtype=torch.float32),
            'city': item['city'],
            'mono_filename': item['mono_filename'],
            'sat_filename': item['sat_filename'],
            'crop_offset': torch.from_numpy(crop_offset),
        }
    
    @staticmethod
    def _euler_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        """ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [  -sp,           cp*sr,           cp*cr    ],
        ], dtype=np.float32)
        return R
    
    def _load_image(self, city: str, view_type: str, filename: str) -> Image.Image:
        """Load an RGB image from the dataset root."""
        img_path = self.data_root / city / view_type / filename
        return Image.open(img_path).convert('RGB')
    
    def _decode_segmentation(self, segmentation, size: int) -> np.ndarray:
        """Decode an RLE segmentation into a binary mask."""
        if segmentation is None:
            return np.zeros((size, size), dtype=np.uint8)
        
        # CMA-Loc stores masks as COCO RLE.
        mask = mask_utils.decode(segmentation)
        return mask.astype(np.uint8)
    
    def _crop_sat_mask(self, mask: np.ndarray, crop_offset: np.ndarray, crop_size: int) -> np.ndarray:
        """Crop the satellite mask with the same offset as the image crop."""
        left, top = int(crop_offset[0]), int(crop_offset[1])
        return mask[top:top+crop_size, left:left+crop_size].astype(np.uint8)

    @staticmethod
    def _clip_bbox_xywh(bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """Clip an xywh box to the visible image region before normalization."""
        x, y, w, h = bbox.astype(np.float32)
        x1 = float(np.clip(x, 0.0, float(img_w)))
        y1 = float(np.clip(y, 0.0, float(img_h)))
        x2 = float(np.clip(x + w, 0.0, float(img_w)))
        y2 = float(np.clip(y + h, 0.0, float(img_h)))

        if x2 <= x1:
            cx = float(np.clip(x + 0.5 * w, 0.0, float(img_w - 1)))
            x1 = max(0.0, cx - 0.5)
            x2 = min(float(img_w), cx + 0.5)
        if y2 <= y1:
            cy = float(np.clip(y + 0.5 * h, 0.0, float(img_h - 1)))
            y1 = max(0.0, cy - 0.5)
            y2 = min(float(img_h), cy + 0.5)

        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    
    def _resize_mono(
        self,
        mono_img: Image.Image,
        mono_point: np.ndarray,
        mono_bbox: np.ndarray,
        mono_mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """Resize the front-view image and transform point, box, and mask."""
        W, H = mono_img.size
        
        # Return directly when the image already has the target size.
        if W == self.mono_size and H == self.mono_size:
            return mono_img, mono_point, mono_bbox, mono_mask
        
        # Resize image.
        resized = mono_img.resize((self.mono_size, self.mono_size), Image.BILINEAR)
        
        # Resize mask with nearest-neighbor interpolation to preserve labels.
        resized_mask = cv2.resize(
            mono_mask, 
            (self.mono_size, self.mono_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Scale point and box coordinates.
        scale = np.array([self.mono_size / W, self.mono_size / H])
        adj_point = mono_point * scale
        adj_bbox = mono_bbox * np.tile(scale, 2)  # [sx, sy, sx, sy]
        
        return resized, adj_point, adj_bbox, resized_mask

    def _resize_satellite_eval(
        self,
        sat_img: Image.Image,
        sat_bbox: np.ndarray,
        camera_position: np.ndarray,
        sat_mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """Resize pre-cropped val/test satellite samples to the configured crop size."""
        W, H = sat_img.size
        if W == self.crop_size and H == self.crop_size:
            return sat_img, sat_bbox, camera_position, sat_mask

        resized = sat_img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        resized_mask = cv2.resize(
            sat_mask,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_NEAREST,
        )
        scale = np.array([self.crop_size / W, self.crop_size / H], dtype=np.float32)
        adj_bbox = sat_bbox * np.tile(scale, 2)
        adj_pos = camera_position * scale
        return resized, adj_bbox, adj_pos, resized_mask
    
    def _crop_satellite(
        self,
        sat_img: Image.Image,
        sat_bbox: np.ndarray,
        camera_position: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """Crop the satellite image while keeping the target and camera visible."""
        W, H = sat_img.size
        cs = min(self.crop_size, W, H)
        
        # Convert bbox from xywh to xyxy.
        bx, by, bw, bh = sat_bbox
        bbox_x1, bbox_y1 = bx, by
        bbox_x2, bbox_y2 = bx + bw, by + bh
        cx, cy = camera_position[0], camera_position[1]
        
        # Valid crop range must include both bbox and camera position.
        min_x = min(bbox_x1, cx)
        max_x = max(bbox_x2, cx)
        min_y = min(bbox_y1, cy)
        max_y = max(bbox_y2, cy)
        
        # left range: [max(0, max_x - cs), min(W - cs, min_x)]
        left_min = max(0, int(np.ceil(max_x)) - cs)
        left_max = min(W - cs, int(np.floor(min_x)))
        # top range: [max(0, max_y - cs), min(H - cs, min_y)]
        top_min = max(0, int(np.ceil(max_y)) - cs)
        top_max = min(H - cs, int(np.floor(min_y)))
        
        # Randomly sample a valid crop, or fall back to a centered crop.
        if left_min <= left_max:
            left = random.randint(left_min, left_max)
        else:
            left = max(0, min(W - cs, int((min_x + max_x) / 2 - cs / 2)))
        
        if top_min <= top_max:
            top = random.randint(top_min, top_max)
        else:
            top = max(0, min(H - cs, int((min_y + max_y) / 2 - cs / 2)))

        
        # Crop and resize.
        cropped = sat_img.crop((left, top, left + cs, top + cs))
        scale = self.crop_size / cs if cs != self.crop_size else 1.0
        if scale != 1.0:
            cropped = cropped.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        
        # Transform coordinates into crop space.
        offset = np.array([left, top], dtype=np.float32)
        adj_bbox = (sat_bbox - np.concatenate([offset, [0, 0]])) * scale
        adj_pos = (camera_position - offset) * scale
        adj_bbox = self._clip_bbox_xywh(adj_bbox, self.crop_size, self.crop_size)
        
        return cropped, adj_bbox, adj_pos, offset
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a float tensor in [0, 1]."""
        return TF.to_tensor(img)
    
    def _normalize_bbox(self, bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """
        Normalize an xywh box to cxcywh in [0, 1].
        
        Args:
            bbox: [x, y, w, h] pixel coordinates.
            img_w, img_h: image size.
        
        Returns:
            normalized_bbox: [cx, cy, w, h] in [0, 1].
        """
        x, y, w, h = bbox
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        return np.array([cx, cy, w_norm, h_norm], dtype=np.float32)


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate fixed-shape tensors and sample metadata."""
    # Stack all tensor fields.
    front_views = torch.stack([item['front_view'] for item in batch])
    satellite_views = torch.stack([item['satellite_view'] for item in batch])
    mono_points = torch.stack([item['mono_point'] for item in batch])
    mono_bboxes = torch.stack([item['mono_bbox'] for item in batch])
    mono_masks = torch.stack([item['mono_mask'] for item in batch])
    sat_masks = torch.stack([item['sat_mask'] for item in batch])
    sat_bboxes = torch.stack([item['sat_bbox'] for item in batch])
    camera_positions = torch.stack([item['camera_position'] for item in batch])
    rotation_matrices = torch.stack([item['rotation_matrix'] for item in batch])
    yaws = torch.stack([item['yaw'] for item in batch])
    pitches = torch.stack([item['pitch'] for item in batch])
    rolls = torch.stack([item['roll'] for item in batch])
    crop_offsets = torch.stack([item['crop_offset'] for item in batch])
    
    return {
        'front_view': front_views,
        'satellite_view': satellite_views,
        'mono_point': mono_points,
        'mono_bbox': mono_bboxes,
        'mono_mask': mono_masks,
        'sat_mask': sat_masks,
        'sat_bbox': sat_bboxes,
        'camera_position': camera_positions,
        'rotation_matrix': rotation_matrices,
        'yaw': yaws,
        'pitch': pitches,
        'roll': rolls,
        'crop_offset': crop_offsets,
        'cities': [item['city'] for item in batch],
        'mono_filenames': [item['mono_filename'] for item in batch],
        'sat_filenames': [item['sat_filename'] for item in batch],
    }
