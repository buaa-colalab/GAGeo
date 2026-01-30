"""
Cross-View Localization Dataset

支持crop数据增强，自动调整bbox和camera position
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CrossViewDataset(Dataset):
    """
    跨视角定位数据集
    
    数据格式:
    {
        "city": "London",
        "mono_filename": "...",
        "mono_point": [x, y],
        "mono_bbox": [x, y, w, h],
        "mono_segmentation": {...},  # RLE or polygon
        "sat_filename": "...",
        "sate_bbox": [x, y, w, h],
        "sate_segmentation": [...],
        "rotation": yaw_degrees,
        "camera_position": [x, y]  # 相机在卫星图中的位置
    }
    """
    
    def __init__(
        self,
        json_path: str,
        data_root: str = "/data/GoogleEarth",
        mono_size: int = 518,
        sat_size: int = 1280,
        crop_sat: bool = True,
        crop_size: int = 518,
        random_crop: bool = True,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            json_path: JSON数据文件路径
            data_root: 图像根目录
            mono_size: 单目图尺寸
            sat_size: 卫星图原始尺寸
            crop_sat: 是否crop卫星图
            crop_size: crop后的尺寸
            random_crop: 训练时随机crop，测试时中心crop
            transform: 额外的数据增强
        """
        self.data_root = Path(data_root)
        self.mono_size = mono_size
        self.sat_size = sat_size
        self.crop_sat = crop_sat
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.transform = transform
        
        # 加载数据
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 加载图像
        mono_img = self._load_image(item['city'], 'mono', item['mono_filename'])
        sat_img = self._load_image(item['city'], 'sate', item['sat_filename'])
        
        # 获取标注
        mono_point = np.array(item['mono_point'], dtype=np.float32)
        mono_bbox = np.array(item['mono_bbox'], dtype=np.float32)  # [x, y, w, h]
        sat_bbox = np.array(item['sate_bbox'], dtype=np.float32)
        
        # 解码segmentation为mask
        mono_mask = self._decode_segmentation(item['mono_segmentation'])
        
        # 相机位置和yaw
        camera_position = np.array(item.get('camera_position', [self.sat_size/2, self.sat_size/2]), dtype=np.float32)
        yaw_degrees = float(item['rotation'])
        yaw_radians = np.deg2rad(yaw_degrees)
        
        # Resize mono图像和mask到目标尺寸
        mono_img, mono_point, mono_bbox, mono_mask = self._resize_mono(
            mono_img, mono_point, mono_bbox, mono_mask
        )
        
        # Crop卫星图（不再处理sat_mask）
        if self.crop_sat:
            sat_img, sat_bbox, camera_position, crop_offset = self._crop_satellite(
                sat_img, sat_bbox, camera_position
            )
        else:
            crop_offset = np.array([0, 0], dtype=np.float32)
        
        # 转换为tensor并归一化
        mono_tensor = self._to_tensor(mono_img)
        sat_tensor = self._to_tensor(sat_img)
        
        # 归一化bbox到[0, 1]
        mono_bbox_norm = self._normalize_bbox(mono_bbox, self.mono_size, self.mono_size)
        sat_bbox_norm = self._normalize_bbox(sat_bbox, sat_tensor.shape[2], sat_tensor.shape[1])
        
        # 归一化camera position到[0, 1]
        camera_position_norm = camera_position / np.array([sat_tensor.shape[2], sat_tensor.shape[1]], dtype=np.float32)
        
        # 转换mono_mask为tensor
        mono_mask_tensor = torch.from_numpy(mono_mask).unsqueeze(0).float()  # [1, H, W]
        
        return {
            'front_view': mono_tensor,
            'satellite_view': sat_tensor,
            'mono_point': torch.from_numpy(mono_point),
            'mono_bbox': torch.from_numpy(mono_bbox_norm),
            'mono_mask': mono_mask_tensor,
            'sat_bbox': torch.from_numpy(sat_bbox_norm),
            'camera_position': torch.from_numpy(camera_position_norm),
            'yaw_radians': torch.tensor(yaw_radians, dtype=torch.float32),
            'yaw_degrees': torch.tensor(yaw_degrees, dtype=torch.float32),
            'city': item['city'],
            'mono_filename': item['mono_filename'],
            'sat_filename': item['sat_filename'],
            'crop_offset': torch.from_numpy(crop_offset),
        }
    
    def _load_image(self, city: str, view_type: str, filename: str) -> Image.Image:
        """加载图像"""
        img_path = self.data_root / city / view_type / filename
        return Image.open(img_path).convert('RGB')
    
    def _decode_segmentation(self, segmentation) -> np.ndarray:
        """解码segmentation为mask"""
        if segmentation is None:
            # 如果没有segmentation，返回全零mask
            return np.zeros((self.mono_size, self.mono_size), dtype=np.uint8)
        
        if isinstance(segmentation, dict) and 'counts' in segmentation:
            # RLE格式
            mask = mask_utils.decode(segmentation)
            return mask.astype(np.uint8)
        elif isinstance(segmentation, list) and len(segmentation) > 0:
            # Polygon格式，转换为mask
            # 获取图像尺寸
            if 'size' in segmentation:
                h, w = segmentation['size']
            else:
                # 从polygon坐标推断尺寸
                polygon = segmentation[0] if isinstance(segmentation[0], list) else segmentation
                coords = np.array(polygon).reshape(-1, 2)
                h = int(coords[:, 1].max()) + 1
                w = int(coords[:, 0].max()) + 1
            
            # 创建空mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # 绘制polygon
            for poly in (segmentation if isinstance(segmentation[0], list) else [segmentation]):
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            
            return mask
        else:
            # 未知格式，返回全零mask
            return np.zeros((self.mono_size, self.mono_size), dtype=np.uint8)
    
    def _resize_mono(
        self,
        mono_img: Image.Image,
        mono_point: np.ndarray,
        mono_bbox: np.ndarray,
        mono_mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """Resize mono图像和mask到目标尺寸，调整坐标"""
        W, H = mono_img.size
        
        # 如果已经是目标尺寸，直接返回
        if W == self.mono_size and H == self.mono_size:
            return mono_img, mono_point, mono_bbox, mono_mask
        
        # Resize图像
        resized = mono_img.resize((self.mono_size, self.mono_size), Image.BILINEAR)
        
        # Resize mask (使用最近邻插值保持二值性)
        resized_mask = cv2.resize(
            mono_mask, 
            (self.mono_size, self.mono_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # 计算缩放比例
        scale_x = self.mono_size / W
        scale_y = self.mono_size / H
        
        # 调整坐标
        adj_point = mono_point.copy()
        adj_point[0] = mono_point[0] * scale_x
        adj_point[1] = mono_point[1] * scale_y
        
        adj_bbox = mono_bbox.copy()
        adj_bbox[0] = mono_bbox[0] * scale_x  # x
        adj_bbox[1] = mono_bbox[1] * scale_y  # y
        adj_bbox[2] = mono_bbox[2] * scale_x  # width
        adj_bbox[3] = mono_bbox[3] * scale_y  # height
        
        return resized, adj_point, adj_bbox, resized_mask
    
    def _crop_satellite(
        self,
        sat_img: Image.Image,
        sat_bbox: np.ndarray,
        camera_position: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """Crop卫星图，调整bbox和camera position"""
        W, H = sat_img.size
        cs = min(self.crop_size, W, H)  # actual crop size
        cx, cy = int(camera_position[0]), int(camera_position[1])
        
        if self.random_crop:
            # 随机crop，确保相机在crop区域内
            left = random.randint(
                max(0, cx - cs + 1),
                min(W - cs, cx)
            ) if cx < W - 1 else max(0, W - cs)
            top = random.randint(
                max(0, cy - cs + 1),
                min(H - cs, cy)
            ) if cy < H - 1 else max(0, H - cs)
        else:
            # 以相机位置为中心crop
            left = np.clip(cx - cs // 2, 0, W - cs)
            top = np.clip(cy - cs // 2, 0, H - cs)
        
        # Crop图像
        cropped = sat_img.crop((left, top, left + cs, top + cs))
        
        scale = 1.0
        if cs != self.crop_size:
            # Resize图像
            cropped = cropped.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            scale = self.crop_size / cs
        
        # 调整坐标
        offset = np.array([left, top], dtype=np.float32)
        adj_bbox = sat_bbox.copy()
        adj_bbox[:2] = (adj_bbox[:2] - offset) * scale
        adj_bbox[2:] = adj_bbox[2:] * scale
        adj_pos = (camera_position - offset) * scale
        
        return cropped, adj_bbox, adj_pos, offset
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """PIL Image转tensor并归一化到[0, 1]"""
        return TF.to_tensor(img)
    
    def _normalize_bbox(self, bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """
        归一化bbox到[0, 1]
        
        Args:
            bbox: [x, y, w, h] 像素坐标
            img_w, img_h: 图像尺寸
        
        Returns:
            normalized_bbox: [cx, cy, w, h] 归一化到[0, 1]
        """
        x, y, w, h = bbox
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        return np.array([cx, cy, w_norm, h_norm], dtype=np.float32)


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义collate函数，处理变长数据
    """
    # 简单stack所有tensor
    front_views = torch.stack([item['front_view'] for item in batch])
    satellite_views = torch.stack([item['satellite_view'] for item in batch])
    mono_points = torch.stack([item['mono_point'] for item in batch])
    mono_bboxes = torch.stack([item['mono_bbox'] for item in batch])
    mono_masks = torch.stack([item['mono_mask'] for item in batch])
    sat_bboxes = torch.stack([item['sat_bbox'] for item in batch])
    camera_positions = torch.stack([item['camera_position'] for item in batch])
    yaw_radians = torch.stack([item['yaw_radians'] for item in batch])
    yaw_degrees = torch.stack([item['yaw_degrees'] for item in batch])
    crop_offsets = torch.stack([item['crop_offset'] for item in batch])
    
    return {
        'front_view': front_views,
        'satellite_view': satellite_views,
        'mono_point': mono_points,
        'mono_bbox': mono_bboxes,
        'mono_mask': mono_masks,
        'sat_bbox': sat_bboxes,
        'camera_position': camera_positions,
        'yaw_radians': yaw_radians,
        'yaw_degrees': yaw_degrees,
        'crop_offset': crop_offsets,
        'cities': [item['city'] for item in batch],
        'mono_filenames': [item['mono_filename'] for item in batch],
        'sat_filenames': [item['sat_filename'] for item in batch],
    }


if __name__ == '__main__':
    # 测试
    dataset = CrossViewDataset(
        json_path='/data/xhj/location/data/test_samples.json',
        crop_sat=True,
        random_crop=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试第一个样本
    sample = dataset[0]
    print("\nSample 0:")
    print(f"  Front view: {sample['front_view'].shape}")
    print(f"  Satellite view: {sample['satellite_view'].shape}")
    print(f"  Mono bbox: {sample['mono_bbox']}")
    print(f"  Sat bbox: {sample['sat_bbox']}")
    print(f"  Camera position: {sample['camera_position']}")
    print(f"  Yaw (degrees): {sample['yaw_degrees']:.1f}")
    print(f"  Crop offset: {sample['crop_offset']}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    batch = next(iter(loader))
    print("\nBatch:")
    print(f"  Front views: {batch['front_view'].shape}")
    print(f"  Satellite views: {batch['satellite_view'].shape}")
    print(f"  Camera positions: {batch['camera_position'].shape}")
    print(f"  Yaw radians: {batch['yaw_radians'].shape}")
