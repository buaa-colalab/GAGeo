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
    跨视角定位数据集 - 支持双向定位
    
    双向定位设计:
    - 定位始终在卫星图上做 (heatmap, camera_position, sat_bbox)
    - 根据方向决定 prompt 来源:
      - 'mono_to_sat': prompt 在 mono 图上 (mono_bbox, mono_point, mono_mask)
      - 'sat_to_mono': prompt 在 sat 图上 (sat_bbox, camera_position)
      - 'both': 训练时随机选择方向
    
    输出字段:
    - mono_view, sat_view: 两个视图图像 (始终固定)
    - prompt_point, prompt_bbox, prompt_mask: prompt 信息 (根据方向变化)
    - target_bbox, target_position: 目标信息 (始终在 sat 图上)
    - yaw_radians, yaw_degrees: 相对旋转角度
    - direction: 当前采样的方向
    
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
        test_mode: bool = False,  # 测试模式：图像已经是crop好的518x518
        direction: str = 'mono_to_sat',  # 'mono_to_sat', 'sat_to_mono', 'both'
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
            test_mode: 测试模式，图像已经是crop好的518x518，跳过crop逻辑
            direction: 定位方向 ('mono_to_sat', 'sat_to_mono', 'both')
        """
        self.data_root = Path(data_root)
        self.mono_size = mono_size
        self.sat_size = sat_size
        self.crop_sat = crop_sat
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.transform = transform
        self.test_mode = test_mode
        self.direction = direction
        
        assert direction in ['mono_to_sat', 'sat_to_mono', 'both'], \
            f"direction must be 'mono_to_sat', 'sat_to_mono', or 'both', got {direction}"
        
        # 加载数据
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {json_path}")
        print(f"Direction: {direction}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 决定本次采样的方向
        if self.direction == 'both':
            current_direction = random.choice(['mono_to_sat', 'sat_to_mono'])
        else:
            current_direction = self.direction
        
        # 加载图像
        mono_img = self._load_image(item['city'], 'mono', item['mono_filename'])
        sat_img = self._load_image(item['city'], 'sate', item['sat_filename'])
        
        # 获取标注
        mono_point = np.array(item['mono_point'], dtype=np.float32)
        mono_bbox = np.array(item['mono_bbox'], dtype=np.float32)  # [x, y, w, h]
        sat_point = np.array(item['sate_point'], dtype=np.float32)
        sat_bbox = np.array(item['sate_bbox'], dtype=np.float32)
        
        # 解码segmentation为mask
        mono_mask = self._decode_segmentation(item.get('mono_segmentation'), self.mono_size)
        sat_mask = self._decode_segmentation(item.get('sate_segmentation'), self.sat_size)
        
        # 相机位置和yaw
        camera_position = np.array(item.get('camera_position', [self.sat_size/2, self.sat_size/2]), dtype=np.float32)
        yaw_degrees = float(item['rotation'])
        yaw_radians = np.deg2rad(yaw_degrees)
        
        # Resize mono图像和mask到目标尺寸
        mono_img, mono_point, mono_bbox, mono_mask = self._resize_mono(
            mono_img, mono_point, mono_bbox, mono_mask
        )
        
        # Crop卫星图（测试模式下跳过crop）
        if self.test_mode:
            # 测试模式：图像已经是crop好的，直接使用
            crop_offset = np.array([0, 0], dtype=np.float32)
        elif self.crop_sat:
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
        
        # ============ 根据方向决定 prompt 和 target ============
        # 双向定位:
        # - mono_to_sat: prompt 在 mono 图，target bbox 在 sat 图
        # - sat_to_mono: prompt 在 sat 图，target bbox 在 mono 图
        # 注意: camera_position 始终在 sat 图上（因为卫星图范围更广）
        if current_direction == 'mono_to_sat':
            # prompt 来自 mono 图，在 sat 图中定位
            prompt_point = torch.from_numpy(mono_point)  # 像素坐标
            prompt_bbox = torch.from_numpy(mono_bbox_norm)  # 归一化 [cx, cy, w, h]
            prompt_mask = mono_mask_tensor
            prompt_view = 'mono'
            # 目标 bbox 在 sat 图上
            target_bbox = torch.from_numpy(sat_bbox_norm)
        else:
            # prompt 来自 sat 图，在 mono 图中定位
            sat_point_cropped = self._adjust_sat_coord(sat_point, crop_offset)
            prompt_point = torch.from_numpy(sat_point_cropped)
            prompt_bbox = torch.from_numpy(sat_bbox_norm)
            sat_mask_cropped = self._crop_sat_mask(sat_mask, crop_offset, self.crop_size)
            prompt_mask = torch.from_numpy(sat_mask_cropped).unsqueeze(0).float()
            prompt_view = 'sat'
            target_bbox = torch.from_numpy(mono_bbox_norm)
        
        # camera_position 始终在 sat 图上（无人机拍摄位置）
        target_position = torch.from_numpy(camera_position_norm)
        target_yaw_radians = torch.tensor(yaw_radians, dtype=torch.float32)
        target_yaw_degrees = torch.tensor(yaw_degrees, dtype=torch.float32)
        
        return {
            # 两个视图 (始终固定)
            'mono_view': mono_tensor,
            'sat_view': sat_tensor,
            
            # Prompt 信息 (根据方向变化)
            'prompt_point': prompt_point,
            'prompt_bbox': prompt_bbox,
            'prompt_mask': prompt_mask,
            
            # 目标信息 (target_bbox 根据方向变化，target_position 始终在 sat 图上)
            'target_bbox': target_bbox,
            'target_position': target_position,
            'yaw_radians': target_yaw_radians,
            'yaw_degrees': target_yaw_degrees,
            
            # 元信息
            'direction': current_direction,
            'prompt_view': prompt_view,
            'city': item['city'],
            'mono_filename': item['mono_filename'],
            'sat_filename': item['sat_filename'],
            'crop_offset': torch.from_numpy(crop_offset),
            
        }
    
    def _load_image(self, city: str, view_type: str, filename: str) -> Image.Image:
        """加载图像"""
        img_path = self.data_root / city / view_type / filename
        return Image.open(img_path).convert('RGB')
    
    def _decode_segmentation(self, segmentation, size: int) -> np.ndarray:
        """解码RLE格式的segmentation为mask"""
        if segmentation is None:
            return np.zeros((size, size), dtype=np.uint8)
        
        mask = mask_utils.decode(segmentation)
        return mask.astype(np.uint8)
    
    def _adjust_sat_coord(self, coord: np.ndarray, crop_offset: np.ndarray) -> np.ndarray:
        """调整卫星图坐标以适应 crop（卫星图只 crop 不 resize）"""
        return (coord - crop_offset).astype(np.float32)
    
    def _crop_sat_mask(self, mask: np.ndarray, crop_offset: np.ndarray, crop_size: int) -> np.ndarray:
        """Crop 卫星图 mask（与 _crop_satellite 使用相同的 crop_offset）"""
        left, top = int(crop_offset[0]), int(crop_offset[1])
        return mask[top:top+crop_size, left:left+crop_size].astype(np.uint8)
    
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
        
        # 计算缩放比例并向量化缩放坐标
        scale = np.array([self.mono_size / W, self.mono_size / H])
        adj_point = mono_point * scale
        adj_bbox = mono_bbox * np.tile(scale, 2)  # [sx, sy, sx, sy]
        
        return resized, adj_point, adj_bbox, resized_mask
    
    def _crop_satellite(
        self,
        sat_img: Image.Image,
        sat_bbox: np.ndarray,
        camera_position: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """Crop卫星图，确保bbox和camera_position都在裁剪区域内"""
        W, H = sat_img.size
        cs = min(self.crop_size, W, H)
        
        # 计算bbox的边界 [x, y, w, h] -> [x1, y1, x2, y2]
        bx, by, bw, bh = sat_bbox
        bbox_x1, bbox_y1 = bx, by
        bbox_x2, bbox_y2 = bx + bw, by + bh
        cx, cy = camera_position[0], camera_position[1]
        
        if self.random_crop:
            # 随机crop，确保bbox和camera都在crop区域内
            # 计算有效的crop范围：必须包含bbox和camera
            min_x = min(bbox_x1, cx)
            max_x = max(bbox_x2, cx)
            min_y = min(bbox_y1, cy)
            max_y = max(bbox_y2, cy)
            
            # left范围：[max(0, max_x - cs), min(W - cs, min_x)]
            left_min = max(0, int(np.ceil(max_x)) - cs)
            left_max = min(W - cs, int(np.floor(min_x)))
            # top范围：[max(0, max_y - cs), min(H - cs, min_y)]
            top_min = max(0, int(np.ceil(max_y)) - cs)
            top_max = min(H - cs, int(np.floor(min_y)))
            
            # 如果范围有效则随机选择，否则以中心为准
            if left_min <= left_max:
                left = random.randint(left_min, left_max)
            else:
                left = max(0, min(W - cs, int((min_x + max_x) / 2 - cs / 2)))
            
            if top_min <= top_max:
                top = random.randint(top_min, top_max)
            else:
                top = max(0, min(H - cs, int((min_y + max_y) / 2 - cs / 2)))
        else:
            # 中心crop：以camera为中心，但确保bbox也在内
            left = int(cx - cs / 2)
            top = int(cy - cs / 2)
            
            # 调整确保bbox不被裁剪
            if bbox_x1 < left:
                left = max(0, int(bbox_x1))
            if bbox_x2 > left + cs:
                left = min(W - cs, int(bbox_x2 - cs))
            if bbox_y1 < top:
                top = max(0, int(bbox_y1))
            if bbox_y2 > top + cs:
                top = min(H - cs, int(bbox_y2 - cs))
            
            # 最终clip到有效范围
            left = np.clip(left, 0, W - cs)
            top = np.clip(top, 0, H - cs)
        
        # Crop并resize
        cropped = sat_img.crop((left, top, left + cs, top + cs))
        scale = self.crop_size / cs if cs != self.crop_size else 1.0
        if scale != 1.0:
            cropped = cropped.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        
        # 调整坐标（向量化）
        offset = np.array([left, top], dtype=np.float32)
        adj_bbox = (sat_bbox - np.concatenate([offset, [0, 0]])) * scale
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
    """自定义collate函数"""
    return {
        # 两个视图
        'mono_view': torch.stack([item['mono_view'] for item in batch]),
        'sat_view': torch.stack([item['sat_view'] for item in batch]),
        
        # Prompt 信息
        'prompt_point': torch.stack([item['prompt_point'] for item in batch]),
        'prompt_bbox': torch.stack([item['prompt_bbox'] for item in batch]),
        'prompt_mask': torch.stack([item['prompt_mask'] for item in batch]),
        
        # 目标信息
        'target_bbox': torch.stack([item['target_bbox'] for item in batch]),
        'target_position': torch.stack([item['target_position'] for item in batch]),
        'yaw_radians': torch.stack([item['yaw_radians'] for item in batch]),
        'yaw_degrees': torch.stack([item['yaw_degrees'] for item in batch]),
        
        # 方向信息
        'directions': [item['direction'] for item in batch],
        'prompt_views': [item['prompt_view'] for item in batch],
        
        # 元信息
        'crop_offset': torch.stack([item['crop_offset'] for item in batch]),
        'cities': [item['city'] for item in batch],
        'mono_filenames': [item['mono_filename'] for item in batch],
        'sat_filenames': [item['sat_filename'] for item in batch],
    }


if __name__ == '__main__':
    # 测试
    dataset = CrossViewDataset(
        json_path='/data/xhj/location/data/single.json',
        crop_sat=True,
        random_crop=False,
        direction='both',
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试第一个样本
    sample = dataset[0]
    print("\nSample 0:")
    print(f"  Mono view: {sample['mono_view'].shape}")
    print(f"  Sat view: {sample['sat_view'].shape}")
    print(f"  Direction: {sample['direction']}")
    print(f"  Prompt view: {sample['prompt_view']}")
    print(f"  Target bbox: {sample['target_bbox']}")
    print(f"  Target position: {sample['target_position']}")
    print(f"  Yaw (degrees): {sample['yaw_degrees']:.1f}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    batch = next(iter(loader))
    print("\nBatch:")
    print(f"  Mono views: {batch['mono_view'].shape}")
    print(f"  Sat views: {batch['sat_view'].shape}")
    print(f"  Directions: {batch['directions']}")
    print(f"  Prompt views: {batch['prompt_views']}")
