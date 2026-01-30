#!/usr/bin/env python3
"""
合并cesium output目录下的cleaned_final_dataset.json文件
转换为location项目所需的格式
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from pycocotools import mask as mask_utils
import cv2


def rle_to_polygon(rle_dict: Dict) -> List[List[float]]:
    """
    将RLE格式转换为polygon格式
    使用pycocotools解码RLE，然后提取轮廓
    """
    try:
        # 解码RLE为mask
        mask = mask_utils.decode(rle_dict)
        
        # 使用OpenCV提取轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # 取最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 转换为polygon格式 [x1, y1, x2, y2, ...]
        polygon = largest_contour.reshape(-1).tolist()
        
        # 如果点太多，进行简化（保留关键点）
        if len(polygon) > 100:  # 超过50个点
            epsilon = 2.0  # 简化参数
            simplified = cv2.approxPolyDP(
                largest_contour, 
                epsilon, 
                closed=True
            )
            polygon = simplified.reshape(-1).tolist()
        
        return [polygon]
    except Exception as e:
        print(f"    警告: RLE转polygon失败: {e}")
        return None


def bbox_to_polygon(bbox: List[float]) -> List[List[float]]:
    """
    从bbox生成polygon格式
    bbox: [x, y, w, h]
    返回: [[x1, y1, x2, y2, ...]] 格式的polygon
    """
    x, y, w, h = bbox
    # 顺时针方向的四个角点
    polygon = [
        x + w/2, y,           # 上中
        x, y + h/2,           # 左中
        x + w/2, y + h,       # 下中
        x + w, y + h/2,       # 右中
    ]
    return [polygon]


def calculate_bbox_center(bbox: List[float]) -> List[float]:
    """
    计算bbox的中心点
    bbox: [x, y, w, h]
    返回: [cx, cy]
    """
    x, y, w, h = bbox
    return [x + w / 2, y + h / 2]


def convert_sample(sample: Dict, city: str) -> Dict:
    """
    将cesium格式转换为location项目格式
    
    输入格式:
    {
        "mono_filename": "...",
        "sate_filename": "...",
        "mono_bbox": [x1, y1, x2, y2],  # 两个角点坐标
        "mono_segmentation": {RLE},
        "sate_bbox": [x1, y1, x2, y2],  # 两个角点坐标
        "sate_segmentation": {RLE},
        "rotation": yaw_degrees
    }
    
    输出格式:
    {
        "city": "...",
        "mono_filename": "...",
        "mono_point": [cx, cy],
        "mono_bbox": [x, y, w, h],  # 左上角+宽高
        "mono_segmentation": {RLE},
        "sat_filename": "...",
        "sate_bbox": [x, y, w, h],  # 左上角+宽高
        "sate_segmentation": [[x1, y1, x2, y2, ...]],  # polygon格式
        "rotation": yaw_degrees,
        "camera_position": [640.0, 640.0]
    }
    """
    # 转换mono_bbox: [x1, y1, x2, y2] -> [x, y, w, h]
    mono_bbox_input = sample['mono_bbox']
    mono_bbox = [
        mono_bbox_input[0],  # x (左上角)
        mono_bbox_input[1],  # y (左上角)
        mono_bbox_input[2] - mono_bbox_input[0],  # width
        mono_bbox_input[3] - mono_bbox_input[1]   # height
    ]
    
    # 转换sate_bbox: [x1, y1, x2, y2] -> [x, y, w, h]
    sate_bbox_input = sample['sate_bbox']
    sate_bbox = [
        sate_bbox_input[0],  # x (左上角)
        sate_bbox_input[1],  # y (左上角)
        sate_bbox_input[2] - sate_bbox_input[0],  # width
        sate_bbox_input[3] - sate_bbox_input[1]   # height
    ]
    
    # 计算mono_point (bbox中心)
    mono_point = calculate_bbox_center(mono_bbox)
    
    # 转换sate_segmentation: RLE -> polygon (使用pycocotools)
    sate_segmentation = rle_to_polygon(sample['sate_segmentation'])
    
    # 如果RLE转换失败，使用bbox生成简单polygon
    if sate_segmentation is None:
        sate_segmentation = bbox_to_polygon(sate_bbox)
    
    # 构建输出格式
    converted = {
        "city": city,
        "mono_filename": sample['mono_filename'],
        "mono_point": mono_point,
        "mono_bbox": mono_bbox,
        "mono_segmentation": sample['mono_segmentation'],  # 保持RLE格式
        "sat_filename": sample['sate_filename'],
        "sate_bbox": sate_bbox,
        "sate_segmentation": sate_segmentation,
        "rotation": sample['rotation'],
        "camera_position": [640.0, 640.0]  # 默认值，假设卫星图是1280x1280，相机在中心
    }
    
    return converted


def merge_datasets(output_dir: Path, output_file: Path):
    """
    合并所有cleaned_final_dataset.json文件
    
    Args:
        output_dir: cesium output目录
        output_file: 输出的合并文件路径
    """
    all_samples = []
    
    # 查找所有cleaned_final_dataset.json文件
    json_files = list(output_dir.glob("*/cleaned_final_dataset.json"))
    
    print(f"找到 {len(json_files)} 个数据文件:")
    for json_file in json_files:
        print(f"  - {json_file}")
    
    # 处理每个文件
    for json_file in json_files:
        city = json_file.parent.name
        print(f"\n处理 {city}...")
        
        # 加载数据
        with open(json_file, 'r') as f:
            samples = json.load(f)
        
        print(f"  原始样本数: {len(samples)}")
        
        # 转换每个样本
        converted_samples = []
        for sample in samples:
            try:
                converted = convert_sample(sample, city)
                converted_samples.append(converted)
            except Exception as e:
                print(f"  警告: 转换样本失败: {e}")
                continue
        
        print(f"  转换后样本数: {len(converted_samples)}")
        all_samples.extend(converted_samples)
    
    # 保存合并后的数据
    print(f"\n总样本数: {len(all_samples)}")
    print(f"保存到: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print("完成!")
    
    # 打印统计信息
    city_counts = {}
    for sample in all_samples:
        city = sample['city']
        city_counts[city] = city_counts.get(city, 0) + 1
    
    print("\n各城市样本数:")
    for city, count in sorted(city_counts.items()):
        print(f"  {city}: {count}")


def main():
    # 设置路径
    output_dir = Path("/data/cesium/output")
    output_file = Path("/data/xhj/location/data/cesium_drone_dataset.json")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("合并Cesium数据集")
    print("="*60)
    print(f"输入目录: {output_dir}")
    print(f"输出文件: {output_file}")
    print(f"\n注意: 子目录已重命名为 mono/ 和 sate/")
    print("="*60 + "\n")
    
    # 合并数据集
    merge_datasets(output_dir, output_file)


if __name__ == '__main__':
    main()
