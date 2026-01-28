#!/usr/bin/env python3
"""
检查并修复卫星图像尺寸
- 标准尺寸: 1280×1280
- 小于标准尺寸: 上采样
- 大于标准尺寸: 下采样
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from collections import defaultdict


def check_and_fix_image(image_path, target_size=(1280, 1280), dry_run=False):
    """
    检查并修复图像尺寸
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸 (width, height)
        dry_run: 是否只检查不修复
    
    Returns:
        tuple: (需要修复, 原始尺寸, 操作类型)
    """
    if not os.path.exists(image_path):
        return False, None, "missing"
    
    try:
        img = Image.open(image_path)
        original_size = img.size  # (width, height)
        
        if original_size == target_size:
            return False, original_size, "ok"
        
        if not dry_run:
            # 使用高质量重采样
            if original_size[0] < target_size[0] or original_size[1] < target_size[1]:
                # 上采样 - 使用 LANCZOS
                resized_img = img.resize(target_size, Image.LANCZOS)
                operation = "upsample"
            else:
                # 下采样 - 使用 LANCZOS
                resized_img = img.resize(target_size, Image.LANCZOS)
                operation = "downsample"
            
            # 保存修复后的图像
            resized_img.save(image_path, quality=95)
            img.close()
            resized_img.close()
        else:
            if original_size[0] < target_size[0] or original_size[1] < target_size[1]:
                operation = "need_upsample"
            else:
                operation = "need_downsample"
        
        return True, original_size, operation
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, None, "error"


def process_json_file(json_path, data_root, target_size=(1280, 1280), dry_run=False):
    """
    处理JSON文件中的所有卫星图像
    
    Args:
        json_path: JSON文件路径
        data_root: 数据根目录
        target_size: 目标尺寸
        dry_run: 是否只检查不修复
    """
    print(f"\n处理文件: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    stats = defaultdict(int)
    stats['total'] = len(data)
    
    problem_images = []
    
    for item in tqdm(data, desc="检查卫星图像"):
        city = item.get('city', '')
        sat_filename = item.get('sat_filename', '')
        
        if not sat_filename:
            stats['no_filename'] += 1
            continue
        
        # 构建图像路径
        image_path = os.path.join(data_root, 'GoogleEarth', city, 'sate', sat_filename)
        
        needs_fix, original_size, operation = check_and_fix_image(image_path, target_size, dry_run)
        
        stats[operation] += 1
        
        if needs_fix or operation in ['missing', 'error']:
            problem_images.append({
                'city': city,
                'filename': sat_filename,
                'path': image_path,
                'original_size': original_size,
                'operation': operation
            })
    
    return stats, problem_images


def main():
    parser = argparse.ArgumentParser(description='检查并修复卫星图像尺寸')
    parser.add_argument('--json', type=str, default='/data/xhj/location/data/train.json',
                        help='JSON文件路径')
    parser.add_argument('--data-root', type=str, default='/data',
                        help='数据根目录')
    parser.add_argument('--width', type=int, default=1280,
                        help='目标宽度')
    parser.add_argument('--height', type=int, default=1280,
                        help='目标高度')
    parser.add_argument('--dry-run', action='store_true',
                        help='只检查不修复')
    parser.add_argument('--report', type=str, default=None,
                        help='保存问题报告的路径')
    
    args = parser.parse_args()
    
    target_size = (args.width, args.height)
    
    print("=" * 60)
    print("卫星图像尺寸检查与修复工具")
    print("=" * 60)
    print(f"目标尺寸: {target_size[0]}×{target_size[1]}")
    print(f"模式: {'仅检查' if args.dry_run else '检查并修复'}")
    print(f"数据根目录: {args.data_root}")
    print("=" * 60)
    
    # 处理JSON文件
    stats, problem_images = process_json_file(
        args.json, 
        args.data_root, 
        target_size, 
        args.dry_run
    )
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总图像数: {stats['total']}")
    print(f"尺寸正确: {stats['ok']}")
    print(f"需要上采样: {stats.get('need_upsample', 0) if args.dry_run else stats.get('upsample', 0)}")
    print(f"需要下采样: {stats.get('need_downsample', 0) if args.dry_run else stats.get('downsample', 0)}")
    print(f"文件缺失: {stats.get('missing', 0)}")
    print(f"处理错误: {stats.get('error', 0)}")
    print(f"无文件名: {stats.get('no_filename', 0)}")
    
    # 保存问题报告
    if problem_images and args.report:
        report_path = args.report
        with open(report_path, 'w') as f:
            json.dump(problem_images, f, indent=2, ensure_ascii=False)
        print(f"\n问题报告已保存到: {report_path}")
    
    # 显示部分问题图像
    if problem_images:
        print("\n" + "=" * 60)
        print(f"问题图像示例 (前10个):")
        print("=" * 60)
        for img_info in problem_images[:10]:
            size_str = f"{img_info['original_size']}" if img_info['original_size'] else "N/A"
            print(f"城市: {img_info['city']}")
            print(f"文件: {img_info['filename']}")
            print(f"原始尺寸: {size_str}")
            print(f"操作: {img_info['operation']}")
            print("-" * 60)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
