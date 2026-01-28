#!/usr/bin/env python3
"""
数据集划分脚本
将完整数据集划分为训练集、验证集和测试集
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, split_by_city=False):
    """
    划分数据集
    
    Args:
        data: 完整数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        split_by_city: 是否按城市分别划分（保证每个城市的数据在各个集合中都有）
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    random.seed(seed)
    
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1.0，当前为 {total_ratio}")
    
    if split_by_city:
        # 按城市分别划分
        city_data = defaultdict(list)
        for item in data:
            city_data[item['city']].append(item)
        
        train_data = []
        val_data = []
        test_data = []
        
        for city, city_items in city_data.items():
            # 打乱当前城市的数据
            random.shuffle(city_items)
            
            n = len(city_items)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_data.extend(city_items[:train_end])
            val_data.extend(city_items[train_end:val_end])
            test_data.extend(city_items[val_end:])
            
            print(f"  {city}: 总数={n}, 训练={train_end}, 验证={val_end-train_end}, 测试={n-val_end}")
    else:
        # 全局随机划分
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        n = len(data_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data = data_copy[:train_end]
        val_data = data_copy[train_end:val_end]
        test_data = data_copy[val_end:]
    
    return train_data, val_data, test_data


def get_statistics(data):
    """获取数据集统计信息"""
    city_count = defaultdict(int)
    for item in data:
        city_count[item['city']] += 1
    return dict(city_count)


def main():
    parser = argparse.ArgumentParser(description='数据集划分脚本')
    parser.add_argument('--input', type=str, default='/data/xhj/location/data/train.json',
                        help='输入JSON文件路径')
    parser.add_argument('--output-dir', type=str, default='/data/xhj/location/data',
                        help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--split-by-city', action='store_true',
                        help='按城市分别划分，保证每个城市的数据在各集合中都有')
    parser.add_argument('--train-name', type=str, default='train1.json',
                        help='训练集文件名 (默认: train.json)')
    parser.add_argument('--val-name', type=str, default='val.json',
                        help='验证集文件名 (默认: val.json)')
    parser.add_argument('--test-name', type=str, default='test.json',
                        help='测试集文件名 (默认: test.json)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据集划分工具")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分比例: 训练={args.train_ratio}, 验证={args.val_ratio}, 测试={args.test_ratio}")
    print(f"随机种子: {args.seed}")
    print(f"按城市划分: {'是' if args.split_by_city else '否'}")
    print("=" * 60)
    
    # 读取数据
    print("\n正在读取数据...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    print(f"总数据量: {len(data)}")
    
    # 显示原始数据统计
    print("\n原始数据城市分布:")
    original_stats = get_statistics(data)
    for city, count in sorted(original_stats.items()):
        print(f"  {city}: {count}")
    
    # 划分数据集
    print("\n正在划分数据集...")
    if args.split_by_city:
        print("按城市划分:")
    
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_by_city=args.split_by_city
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存划分后的数据
    print("\n正在保存数据...")
    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    test_path = output_dir / args.test_name
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"训练集已保存: {train_path}")
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"验证集已保存: {val_path}")
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"测试集已保存: {test_path}")
    
    # 显示划分后的统计信息
    print("\n" + "=" * 60)
    print("划分结果统计")
    print("=" * 60)
    
    print(f"\n训练集: {len(train_data)} 条 ({len(train_data)/len(data)*100:.2f}%)")
    train_stats = get_statistics(train_data)
    for city, count in sorted(train_stats.items()):
        print(f"  {city}: {count}")
    
    print(f"\n验证集: {len(val_data)} 条 ({len(val_data)/len(data)*100:.2f}%)")
    val_stats = get_statistics(val_data)
    for city, count in sorted(val_stats.items()):
        print(f"  {city}: {count}")
    
    print(f"\n测试集: {len(test_data)} 条 ({len(test_data)/len(data)*100:.2f}%)")
    test_stats = get_statistics(test_data)
    for city, count in sorted(test_stats.items()):
        print(f"  {city}: {count}")
    
    print("\n完成!")


if __name__ == '__main__':
    main()
