#!/usr/bin/env python3
"""
从评估日志中提取数据并生成消融实验表格
"""

import re
from pathlib import Path

# 定义实验映射
EXPERIMENTS = {
    'ab4_all': {
        'name': 'All Components (Ours)',
        'file': 'slurm_v3_eval_ab4_all_63924.out'
    },
    'ab3_dsctr': {
        'name': 'w/o Camera Loss',
        'file': 'slurm_v3_eval_ab3_dsctr_63923.out'
    },
    'ab1_ds': {
        'name': 'w/o Contrastive & Camera',
        'file': 'slurm_v3_eval_ab1_ds_63922.out'
    },
    'ab5_off': {
        'name': 'w/o All',
        'file': 'slurm_v3_eval_ab5_off_63921.out'
    }
}

# 定义需要提取的指标
METRICS = ['A@.5:.95', 'ACC@0.5', 'ACC@0.75', 'RotErr', 'PosErrPx', 'M_mIoU', 'M_mDice', 'M_AAE', 'M_ME']

# 定义评估场景
SCENARIOS = {
    'test': ['bbox', 'mask', 'point'],
    'unseen_test': ['bbox', 'mask', 'point']
}

def parse_log_line(line):
    """解析日志中的一行数据"""
    # 格式: task/drone   2184    0.7568   0.9721    0.8713   180.05     1.59    141.22 │   0.8279   0.9003     511.9     4.19 │   0.8055   0.8855     624.7     5.11
    parts = line.split('│')
    if len(parts) < 3:
        return None
    
    # 提取基础指标
    base_part = parts[0].strip().split()
    if len(base_part) < 8:
        return None
    
    # 提取M指标 (mask metrics)
    m_part = parts[1].strip().split()
    if len(m_part) < 9:
        return None
    
    return {
        'A@.5:.95': float(base_part[2]),
        'ACC@0.5': float(base_part[3]),
        'ACC@0.75': float(base_part[4]),
        'RotErr': float(base_part[6]),
        'PosErrPx': float(base_part[7]),
        'M_mIoU': float(m_part[0]),
        'M_mDice': float(m_part[1]),
        'M_AAE': float(m_part[2]),
        'M_ME': float(m_part[3]),
    }

def extract_data_from_log(log_file, split_name, prompt_type):
    """从日志文件中提取指定split和prompt的数据"""
    log_path = Path('/data/home/scxi704/run/eval_logs') / log_file
    
    if not log_path.exists():
        print(f"Warning: {log_path} not found")
        return None, None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 查找对应的评估部分
    pattern = rf'Cross-View V2 Evaluation — {split_name} \| prompt={prompt_type}.*?task/drone\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+│\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+).*?task/ground\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+│\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        # 尝试更简单的匹配方式
        lines = content.split('\n')
        drone_data = None
        ground_data = None
        
        in_section = False
        for i, line in enumerate(lines):
            if f'Cross-View V2 Evaluation — {split_name} | prompt={prompt_type}' in line:
                in_section = True
                continue
            
            if in_section:
                if 'task/drone' in line and 'task/ground' not in line:
                    drone_data = parse_log_line(line)
                elif 'task/ground' in line:
                    ground_data = parse_log_line(line)
                    break
        
        return drone_data, ground_data
    
    # 从match中提取数据
    drone_data = {
        'A@.5:.95': float(match.group(2)),
        'ACC@0.5': float(match.group(3)),
        'ACC@0.75': float(match.group(4)),
        'RotErr': float(match.group(5)),
        'PosErrPx': float(match.group(6)),
        'M_mIoU': float(match.group(7)),
        'M_mDice': float(match.group(8)),
        'M_AAE': float(match.group(9)),
        'M_ME': float(match.group(10)),
    }
    
    ground_data = {
        'A@.5:.95': float(match.group(12)),
        'ACC@0.5': float(match.group(13)),
        'ACC@0.75': float(match.group(14)),
        'RotErr': float(match.group(15)),
        'PosErrPx': float(match.group(16)),
        'M_mIoU': float(match.group(17)),
        'M_mDice': float(match.group(18)),
        'M_AAE': float(match.group(19)),
        'M_ME': float(match.group(20)),
    }
    
    return drone_data, ground_data

def format_value(value, metric, is_best=False):
    """格式化数值"""
    if metric in ['M_mIoU', 'M_mDice']:
        # 百分比格式，保留2位小数
        formatted = f"{value*100:.2f}%"
    elif metric in ['A@.5:.95', 'ACC@0.5', 'ACC@0.75']:
        # 保留4位小数
        formatted = f"{value:.4f}"
    elif metric == 'M_AAE':
        # 平方米，保留2位小数（不带单位）
        formatted = f"{value:.2f}"
    elif metric == 'M_ME':
        # 米，保留2位小数（不带单位）
        formatted = f"{value:.2f}"
    elif metric == 'RotErr':
        # 度，保留2位小数
        formatted = f"{value:.2f}"
    elif metric == 'PosErrPx':
        # 像素，保留2位小数
        formatted = f"{value:.2f}"
    else:
        formatted = f"{value:.4f}"
    
    if is_best:
        return f"**{formatted}**"
    return formatted

def generate_table(split_name, prompt_type):
    """生成一个消融表格"""
    print(f"\n{'='*150}")
    print(f"  Ablation Study — {split_name} | prompt={prompt_type}")
    print(f"{'='*150}\n")
    
    # 收集所有数据
    all_data = {}
    for exp_key, exp_info in EXPERIMENTS.items():
        drone_data, ground_data = extract_data_from_log(exp_info['file'], split_name, prompt_type)
        if drone_data is None or ground_data is None:
            print(f"Warning: Could not extract data for {exp_info['name']}")
            continue
        all_data[exp_key] = {
            'name': exp_info['name'],
            'drone': drone_data,
            'ground': ground_data
        }
    
    # 表头
    header = f"{'Setting':<30}"
    for metric in METRICS:
        if metric in ['A@.5:.95', 'ACC@0.5', 'ACC@0.75', 'M_mIoU', 'M_mDice']:
            header += f"  {metric:>10}"
        else:
            header += f"  {metric:>10}"
    print(header)
    print("-" * 150)
    
    # Drone → Satellite 部分
    print("Drone → Satellite")
    print("-" * 150)
    
    for exp_key in ['ab4_all', 'ab3_dsctr', 'ab1_ds', 'ab5_off']:
        if exp_key not in all_data:
            continue
        
        exp_info = all_data[exp_key]
        is_best = (exp_key == 'ab4_all')
        
        row = f"{exp_info['name']:<30}"
        for metric in METRICS:
            value = exp_info['drone'][metric]
            formatted = format_value(value, metric, is_best)
            row += f"  {formatted:>10}"
        print(row)
    
    print()
    print("Ground → Satellite")
    print("-" * 150)
    
    for exp_key in ['ab4_all', 'ab3_dsctr', 'ab1_ds', 'ab5_off']:
        if exp_key not in all_data:
            continue
        
        exp_info = all_data[exp_key]
        is_best = (exp_key == 'ab4_all')
        
        row = f"{exp_info['name']:<30}"
        for metric in METRICS:
            value = exp_info['ground'][metric]
            formatted = format_value(value, metric, is_best)
            row += f"  {formatted:>10}"
        print(row)
    
    print(f"{'='*150}\n")

def main():
    """主函数"""
    output_file = Path('/data/home/scxi704/run/xhj/location_v3/ablation_tables.txt')
    
    # 重定向输出到文件
    import sys
    original_stdout = sys.stdout
    
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        # 生成所有6个表格
        for split_name in ['test', 'unseen_test']:
            for prompt_type in ['bbox', 'mask', 'point']:
                generate_table(split_name, prompt_type)
        sys.stdout = original_stdout
    
    print(f"所有表格已保存到: {output_file}")
    
    # 同时在控制台显示
    with open(output_file, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    main()
