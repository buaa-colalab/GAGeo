#!/usr/bin/env python3
"""
修复下载后产生的嵌套目录结构
将 output_v3/ablation_xxx/ 中的内容移动到正确位置
"""

import os
import shutil
from pathlib import Path

LOCAL_BASE_DIR = Path("/data/home/scxi704/run/xhj/location_v4/output_v3")

FOLDERS = [
    "ablation_1_ds_only",
    "ablation_3_ds_contrastive", 
    "ablation_4_all_on",
    "ablation_5_all_off"
]


def fix_nested_structure(folder_name, base_dir):
    """修复嵌套目录结构"""
    folder_dir = base_dir / folder_name
    
    if not folder_dir.exists():
        print(f"  跳过: {folder_name} 不存在")
        return
    
    # 检查是否存在嵌套结构: output_v3/ablation_xxx/
    nested_dir = folder_dir / "output_v3" / folder_name
    
    if nested_dir.exists():
        print(f"  发现嵌套结构: {folder_name}")
        print(f"    嵌套路径: {nested_dir}")
        
        # 移动嵌套目录中的内容到正确位置
        for item in nested_dir.iterdir():
            target = folder_dir / item.name
            
            if target.exists():
                if target.is_dir() and item.is_dir():
                    # 合并目录内容
                    print(f"    合并目录: {item.name}")
                    for subitem in item.rglob("*"):
                        rel_path = subitem.relative_to(item)
                        target_subitem = target / rel_path
                        if subitem.is_file():
                            target_subitem.parent.mkdir(parents=True, exist_ok=True)
                            if target_subitem.exists():
                                target_subitem.unlink()
                            shutil.copy2(subitem, target_subitem)
                    # 删除嵌套目录
                    shutil.rmtree(item)
                elif target.is_file() and item.is_file():
                    # 如果文件已存在，跳过
                    print(f"    跳过已存在的文件: {item.name}")
                    item.unlink()
                else:
                    # 移动/重命名
                    print(f"    移动: {item.name}")
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(target))
            else:
                # 直接移动
                print(f"    移动: {item.name}")
                shutil.move(str(item), str(target))
        
        # 删除空的嵌套目录
        try:
            if (folder_dir / "output_v3").exists():
                shutil.rmtree(folder_dir / "output_v3")
            print(f"  ✓ 已修复: {folder_name}")
        except Exception as e:
            print(f"  ⚠ 警告: 清理嵌套目录时出错: {e}")
    else:
        print(f"  ✓ 无需修复: {folder_name} (无嵌套结构)")


def main():
    """主函数"""
    print("开始修复嵌套目录结构...")
    print(f"目标目录: {LOCAL_BASE_DIR}")
    
    for folder in FOLDERS:
        print(f"\n处理文件夹: {folder}")
        fix_nested_structure(folder, LOCAL_BASE_DIR)
    
    print("\n修复完成！")


if __name__ == "__main__":
    main()
