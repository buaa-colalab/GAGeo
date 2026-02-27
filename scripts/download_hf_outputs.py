#!/usr/bin/env python3
"""
从 Hugging Face 下载 output_v3 下的所有文件夹的特定内容
下载每个文件夹下的: best/, cross_view_v3/, vis/, config.yaml

使用方法:
    python3 download_hf_outputs.py

依赖:
    pip install huggingface_hub
"""

import os
import shutil
import sys
from pathlib import Path
from fnmatch import fnmatch

# 检查依赖
try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    print("错误: 未安装 huggingface_hub")
    print("请运行: pip install huggingface_hub")
    sys.exit(1)

# 配置
REPO_ID = "caryK/location"
REPO_PATH = "output_v3"
LOCAL_BASE_DIR = Path("/data/home/scxi704/run/xhj/location_v3/output_v3")

# 需要下载的文件夹列表（从网页信息中获取）
FOLDERS = [
    "ablation_1_ds_only",
    "ablation_3_ds_contrastive", 
    "ablation_4_all_on",
    "ablation_5_all_off"
]

# 需要下载的子文件夹和文件模式
TARGET_PATTERNS = [
    "best/pytorch_model/mp_rank_00_model_states.pt",
    # "cross_view_v3/*",
    # "vis/*",
    "config.yaml"
]


def should_download_file(relative_path):
    """判断文件是否应该被下载"""
    for pattern in TARGET_PATTERNS:
        # 检查完全匹配或前缀匹配
        if relative_path == pattern.replace("/*", ""):
            return True
        if pattern.endswith("/*") and relative_path.startswith(pattern.replace("/*", "") + "/"):
            return True
        # 使用 fnmatch 进行模式匹配
        if fnmatch(relative_path, pattern) or fnmatch(relative_path, pattern.replace("/*", "/**")):
            return True
    return False


def download_folder_contents(api, repo_id, folder_name, local_dir):
    """下载指定文件夹下的特定内容"""
    print(f"\n正在处理文件夹: {folder_name}")
    
    folder_path = f"{REPO_PATH}/{folder_name}"
    local_folder_dir = local_dir / folder_name
    local_folder_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 列出仓库中的所有文件
        print(f"  正在列出文件...")
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        
        if not all_files:
            print(f"  ⚠ 警告: 仓库为空或无法访问")
            return
        
        # 筛选出目标文件夹中的文件
        folder_files = [f for f in all_files if f.startswith(f"{folder_path}/")]
        
        if not folder_files:
            print(f"  ⚠ 警告: 文件夹 {folder_name} 为空或不存在")
            return
        
        # 筛选出需要下载的文件
        files_to_download = []
        for file_path in folder_files:
            # 移除前缀，获取相对路径
            relative_path = file_path[len(f"{folder_path}/"):]
            
            if should_download_file(relative_path):
                files_to_download.append((file_path, relative_path))
        
        if not files_to_download:
            print(f"  ⚠ 警告: 在 {folder_name} 中未找到目标文件")
            print(f"  可用文件示例: {folder_files[:3] if len(folder_files) > 0 else '无'}")
            return
        
        print(f"  找到 {len(files_to_download)} 个文件需要下载")
        
        # 下载每个文件
        downloaded_count = 0
        skipped_count = 0
        for file_path, relative_path in files_to_download:
            local_file_path = local_folder_dir / relative_path
            
            # 如果文件已存在，跳过
            if local_file_path.exists() and local_file_path.is_file():
                skipped_count += 1
                if skipped_count <= 3:  # 只显示前几个跳过的文件
                    print(f"  跳过已存在的文件: {relative_path}")
                continue
            
            # 创建父目录
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if downloaded_count < 5 or (downloaded_count + 1) % 10 == 0:
                    print(f"  下载 ({downloaded_count + 1}/{len(files_to_download)}): {relative_path}")
                
                # 下载文件（不使用 local_dir，避免路径嵌套问题）
                # hf_hub_download 会返回下载文件的路径
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="model"
                )
                
                # 复制文件到目标位置（保持目录结构）
                if os.path.exists(downloaded_file):
                    # 如果目标是目录，需要复制整个目录
                    if os.path.isdir(downloaded_file):
                        if local_file_path.exists():
                            shutil.rmtree(local_file_path)
                        shutil.copytree(downloaded_file, str(local_file_path))
                    else:
                        # 如果是文件，直接复制
                        if local_file_path.exists() and local_file_path.is_dir():
                            shutil.rmtree(local_file_path)
                        shutil.copy2(downloaded_file, str(local_file_path))
                    downloaded_count += 1
                else:
                    print(f"  ⚠ 警告: 下载的文件不存在: {downloaded_file}")
                
            except Exception as e:
                print(f"  ✗ 下载失败 {relative_path}: {e}")
        
        print(f"  ✓ 完成: 下载了 {downloaded_count} 个文件，跳过了 {skipped_count} 个已存在的文件")
                
    except Exception as e:
        print(f"  ✗ 错误: 处理 {folder_name} 时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print(f"开始从 {REPO_ID} 下载文件")
    print(f"目标目录: {LOCAL_BASE_DIR}")
    print(f"要下载的文件夹: {FOLDERS}")
    print(f"要下载的内容: {TARGET_PATTERNS}")
    
    # 确保本地目录存在
    LOCAL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建 HfApi 实例
    api = HfApi()
    
    # 下载每个文件夹
    for folder in FOLDERS:
        download_folder_contents(api, REPO_ID, folder, LOCAL_BASE_DIR)
    
    print("\n所有下载任务完成！")


if __name__ == "__main__":
    main()
