#!/bin/bash
# 从 Hugging Face 下载 output_v3 下的所有文件夹的特定内容
# 下载每个文件夹下的: best/, cross_view_v3/, vis/, config.yaml

set -e

REPO_ID="caryK/location"
REPO_PATH="output_v3"
LOCAL_BASE_DIR="/data/home/scxi704/run/xhj/location_v3/output_v3"

# 需要下载的文件夹列表
FOLDERS=(
    "ablation_1_ds_only"
    "ablation_3_ds_contrastive"
    "ablation_4_all_on"
    "ablation_5_all_off"
)

echo "开始从 $REPO_ID 下载文件"
echo "目标目录: $LOCAL_BASE_DIR"
echo "要下载的文件夹: ${FOLDERS[@]}"

# 确保本地目录存在
mkdir -p "$LOCAL_BASE_DIR"

# 下载每个文件夹
for folder in "${FOLDERS[@]}"; do
    echo ""
    echo "正在处理文件夹: $folder"
    
    folder_path="$REPO_PATH/$folder"
    local_folder_dir="$LOCAL_BASE_DIR/$folder"
    mkdir -p "$local_folder_dir"
    
    # 下载 best/ 文件夹
    echo "  下载 best/ ..."
    huggingface-cli download "$REPO_ID" \
        --repo-type model \
        --local-dir "$local_folder_dir" \
        --local-dir-use-symlinks False \
        "$folder_path/best" || echo "    ⚠ best/ 不存在或下载失败"
    
    # 下载 cross_view_v3/ 文件夹
    echo "  下载 cross_view_v3/ ..."
    huggingface-cli download "$REPO_ID" \
        --repo-type model \
        --local-dir "$local_folder_dir" \
        --local-dir-use-symlinks False \
        "$folder_path/cross_view_v3" || echo "    ⚠ cross_view_v3/ 不存在或下载失败"
    
    # 下载 vis/ 文件夹
    echo "  下载 vis/ ..."
    huggingface-cli download "$REPO_ID" \
        --repo-type model \
        --local-dir "$local_folder_dir" \
        --local-dir-use-symlinks False \
        "$folder_path/vis" || echo "    ⚠ vis/ 不存在或下载失败"
    
    # 下载 config.yaml
    echo "  下载 config.yaml ..."
    huggingface-cli download "$REPO_ID" \
        --repo-type model \
        --local-dir "$local_folder_dir" \
        --local-dir-use-symlinks False \
        "$folder_path/config.yaml" || echo "    ⚠ config.yaml 不存在或下载失败"
    
    echo "  ✓ 完成: $folder"
done

echo ""
echo "所有下载任务完成！"
