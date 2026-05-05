# 新机器 Setup 说明

本文档面向一台新的训练机器，目标是把下面这些实验跑起来：

- `gageo_dinov2_vit_b16_joint`
- `gageo_dinov2_vit_h14_joint`
- `gageo_moco_q4096`
- `gageo_pi3_frame_pos_cmaloc`
- `trogeo_pi3`

文档按下面的顺序组织：

1. 下载数据集
2. 拉代码并配置环境
3. 准备训练所需 checkpoint
4. 修改 YAML 路径
5. 启动训练

## 1. 下载数据集

数据集页面：

- `https://huggingface.co/datasets/cipual/Urban-CVOGL/tree/main`

推荐使用 Hugging Face 官方 CLI 的 `hf download` 命令下载整个数据集仓库。官方文档说明 `hf download ... --repo-type dataset --local-dir ...` 可以把整个 dataset repo 拉到指定目录。

参考：

- Hugging Face CLI 文档：`https://huggingface.co/docs/huggingface_hub/main/en/guides/cli`
- Hub 下载文档：`https://huggingface.co/docs/huggingface_hub/guides/download`

### 1.1 安装 Hugging Face CLI

如果机器上还没有 `hf` 命令：

```bash
python -m pip install -U "huggingface_hub[cli]"
```

如果数据集需要登录权限，再执行：

```bash
hf auth login
```

### 1.2 下载数据集到本地目录

下面示例把数据集放到 `/data/Urban-CVOGL`：

```bash
hf download cipual/Urban-CVOGL \
  --repo-type dataset \
  --local-dir /data/Urban-CVOGL
```

下载完成后，建议你先检查目录结构。训练代码实际需要两类路径：

- JSON 标注目录，例如：`/data/Urban-CVOGL/json`
- 图像根目录，例如：`/data/Urban-CVOGL/urban`

对当前 `location_v4` 代码，图像根目录需要满足下面这种层级：

```text
<DATA_ROOT>/<city>/mono/<image>
<DATA_ROOT>/<city>/sate/<image>
<DATA_ROOT>/<city>/crop_sate/<image>
```

JSON 标注通常需要：

```text
<JSON_ROOT>/train_all.json
<JSON_ROOT>/val_all.json
<JSON_ROOT>/test_all.json
```

如果你下载后的目录名和这里不同，不需要改代码，后面直接改 YAML 里的路径即可。

## 2. 拉代码并配置环境

环境依赖以你给的 `uv.lock` 为准：

- `https://github.com/xuhaojun1/location/blob/test/uv.lock`

### 2.1 安装 uv

如果机器上还没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后重新打开一个 shell，或者执行：

```bash
source ~/.bashrc
```

### 2.2 拉代码

建议把主仓库放到一个固定目录，例如 `/workspace/location`：

```bash
git clone https://github.com/xuhaojun1/location.git /workspace/location
cd /workspace/location
git checkout test
```

### 2.3 用 uv.lock 创建环境

在仓库根目录执行：

```bash
uv sync
```

如果你希望严格按照 lock 文件，不接受版本漂移：

```bash
uv sync --frozen
```

之后进入环境有两种常用方式。

直接在 uv 环境里运行：

```bash
uv run python -V
```

或者先激活：

```bash
source .venv/bin/activate
python -V
```

## 3. 准备训练所需 checkpoint

`location_v4` 这几组 rebuttal 实验需要以下预训练权重：

- `pi3_model.safetensors`
- `sam2.1_hiera_large.pt`
- `vit_b_16_imagenet1k_v1.pth`
- `vit_h_14_imagenet1k_swag_e2e_v1.pth`

我已经在仓库里加了自动准备脚本：

- [download_required_checkpoints.py](/mnt/data/wrp/location_v4/scripts/download_required_checkpoints.py)
- [download_required_checkpoints.sh](/mnt/data/wrp/location_v4/scripts/download_required_checkpoints.sh)

### 3.1 在线机器直接准备

假设你的 `location_v4` 位于 `/workspace/location/location_v4`：

```bash
cd /workspace/location/location_v4
uv run python scripts/download_required_checkpoints.py \
  --output_dir /workspace/checkpoints_offline
```

或者：

```bash
bash scripts/download_required_checkpoints.sh \
  --output_dir /workspace/checkpoints_offline
```

如果你本地已经有 Pi3 和 SAM2 权重，也可以显式指定：

```bash
uv run python scripts/download_required_checkpoints.py \
  --output_dir /workspace/checkpoints_offline \
  --pi3_source /your/local/pi3/model.safetensors \
  --sam_source /your/local/sam2.1_hiera_large.pt
```

### 3.2 离线机器

如果训练机器没网，在一台有网的机器上先执行上面的脚本，然后把整个目录拷过去，例如：

```text
/workspace/checkpoints_offline
```

目录里应该至少有：

```text
pi3_model.safetensors
sam2.1_hiera_large.pt
vit_b_16_imagenet1k_v1.pth
vit_h_14_imagenet1k_swag_e2e_v1.pth
offline_paths.yaml
```

## 4. 修改 YAML 路径

当前代码支持两种路径配置方式：

1. 用环境变量
2. 直接改 YAML 成绝对路径

为了减少歧义，推荐直接把 YAML 改成绝对路径。这样迁移到新机器时最稳。

### 4.1 GAGeo 相关 YAML

主要会改这些文件：

- [default_v3.yaml](/mnt/data/wrp/location_v4/configs/default_v3.yaml)
- [gageo_dinov2_vit_b16_joint.yaml](/mnt/data/wrp/location_v4/configs/gageo_dinov2_vit_b16_joint.yaml)
- [gageo_dinov2_vit_h14_joint.yaml](/mnt/data/wrp/location_v4/configs/gageo_dinov2_vit_h14_joint.yaml)
- [gageo_moco_q4096.yaml](/mnt/data/wrp/location_v4/configs/gageo_moco_q4096.yaml)
- [gageo_pi3_frame_pos_cmaloc.yaml](/mnt/data/wrp/location_v4/configs/gageo_pi3_frame_pos_cmaloc.yaml)

下面给一个 ViT-B 的示例。假设：

- JSON 在 `/data/Urban-CVOGL/json`
- 图像在 `/data/Urban-CVOGL/urban`
- checkpoint 在 `/workspace/checkpoints_offline`
- 输出在 `/exp/location_v4`

那么可以写成：

```yaml
data:
  train_json: /data/Urban-CVOGL/json/train_all.json
  val_json: /data/Urban-CVOGL/json/val_all.json
  data_root: /data/Urban-CVOGL/urban

model:
  pi3_weights: /workspace/checkpoints_offline/pi3_model.safetensors
  sam_weights: /workspace/checkpoints_offline/sam2.1_hiera_large.pt
  joint_vit_weights: /workspace/checkpoints_offline/vit_b_16_imagenet1k_v1.pth

checkpoint:
  output_dir: /exp/location_v4/gageo_dinov2_vit_b16_joint
```

ViT-H 同理，只是把：

```yaml
joint_vit_weights: /workspace/checkpoints_offline/vit_h_14_imagenet1k_swag_e2e_v1.pth
```

写进去。

`gageo_moco_q4096.yaml` 不需要 `joint_vit_weights`，但仍然需要：

```yaml
model:
  pi3_weights: /workspace/checkpoints_offline/pi3_model.safetensors
  sam_weights: /workspace/checkpoints_offline/sam2.1_hiera_large.pt
```

`gageo_pi3_frame_pos_cmaloc.yaml` 也是同样的改法。

### 4.2 如果你不想改成绝对路径

也可以保留变量写法，例如：

```yaml
data:
  train_json: "${JSON_ROOT}/train_all.json"
  val_json: "${JSON_ROOT}/val_all.json"
  data_root: "${DATA_ROOT}"

model:
  pi3_weights: "${CHECKPOINT_DIR}/pi3_model.safetensors"
  sam_weights: "${CHECKPOINT_DIR}/sam2.1_hiera_large.pt"
  joint_vit_weights: "${CHECKPOINT_DIR}/vit_b_16_imagenet1k_v1.pth"

checkpoint:
  output_dir: "${OUTPUT_ROOT}/gageo_dinov2_vit_b16_joint"
```

然后在 shell 里设置：

```bash
export CHECKPOINT_DIR=/workspace/checkpoints_offline
export JSON_ROOT=/data/Urban-CVOGL/json
export DATA_ROOT=/data/Urban-CVOGL/urban
export OUTPUT_ROOT=/exp/location_v4
```

### 4.3 TROGeo-Pi3 的 YAML

TROGeo-Pi3 我也补了一个离线配置：

- [/mnt/data/wrp/CVOS-Code/configs/trogeo_pi3_offline.yaml](/mnt/data/wrp/CVOS-Code/configs/trogeo_pi3_offline.yaml)

需要重点检查这些字段：

```yaml
data:
  data_root: /data/Urban-CVOGL/urban
  train_json: /data/Urban-CVOGL/json/train_all.json
  val_json: /data/Urban-CVOGL/json/val_all.json
  test_json: /data/Urban-CVOGL/json/test_all.json

model:
  pi3_pretrain: /workspace/checkpoints_offline/pi3_model.safetensors

checkpoint:
  output_dir: /exp/CVOS-Code/trogeo_pi3_eccv

logging:
  tb_logdir: /exp/CVOS-Code/trogeo_pi3_eccv/tensorboard
```

## 5. 启动训练

下面默认你已经完成：

- 数据集下载
- `uv sync`
- checkpoint 准备
- YAML 路径修改

### 5.1 GAGeo ViT-B

```bash
cd /workspace/location/location_v4
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29601 \
  bash scripts/train_gageo_dinov2_vit_b16_joint_ddp_terminal.sh
```

### 5.2 GAGeo ViT-H

```bash
cd /workspace/location/location_v4
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29601 \
  bash scripts/train_gageo_dinov2_vit_h14_joint_ddp_terminal.sh
```

### 5.3 GAGeo MoCo queue size 4096

```bash
cd /workspace/location/location_v4
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29601 \
  bash scripts/train_gageo_moco_q4096_ddp_terminal.sh
```

### 5.4 GAGeo frame-wise positional embedding

```bash
cd /workspace/location/location_v4
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29601 \
  bash scripts/train_gageo_pi3_frame_pos_cmaloc_ddp_terminal.sh
```

### 5.5 TROGeo-Pi3

```bash
cd /workspace/location/CVOS-Code
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29601 \
  bash scripts/train_trogeo_pi3_offline_terminal.sh
```

## 6. 常见问题

### 6.1 机器没网会不会影响训练

如果你已经把下面这些都写成本地路径，就不会：

- 数据集 JSON 路径
- 数据图像根目录
- Pi3 checkpoint
- SAM2 checkpoint
- ViT-B 或 ViT-H checkpoint
- 输出目录

其中 ViT-B / ViT-H 这两个实验，务必在 YAML 里显式写 `joint_vit_weights`，否则如果 `encoder_pretrained: true`，代码可能会尝试去下载 torchvision 权重。

### 6.2 必须用环境变量吗

不必须。

直接把 YAML 改成绝对路径是最稳的方式。环境变量只是为了方便批量切换机器路径。

### 6.3 输出目录是脚本决定还是 YAML 决定

现在默认以 YAML 里的 `checkpoint.output_dir` 为准。

只有当你显式传了 `--output_dir`，或者设置了 `RUN_OUTPUT_DIR` 时，脚本才会覆盖 YAML。
