<div align="center">

<h1>GAGeo: Geometry-Aware Cross-View Object Geo-Localization</h1>

<div>
    Liyao Wang*<sup>1</sup>&emsp;
    Ruipu Wu*<sup>1</sup>&emsp;
    Haojun Xu*<sup>1</sup>&emsp;
    Lei Shi<sup>2</sup>&emsp;
    Linjiang Huang<sup>1</sup>&emsp;
    Si Liu<sup>1</sup>
</div>

<div>
    <sup>1</sup>Beihang University&emsp;
    <sup>2</sup>Meituan
</div>

<div>
    <strong>Beyond 2D Matching: A Unified Single-Stage Framework for Geometry-Aware Cross-View Object Geo-Localization</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://cipual.github.io/GAGeo-project-page/" target="_blank">
        <img src="https://img.shields.io/badge/Project-Page-green">
        </a>
        <a href="https://arxiv.org/abs/2606.30576" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2606.30576-b31b1b.svg">
        </a>
        <a href="#citation">
        <img src="https://img.shields.io/badge/Cite-BibTeX-blue">
        </a>
        <a href="https://huggingface.co/datasets/cipual/CMA-Loc" target="_blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow">
        </a>
        <a href="https://huggingface.co/cipual/GAGeo" target="_blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange">
        </a>
    </h4>
</div>

<strong>GAGeo is a single-stage geometry-aware framework for cross-view object geo-localization. Given a ground-view or drone-view query and a point, box, or mask prompt, it localizes the target object in the satellite view and predicts detection, segmentation, camera-position, and pose outputs.</strong>

</div>

## 📢 News

* **[2026-06-25]** The public training and evaluation code has been cleaned for release.
* **[2026-06-18]** GAGeo was accepted by ECCV 2026. 🎉

## 💡 Highlights

* **Geometry-aware single-stage localization.** GAGeo adapts a 3D foundation model backbone to cross-view object geo-localization and predicts boxes, masks, camera position, and pose in one forward pass.
* **Multi-prompt target referring.** The model supports point, bounding-box, and mask prompts for both ground-to-satellite and drone-to-satellite localization.
* **Unified CMA-Loc benchmark.** CMA-Loc provides ground-satellite and drone-satellite instance pairs with object masks, prompt annotations, and geometric supervision.

## 🛠️ Usage

### Installation

#### Clone Repository

```bash
git clone <repo-url> GAGeo
cd GAGeo
```

#### Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can also use the provided installer:

```bash
bash install.sh
```

RoPE is implemented in PyTorch and does not require building a custom CUDA extension.

---

### Repository Layout

```text
GAGeo/
├── configs/                     # Training and evaluation configs
│   ├── default.yaml             # Public CMA-Loc training/evaluation config
│   └── accelerate_deepspeed_zero2.yaml
├── data/
│   ├── README.md
│   ├── json/                    # CMA-Loc split annotations
│   └── urban/                   # Local CMA-Loc images
├── datasets/
│   ├── __init__.py
│   └── cmaloc.py                # CMA-Loc dataset loader
├── models/
│   ├── backbone/                # Pi3/DINOv2 backbone wrapper
│   ├── decoder/                 # Mask and task decoders
│   ├── encoder/                 # Prompt encoder
│   ├── heads/                   # Detection, segmentation, and pose heads
│   ├── layers/                  # Transformer and RoPE layers
│   └── cross_view_localizer.py  # Main GAGeo model
├── scripts/
│   ├── train.sh
│   ├── evaluate_cmaloc.sh
│   └── download_required_checkpoints.py
├── utils/                       # Losses, metrics, prompts, and runtime helpers
├── train.py                     # Accelerate/DeepSpeed training entrypoint
├── train_ddp.py                 # Native PyTorch DDP training entrypoint
├── evaluate_cmaloc.py           # CMA-Loc evaluation entrypoint
└── GAGeo_ckpt/gageo/            # Released GAGeo checkpoint location
```

---

### Dataset Preparation

Place the CMA-Loc data under the project data directory:

```text
data/
├── json/
│   ├── train_all.json
│   ├── val_all.json
│   ├── test_all.json
│   └── unseen_test.json
└── urban/
    └── <city>/
        ├── mono/
        ├── sate/
        └── crop_sate/
```

The scripts use this layout by default:

```bash
export DATA_ROOT=$PWD/data/urban
export JSON_ROOT=$PWD/data/json
export OUTPUT_ROOT=$PWD/outputs
```

Expected image paths:

```text
$DATA_ROOT/<city>/mono/<mono_filename>
$DATA_ROOT/<city>/sate/<sat_filename>
$DATA_ROOT/<city>/crop_sate/<sat_filename>
```

---

### Pretrained Checkpoints

The released GAGeo checkpoint should be placed at:

```text
GAGeo_ckpt/gageo/mp_rank_00_model_states.pt
```

Training also requires the upstream Pi3, SAM2.1, and torchvision ViT checkpoints expected by `configs/default.yaml`. Prepare them under `CHECKPOINT_DIR`:

```bash
export CHECKPOINT_DIR=$PWD/checkpoints_offline
python scripts/download_required_checkpoints.py --output_dir "$CHECKPOINT_DIR"
```

If your checkpoint files are stored elsewhere, set `CHECKPOINT_DIR` before launching training.

---

### Training

The default public training config is:

```text
configs/default.yaml
```

Launch training with:

```bash
bash scripts/train.sh gageo configs/default.yaml
```

The launcher automatically selects single-process, Accelerate, or native PyTorch DDP mode based on `CUDA_VISIBLE_DEVICES`, `NUM_PROCESSES`, and `DISTRIBUTED_BACKEND`.

Common examples:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh gageo configs/default.yaml

# Multi-GPU with Accelerate
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash scripts/train.sh gageo configs/default.yaml

# Native PyTorch DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 DISTRIBUTED_BACKEND=ddp \
    bash scripts/train.sh gageo configs/default.yaml
```

---

### Evaluation

Evaluate the released checkpoint on CMA-Loc:

```bash
bash scripts/evaluate_cmaloc.sh
```

Equivalent direct command:

```bash
python evaluate_cmaloc.py \
    --config configs/default.yaml \
    --checkpoint GAGeo_ckpt/gageo/mp_rank_00_model_states.pt \
    --image_root data/urban \
    --splits test unseen_test \
    --prompt_types point bbox mask \
    --batch_size 8 \
    --num_workers 8 \
    --skip_sam \
    --save_json outputs/cmaloc_metrics.json
```

Use `--view_subset drone_to_satellite` or `--view_subset ground_to_satellite` to report a single CMA-Loc task direction.

---

## 📊 Reproduced CMA-Loc Results

Using `configs/default.yaml` and the released checkpoint at `GAGeo_ckpt/gageo/mp_rank_00_model_states.pt`, the retained CMA-Loc experiment reproduces the paper tables within rounding tolerance for point, box, and mask prompts on both seen and unseen splits.

The reproduced metrics are saved under:

```text
outputs/reproduce_paper/cmaloc_seen_test.json
outputs/reproduce_paper/cmaloc_unseen_test.json
```

Note that `data/json/test_all.json` corresponds to the seen split, while `data/json/unseen_test.json` corresponds to the unseen split.

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{wang2026gageo,
  title={Beyond 2D Matching: A Unified Single-Stage Framework for Geometry-Aware Cross-View Object Geo-Localization},
  author={Wang, Liyao and Wu, Ruipu and Xu, Haojun and Shi, Lei and Huang, Linjiang and Liu, Si},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```

## 📄 License

Please refer to the project license file when it is released.

## 🙏 Acknowledgement

This project builds upon several excellent open-source projects and models:

* [Pi3](https://github.com/yyfz/Pi3)
* [DINOv2](https://github.com/facebookresearch/dinov2)
* [SAM2](https://github.com/facebookresearch/sam2)

We thank the authors for releasing their code and models to the community.
