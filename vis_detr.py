"""
Visualization Script for Cross-View Localization DETR Model
可视化预测的相机位置、bbox和热力图
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import torch
from tqdm import tqdm

from models import CrossViewLocalizerPi3
from data import CrossViewDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Cross-View Localization DETR Results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data_json', type=str, default=None,
                        help='Override data json path')
    parser.add_argument('--output_dir', type=str, default='./visualizations')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    parser.add_argument('--show_heatmap', action='store_true',
                        help='Show heatmap overlay')
    return parser.parse_args()


def load_model(checkpoint_path, cfg, device):
    """Load model from checkpoint."""
    from models import build_cross_view_localizer_pi3
    model = build_cross_view_localizer_pi3(
        pretrained_pi3=None,
        freeze_backbone=False,
        img_size=cfg['data']['img_size'],
        decoder_size=cfg['model'].get('decoder_size', 'large'),
        num_heads=cfg['model']['num_heads'],
        num_decoder_layers=cfg['model'].get('num_decoder_layers', 6),
        num_object_queries=cfg['model'].get('num_object_queries', 10),
        num_location_queries=cfg['model'].get('num_location_queries', 16),
    )
    
    ckpt_path = Path(checkpoint_path)
    
    # Try Accelerate format
    model_file = ckpt_path / 'pytorch_model.bin'
    if not model_file.exists():
        model_file = ckpt_path / 'model.safetensors'
    if not model_file.exists():
        model_file = ckpt_path
    
    if model_file.exists() and model_file.is_file():
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Cannot find model checkpoint at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model


def to_numpy_img(img_tensor):
    """Convert tensor to numpy image."""
    return img_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def draw_bbox(ax, bbox, img_size, color, linestyle, linewidth, label):
    """Draw bounding box on axis."""
    cx, cy, w, h = bbox
    x1 = (cx - w/2) * img_size
    y1 = (cy - h/2) * img_size
    width = w * img_size
    height = h * img_size
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        linestyle=linestyle,
        label=label
    )
    ax.add_patch(rect)


def draw_camera_pose(ax, position, yaw, img_size, color, marker, label_pos):
    """Draw camera position and yaw direction."""
    x = position[0] * img_size
    y = position[1] * img_size
    ax.scatter(x, y, c=color, marker=marker, s=150, label=label_pos,
               zorder=5, edgecolors='white', linewidths=2)
    
    # Draw yaw direction arrow
    arrow_len = 40
    dx = arrow_len * np.sin(yaw)
    dy = -arrow_len * np.cos(yaw)
    arrow = FancyArrowPatch(
        (x, y), (x + dx, y + dy),
        arrowstyle='->',
        mutation_scale=15,
        color=color,
        linewidth=2
    )
    ax.add_patch(arrow)


def visualize_sample(model, sample, device, img_size, output_path, show, show_heatmap):
    """Visualize a single sample."""
    with torch.no_grad():
        front_view = sample['front_view'].unsqueeze(0).to(device)
        sat_view = sample['satellite_view'].unsqueeze(0).to(device)
        mono_point = sample['mono_point'].unsqueeze(0).to(device)
        
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(1, 1, device=device)
        
        outputs = model(
            front_view=front_view,
            satellite_view=sat_view,
            points=(point_coords, point_labels),
        )
    
    # Extract predictions
    pred_boxes = outputs['pred_boxes'][0].cpu().numpy()  # [N, 4]
    bbox_scores = outputs['bbox_scores'][0].cpu().numpy()  # [N]
    best_idx = bbox_scores.argmax()
    pred_bbox = pred_boxes[best_idx]
    pred_score = bbox_scores[best_idx]
    
    pred_yaw = outputs['yaw_radians'][0].cpu().item() if 'yaw_radians' in outputs else None
    pred_position = outputs['position'][0].cpu().numpy() if 'position' in outputs else None
    heatmap = outputs['heatmap'][0].cpu().numpy() if 'heatmap' in outputs else None
    
    # Ground truth
    gt_bbox = sample['sat_bbox'].numpy()
    gt_yaw = sample['yaw_radians'].item()
    gt_position = sample['camera_position'].numpy()
    
    # Create figure
    n_cols = 4 if show_heatmap and heatmap is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    
    # 1. Front view with prompt point
    front_img = to_numpy_img(sample['front_view'])
    axes[0].imshow(front_img)
    mono_pt = sample['mono_point'].numpy()
    axes[0].scatter(mono_pt[0], mono_pt[1], c='lime', marker='x', s=200,
                    linewidths=3, label='Prompt Point')
    axes[0].set_title('Front View', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].axis('off')
    
    # 2. Satellite view with BBox
    sat_img = to_numpy_img(sample['satellite_view'])
    axes[1].imshow(sat_img)
    draw_bbox(axes[1], gt_bbox, img_size, 'lime', '-', 3, 'GT BBox')
    draw_bbox(axes[1], pred_bbox, img_size, 'red', '--', 2, f'Pred (score={pred_score:.2f})')
    axes[1].set_title('Satellite - BBox', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].axis('off')
    
    # 3. Satellite view with Camera Pose
    axes[2].imshow(sat_img)
    draw_camera_pose(axes[2], gt_position, gt_yaw, img_size, 'lime', 'o', 'GT Position')
    if pred_position is not None and pred_yaw is not None:
        draw_camera_pose(axes[2], pred_position, pred_yaw, img_size, 'red', '^', 'Pred Position')
    axes[2].set_title('Satellite - Camera Pose', fontsize=14)
    axes[2].legend(loc='upper right')
    axes[2].axis('off')
    
    # 4. Heatmap (optional)
    if show_heatmap and heatmap is not None and n_cols == 4:
        axes[3].imshow(sat_img)
        axes[3].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[3].scatter(gt_position[0] * img_size, gt_position[1] * img_size,
                        c='lime', marker='o', s=150, edgecolors='white', linewidths=2, label='GT')
        if pred_position is not None:
            axes[3].scatter(pred_position[0] * img_size, pred_position[1] * img_size,
                            c='red', marker='^', s=150, edgecolors='white', linewidths=2, label='Pred')
        axes[3].set_title('Position Heatmap', fontsize=14)
        axes[3].legend(loc='upper right')
        axes[3].axis('off')
    
    # Compute metrics
    metrics = []
    
    # BBox IoU
    from utils import box_cxcywh_to_xyxy, compute_iou
    pred_xyxy = box_cxcywh_to_xyxy(torch.tensor(pred_bbox).unsqueeze(0))
    gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(gt_bbox).unsqueeze(0))
    iou = compute_iou(pred_xyxy, gt_xyxy)[0, 0].item()
    metrics.append(f'IoU: {iou:.3f}')
    
    if pred_yaw is not None:
        yaw_diff = np.arctan2(np.sin(pred_yaw - gt_yaw), np.cos(pred_yaw - gt_yaw))
        metrics.append(f'Yaw Err: {np.abs(np.degrees(yaw_diff)):.1f}°')
    
    if pred_position is not None:
        pos_err = np.linalg.norm(pred_position - gt_position)
        pos_err_px = pos_err * img_size
        metrics.append(f'Pos Err: {pos_err_px:.1f}px')
    
    fig.suptitle(' | '.join(metrics), fontsize=12, y=0.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'pred_bbox': pred_bbox,
        'pred_yaw': pred_yaw,
        'pred_position': pred_position,
        'pred_score': pred_score,
        'gt_bbox': gt_bbox,
        'gt_yaw': gt_yaw,
        'gt_position': gt_position,
        'iou': iou,
    }


def plot_summary(results, output_dir, show):
    """Plot summary statistics."""
    ious = [r['iou'] for r in results]
    yaw_errors = [np.abs(np.degrees(np.arctan2(
        np.sin(r['pred_yaw'] - r['gt_yaw']),
        np.cos(r['pred_yaw'] - r['gt_yaw'])
    ))) for r in results if r['pred_yaw'] is not None]
    pos_errors = [np.linalg.norm(r['pred_position'] - r['gt_position'])
                  for r in results if r['pred_position'] is not None]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # IoU
    axes[0].bar(range(len(ious)), ious, color='steelblue')
    axes[0].axhline(np.mean(ious), color='red', linestyle='--',
                    label=f'Mean: {np.mean(ious):.3f}')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('IoU')
    axes[0].set_title('BBox IoU')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    
    # Yaw error
    if yaw_errors:
        axes[1].bar(range(len(yaw_errors)), yaw_errors, color='coral')
        axes[1].axhline(np.mean(yaw_errors), color='red', linestyle='--',
                        label=f'Mean: {np.mean(yaw_errors):.1f}°')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Yaw Error (°)')
        axes[1].set_title('Yaw Error')
        axes[1].legend()
    
    # Position error
    if pos_errors:
        axes[2].bar(range(len(pos_errors)), pos_errors, color='seagreen')
        axes[2].axhline(np.mean(pos_errors), color='red', linestyle='--',
                        label=f'Mean: {np.mean(pos_errors):.4f}')
        axes[2].set_xlabel('Sample')
        axes[2].set_ylabel('Position Error (normalized)')
        axes[2].set_title('Position Error')
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Loading model from {args.checkpoint}')
    model = load_model(args.checkpoint, cfg, device)
    
    data_json = args.data_json or cfg['data']['val_json']
    dataset = CrossViewDataset(
        json_path=data_json,
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        random_crop=False,
    )
    
    img_size = cfg['data']['img_size']
    num_samples = min(args.num_samples, len(dataset))
    print(f'Visualizing {num_samples} samples...')
    
    results = []
    for i in tqdm(range(num_samples)):
        result = visualize_sample(
            model, dataset[i], device, img_size,
            output_dir / f'sample_{i:04d}.png',
            args.show, args.show_heatmap
        )
        results.append(result)
    
    plot_summary(results, output_dir, args.show)
    
    # Print summary
    print(f'\n{"="*50}')
    print('Summary Statistics')
    print(f'{"="*50}')
    
    ious = [r['iou'] for r in results]
    print(f'BBox IoU: {np.mean(ious):.3f} ± {np.std(ious):.3f}')
    print(f'IoU@0.5: {np.mean([iou >= 0.5 for iou in ious]):.1%}')
    print(f'IoU@0.75: {np.mean([iou >= 0.75 for iou in ious]):.1%}')
    
    yaw_errors = [np.abs(np.degrees(np.arctan2(
        np.sin(r['pred_yaw'] - r['gt_yaw']),
        np.cos(r['pred_yaw'] - r['gt_yaw'])
    ))) for r in results if r['pred_yaw'] is not None]
    if yaw_errors:
        print(f'Yaw Error: {np.mean(yaw_errors):.1f}° ± {np.std(yaw_errors):.1f}°')
    
    pos_errors = [np.linalg.norm(r['pred_position'] - r['gt_position'])
                  for r in results if r['pred_position'] is not None]
    if pos_errors:
        print(f'Position Error: {np.mean(pos_errors):.4f} ± {np.std(pos_errors):.4f}')
        print(f'Position Error (px): {np.mean(pos_errors) * img_size:.1f}px')
    
    print(f'\nVisualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
