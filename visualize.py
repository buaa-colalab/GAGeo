"""
Visualization Script for Cross-View Localization
可视化预测的相机位置和bbox
"""

import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import torch
from PIL import Image
import cv2

from models import CrossViewLocalizer
from data.dataset import CrossViewDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Cross-View Localization Results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data_json', type=str, default=None,
                        help='Path to data JSON file (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device to use')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, cfg: dict, device: torch.device):
    """Load model from checkpoint."""
    model = CrossViewLocalizer(
        img_size=cfg['data']['img_size'],
        embed_dim=cfg['model']['embed_dim'],
        vggt_depth=cfg['model']['vggt_depth'],
        num_heads=cfg['model']['num_heads'],
        num_decoder_layers=cfg['model']['num_decoder_layers'],
        enable_bbox=cfg['model']['enable_bbox'],
        enable_seg=cfg['model']['enable_seg'],
        enable_camera=cfg['model']['enable_camera'],
        enable_position=cfg['model']['enable_position'],
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model


def denormalize_image(img_tensor):
    """Denormalize image tensor for visualization."""
    # 图像已经在[0,1]范围内，直接转换即可
    img = img_tensor.cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


def cxcywh_to_xyxy(bbox, img_size):
    """Convert center format bbox to corner format."""
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return x1, y1, x2, y2


def draw_bbox(ax, bbox, img_size, color='red', linestyle='-', linewidth=2, label=None):
    """Draw bounding box on axis."""
    x1, y1, x2, y2 = cxcywh_to_xyxy(bbox, img_size)
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color, facecolor='none',
        linestyle=linestyle, label=label
    )
    ax.add_patch(rect)
    return rect


def draw_camera_position(ax, position, img_size, color='blue', marker='o', size=100, label=None):
    """Draw camera position on satellite image."""
    x = position[0] * img_size
    y = position[1] * img_size
    ax.scatter(x, y, c=color, marker=marker, s=size, label=label, zorder=5, edgecolors='white', linewidths=2)
    return x, y


def draw_yaw_arrow(ax, position, yaw_radians, img_size, color='blue', length=30, label=None):
    """Draw yaw direction arrow."""
    x = position[0] * img_size
    y = position[1] * img_size
    
    # yaw is angle from north (up), clockwise positive
    # In image coordinates: up is -y, right is +x
    dx = length * np.sin(yaw_radians)
    dy = -length * np.cos(yaw_radians)  # negative because y-axis is inverted
    
    arrow = FancyArrowPatch(
        (x, y), (x + dx, y + dy),
        arrowstyle='->', mutation_scale=15,
        color=color, linewidth=2, label=label
    )
    ax.add_patch(arrow)
    return arrow


def visualize_single_sample(
    model,
    sample,
    device,
    img_size,
    output_path=None,
    show=False,
    sample_idx=0,
):
    """Visualize a single sample with predictions."""
    model.eval()
    
    # Debug: 检查图像质量
    if sample_idx == 0:
        print(f"\nSample {sample_idx} info:")
        print(f"  Front view shape: {sample['front_view'].shape}")
        print(f"  Front view range: [{sample['front_view'].min():.3f}, {sample['front_view'].max():.3f}]")
        print(f"  Satellite view shape: {sample['satellite_view'].shape}")
        print(f"  Satellite view range: [{sample['satellite_view'].min():.3f}, {sample['satellite_view'].max():.3f}]")
        print(f"  Mono point: {sample['mono_point']}")
        print(f"  City: {sample.get('city', 'N/A')}")
        print(f"  Mono filename: {sample.get('mono_filename', 'N/A')}")
    
    with torch.no_grad():
        # Prepare input
        front_view = sample['front_view'].unsqueeze(0).to(device)
        sat_view = sample['satellite_view'].unsqueeze(0).to(device)
        mono_point = sample['mono_point'].unsqueeze(0).to(device)
        
        # Point prompt
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(1, 1, device=device)
        
        # Forward
        outputs = model(
            front_view=front_view,
            satellite_view=sat_view,
            points=(point_coords, point_labels),
        )
    
    # Get predictions
    pred_bbox = outputs['pred_boxes'][0, 0].cpu().numpy() if 'pred_boxes' in outputs else None
    pred_yaw = outputs['yaw_radians'][0].cpu().item() if 'yaw_radians' in outputs else None
    pred_position = outputs['position'][0].cpu().numpy() if 'position' in outputs else None
    
    # Get ground truth
    gt_bbox = sample['sat_bbox'].numpy()
    gt_yaw = sample['yaw_radians'].item()
    gt_position = sample['camera_position'].numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Front view with mono point
    ax1 = axes[0]
    front_img = denormalize_image(sample['front_view'])
    ax1.imshow(front_img)
    
    # Draw mono point
    mono_pt = sample['mono_point'].numpy()
    # mono_point在dataset中是像素坐标，需要检查是否在合理范围内
    h, w = front_img.shape[:2]
    if 0 <= mono_pt[0] < w and 0 <= mono_pt[1] < h:
        ax1.scatter(mono_pt[0], mono_pt[1], 
                    c='lime', marker='x', s=200, linewidths=3, label='Mono Point')
    else:
        # 如果mono_point是归一化坐标，需要转换
        ax1.scatter(mono_pt[0] * w, mono_pt[1] * h, 
                    c='lime', marker='x', s=200, linewidths=3, label='Mono Point (normalized)')
        if sample_idx == 0:
            print(f"  Warning: mono_point seems normalized, converting to pixels")
    
    ax1.set_title(f'Front View (Sample {sample_idx})\n{sample.get("mono_filename", "")}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.axis('off')
    
    # 2. Satellite view with bbox
    ax2 = axes[1]
    sat_img = denormalize_image(sample['satellite_view'])
    ax2.imshow(sat_img)
    
    # Draw GT bbox
    draw_bbox(ax2, gt_bbox, img_size, color='lime', linestyle='-', linewidth=3, label='GT BBox')
    
    # Draw predicted bbox
    if pred_bbox is not None:
        draw_bbox(ax2, pred_bbox, img_size, color='red', linestyle='--', linewidth=2, label='Pred BBox')
    
    ax2.set_title('Satellite View - BBox', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.axis('off')
    
    # 3. Satellite view with camera position and yaw
    ax3 = axes[2]
    ax3.imshow(sat_img)
    
    # Draw GT position and yaw
    draw_camera_position(ax3, gt_position, img_size, color='lime', marker='o', size=150, label='GT Position')
    draw_yaw_arrow(ax3, gt_position, gt_yaw, img_size, color='lime', length=40, label='GT Yaw')
    
    # Draw predicted position and yaw
    if pred_position is not None:
        draw_camera_position(ax3, pred_position, img_size, color='red', marker='^', size=150, label='Pred Position')
    if pred_yaw is not None and pred_position is not None:
        draw_yaw_arrow(ax3, pred_position, pred_yaw, img_size, color='red', length=40, label='Pred Yaw')
    
    ax3.set_title('Satellite View - Camera Pose', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.axis('off')
    
    # Add metrics text
    metrics_text = []
    if pred_bbox is not None:
        bbox_error = np.abs(pred_bbox - gt_bbox).mean()
        metrics_text.append(f'BBox MAE: {bbox_error:.4f}')
    if pred_yaw is not None:
        yaw_diff = pred_yaw - gt_yaw
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
        yaw_error_deg = np.abs(np.degrees(yaw_diff))
        metrics_text.append(f'Yaw Error: {yaw_error_deg:.1f}°')
    if pred_position is not None:
        pos_error = np.linalg.norm(pred_position - gt_position)
        metrics_text.append(f'Pos Error: {pos_error:.4f}')
    
    if metrics_text:
        fig.suptitle(' | '.join(metrics_text), fontsize=12, y=0.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {output_path}')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'pred_bbox': pred_bbox,
        'pred_yaw': pred_yaw,
        'pred_position': pred_position,
        'gt_bbox': gt_bbox,
        'gt_yaw': gt_yaw,
        'gt_position': gt_position,
    }


def visualize_batch_comparison(results, output_path=None, show=False):
    """Create a summary comparison plot for multiple samples."""
    n_samples = len(results)
    
    # Compute errors
    bbox_errors = []
    yaw_errors = []
    pos_errors = []
    
    for r in results:
        if r['pred_bbox'] is not None:
            bbox_errors.append(np.abs(r['pred_bbox'] - r['gt_bbox']).mean())
        if r['pred_yaw'] is not None:
            yaw_diff = r['pred_yaw'] - r['gt_yaw']
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            yaw_errors.append(np.abs(np.degrees(yaw_diff)))
        if r['pred_position'] is not None:
            pos_errors.append(np.linalg.norm(r['pred_position'] - r['gt_position']))
    
    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # BBox errors
    if bbox_errors:
        ax1 = axes[0]
        ax1.bar(range(len(bbox_errors)), bbox_errors, color='steelblue')
        ax1.axhline(y=np.mean(bbox_errors), color='red', linestyle='--', label=f'Mean: {np.mean(bbox_errors):.4f}')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('BBox MAE')
        ax1.set_title('BBox Prediction Error')
        ax1.legend()
    
    # Yaw errors
    if yaw_errors:
        ax2 = axes[1]
        ax2.bar(range(len(yaw_errors)), yaw_errors, color='coral')
        ax2.axhline(y=np.mean(yaw_errors), color='red', linestyle='--', label=f'Mean: {np.mean(yaw_errors):.1f}°')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Yaw Error (degrees)')
        ax2.set_title('Yaw Prediction Error')
        ax2.legend()
    
    # Position errors
    if pos_errors:
        ax3 = axes[2]
        ax3.bar(range(len(pos_errors)), pos_errors, color='seagreen')
        ax3.axhline(y=np.mean(pos_errors), color='red', linestyle='--', label=f'Mean: {np.mean(pos_errors):.4f}')
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Position Error')
        ax3.set_title('Position Prediction Error')
        ax3.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved summary to {output_path}')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_position_scatter(results, output_path=None, show=False):
    """Create scatter plot of GT vs Predicted positions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gt_x, gt_y = [], []
    pred_x, pred_y = [], []
    
    for r in results:
        if r['pred_position'] is not None:
            gt_x.append(r['gt_position'][0])
            gt_y.append(r['gt_position'][1])
            pred_x.append(r['pred_position'][0])
            pred_y.append(r['pred_position'][1])
    
    if gt_x:
        # X coordinate comparison
        ax1 = axes[0]
        ax1.scatter(gt_x, pred_x, c='steelblue', alpha=0.7, s=100)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        ax1.set_xlabel('GT X')
        ax1.set_ylabel('Pred X')
        ax1.set_title('Position X: GT vs Predicted')
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        
        # Y coordinate comparison
        ax2 = axes[1]
        ax2.scatter(gt_y, pred_y, c='coral', alpha=0.7, s=100)
        ax2.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        ax2.set_xlabel('GT Y')
        ax2.set_ylabel('Pred Y')
        ax2.set_title('Position Y: GT vs Predicted')
        ax2.legend()
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved position scatter to {output_path}')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()
    
    # Setup
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load config
    cfg = load_config(args.config)
    img_size = cfg['data']['img_size']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f'Loading model from {args.checkpoint}')
    model = load_model(args.checkpoint, cfg, device)
    
    # Load dataset
    data_json = args.data_json or cfg['data']['val_json']
    print(f'Loading data from {data_json}')
    
    dataset = CrossViewDataset(
        json_path=data_json,
        data_root=cfg['data']['data_root'],
        crop_size=cfg['data']['crop_size'],
        random_crop=False,
    )
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    print(f'Visualizing {num_samples} samples...')
    
    results = []
    for i in range(num_samples):
        sample = dataset[i]
        output_path = output_dir / f'sample_{i:04d}.png'
        
        result = visualize_single_sample(
            model, sample, device, img_size,
            output_path=output_path,
            show=args.show,
            sample_idx=i,
        )
        results.append(result)
    
    # Create summary plots
    print('Creating summary plots...')
    visualize_batch_comparison(results, output_dir / 'summary_errors.png', show=args.show)
    visualize_position_scatter(results, output_dir / 'position_scatter.png', show=args.show)
    
    # Print summary statistics
    print('\n' + '='*50)
    print('Summary Statistics')
    print('='*50)
    
    bbox_errors = [np.abs(r['pred_bbox'] - r['gt_bbox']).mean() for r in results if r['pred_bbox'] is not None]
    if bbox_errors:
        print(f'BBox MAE: {np.mean(bbox_errors):.4f} ± {np.std(bbox_errors):.4f}')
    
    yaw_errors = []
    for r in results:
        if r['pred_yaw'] is not None:
            yaw_diff = r['pred_yaw'] - r['gt_yaw']
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            yaw_errors.append(np.abs(np.degrees(yaw_diff)))
    if yaw_errors:
        print(f'Yaw Error: {np.mean(yaw_errors):.1f}° ± {np.std(yaw_errors):.1f}°')
    
    pos_errors = [np.linalg.norm(r['pred_position'] - r['gt_position']) for r in results if r['pred_position'] is not None]
    if pos_errors:
        print(f'Position Error: {np.mean(pos_errors):.4f} ± {np.std(pos_errors):.4f}')
    
    print(f'\nVisualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
