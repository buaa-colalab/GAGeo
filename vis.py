"""
Visualization Script for Cross-View Localization
可视化预测的相机位置和bbox - 精简版
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

from models import CrossViewLocalizer
from data.dataset import CrossViewDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Cross-View Localization Results')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data_json', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./visualizations')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def load_model(checkpoint_path, cfg, device):
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
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    return model


def to_numpy_img(img_tensor):
    return img_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def draw_bbox(ax, bbox, img_size, color, linestyle, linewidth, label):
    cx, cy, w, h = bbox
    x1, y1 = (cx - w/2) * img_size, (cy - h/2) * img_size
    x2, y2 = (cx + w/2) * img_size, (cy + h/2) * img_size
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=linewidth, 
                            edgecolor=color, facecolor='none', linestyle=linestyle, label=label)
    ax.add_patch(rect)


def draw_camera_pose(ax, position, yaw, img_size, color, marker, label_pos, label_yaw):
    x, y = position[0] * img_size, position[1] * img_size
    ax.scatter(x, y, c=color, marker=marker, s=150, label=label_pos, zorder=5, edgecolors='white', linewidths=2)
    
    dx, dy = 40 * np.sin(yaw), -40 * np.cos(yaw)
    arrow = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=15, 
                           color=color, linewidth=2, label=label_yaw)
    ax.add_patch(arrow)


def visualize_sample(model, sample, device, img_size, output_path, show):
    with torch.no_grad():
        front_view = sample['front_view'].unsqueeze(0).to(device)
        sat_view = sample['satellite_view'].unsqueeze(0).to(device)
        mono_point = sample['mono_point'].unsqueeze(0).to(device)
        mono_mask = sample['mono_mask'].unsqueeze(0).to(device) if 'mono_mask' in sample else None
        
        point_coords = mono_point.unsqueeze(1)
        point_labels = torch.ones(1, 1, device=device)
        
        outputs = model(
            front_view=front_view, 
            satellite_view=sat_view, 
            points=(point_coords, point_labels),
            masks=mono_mask
        )
    
    pred_bbox = outputs['pred_boxes'][0, 0].cpu().numpy() if 'pred_boxes' in outputs else None
    pred_yaw = outputs['yaw_radians'][0].cpu().item() if 'yaw_radians' in outputs else None
    pred_position = outputs['position'][0].cpu().numpy() if 'position' in outputs else None
    
    gt_bbox = sample['sat_bbox'].numpy()
    gt_yaw = sample['yaw_radians'].item()
    gt_position = sample['camera_position'].numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Front view
    front_img = to_numpy_img(sample['front_view'])
    axes[0].imshow(front_img)
    mono_pt = sample['mono_point'].numpy()
    h, w = front_img.shape[:2]
    pt_x, pt_y = (mono_pt[0], mono_pt[1]) if 0 <= mono_pt[0] < w else (mono_pt[0]*w, mono_pt[1]*h)
    axes[0].scatter(pt_x, pt_y, c='lime', marker='x', s=200, linewidths=3, label='Mono Point')
    axes[0].set_title('Front View', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].axis('off')
    
    # Satellite - BBox
    sat_img = to_numpy_img(sample['satellite_view'])
    axes[1].imshow(sat_img)
    draw_bbox(axes[1], gt_bbox, img_size, 'lime', '-', 3, 'GT BBox')
    if pred_bbox is not None:
        draw_bbox(axes[1], pred_bbox, img_size, 'red', '--', 2, 'Pred BBox')
    axes[1].set_title('Satellite View - BBox', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].axis('off')
    
    # Satellite - Camera Pose
    axes[2].imshow(sat_img)
    draw_camera_pose(axes[2], gt_position, gt_yaw, img_size, 'lime', 'o', 'GT Position', 'GT Yaw')
    if pred_position is not None and pred_yaw is not None:
        draw_camera_pose(axes[2], pred_position, pred_yaw, img_size, 'red', '^', 'Pred Position', 'Pred Yaw')
    axes[2].set_title('Satellite View - Camera Pose', fontsize=14)
    axes[2].legend(loc='upper right')
    axes[2].axis('off')
    
    # Metrics
    metrics = []
    if pred_bbox is not None:
        metrics.append(f'BBox MAE: {np.abs(pred_bbox - gt_bbox).mean():.4f}')
    if pred_yaw is not None:
        yaw_diff = np.arctan2(np.sin(pred_yaw - gt_yaw), np.cos(pred_yaw - gt_yaw))
        metrics.append(f'Yaw Error: {np.abs(np.degrees(yaw_diff)):.1f}°')
    if pred_position is not None:
        metrics.append(f'Pos Error: {np.linalg.norm(pred_position - gt_position):.4f}')
    
    if metrics:
        fig.suptitle(' | '.join(metrics), fontsize=12, y=0.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return {'pred_bbox': pred_bbox, 'pred_yaw': pred_yaw, 'pred_position': pred_position,
            'gt_bbox': gt_bbox, 'gt_yaw': gt_yaw, 'gt_position': gt_position}


def plot_summary(results, output_dir, show):
    bbox_errors = [np.abs(r['pred_bbox'] - r['gt_bbox']).mean() for r in results if r['pred_bbox'] is not None]
    yaw_errors = [np.abs(np.degrees(np.arctan2(np.sin(r['pred_yaw'] - r['gt_yaw']), 
                  np.cos(r['pred_yaw'] - r['gt_yaw'])))) for r in results if r['pred_yaw'] is not None]
    pos_errors = [np.linalg.norm(r['pred_position'] - r['gt_position']) for r in results if r['pred_position'] is not None]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if bbox_errors:
        axes[0].bar(range(len(bbox_errors)), bbox_errors, color='steelblue')
        axes[0].axhline(np.mean(bbox_errors), color='red', linestyle='--', label=f'Mean: {np.mean(bbox_errors):.4f}')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('BBox MAE')
        axes[0].set_title('BBox Error')
        axes[0].legend()
    
    if yaw_errors:
        axes[1].bar(range(len(yaw_errors)), yaw_errors, color='coral')
        axes[1].axhline(np.mean(yaw_errors), color='red', linestyle='--', label=f'Mean: {np.mean(yaw_errors):.1f}°')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Yaw Error (°)')
        axes[1].set_title('Yaw Error')
        axes[1].legend()
    
    if pos_errors:
        axes[2].bar(range(len(pos_errors)), pos_errors, color='seagreen')
        axes[2].axhline(np.mean(pos_errors), color='red', linestyle='--', label=f'Mean: {np.mean(pos_errors):.4f}')
        axes[2].set_xlabel('Sample')
        axes[2].set_ylabel('Position Error')
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
    dataset = CrossViewDataset(json_path=data_json, data_root=cfg['data']['data_root'],
                               crop_size=cfg['data']['crop_size'], random_crop=False)
    
    num_samples = min(args.num_samples, len(dataset))
    print(f'Visualizing {num_samples} samples...')
    
    results = []
    for i in tqdm(range(num_samples)):
        result = visualize_sample(model, dataset[i], device, cfg['data']['img_size'],
                                 output_dir / f'sample_{i:04d}.png', args.show)
        results.append(result)
    
    plot_summary(results, output_dir, args.show)
    
    print(f'\n{"="*50}')
    print('Summary Statistics')
    print(f'{"="*50}')
    
    bbox_errors = [np.abs(r['pred_bbox'] - r['gt_bbox']).mean() for r in results if r['pred_bbox'] is not None]
    if bbox_errors:
        print(f'BBox MAE: {np.mean(bbox_errors):.4f} ± {np.std(bbox_errors):.4f}')
    
    yaw_errors = [np.abs(np.degrees(np.arctan2(np.sin(r['pred_yaw'] - r['gt_yaw']), 
                  np.cos(r['pred_yaw'] - r['gt_yaw'])))) for r in results if r['pred_yaw'] is not None]
    if yaw_errors:
        print(f'Yaw Error: {np.mean(yaw_errors):.1f}° ± {np.std(yaw_errors):.1f}°')
    
    pos_errors = [np.linalg.norm(r['pred_position'] - r['gt_position']) for r in results if r['pred_position'] is not None]
    if pos_errors:
        print(f'Position Error: {np.mean(pos_errors):.4f} ± {np.std(pos_errors):.4f}')
    
    print(f'\nVisualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
