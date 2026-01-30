"""
Training-time visualization utilities
轻量级可视化函数，用于训练过程中的快速检查
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


def to_numpy_img(img_tensor):
    """Convert tensor to numpy image [H, W, 3]"""
    return img_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def draw_bbox(ax, bbox, img_size, color, linestyle, linewidth, label):
    """Draw bounding box on axis"""
    cx, cy, w, h = bbox
    x1, y1 = (cx - w/2) * img_size, (cy - h/2) * img_size
    x2, y2 = (cx + w/2) * img_size, (cy + h/2) * img_size
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=linewidth, 
                            edgecolor=color, facecolor='none', linestyle=linestyle, label=label)
    ax.add_patch(rect)


def draw_camera_pose(ax, position, yaw, img_size, color, marker, label_pos):
    """Draw camera position and orientation"""
    x, y = position[0] * img_size, position[1] * img_size
    ax.scatter(x, y, c=color, marker=marker, s=150, label=label_pos, zorder=5, 
              edgecolors='white', linewidths=2)
    
    dx, dy = 40 * np.sin(yaw), -40 * np.cos(yaw)
    arrow = FancyArrowPatch((x, y), (x+dx, y+dy), arrowstyle='->', mutation_scale=15, 
                           color=color, linewidth=2)
    ax.add_patch(arrow)


def visualize_batch_sample(batch, outputs, idx, img_size, save_path):
    """
    Visualize a single sample from batch during training
    
    Args:
        batch: dict with 'front_view', 'satellite_view', 'mono_point', 'sat_bbox', etc.
        outputs: model outputs dict
        idx: index in batch to visualize
        img_size: image size for coordinate conversion
        save_path: where to save the figure
    """
    # Extract data for this sample
    front_img = to_numpy_img(batch['front_view'][idx])
    sat_img = to_numpy_img(batch['satellite_view'][idx])
    mono_pt = batch['mono_point'][idx].cpu().numpy()
    
    gt_bbox = batch['sat_bbox'][idx].cpu().numpy()
    gt_yaw = batch['yaw_radians'][idx].cpu().item()
    gt_position = batch['camera_position'][idx].cpu().numpy()
    
    # Extract predictions
    pred_bbox = outputs['pred_boxes'][idx, 0].detach().cpu().numpy() if 'pred_boxes' in outputs else None
    pred_yaw = outputs['yaw_radians'][idx].detach().cpu().item() if 'yaw_radians' in outputs else None
    pred_position = outputs['position'][idx].detach().cpu().numpy() if 'position' in outputs else None
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Front view with mono point
    axes[0].imshow(front_img)
    h, w = front_img.shape[:2]
    pt_x, pt_y = mono_pt[0], mono_pt[1]
    axes[0].scatter(pt_x, pt_y, c='lime', marker='x', s=200, linewidths=3, label='Mono Point')
    axes[0].set_title('Front View', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].axis('off')
    
    # Satellite - BBox
    axes[1].imshow(sat_img)
    draw_bbox(axes[1], gt_bbox, img_size, 'lime', '-', 3, 'GT BBox')
    if pred_bbox is not None:
        draw_bbox(axes[1], pred_bbox, img_size, 'red', '--', 2, 'Pred BBox')
    axes[1].set_title('Satellite - BBox', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].axis('off')
    
    # Satellite - Camera Pose
    axes[2].imshow(sat_img)
    draw_camera_pose(axes[2], gt_position, gt_yaw, img_size, 'lime', 'o', 'GT')
    if pred_position is not None and pred_yaw is not None:
        draw_camera_pose(axes[2], pred_position, pred_yaw, img_size, 'red', '^', 'Pred')
    axes[2].set_title('Satellite - Pose', fontsize=14)
    axes[2].legend(loc='upper right')
    axes[2].axis('off')
    
    # Add metrics as title
    metrics = []
    if pred_bbox is not None:
        metrics.append(f'BBox MAE: {np.abs(pred_bbox - gt_bbox).mean():.4f}')
    if pred_yaw is not None:
        yaw_diff = np.arctan2(np.sin(pred_yaw - gt_yaw), np.cos(pred_yaw - gt_yaw))
        metrics.append(f'Yaw: {np.abs(np.degrees(yaw_diff)):.1f}°')
    if pred_position is not None:
        metrics.append(f'Pos: {np.linalg.norm(pred_position - gt_position):.4f}')
    
    if metrics:
        fig.suptitle(' | '.join(metrics), fontsize=12, y=0.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_validation_samples(model, dataloader, accelerator, cfg, epoch, num_samples=10):
    """
    Visualize a few validation samples during training
    Only runs on main process to avoid duplicate saves
    
    Args:
        model: the model (already wrapped by accelerator)
        dataloader: validation dataloader (already prepared by accelerator)
        accelerator: Accelerator instance
        cfg: config dict
        epoch: current epoch number
        num_samples: number of samples to visualize
    """
    if not accelerator.is_main_process:
        return
    
    import torch
    
    model.eval()
    output_dir = Path(cfg['checkpoint']['output_dir']) / 'vis' / f'epoch_{epoch}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_size = cfg['data']['img_size']
    samples_saved = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_saved >= num_samples:
                break
            
            # Prepare inputs
            front_view = batch['front_view']
            sat_view = batch['satellite_view']
            mono_point = batch['mono_point']
            
            B = front_view.shape[0]
            point_coords = mono_point.unsqueeze(1)
            point_labels = torch.ones(B, 1, device=front_view.device)
            
            # Forward pass
            with accelerator.autocast():
                outputs = model(
                    front_view=front_view,
                    satellite_view=sat_view,
                    points=(point_coords, point_labels),
                )
            
            # Visualize samples from this batch
            batch_size = min(B, num_samples - samples_saved)
            for i in range(batch_size):
                save_path = output_dir / f'sample_{samples_saved:03d}.png'
                visualize_batch_sample(batch, outputs, i, img_size, save_path)
                samples_saved += 1
                
                if samples_saved >= num_samples:
                    break
    
    accelerator.print(f'Saved {samples_saved} visualizations to {output_dir}')
    model.train()
