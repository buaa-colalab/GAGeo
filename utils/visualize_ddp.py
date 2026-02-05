"""
DDP-compatible visualization utilities
适用于原生 DDP 训练的可视化函数
"""

import torch
from pathlib import Path
from tqdm import tqdm
from .visualize import visualize_batch_sample


def visualize_validation_samples_ddp(
    model, 
    dataloader, 
    device, 
    cfg, 
    epoch, 
    is_main_process=True,
    num_samples=10,
    prompt_type='point'
):
    """
    Visualize a few validation samples during DDP training
    Only runs on main process to avoid duplicate saves
    
    Args:
        model: the model (already wrapped with DDP)
        dataloader: validation dataloader
        device: torch device
        cfg: config dict
        epoch: current epoch number
        is_main_process: whether this is the main process
        num_samples: number of samples to visualize
        prompt_type: type of prompt to use ('point', 'bbox', or 'mask')
    """
    if not is_main_process:
        return
    
    model.eval()
    output_dir = Path(cfg['checkpoint']['output_dir']) / 'vis' / f'epoch_{epoch}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_size = cfg['data']['img_size']
    samples_saved = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Visualizing', disable=not is_main_process):
            if samples_saved >= num_samples:
                break
            
            # Move to device
            front_view = batch['front_view'].to(device)
            sat_view = batch['satellite_view'].to(device)
            
            # Use specified prompt type
            from utils.prompt_utils import prepare_single_prompt
            points, boxes, masks = prepare_single_prompt(batch, device, prompt_type=prompt_type)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                outputs = model(
                    front_view=front_view,
                    satellite_view=sat_view,
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )
            
            # Visualize samples from this batch
            batch_size = min(front_view.shape[0], num_samples - samples_saved)
            for i in range(batch_size):
                save_path = output_dir / f'sample_{samples_saved:03d}_{prompt_type}.png'
                visualize_batch_sample(batch, outputs, i, img_size, save_path, prompt_type=prompt_type)
                samples_saved += 1
                
                if samples_saved >= num_samples:
                    break
    
    print(f'Saved {samples_saved} visualizations to {output_dir}')
    model.train()
