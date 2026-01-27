"""
TensorBoard Logger for Training
"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard logger with distributed training support."""
    
    def __init__(self, log_dir: str, enabled: bool = True, rank: int = 0):
        """
        Args:
            log_dir: Directory to save TensorBoard logs
            enabled: Whether to enable logging
            rank: Process rank (only rank 0 logs)
        """
        self.enabled = enabled and (rank == 0)
        self.writer = None
        
        if self.enabled:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to: {log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalar values."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_dict(self, prefix: str, metrics: dict, step: int):
        """Log a dictionary of metrics."""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)
    
    def log_image(self, tag: str, img_tensor, step: int):
        """Log an image."""
        if self.writer:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def flush(self):
        """Flush pending logs to disk."""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()
            self.writer = None
