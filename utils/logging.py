"""
Enhanced logging utilities for CBM Library
"""

import logging
import time
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json


def setup_enhanced_logging(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup enhanced logging with colored output and optional file logging
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Color mapping for different log levels
    def add_color_to_record(record):
        color_map = {
            'DEBUG': '37',    # White
            'INFO': '32',     # Green
            'WARNING': '33',  # Yellow
            'ERROR': '31',    # Red
            'CRITICAL': '35'  # Magenta
        }
        record.levelcolor = color_map.get(record.levelname, '37')
        return True
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '\033[96m%(asctime)s\033[0m - \033[94m%(name)s\033[0m - '
        '\033[%(levelcolor)sm%(levelname)s\033[0m - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.addFilter(add_color_to_record)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class ProgressTracker:
    """Enhanced progress tracking with ETA estimation and metrics history"""
    
    def __init__(self, total_steps: int, update_frequency: int = 10, log_file: Optional[str] = None):
        self.total_steps = total_steps
        self.update_frequency = update_frequency
        self.start_time = time.time()
        self.last_update = self.start_time
        self.step_times = []
        self.metrics_history = []
        self.log_file = log_file
        
        self.logger = setup_enhanced_logging(f"{__name__}.ProgressTracker")
        
    def update(self, step: int, metrics: Optional[Dict[str, float]] = None):
        """
        Update progress with current step and optional metrics
        
        Args:
            step: Current step number
            metrics: Dictionary of metric names and values
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate ETA
        if step > 0:
            avg_step_time = elapsed / step
            eta = avg_step_time * (self.total_steps - step)
        else:
            eta = 0
        
        # Store metrics with timestamp
        if metrics:
            self.metrics_history.append({
                'step': step,
                'timestamp': current_time,
                'elapsed': elapsed,
                'eta': eta,
                **metrics
            })
        
        # Display progress
        if step % self.update_frequency == 0 or step == self.total_steps - 1:
            self._display_progress(step, elapsed, eta, metrics)
        
        # Save to file if specified
        if self.log_file and metrics:
            self._save_metrics_to_file()
    
    def _display_progress(self, step: int, elapsed: float, eta: float, metrics: Optional[Dict[str, float]]):
        """Display formatted progress bar and metrics"""
        progress_pct = (step / self.total_steps) * 100
        bar_length = 30
        filled_length = int(bar_length * step // self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        eta_str = self._format_time(eta)
        elapsed_str = self._format_time(elapsed)
        
        # Format metrics
        metrics_str = ""
        if metrics:
            formatted_metrics = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    if abs(v) < 1e-3 or abs(v) > 1e3:
                        formatted_metrics.append(f"{k}: {v:.2e}")
                    else:
                        formatted_metrics.append(f"{k}: {v:.6f}")
                else:
                    formatted_metrics.append(f"{k}: {v}")
            metrics_str = " | " + " | ".join(formatted_metrics)
        
        # Display progress
        progress_msg = (f"Progress: |{bar}| {progress_pct:6.2f}% | "
                       f"{step:4d}/{self.total_steps} | "
                       f"⏱️ {elapsed_str} | 🔮 ETA: {eta_str}{metrics_str}")
        
        self.logger.info(progress_msg)
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _save_metrics_to_file(self):
        """Save metrics history to JSON file"""
        if not self.log_file:
            return
        
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save metrics to file: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        if not self.metrics_history:
            return {}
        
        total_time = time.time() - self.start_time
        
        summary = {
            'total_time': total_time,
            'total_steps': len(self.metrics_history),
            'avg_step_time': total_time / len(self.metrics_history) if self.metrics_history else 0,
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {}
        }
        
        # Calculate metric statistics
        if self.metrics_history:
            metrics_keys = set()
            for entry in self.metrics_history:
                metrics_keys.update(k for k in entry.keys() 
                                  if k not in ['step', 'timestamp', 'elapsed', 'eta'])
            
            for metric in metrics_keys:
                values = [entry[metric] for entry in self.metrics_history if metric in entry]
                if values:
                    summary[f'{metric}_min'] = min(values)
                    summary[f'{metric}_max'] = max(values)
                    summary[f'{metric}_avg'] = sum(values) / len(values)
        
        return summary


@dataclass
class TrainingProgress:
    """Data class for tracking training progress state"""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = float('inf')
    best_loss: float = float('inf')
    best_metric: float = 0.0
    patience_counter: int = 0
    learning_rate: float = 0.001
    elapsed_time: float = 0.0
    eta: float = 0.0
    
    def update(self, **kwargs):
        """Update progress with validation"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def is_best(self, metric_value: float, minimize: bool = True) -> bool:
        """Check if current metric is the best so far"""
        if minimize:
            return metric_value < self.best_loss
        else:
            return metric_value > self.best_metric


class MetricsLogger:
    """Advanced metrics logging with statistical analysis"""
    
    def __init__(self, save_dir: str, experiment_name: str = "cbm_training"):
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.metrics_history = []
        self.start_time = time.time()
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_enhanced_logging(f"{__name__}.MetricsLogger")
        self.logger.info(f"📊 Metrics logger initialized: {self.save_dir}")
    
    def log_metrics(self, step: int, metrics: Dict[str, Any], stage: str = "training"):
        """Log metrics with timestamp and stage information"""
        entry = {
            'step': step,
            'stage': stage,
            'timestamp': time.time() - self.start_time,
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
            **metrics
        }
        
        self.metrics_history.append(entry)
        
        # Auto-save every 100 entries
        if len(self.metrics_history) % 100 == 0:
            self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        filename = f"{self.experiment_name}_metrics.json"
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            self.logger.debug(f"💾 Saved metrics to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save metrics: {e}")
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a specific metric"""
        values = [entry[metric_name] for entry in self.metrics_history 
                 if metric_name in entry and isinstance(entry[metric_name], (int, float))]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1]
        }
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model configuration and architecture info"""
        info_file = self.save_dir / f"{self.experiment_name}_model_info.json"
        
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        self.logger.info(f"📋 Model info saved to {info_file}")


# Convenience function for quick setup
def setup_training_logging(experiment_name: str, 
                          log_dir: str = "logs",
                          level: int = logging.INFO) -> tuple[logging.Logger, MetricsLogger]:
    """
    Quick setup for training logging
    
    Returns:
        Tuple of (logger, metrics_logger)
    """
    # Setup main logger
    log_file = Path(log_dir) / f"{experiment_name}.log"
    logger = setup_enhanced_logging(experiment_name, level, str(log_file))
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(log_dir, experiment_name)
    
    logger.info(f"🚀 Training logging setup complete for {experiment_name}")
    
    return logger, metrics_logger