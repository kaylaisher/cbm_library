"""
Enhanced early stopping utilities for CBM training
"""

import torch
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
import copy

from ..utils.logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation and early stopping"""
    enabled: bool = True
    patience: int = 50
    min_delta: float = 1e-4
    metric_name: str = "loss"
    minimize: bool = True
    save_best: bool = True
    validation_frequency: int = 100
    restore_best_weights: bool = True
    
    # Advanced options
    warmup_steps: int = 0  # Skip early stopping for first N steps
    decay_patience: bool = False  # Reduce patience over time
    min_patience: int = 10  # Minimum patience when using decay
    
    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        issues = []
        
        if self.patience <= 0:
            issues.append("Patience must be positive")
        if self.min_delta < 0:
            issues.append("Min delta must be non-negative")
        if self.validation_frequency <= 0:
            issues.append("Validation frequency must be positive")
        if self.warmup_steps < 0:
            issues.append("Warmup steps must be non-negative")
        if self.decay_patience and self.min_patience <= 0:
            issues.append("Min patience must be positive when using decay")
        
        return issues


class EarlyStopping:
    """
    Enhanced early stopping with multiple metrics and advanced features
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Validate configuration
        config_issues = config.validate()
        if config_issues:
            logger.warning(f"⚠️ Configuration issues: {config_issues}")
        
        # Initialize state
        self.best_score = float('inf') if config.minimize else float('-inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.step_counter = 0
        self.stopped = False
        
        # History tracking
        self.score_history = []
        self.patience_history = []
        self.improvement_history = []
        
        # Timing
        self.start_time = time.time()
        self.best_score_time = self.start_time
        
        logger.info(f"🛡️ Early stopping initialized: patience={config.patience}, "
                   f"metric={config.metric_name}, minimize={config.minimize}")
    
    def __call__(self, 
                 metric_value: float, 
                 model_state: Optional[Dict] = None,
                 step: Optional[int] = None) -> bool:
        """
        Check if training should stop
        
        Args:
            metric_value: Current metric value
            model_state: Current model state for saving best model
            step: Current training step
        
        Returns:
            True if training should stop, False otherwise
        """
        if not self.config.enabled:
            return False
        
        self.step_counter += 1
        current_step = step if step is not None else self.step_counter
        
        # Skip early stopping during warmup
        if current_step < self.config.warmup_steps:
            logger.debug(f"Early stopping skipped (warmup): {current_step}/{self.config.warmup_steps}")
            return False
        
        # Check for improvement
        improved = self._check_improvement(metric_value)
        
        # Update history
        self.score_history.append({
            'step': current_step,
            'score': metric_value,
            'improved': improved,
            'timestamp': time.time() - self.start_time
        })
        
        if improved:
            self._handle_improvement(metric_value, model_state, current_step)
        else:
            self._handle_no_improvement(current_step)
        
        # Check if should stop
        current_patience = self._get_current_patience()
        should_stop = self.patience_counter >= current_patience
        
        if should_stop:
            self._trigger_early_stopping(current_step)
            return True
        
        return False
    
    def _check_improvement(self, metric_value: float) -> bool:
        """Check if the current metric represents an improvement"""
        if self.config.minimize:
            return metric_value < self.best_score - self.config.min_delta
        else:
            return metric_value > self.best_score + self.config.min_delta
    
    def _handle_improvement(self, metric_value: float, model_state: Optional[Dict], step: int):
        """Handle case when metric improved"""
        old_best = self.best_score
        self.best_score = metric_value