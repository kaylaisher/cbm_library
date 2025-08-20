"""
Base configuration for CBM models
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json


@dataclass
class CBMConfig:
    """Simple base configuration for all CBM models"""
    
    # Model Architecture
    num_concepts: int = 100
    num_classes: int = 10
    device: str = "cuda"
    backbone: str = "resnet18"
    
    # Training Parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 100
    weight_decay: float = 1e-4
    
    # Dataset
    dataset: str = "cifar10"
    data_dir: str = "./data"
    
    # Validation & Early Stopping
    validation_split: float = 0.2
    patience: int = 50
    min_delta: float = 1e-4
    
    # Logging & Saving
    save_dir: str = "./saved_models"
    log_dir: str = "./logs"
    save_frequency: int = 10
    experiment_name: str = "cbm_experiment"
    
    # Evaluation
    eval_batch_size: int = 128
    top_k: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CBMConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CBMConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """Simple validation - returns list of issues"""
        issues = []
        
        if self.num_concepts <= 0:
            issues.append("num_concepts must be positive")
        
        if self.num_classes <= 0:
            issues.append("num_classes must be positive")
            
        if self.batch_size <= 0:
            issues.append("batch_size must be positive")
            
        if self.learning_rate <= 0:
            issues.append("learning_rate must be positive")
            
        if not (0 < self.validation_split < 1):
            issues.append("validation_split must be between 0 and 1")
            
        valid_devices = ['cpu', 'cuda', 'mps']
        if self.device not in valid_devices and not self.device.startswith('cuda:'):
            issues.append(f"device must be one of {valid_devices} or 'cuda:N'")

        return issues
    
    def __repr__(self) -> str:
        return f"CBMConfig(dataset={self.dataset}, concepts={self.num_concepts}, classes={self.num_classes})"

