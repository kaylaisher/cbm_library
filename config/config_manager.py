#cbm_library/config/config_manager.py
"""
Enhanced configuration management for CBM library
"""

import json
import yaml
from typing import Dict, Any, List, Optional, Union, Type
from pathlib import Path
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

from ..utils.logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)


class ConfigManager:
    """
    Enhanced configuration management with validation, history tracking, and templates
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict.copy() if config_dict else {}
        self.history = [self.config.copy()]
        self.validators = {}
        self.defaults = {}
        self.locked_keys = set()
        self.change_log = []
        self.max_history = 100  # keep last 100 snapshots
        
        logger.debug(f"ConfigManager initialized with {len(self.config)} parameters")
    
    def update(self, **kwargs) -> List[str]:
        """
        Update config with validation and history tracking
        
        Returns:
            List of validation issues
        """
        old_config = self.config.copy()
        issues = []
        changes = []
        
        for key, value in kwargs.items():
            # Check if key is locked
            if key in self.locked_keys:
                logger.warning(f"⚠️ Attempted to modify locked parameter: {key}")
                issues.append(f"Parameter '{key}' is locked and cannot be modified")
                continue
            
            # Validate parameter
            validation_issues = self._validate_parameter(key, value)
            if validation_issues:
                issues.extend(validation_issues)
                logger.warning(f"⚠️ Validation failed for {key}={value}: {validation_issues}")
                continue
            
            # Track changes
            old_value = self.config.get(key)
            if old_value != value:
                changes.append({
                    'key': key,
                    'old_value': old_value,
                    'new_value': value,
                    'timestamp': time.time()
                })
            
            self.config[key] = value
        
        # Update history and change log
        if changes:
            self.history.append(self.config.copy())
            self.change_log.extend(changes)
            logger.debug(f"Updated config: {[c['key'] for c in changes]}")
        
        return issues
    
    def _validate_parameter(self, key: str, value: Any) -> List[str]:
        """Validate a single parameter"""
        issues = []
        
        # Use custom validator if available
        if key in self.validators:
            validator = self.validators[key]
            if callable(validator):
                try:
                    is_valid = validator(value)
                    if not is_valid:
                        issues.append(f"Custom validation failed for {key}")
                except Exception as e:
                    issues.append(f"Validator error for {key}: {e}")
            elif isinstance(validator, dict):
                # Validator dictionary with rules
                if 'type' in validator and not isinstance(value, validator['type']):
                    issues.append(f"{key} must be of type {validator['type'].__name__}")
                
                if 'min' in validator and value < validator['min']:
                    issues.append(f"{key} must be >= {validator['min']}")
                
                if 'max' in validator and value > validator['max']:
                    issues.append(f"{key} must be <= {validator['max']}")
                
                if 'choices' in validator and value not in validator['choices']:
                    issues.append(f"{key} must be one of {validator['choices']}")
        
        # Built-in validations for common parameters
        builtin_issues = self._builtin_validations(key, value)
        issues.extend(builtin_issues)
        
        return issues
    
    def _builtin_validations(self, key: str, value: Any) -> List[str]:
        """Built-in validations for common parameter patterns"""
        issues = []
        
        # Learning rate validation
        if 'learning_rate' in key.lower() or key == 'lr':
            if not isinstance(value, (int, float)) or value <= 0:
                issues.append(f"Learning rate must be positive number, got {value}")
            elif value > 1.0:
                issues.append(f"Learning rate {value} is unusually high (>1.0)")
        
        # Batch size validation
        elif 'batch_size' in key.lower():
            if not isinstance(value, int) or value <= 0:
                issues.append(f"Batch size must be positive integer, got {value}")
        
        # Epoch validation
        elif 'epoch' in key.lower():
            if not isinstance(value, int) or value <= 0:
                issues.append(f"Epochs must be positive integer, got {value}")
        
        # Patience validation
        elif 'patience' in key.lower():
            if not isinstance(value, int) or value <= 0:
                issues.append(f"Patience must be positive integer, got {value}")
        
        # Probability/ratio validation
        elif any(term in key.lower() for term in ['prob', 'ratio', 'rate', 'dropout']):
            if isinstance(value, (int, float)) and not (0 <= value <= 1):
                issues.append(f"{key} should be between 0 and 1, got {value}")
        
        # Device validation
        elif 'device' in key.lower():
            if isinstance(value, str):
                valid_devices = ['cpu', 'cuda', 'mps']
                if not (value in valid_devices or value.startswith('cuda:')):
                    issues.append(f"Device should be one of {valid_devices} or 'cuda:N', got {value}")
        
        return issues
    
    def add_validator(self, key: str, validator: Union[callable, Dict[str, Any]]):
        """Add custom validator for a parameter"""
        self.validators[key] = validator
        logger.debug(f"Added validator for parameter: {key}")
    
    def set_default(self, key: str, value: Any):
        """Set default value for a parameter"""
        self.defaults[key] = value
        if key not in self.config:
            self.config[key] = value
            self.change_log.append({
                'key': key, 'old_value': None, 'new_value': value,
                'timestamp': time.time(), 'action': 'set_default'
            })
            self.history.append(self.config.copy())

    def lock_parameter(self, key: str):
        """Lock a parameter to prevent modification"""
        self.locked_keys.add(key)
        logger.debug(f"Locked parameter: {key}")
    
    def unlock_parameter(self, key: str):
        """Unlock a parameter to allow modification"""
        self.locked_keys.discard(key)
        logger.debug(f"Unlocked parameter: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default fallback"""
        return self.config.get(key, self.defaults.get(key, default))
    
    def get_nested(self, key_path: str, separator: str = '.', default: Any = None) -> Any:
        """Get nested config value using dot notation"""
        keys = key_path.split(separator)
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_nested(self, key_path: str, value: Any, separator: str = '.'):
        """Set nested config value using dot notation"""
        keys = key_path.split(separator)
        config_ref = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set final value
        config_ref[keys[-1]] = value
    
    def merge(self, other_config: Dict[str, Any], overwrite: bool = True) -> List[str]:
        """Merge another configuration"""
        issues = []
        
        for key, value in other_config.items():
            if not overwrite and key in self.config:
                continue
            
            merge_issues = self.update(**{key: value})
            issues.extend(merge_issues)
        
        return issues
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        old_config = self.config.copy()
        self.config = self.defaults.copy()
        self.history.append(self.config.copy())
        
        self.change_log.append({
            'action': 'reset_to_defaults',
            'old_config': old_config,
            'timestamp': time.time()
        })
        
        logger.info("🔄 Configuration reset to defaults")
    
    def revert_to_previous(self, steps_back: int = 1) -> bool:
        """Revert to previous configuration state"""
        if len(self.history) <= steps_back:
            logger.warning(f"⚠️ Cannot revert {steps_back} steps, only {len(self.history)-1} available")
            return False
        
        old_config = self.config.copy()
        self.config = self.history[-(steps_back + 1)].copy()
        
        self.change_log.append({
            'action': f'revert_{steps_back}_steps',
            'old_config': old_config,
            'new_config': self.config.copy(),
            'timestamp': time.time()
        })
        
        logger.info(f"🔄 Reverted configuration {steps_back} steps back")
        return True
    
    def save(self, path: str, format: str = 'auto'):
        """Save configuration to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if format == 'auto':
            format = path.suffix.lower() or 'json'
        
        save_data = {
            'config': self.config,
            'defaults': self.defaults,
            'locked_keys': list(self.locked_keys),
            'history': self.history,
            'change_log': self.change_log,
            'metadata': {
                'saved_at': time.time(),
                'version': '1.0'
            }
        }
        
        try:
            if format in ['.json', 'json']:
                with open(path, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
            elif format in ['.yaml', '.yml', 'yaml']:
                with open(path, 'w') as f:
                    yaml.dump(save_data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"💾 Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            raise
    
    def load(self, path: str, merge_with_current: bool = False):
        """Load configuration from file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            # Determine format and load
            if path.suffix.lower() in ['.json']:
                with open(path, 'r') as f:
                    data = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # Handle different data formats
            if isinstance(data, dict) and 'config' in data:
                # Full save format
                if not merge_with_current:
                    self.config = data['config'].copy()
                    self.defaults = data.get('defaults', {})
                    self.locked_keys = set(data.get('locked_keys', []))
                    self.history = data.get('history', [self.config.copy()])
                    self.change_log = data.get('change_log', [])
                else:
                    self.merge(data['config'])
            else:
                # Simple config format
                if not merge_with_current:
                    self.config = data.copy()
                    self.history = [self.config.copy()]
                else:
                    self.merge(data)
            
            logger.info(f"📂 Configuration loaded from {path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise
    
    def export_for_reproducibility(self) -> Dict[str, Any]:
        """Export minimal config needed for reproducibility"""
        return {
            'config': self.config.copy(),
            'defaults_used': {k: v for k, v in self.defaults.items() if k not in self.config},
            'timestamp': time.time(),
            'git_hash': self._get_git_hash()  # If available
        }
    
    def _get_git_hash(self) -> Optional[str]:
        """Get current git hash if available"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def get_diff(self, other_config: Union['ConfigManager', Dict[str, Any]]) -> Dict[str, Any]:
        """Get difference between this config and another"""
        if isinstance(other_config, ConfigManager):
            other_dict = other_config.config
        else:
            other_dict = other_config
        
        diff = {
            'added': {},
            'modified': {},
            'removed': {}
        }
        
        # Find added and modified
        for key, value in self.config.items():
            if key not in other_dict:
                diff['added'][key] = value
            elif other_dict[key] != value:
                diff['modified'][key] = {
                    'old': other_dict[key],
                    'new': value
                }
        
        # Find removed
        for key, value in other_dict.items():
            if key not in self.config:
                diff['removed'][key] = value
        
        return diff
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all current configuration parameters"""
        all_issues = {}
        
        for key, value in self.config.items():
            issues = self._validate_parameter(key, value)
            if issues:
                all_issues[key] = issues
        
        return all_issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'total_parameters': len(self.config),
            'locked_parameters': len(self.locked_keys),
            'default_parameters': len(self.defaults),
            'history_length': len(self.history),
            'change_count': len(self.change_log),
            'validation_issues': len(self.validate_all()),
            'parameter_types': self._get_parameter_types()
        }
    
    def _get_parameter_types(self) -> Dict[str, int]:
        """Get count of parameters by type"""
        type_counts = defaultdict(int)
        for value in self.config.values():
            type_counts[type(value).__name__] += 1
        return dict(type_counts)
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        return (f"ConfigManager(parameters={summary['total_parameters']}, "
                f"locked={summary['locked_parameters']}, "
                f"history={summary['history_length']})")


@dataclass
class CBMBaseConfig:
    """Base configuration for all CBM methods"""
    # Model architecture
    num_concepts: int = 100
    num_classes: int = 10
    device: str = "cuda"
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 100
    
    # Validation and early stopping
    validation_split: float = 0.2
    patience: int = 50
    min_delta: float = 1e-4
    
    # Logging and saving
    save_dir: str = "./saved_models"
    log_dir: str = "./logs"
    save_frequency: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CBMBaseConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        issues = []
        
        if self.num_concepts <= 0:
            issues.append("num_concepts must be positive")
        
        if self.num_classes <= 0:
            issues.append("num_classes must be positive")
            
        if self.batch_size <= 0:
            issues.append("batch_size must be positive")
            
        if self.learning_rate <= 0 or self.learning_rate > 1:
            issues.append("learning_rate must be between 0 and 1")
            
        if self.max_epochs <= 0:
            issues.append("max_epochs must be positive")
            
        if not (0 < self.validation_split < 1):
            issues.append("validation_split must be between 0 and 1")
            
        if self.patience <= 0:
            issues.append("patience must be positive")
            
        if self.min_delta < 0:
            issues.append("min_delta must be non-negative")
        
        return issues
    
    def __repr__(self) -> str:
        return f"CBMBaseConfig(concepts={self.num_concepts}, classes={self.num_classes}, device={self.device})"
