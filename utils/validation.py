"""
Enhanced validation utilities for CBM models and training data
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import warnings

from .logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)


@dataclass
class ValidationResult:
    """Structured validation result"""
    valid: bool
    issues: List[str]
    warnings: List[str]
    info: Dict[str, Any]
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []
        if self.info is None:
            self.info = {}
    
    def add_issue(self, issue: str):
        """Add a validation issue"""
        self.issues.append(issue)
        self.valid = False
    
    def add_warning(self, warning: str):
        """Add a validation warning"""
        self.warnings.append(warning)
    
    def add_info(self, key: str, value: Any):
        """Add information"""
        self.info[key] = value
    
    def log_results(self, logger_instance=None):
        """Log validation results"""
        log = logger_instance or logger
        
        if self.valid:
            log.info("✅ Validation passed")
        else:
            log.error("❌ Validation failed")
        
        for issue in self.issues:
            log.error(f"  🔴 {issue}")
        
        for warning in self.warnings:
            log.warning(f"  🟡 {warning}")


class ModelValidator:
    """Comprehensive model validation utilities"""
    
    @staticmethod
    def validate_model_components(model) -> ValidationResult:
        """Validate that all model components are properly initialized"""
        result = ValidationResult(valid=True, issues=[], warnings=[], info={})
        
        # Check backbone
        if not hasattr(model, 'backbone') or model.backbone is None:
            result.add_issue("Backbone model is None")
        elif not isinstance(model.backbone, nn.Module):
            result.add_issue("Backbone is not a PyTorch module")
        else:
            result.add_info("backbone_type", type(model.backbone).__name__)
            
            # Check if backbone has parameters
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            result.add_info("backbone_parameters", backbone_params)
            
            if backbone_params == 0:
                result.add_warning("Backbone has no parameters")
        
        # Check concept layer
        if hasattr(model, 'concept_layer'):
            if model.concept_layer is None:
                result.add_warning("Concept layer is None (may not be trained yet)")
            elif not isinstance(model.concept_layer, nn.Module):
                result.add_issue("Concept layer is not a PyTorch module")
            else:
                concept_params = sum(p.numel() for p in model.concept_layer.parameters())
                result.add_info("concept_layer_parameters", concept_params)
        
        # Check final layer
        if hasattr(model, 'final_layer'):
            if model.final_layer is None:
                result.add_warning("Final layer is None (may not be trained yet)")
            elif not isinstance(model.final_layer, nn.Module):
                result.add_issue("Final layer is not a PyTorch module")
            else:
                final_params = sum(p.numel() for p in model.final_layer.parameters())
                result.add_info("final_layer_parameters", final_params)
        
        # Check concepts
        if hasattr(model, 'concept_names'):
            if not isinstance(model.concept_names, list):
                result.add_issue("Concept names should be a list")
            elif len(model.concept_names) == 0:
                result.add_warning("No concepts defined")
            else:
                result.add_info("num_concepts", len(model.concept_names))
                
                # Check for duplicate concepts
                unique_concepts = set(model.concept_names)
                if len(unique_concepts) != len(model.concept_names):
                    result.add_warning("Duplicate concepts detected")
                    result.add_info("unique_concepts", len(unique_concepts))
        
        # Check device consistency
        devices = set()
        if hasattr(model, 'backbone') and model.backbone is not None:
            try:
                backbone_device = next(model.backbone.parameters()).device
                devices.add(str(backbone_device))
            except StopIteration:
                result.add_warning("Backbone has no parameters to check device")
        
        if hasattr(model, 'concept_layer') and model.concept_layer is not None:
            try:
                concept_device = next(model.concept_layer.parameters()).device
                devices.add(str(concept_device))
            except StopIteration:
                pass
        
        if hasattr(model, 'final_layer') and model.final_layer is not None:
            try:
                final_device = next(model.final_layer.parameters()).device
                devices.add(str(final_device))
            except StopIteration:
                pass
        
        if len(devices) > 1:
            result.add_issue(f"Model components on different devices: {devices}")
        elif len(devices) == 1:
            result.add_info("device", list(devices)[0])
        
        return result
    
    @staticmethod
    def validate_training_data(dataset, labels=None) -> ValidationResult:
        """Validate training data format and content"""
        result = ValidationResult(valid=True, issues=[], warnings=[], info={})
        
        try:
            # Check dataset basic properties
            if not hasattr(dataset, '__len__'):
                result.add_issue("Dataset doesn't have __len__ method")
                return result
            
            dataset_len = len(dataset)
            if dataset_len == 0:
                result.add_issue("Dataset is empty")
                return result
            
            result.add_info("dataset_length", dataset_len)
            
            # Check data consistency by sampling
            if hasattr(dataset, '__getitem__'):
                try:
                    # Sample first item
                    sample = dataset[0]
                    
                    if isinstance(sample, (tuple, list)):
                        if len(sample) < 2:
                            result.add_warning("Dataset samples have less than 2 elements")
                        else:
                            data, target = sample[0], sample[1]
                            
                            # Validate data tensor
                            if not isinstance(data, torch.Tensor):
                                result.add_issue("Data samples are not tensors")
                            else:
                                result.add_info("data_shape", list(data.shape))
                                result.add_info("data_dtype", str(data.dtype))
                                
                                # Check for NaN or inf values
                                if torch.isnan(data).any():
                                    result.add_issue("Data contains NaN values")
                                if torch.isinf(data).any():
                                    result.add_issue("Data contains infinite values")
                            
                            # Validate target
                            if isinstance(target, torch.Tensor):
                                result.add_info("target_shape", list(target.shape))
                                result.add_info("target_dtype", str(target.dtype))
                            elif isinstance(target, (int, float)):
                                result.add_info("target_type", "scalar")
                            else:
                                result.add_warning(f"Unexpected target type: {type(target)}")
                    
                    # Sample a few more items to check consistency
                    sample_indices = np.linspace(0, dataset_len-1, min(10, dataset_len), dtype=int)
                    shapes = []
                    dtypes = []
                    
                    for idx in sample_indices:
                        try:
                            sample = dataset[idx]
                            if isinstance(sample, (tuple, list)) and len(sample) >= 1:
                                data = sample[0]
                                if isinstance(data, torch.Tensor):
                                    shapes.append(data.shape)
                                    dtypes.append(data.dtype)
                        except Exception as e:
                            result.add_warning(f"Error accessing sample {idx}: {e}")
                    
                    # Check shape consistency
                    if shapes:
                        unique_shapes = set(shapes)
                        if len(unique_shapes) > 1:
                            result.add_warning(f"Inconsistent data shapes: {unique_shapes}")
                        
                        unique_dtypes = set(dtypes)
                        if len(unique_dtypes) > 1:
                            result.add_warning(f"Inconsistent data types: {unique_dtypes}")
                
                except Exception as e:
                    result.add_issue(f"Error accessing dataset sample: {e}")
            
            # Check labels if provided separately
            if labels is not None:
                if not isinstance(labels, torch.Tensor):
                    result.add_issue("Labels should be a tensor")
                elif len(labels) != dataset_len:
                    result.add_issue(f"Labels length ({len(labels)}) doesn't match dataset length ({dataset_len})")
                else:
                    result.add_info("labels_shape", list(labels.shape))
                    result.add_info("labels_dtype", str(labels.dtype))
                    
                    # Check label range for classification
                    if labels.dtype in [torch.long, torch.int]:
                        unique_labels = torch.unique(labels)
                        result.add_info("num_classes", len(unique_labels))
                        result.add_info("label_range", (unique_labels.min().item(), unique_labels.max().item()))
                        
                        if unique_labels.min() < 0:
                            result.add_warning("Labels contain negative values")
        
        except Exception as e:
            result.add_issue(f"Error validating dataset: {e}")
        
        return result
    
    @staticmethod
    def validate_concept_activations(concept_activations: torch.Tensor, 
                                   num_concepts: int, 
                                   num_samples: int) -> ValidationResult:
        """Validate concept activation tensor"""
        result = ValidationResult(valid=True, issues=[], warnings=[], info={})
        
        # Basic tensor validation
        if not isinstance(concept_activations, torch.Tensor):
            result.add_issue("Concept activations must be a tensor")
            return result
        
        # Shape validation
        expected_shape = (num_samples, num_concepts)
        if concept_activations.shape != expected_shape:
            result.add_issue(f"Concept activations shape {concept_activations.shape} doesn't match expected {expected_shape}")
        
        result.add_info("activations_shape", list(concept_activations.shape))
        result.add_info("activations_dtype", str(concept_activations.dtype))
        
        # Value validation
        if torch.isnan(concept_activations).any():
            result.add_issue("Concept activations contain NaN values")
        
        if torch.isinf(concept_activations).any():
            result.add_issue("Concept activations contain infinite values")
        
        # Statistical information
        result.add_info("activations_min", concept_activations.min().item())
        result.add_info("activations_max", concept_activations.max().item())
        result.add_info("activations_mean", concept_activations.mean().item())
        result.add_info("activations_std", concept_activations.std().item())
        
        # Check for dead concepts (always zero)
        zero_concepts = (concept_activations == 0).all(dim=0).sum().item()
        if zero_concepts > 0:
            result.add_warning(f"{zero_concepts} concepts are always zero")
        
        # Check for saturated concepts (always same value)
        for i in range(concept_activations.shape[1]):
            concept_values = concept_activations[:, i]
            if concept_values.std() < 1e-6:
                result.add_warning(f"Concept {i} has very low variance (std={concept_values.std().item():.2e})")
        
        return result


class DataValidator:
    """Enhanced data validation utilities"""
    
    @staticmethod
    def validate_tensor_compatibility(tensor1: torch.Tensor, 
                                    tensor2: torch.Tensor, 
                                    dim: int = 0) -> ValidationResult:
        """Validate that two tensors are compatible for operations"""
        result = ValidationResult(valid=True, issues=[], warnings=[], info={})
        
        # Device check
        if tensor1.device != tensor2.device:
            result.add_issue(f"Tensors on different devices: {tensor1.device} vs {tensor2.device}")
        
        # Dtype check
        if tensor1.dtype != tensor2.dtype:
            result.add_warning(f"Tensors have different dtypes: {tensor1.dtype} vs {tensor2.dtype}")
        
        # Dimension compatibility
        if dim < len(tensor1.shape) and dim < len(tensor2.shape):
            if tensor1.shape[dim] != tensor2.shape[dim]:
                result.add_issue(f"Tensors incompatible at dimension {dim}: {tensor1.shape[dim]} vs {tensor2.shape[dim]}")
        
        result.add_info("tensor1_shape", list(tensor1.shape))
        result.add_info("tensor2_shape", list(tensor2.shape))
        
        return result
    
    @staticmethod
    def validate_batch_size(batch_size: int, dataset_size: int, min_batch_size: int = 1) -> ValidationResult:
        """Validate batch size parameters"""
        result = ValidationResult(valid=True, issues=[], warnings=[], info={})
        
        if batch_size < min_batch_size:
            result.add_issue(f"Batch size {batch_size} is below minimum {min_batch_size}")
        
        if batch_size > dataset_size:
            result.add_warning(f"Batch size {batch_size} larger than dataset size {dataset_size}")
        
        num_batches = (dataset_size + batch_size - 1) // batch_size
        result.add_info("num_batches", num_batches)
        result.add_info("last_batch_size", dataset_size % batch_size if dataset_size % batch_size != 0 else batch_size)
        
        return result


class ConfigValidator:
    """Configuration validation utilities"""
    
    @staticmethod
    def validate_hyperparameters(config: Dict[str, Any]) -> ValidationResult:
        """Validate hyperparameter values"""
        result = ValidationResult(valid=True, issues=[], warnings=[], info={})
        
        # Learning rate validation
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                result.add_issue(f"Learning rate must be positive, got {lr}")
            elif lr > 1.0:
                result.add_warning(f"Learning rate {lr} is unusually high")
            elif lr < 1e-6:
                result.add_warning(f"Learning rate {lr} is unusually low")
        
        # Batch size validation
        if 'batch_size' in config:
            bs = config['batch_size']
            if not isinstance(bs, int) or bs <= 0:
                result.add_issue(f"Batch size must be positive integer, got {bs}")
            elif bs > 1024:
                result.add_warning(f"Batch size {bs} is very large")
        
        # Epochs validation
        if 'max_epochs' in config:
            epochs = config['max_epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                result.add_issue(f"Max epochs must be positive integer, got {epochs}")
            elif epochs > 1000:
                result.add_warning(f"Max epochs {epochs} is very large")
        
        # Patience validation
        if 'patience' in config:
            patience = config['patience']
            if not isinstance(patience, int) or patience <= 0:
                result.add_issue(f"Patience must be positive integer, got {patience}")
        
        # Regularization parameters
        for param in ['weight_decay', 'dropout', 'lam']:
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)) or value < 0:
                    result.add_issue(f"{param} must be non-negative, got {value}")
                elif value > 1.0:
                    result.add_warning(f"{param} {value} is greater than 1.0")
        
        result.add_info("validated_parameters", list(config.keys()))
        
        return result


def validate_save_path(save_path: Union[str, Path], create_if_missing: bool = True) -> ValidationResult:
    """Validate save path for model saving"""
    result = ValidationResult(valid=True, issues=[], warnings=[], info={})
    save_path = Path(save_path)

    # Ensure parent dir exists
    if not save_path.parent.exists():
        if create_if_missing:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                result.add_info("created_directory", str(save_path.parent))
            except Exception as e:
                result.add_issue(f"Cannot create directory {save_path.parent}: {e}")
        else:
            result.add_issue(f"Parent directory does not exist: {save_path.parent}")

    # Write permission
    try:
        test_file = save_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
        result.add_info("write_permission", True)
    except Exception as e:
        result.add_issue(f"No write permission for {save_path.parent}: {e}")

    # If exists, warn
    if save_path.exists():
        if save_path.is_file():
            result.add_warning(f"File {save_path} will be overwritten")
        elif save_path.is_dir():
            result.add_warning(f"Directory {save_path} already exists; files may be overwritten")

    return result

# at the bottom of cbm_library/utils/validation.py
__all__ = [
    "ValidationResult",
    "ModelValidator",
    "DataValidator", 
    "ConfigValidator",
    "validate_save_path",
    "ComprehensiveValidator",
]

# Backwards compatibility alias
class ComprehensiveValidator(ModelValidator):
    """Deprecated: use ModelValidator instead."""
    pass
