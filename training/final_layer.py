# cbm_library/training/final_layer.py - INTEGRATED VERSION
"""
Unified Final Layer Training Module with Original LF-CBM Integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from abc import ABC, abstractmethod
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..utils.logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)

class FinalLayerType(Enum):
    SPARSE_GLM = "sparse_glm"  # Label-free CBM, VLG-CBM
    DENSE_LINEAR = "dense_linear"  # Standard dense training
    SPARSE_LINEAR = "sparse_linear"  # Manual sparsity constraints
    ELASTIC_NET = "elastic_net"  # Elastic net regularization

@dataclass
class FinalLayerConfig:
    """Configuration for final layer training"""
    layer_type: FinalLayerType
    num_concepts: int
    num_classes: int
    
    # Sparsity control
    sparsity_lambda: float = 0.0007
    target_sparsity_per_class: int = 30
    sparsity_percentage: Optional[float] = None
    
    # GLM-SAGA specific (matching original)
    glm_step_size: float = 0.1
    glm_alpha: float = 0.99
    glm_max_iters: int = 1000
    glm_epsilon: float = 1.0
    saga_batch_size: int = 256  # Added from original
    
    # Standard optimization
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 100
    weight_decay: float = 0.0
    
    # Normalization (matching original behavior)
    normalize_concepts: bool = True
    concept_mean: Optional[torch.Tensor] = None
    concept_std: Optional[torch.Tensor] = None
    
    # Device
    device: str = "cuda"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['layer_type'] = self.layer_type.value
        # Convert tensors to None for serialization
        if self.concept_mean is not None:
            result['concept_mean'] = None
        if self.concept_std is not None:
            result['concept_std'] = None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        data = data.copy()
        data['layer_type'] = FinalLayerType(data['layer_type'])
        return cls(**data)

class BaseFinalLayerMethod(ABC):
    """Abstract base class for final layer training methods"""
    
    @abstractmethod
    def train(self, 
              concept_activations: torch.Tensor,
              labels: torch.Tensor,
              config: FinalLayerConfig,
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Train the final layer and return weights + metadata"""
        pass
    
    @abstractmethod
    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        """Create the final layer module"""
        pass

class SparseGLMMethod(BaseFinalLayerMethod):
    """GLM-SAGA based sparse training matching original Label-free CBM exactly"""
    
    def __init__(self):
        try:
            from glm_saga.elasticnet import glm_saga, IndexedTensorDataset
            self.glm_saga = glm_saga
            self.IndexedTensorDataset = IndexedTensorDataset
            self.available = True
            logger.info("✅ GLM-SAGA available for sparse training")
        except ImportError:
            logger.warning("❌ GLM-SAGA not available. Install: pip install glm-saga")
            self.available = False
    
    def train(self, 
              concept_activations: torch.Tensor,
              labels: torch.Tensor,
              config: FinalLayerConfig,
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        
        if not self.available:
            raise ImportError("GLM-SAGA not available. Install: pip install glm-saga")
        
        logger.info(f"🚀 Training sparse GLM final layer (original method)")
        logger.info(f"📊 {config.num_concepts} concepts → {config.num_classes} classes")
        
        # Step 1: Normalize concept activations (matching original exactly)
        if config.normalize_concepts:
            train_mean = torch.mean(concept_activations, dim=0, keepdim=True)
            train_std = torch.std(concept_activations, dim=0, keepdim=True)
            normalized_concepts = (concept_activations - train_mean) / train_std
        else:
            normalized_concepts = concept_activations
            train_mean = torch.zeros(1, concept_activations.shape[1])
            train_std = torch.ones(1, concept_activations.shape[1])
        
        # Convert to expected format
        concept_mean = train_mean.squeeze(0)
        concept_std = train_std.squeeze(0)
        
        # Step 2: Prepare training data (matching original)
        train_y = labels.long()
        indexed_train_ds = self.IndexedTensorDataset(normalized_concepts, train_y)
        indexed_train_loader = DataLoader(
            indexed_train_ds, 
            batch_size=config.saga_batch_size, 
            shuffle=True
        )
        
        # Step 3: Prepare validation data if provided
        val_loader = None
        if validation_data is not None:
            val_concepts, val_labels = validation_data
            if config.normalize_concepts:
                val_concepts = (val_concepts - train_mean) / train_std
            val_ds = TensorDataset(val_concepts, val_labels.long())
            val_loader = DataLoader(val_ds, batch_size=config.saga_batch_size, shuffle=False)
        
        # Step 4: Create linear model and zero initialize (matching original)
        linear = nn.Linear(config.num_concepts, config.num_classes).to(config.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()
        
        # Step 5: GLM-SAGA metadata (matching original exactly)
        metadata = {
            'max_reg': {
                'nongrouped': config.sparsity_lambda
            }
        }
        
        logger.info(f"🔧 GLM-SAGA parameters:")
        logger.info(f"   λ (sparsity): {config.sparsity_lambda}")
        logger.info(f"   α (momentum): {config.glm_alpha}")
        logger.info(f"   step_size: {config.glm_step_size}")
        logger.info(f"   max_iters: {config.glm_max_iters}")
        
        # Step 6: Run GLM-SAGA (matching original call exactly)
        output_proj = self.glm_saga(
            linear, 
            indexed_train_loader,
            config.glm_step_size,
            config.glm_max_iters, 
            config.glm_alpha,
            epsilon=config.glm_epsilon,
            k=1,
            val_loader=val_loader,
            do_zero=False,
            metadata=metadata,
            n_ex=len(concept_activations),
            n_classes=config.num_classes
        )
        
        # Step 7: Extract results (matching original)
        best_result = output_proj['path'][0]
        W_g = best_result['weight']
        b_g = best_result['bias']
        
        # Step 8: Calculate sparsity statistics
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        sparsity_per_class = nnz / config.num_classes
        
        logger.info(f"✅ Sparse GLM training completed:")
        logger.info(f"   Non-zero weights: {nnz}/{total}")
        logger.info(f"   Sparsity per class: {sparsity_per_class:.1f}")
        logger.info(f"   Final λ: {best_result.get('lam', config.sparsity_lambda)}")
        
        # Progress callback for integration with enhanced features
        if progress_callback:
            final_metrics = {
                'loss': float(best_result.get('loss', 0.0)),
                'sparsity': nnz / total,
                'lambda': float(best_result.get('lam', config.sparsity_lambda))
            }
            progress_callback(config.glm_max_iters, final_metrics)
        
        return {
            'weight': W_g,
            'bias': b_g,
            'concept_mean': concept_mean,
            'concept_std': concept_std,
            'training_metrics': best_result.get('metrics', {}),
            'sparsity_stats': {
                'non_zero_weights': nnz,
                'total_weights': total,
                'sparsity_percentage': nnz / total,
                'sparsity_per_class': sparsity_per_class
            },
            'glm_metadata': {
                'lambda': float(best_result.get('lam', config.sparsity_lambda)),
                'alpha': float(best_result.get('alpha', config.glm_alpha)),
                'final_loss': float(best_result.get('loss', 0.0)),
                'time': float(best_result.get('time', 0.0))
            }
        }
    
    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        return nn.Linear(config.num_concepts, config.num_classes)

class DenseLinearMethod(BaseFinalLayerMethod):
    """Standard dense linear training"""
    
    def train(self, 
              concept_activations: torch.Tensor,
              labels: torch.Tensor,
              config: FinalLayerConfig,
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        
        logger.info(f"🚀 Training dense linear final layer")
        logger.info(f"📊 {config.num_concepts} concepts → {config.num_classes} classes")
        
        # Normalize if requested
        if config.normalize_concepts:
            concept_mean = concept_activations.mean(dim=0)
            concept_std = concept_activations.std(dim=0) + 1e-8
            normalized_concepts = (concept_activations - concept_mean) / concept_std
        else:
            normalized_concepts = concept_activations
            concept_mean = torch.zeros(concept_activations.shape[1])
            concept_std = torch.ones(concept_activations.shape[1])
        
        # Create model
        model = nn.Linear(config.num_concepts, config.num_classes).to(config.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Prepare data
        train_dataset = TensorDataset(normalized_concepts, labels.long())
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(config.max_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_concepts, batch_labels in train_loader:
                batch_concepts = batch_concepts.to(config.device)
                batch_labels = batch_labels.to(config.device)
                
                optimizer.zero_grad()
                outputs = model(batch_concepts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            val_acc = 0.0
            if validation_data is not None:
                val_concepts, val_labels = validation_data
                if config.normalize_concepts:
                    val_concepts = (val_concepts - concept_mean) / concept_std
                
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_concepts.to(config.device))
                    val_pred = val_outputs.argmax(dim=1)
                    val_acc = (val_pred == val_labels.to(config.device)).float().mean().item()
                    val_accuracies.append(val_acc)
            
            # Progress callback
            if progress_callback and epoch % 10 == 0:
                progress_callback(epoch, {'loss': avg_loss, 'val_accuracy': val_acc})
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
        
        logger.info(f"✅ Dense training completed in {config.max_epochs} epochs")
        
        return {
            'weight': model.weight.detach().cpu(),
            'bias': model.bias.detach().cpu(),
            'concept_mean': concept_mean,
            'concept_std': concept_std,
            'training_metrics': {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'final_train_loss': train_losses[-1] if train_losses else 0.0
            },
            'sparsity_stats': {
                'non_zero_weights': model.weight.numel(),
                'total_weights': model.weight.numel(),
                'sparsity_percentage': 1.0,
                'sparsity_per_class': model.weight.shape[0]  # Fixed: weight.shape[0] is num_classes
            }
        }
    
    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        return nn.Linear(config.num_concepts, config.num_classes)

class SparseLinearMethod(BaseFinalLayerMethod):
    """Manual sparsity constraints during training"""
    
    def train(self, 
              concept_activations: torch.Tensor,
              labels: torch.Tensor,
              config: FinalLayerConfig,
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        
        logger.info(f"🚀 Training sparse linear final layer")
        
        # Use dense method as base
        dense_method = DenseLinearMethod()
        result = dense_method.train(concept_activations, labels, config, validation_data, progress_callback)
        
        # Apply sparsity constraints
        weight = result['weight']
        
        if config.sparsity_percentage is not None:
            # Percentage-based sparsity
            k = int(weight.numel() * config.sparsity_percentage)
        else:
            # Per-class sparsity
            k = config.target_sparsity_per_class * config.num_classes
        
        # Keep top-k weights by absolute value
        flat_weights = weight.flatten()
        _, top_k_indices = torch.topk(torch.abs(flat_weights), k)
        
        # Create sparse weight mask
        sparse_mask = torch.zeros_like(flat_weights, dtype=torch.bool)
        sparse_mask[top_k_indices] = True
        sparse_mask = sparse_mask.reshape(weight.shape)
        
        # Apply sparsity
        sparse_weight = weight * sparse_mask.float()
        
        # Update sparsity stats
        nnz = sparse_mask.sum().item()
        result['weight'] = sparse_weight
        result['sparsity_stats'] = {
            'non_zero_weights': nnz,
            'total_weights': weight.numel(),
            'sparsity_percentage': nnz / weight.numel(),
            'sparsity_per_class': nnz / config.num_classes
        }
        
        logger.info(f"✅ Applied sparsity: {nnz}/{weight.numel()} non-zero weights")
        
        return result
    
    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        return nn.Linear(config.num_concepts, config.num_classes)

class UnifiedFinalTrainer:
    """
    Unified trainer for final layers across all CBM methods
    Integrates with enhanced BaseCBM features while maintaining original method compatibility
    """
    
    def __init__(self):
        self.methods = {
            FinalLayerType.SPARSE_GLM: SparseGLMMethod(),
            FinalLayerType.DENSE_LINEAR: DenseLinearMethod(),
            FinalLayerType.SPARSE_LINEAR: SparseLinearMethod(),
            FinalLayerType.ELASTIC_NET: DenseLinearMethod(),
        }
    
    def train(self, 
              concept_activations: torch.Tensor,
              labels: torch.Tensor,
              config: FinalLayerConfig,
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Train final layer using specified method
        
        Args:
            concept_activations: [N, num_concepts] concept activations
            labels: [N] class labels
            config: Training configuration
            validation_data: Optional (val_concepts, val_labels) tuple
            progress_callback: Optional callback for progress monitoring (for BaseCBM integration)
        
        Returns:
            Training results dictionary
        """
        
        if config.layer_type not in self.methods:
            raise ValueError(f"Unsupported layer type: {config.layer_type}")
        
        # Update config with inferred dimensions if not set
        if config.num_concepts == 0:
            config.num_concepts = concept_activations.shape[1]
        if config.num_classes == 0:
            config.num_classes = len(torch.unique(labels))
        
        logger.info(f"🎯 Training {config.layer_type.value} final layer")
        logger.info(f"📊 Input shape: {concept_activations.shape}")
        logger.info(f"🏷️ Labels shape: {labels.shape}")
        
        method = self.methods[config.layer_type]
        result = method.train(concept_activations, labels, config, validation_data, progress_callback)
        
        # Add configuration to result
        result['config'] = config
        result['method'] = config.layer_type.value
        
        return result
    
    def create_final_layer(self, config: FinalLayerConfig, training_result: Dict[str, Any]) -> nn.Module:
        """Create a final layer module from training results"""
        
        method = self.methods[config.layer_type]
        layer = method.create_layer(config)
        
        # Load trained weights
        layer.weight.data = training_result['weight'].to(config.device)
        layer.bias.data = training_result['bias'].to(config.device)
        
        return layer
    
    def save_training_result(self, result: Dict[str, Any], save_path: str):
        """Save training results in both enhanced and original formats"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save in original format (for compatibility)
        torch.save(result['weight'], os.path.join(save_path, 'W_g.pt'))
        torch.save(result['bias'], os.path.join(save_path, 'b_g.pt'))
        torch.save(result['concept_mean'], os.path.join(save_path, 'proj_mean.pt'))
        torch.save(result['concept_std'], os.path.join(save_path, 'proj_std.pt'))
        
        # Save enhanced metadata
        metadata = {
            'config': result['config'].to_dict(),
            'method': result['method'],
            'training_metrics': result['training_metrics'],
            'sparsity_stats': result['sparsity_stats']
        }
        
        # Handle GLM-specific metadata
        if 'glm_metadata' in result:
            metadata['glm_metadata'] = result['glm_metadata']
        
        # Save original-style metrics.txt for compatibility
        if result['method'] == 'sparse_glm' and 'glm_metadata' in result:
            glm_meta = result['glm_metadata']
            original_metrics = {
                'lam': glm_meta['lambda'],
                'alpha': glm_meta['alpha'], 
                'time': glm_meta.get('time', 0.0),
                'metrics': result['training_metrics'],
                'sparsity': {
                    "Non-zero weights": result['sparsity_stats']['non_zero_weights'],
                    "Total weights": result['sparsity_stats']['total_weights'],
                    "Percentage non-zero": result['sparsity_stats']['sparsity_percentage']
                }
            }
            
            with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
                json.dump(original_metrics, f, indent=2)
        
        # Save enhanced metadata
        with open(os.path.join(save_path, 'final_layer_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Training results saved to {save_path}")
    
    def load_training_result(self, load_path: str, device: str = "cuda") -> Dict[str, Any]:
        """Load training results from disk"""
        
        result = {}
        
        # Load tensors
        result['weight'] = torch.load(os.path.join(load_path, 'W_g.pt'), map_location=device)
        result['bias'] = torch.load(os.path.join(load_path, 'b_g.pt'), map_location=device)
        result['concept_mean'] = torch.load(os.path.join(load_path, 'proj_mean.pt'), map_location=device)
        result['concept_std'] = torch.load(os.path.join(load_path, 'proj_std.pt'), map_location=device)
        
        # Load metadata
        metadata_path = os.path.join(load_path, 'final_layer_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Reconstruct config
            result['config'] = FinalLayerConfig.from_dict(metadata['config'])
            result.update(metadata)
        else:
            # Fallback: try to load from original metrics.txt
            logger.warning("Enhanced metadata not found, trying original format...")
            # Add fallback loading logic here if needed
        
        logger.info(f"📂 Training results loaded from {load_path}")
        
        return result

# Method-specific configuration helpers (updated for original compatibility)
def get_label_free_cbm_config(num_concepts: int, num_classes: int, **kwargs) -> FinalLayerConfig:
    """Standard configuration for Label-free CBM (matching original exactly)"""
    defaults = {
        'layer_type': FinalLayerType.SPARSE_GLM,
        'sparsity_lambda': 0.0007,
        'target_sparsity_per_class': 30,
        'glm_alpha': 0.99,
        'glm_step_size': 0.1,
        'glm_max_iters': 1000,
        'glm_epsilon': 1.0,
        'saga_batch_size': 256,  # From original
        'normalize_concepts': True
    }
    defaults.update(kwargs)
    
    return FinalLayerConfig(
        num_concepts=num_concepts,
        num_classes=num_classes,
        **defaults
    )

def get_vlg_cbm_config(num_concepts: int, num_classes: int, **kwargs) -> FinalLayerConfig:
    """Standard configuration for VLG-CBM (similar to Label-free)"""
    return get_label_free_cbm_config(num_concepts, num_classes, **kwargs)

def get_dense_cbm_config(num_concepts: int, num_classes: int, **kwargs) -> FinalLayerConfig:
    """Configuration for dense CBM training"""
    defaults = {
        'layer_type': FinalLayerType.DENSE_LINEAR,
        'learning_rate': 0.001,
        'max_epochs': 100,
        'normalize_concepts': True
    }
    defaults.update(kwargs)
    
    return FinalLayerConfig(
        num_concepts=num_concepts,
        num_classes=num_classes,
        **defaults
    )

def get_cb_llm_config(num_concepts: int, num_classes: int, **kwargs) -> FinalLayerConfig:
    """Configuration for CB-LLM final layer"""
    defaults = {
        'layer_type': FinalLayerType.DENSE_LINEAR,
        'learning_rate': 0.0001,
        'max_epochs': 50,
        'normalize_concepts': True,
        'weight_decay': 0.01
    }
    defaults.update(kwargs)
    
    return FinalLayerConfig(
        num_concepts=num_concepts,
        num_classes=num_classes,
        **defaults
    )

def get_labo_config(num_concepts: int, num_classes: int, **kwargs) -> FinalLayerConfig:
    """Configuration for LaBo CBM"""
    defaults = {
        'layer_type': FinalLayerType.SPARSE_LINEAR,
        'target_sparsity_per_class': 25,
        'learning_rate': 0.001,
        'max_epochs': 100,
        'normalize_concepts': True
    }
    defaults.update(kwargs)
    
    return FinalLayerConfig(
        num_concepts=num_concepts,
        num_classes=num_classes,
        **defaults
    )