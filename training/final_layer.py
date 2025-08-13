"""
Unified Final Layer Training Module
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

logger = logging.getLogger(__name__)

class FinalLayerType(Enum):
    SPARSE_GLM = "sparse_glm" 
    DENSE_LINEAR = "dense_linear" 
    SPARSE_LINEAR = "sparse_linear"  
    ELASTIC_NET = "elastic_net"  

@dataclass
class FinalLayerConfig:
    """Configuration for final layer training"""
    layer_type: FinalLayerType
    num_concepts: int
    num_classes: int
    
    sparsity_lambda: float = 0.0007
    target_sparsity_per_class: int = 30
    sparsity_percentage: Optional[float] = None
    
    glm_step_size: float = 0.1
    glm_alpha: float = 0.99
    glm_max_iters: int = 1000
    glm_epsilon: float = 1.0
    
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 100
    weight_decay: float = 0.0
    
    
    normalize_concepts: bool = True
    concept_mean: Optional[torch.Tensor] = None
    concept_std: Optional[torch.Tensor] = None
    
    device: str = "cuda"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['layer_type'] = self.layer_type.value
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
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """Train the final layer and return weights + metadata"""
        pass
    
    @abstractmethod
    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        """Create the final layer module"""
        pass

class SparseGLMMethod(BaseFinalLayerMethod):
    """GLM-SAGA based sparse training (Label-free CBM, VLG-CBM)"""
    
    def __init__(self):
        try:
            from glm_saga.elasticnet import glm_saga
            self.glm_saga = glm_saga
            self.available = True
        except ImportError:
            logger.warning("GLM-SAGA not available. Sparse GLM training will not work.")
            self.available = False
    
    def train(self, 
              concept_activations: torch.Tensor,
              labels: torch.Tensor,
              config: FinalLayerConfig,
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        
        if not self.available:
            raise ImportError("GLM-SAGA not available. Install: pip install glm-saga")
        
        logger.info(f"Training sparse GLM final layer with {config.num_concepts} concepts, {config.num_classes} classes")
        
        if config.normalize_concepts:
            concept_mean = concept_activations.mean(dim=0)
            concept_std = concept_activations.std(dim=0) + 1e-8
            normalized_concepts = (concept_activations - concept_mean) / concept_std
        else:
            normalized_concepts = concept_activations
            concept_mean = torch.zeros(concept_activations.shape[1])
            concept_std = torch.ones(concept_activations.shape[1])
        
        train_dataset = TensorDataset(normalized_concepts, labels.long())
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        linear = nn.Linear(config.num_concepts, config.num_classes).to(config.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()
        
        val_loader = None
        if validation_data is not None:
            val_concepts, val_labels = validation_data
            if config.normalize_concepts:
                val_concepts = (val_concepts - concept_mean) / concept_std
            val_dataset = TensorDataset(val_concepts, val_labels.long())
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        metadata = {
            'max_reg': {
                'nongrouped': config.sparsity_lambda
            }
        }
        
        logger.info(f"Running GLM-SAGA with lambda={config.sparsity_lambda}, alpha={config.glm_alpha}")
        
        output_proj = self.glm_saga(
            linear, train_loader, 
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
        
        best_result = output_proj['path'][0]
        W_g = best_result['weight']
        b_g = best_result['bias']
        
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        sparsity_per_class = nnz / config.num_classes
        
        logger.info(f"Final layer trained: {nnz}/{total} non-zero weights ({sparsity_per_class:.1f} per class)")
        
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
                'final_loss': float(best_result.get('loss', 0.0))
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
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        
        logger.info(f"Training dense linear final layer with {config.num_concepts} concepts, {config.num_classes} classes")
        
        if config.normalize_concepts:
            concept_mean = concept_activations.mean(dim=0)
            concept_std = concept_activations.std(dim=0) + 1e-8
            normalized_concepts = (concept_activations - concept_mean) / concept_std
        else:
            normalized_concepts = concept_activations
            concept_mean = torch.zeros(concept_activations.shape[1])
            concept_std = torch.ones(concept_activations.shape[1])
        
        model = nn.Linear(config.num_concepts, config.num_classes).to(config.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        train_dataset = TensorDataset(normalized_concepts, labels.long())
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
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
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")
        
        logger.info(f"Dense final layer training completed in {config.max_epochs} epochs")
        
        return {
            'weight': model.weight.detach().cpu(),
            'bias': model.bias.detach().cpu(),
            'concept_mean': concept_mean,
            'concept_std': concept_std,
            'training_metrics': {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'final_train_loss': train_losses[-1]
            },
            'sparsity_stats': {
                'non_zero_weights': model.weight.numel(),
                'total_weights': model.weight.numel(),
                'sparsity_percentage': 1.0,
                'sparsity_per_class': model.weight.shape[1]
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
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        
        logger.info(f"Training sparse linear final layer with {config.num_concepts} concepts, {config.num_classes} classes")
        
        dense_method = DenseLinearMethod()
        result = dense_method.train(concept_activations, labels, config, validation_data)
        
        weight = result['weight']
        
        if config.sparsity_percentage is not None:
            k = int(weight.numel() * config.sparsity_percentage)
        else:
            k = config.target_sparsity_per_class * config.num_classes
        
        flat_weights = weight.flatten()
        _, top_k_indices = torch.topk(torch.abs(flat_weights), k)
        
        sparse_mask = torch.zeros_like(flat_weights, dtype=torch.bool)
        sparse_mask[top_k_indices] = True
        sparse_mask = sparse_mask.reshape(weight.shape)
        
        sparse_weight = weight * sparse_mask.float()
        
        nnz = sparse_mask.sum().item()
        result['weight'] = sparse_weight
        result['sparsity_stats'] = {
            'non_zero_weights': nnz,
            'total_weights': weight.numel(),
            'sparsity_percentage': nnz / weight.numel(),
            'sparsity_per_class': nnz / config.num_classes
        }
        
        logger.info(f"Applied sparsity: {nnz}/{weight.numel()} non-zero weights")
        
        return result
    
    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        return nn.Linear(config.num_concepts, config.num_classes)

class UnifiedFinalTrainer:
    """
    Unified trainer for final layers across all CBM methods
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
              validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Train final layer using specified method
        """
        
        if config.layer_type not in self.methods:
            raise ValueError(f"Unsupported layer type: {config.layer_type}")
        
        if config.num_concepts == 0:
            config.num_concepts = concept_activations.shape[1]
        if config.num_classes == 0:
            config.num_classes = len(torch.unique(labels))
        
        method = self.methods[config.layer_type]
        result = method.train(concept_activations, labels, config, validation_data)
        
        result['config'] = config
        result['method'] = config.layer_type.value
        
        return result
    
    def create_final_layer(self, config: FinalLayerConfig, training_result: Dict[str, Any]) -> nn.Module:
        """Create a final layer module from training results"""
        
        method = self.methods[config.layer_type]
        layer = method.create_layer(config)
        
        layer.weight.data = training_result['weight'].to(config.device)
        layer.bias.data = training_result['bias'].to(config.device)
        
        return layer
    
    def save_training_result(self, result: Dict[str, Any], save_path: str):
        """Save training results to disk"""
        os.makedirs(save_path, exist_ok=True)
        
        torch.save(result['weight'], os.path.join(save_path, 'W_g.pt'))
        torch.save(result['bias'], os.path.join(save_path, 'b_g.pt'))
        torch.save(result['concept_mean'], os.path.join(save_path, 'concept_mean.pt'))
        torch.save(result['concept_std'], os.path.join(save_path, 'concept_std.pt'))
        
        metadata = {
            'config': result['config'].to_dict(),
            'method': result['method'],
            'training_metrics': result['training_metrics'],
            'sparsity_stats': result['sparsity_stats']
        }
        
        if 'glm_metadata' in result:
            metadata['glm_metadata'] = result['glm_metadata']
        
        with open(os.path.join(save_path, 'final_layer_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved final layer training results to {save_path}")
    
    def load_training_result(self, load_path: str, device: str = "cuda") -> Dict[str, Any]:
        """Load training results from disk"""
        
        result = {}
        
        result['weight'] = torch.load(os.path.join(load_path, 'W_g.pt'), map_location=device)
        result['bias'] = torch.load(os.path.join(load_path, 'b_g.pt'), map_location=device)
        result['concept_mean'] = torch.load(os.path.join(load_path, 'concept_mean.pt'), map_location=device)
        result['concept_std'] = torch.load(os.path.join(load_path, 'concept_std.pt'), map_location=device)
        
        with open(os.path.join(load_path, 'final_layer_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        result['config'] = FinalLayerConfig.from_dict(metadata['config'])
        result.update(metadata)
        
        logger.info(f"Loaded final layer training results from {load_path}")
        
        return result

def get_label_free_cbm_config(num_concepts: int, num_classes: int, **kwargs) -> FinalLayerConfig:
    """Standard configuration for Label-free CBM"""
    defaults = {
        'layer_type': FinalLayerType.SPARSE_GLM,
        'sparsity_lambda': 0.0007,
        'target_sparsity_per_class': 30,
        'glm_alpha': 0.99,
        'glm_step_size': 0.1,
        'glm_max_iters': 1000,
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
