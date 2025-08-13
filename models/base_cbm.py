import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)

class BaseCBM(nn.Module, ABC):
    """
    Abstract base class for all Concept Bottleneck Models
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 num_concepts: int,
                 num_classes: int,
                 device: str = "cuda"):
        super().__init__()
        
        self.backbone = backbone
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.device = device
        
        # Components that will be set during training
        self.concept_layer = None  # Concept Bottleneck Layer (CBL)
        self.final_layer = None    # Final classification layer
        self.concept_mean = None   # Normalization parameters
        self.concept_std = None
        
        # Metadata
        self.concept_names = []
        self.training_config = {}
        self.is_trained = False
    
    @abstractmethod
    def train_concept_layer(self, 
                          dataset: Any, 
                          concepts: List[str],
                          config: Dict[str, Any]) -> torch.Tensor:
        """
        Train the concept bottleneck layer (CBL)
        
        Returns:
            concept_activations: [N, num_concepts] tensor of concept activations
        """
        pass
    
    def train_final_layer(self, 
                         concept_activations: torch.Tensor,
                         labels: torch.Tensor,
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the final classification layer
        """
        from ..training.final_layer import UnifiedFinalTrainer, FinalLayerConfig, FinalLayerType
        
        trainer = UnifiedFinalTrainer()
        
        # Create configuration
        final_config = FinalLayerConfig(
            layer_type=config.get('layer_type', FinalLayerType.DENSE_LINEAR),
            num_concepts=self.num_concepts,
            num_classes=self.num_classes,
            device=self.device,
            **config
        )
        
        # Train final layer
        result = trainer.train(concept_activations, labels, final_config)
        
        # Create and store final layer
        self.final_layer = trainer.create_final_layer(final_config, result)
        self.concept_mean = result['concept_mean']
        self.concept_std = result['concept_std']
        
        logger.info(f"Final layer trained with {result['sparsity_stats']['non_zero_weights']} non-zero weights")
        
        return result
    
    def get_concept_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get concept activations for input x"""
        if self.concept_layer is None:
            raise ValueError("Concept layer not trained yet")
        
        with torch.no_grad():
            features = self.get_backbone_features(x)
            concept_activations = self.concept_layer(features)
            
            # Normalize if needed
            if self.concept_mean is not None and self.concept_std is not None:
                concept_activations = (concept_activations - self.concept_mean) / self.concept_std
            
            return concept_activations
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone model"""
        features = self.backbone(x)
        return torch.flatten(features, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CBM
        
        Returns:
            logits: [batch_size, num_classes] classification logits
            concept_activations: [batch_size, num_concepts] concept activations
        """
        if self.concept_layer is None or self.final_layer is None:
            raise ValueError("Model not fully trained yet")
        
        # Get backbone features
        features = self.get_backbone_features(x)
        
        # Get concept activations
        concept_activations = self.concept_layer(features)
        
        # Normalize concept activations
        if self.concept_mean is not None and self.concept_std is not None:
            normalized_concepts = (concept_activations - self.concept_mean) / self.concept_std
        else:
            normalized_concepts = concept_activations
        
        # Final classification
        logits = self.final_layer(normalized_concepts)
        
        return logits, concept_activations
    
    def explain_prediction(self, 
                          x: torch.Tensor, 
                          top_k: int = 10,
                          return_contributions: bool = False) -> Dict[str, Any]:
        """Explain model prediction for input x"""
        if len(x.shape) > 1 and x.shape[0] > 1:
            raise ValueError("Explanation only supports single samples")
        
        # Get prediction
        logits, concept_activations = self.forward(x)
        predicted_class = logits.argmax(dim=1).item()
        
        # Normalize concept activations
        if self.concept_mean is not None and self.concept_std is not None:
            normalized_concepts = (concept_activations - self.concept_mean) / self.concept_std
        else:
            normalized_concepts = concept_activations
        
        # Calculate contributions (weight * activation)
        final_weights = self.final_layer.weight[predicted_class]  # [num_concepts]
        contributions = final_weights * normalized_concepts.squeeze(0)  # [num_concepts]
        
        # Get top contributing concepts
        top_indices = torch.topk(torch.abs(contributions), min(top_k, len(contributions))).indices
        
        explanation = {
            'predicted_class': predicted_class,
            'prediction_confidence': torch.softmax(logits, dim=1).max().item(),
            'top_concepts': []
        }
        
        for idx in top_indices:
            concept_idx = idx.item()
            concept_name = self.concept_names[concept_idx] if concept_idx < len(self.concept_names) else f"concept_{concept_idx}"
            
            explanation['top_concepts'].append({
                'concept_name': concept_name,
                'concept_index': concept_idx,
                'activation': concept_activations[0, concept_idx].item(),
                'weight': final_weights[concept_idx].item(),
                'contribution': contributions[concept_idx].item(),
                'positive_contribution': contributions[concept_idx].item() > 0
            })
        
        if return_contributions:
            explanation['all_contributions'] = contributions.cpu().numpy()
            explanation['all_activations'] = concept_activations.squeeze(0).cpu().numpy()
        
        return explanation
    
    def intervene_on_concepts(self, 
                             x: torch.Tensor,
                             concept_interventions: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform concept intervention by manually setting concept values"""
        # Get original concept activations
        original_concepts = self.get_concept_activations(x)
        modified_concepts = original_concepts.clone()
        
        intervention_info = {
            'interventions_applied': [],
            'original_prediction': None,
            'new_prediction': None
        }
        
        # Get original prediction
        with torch.no_grad():
            if self.concept_mean is not None and self.concept_std is not None:
                normalized_original = (original_concepts - self.concept_mean) / self.concept_std
            else:
                normalized_original = original_concepts
            original_logits = self.final_layer(normalized_original)
            intervention_info['original_prediction'] = original_logits.argmax(dim=1).item()
        
        # Apply interventions
        for concept_name, value in concept_interventions.items():
            if concept_name in self.concept_names:
                concept_idx = self.concept_names.index(concept_name)
                original_value = modified_concepts[0, concept_idx].item()
                modified_concepts[0, concept_idx] = value
                
                intervention_info['interventions_applied'].append({
                    'concept_name': concept_name,
                    'concept_index': concept_idx,
                    'original_value': original_value,
                    'new_value': value
                })
            else:
                logger.warning(f"Concept '{concept_name}' not found in model")
        
        # Get new prediction
        if self.concept_mean is not None and self.concept_std is not None:
            normalized_modified = (modified_concepts - self.concept_mean) / self.concept_std
        else:
            normalized_modified = modified_concepts
        
        new_logits = self.final_layer(normalized_modified)
        intervention_info['new_prediction'] = new_logits.argmax(dim=1).item()
        
        return new_logits, intervention_info