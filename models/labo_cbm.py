"""
LaBo CBM Model Implementation
Integrates with the unified final layer training system
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
import logging
import clip

from .base_cbm import BaseCBM
from ..training.concept_projection.labo_trainer import LaBoConceptProjectionTrainer, LaBoProjectionConfig
from ..training.final_layer import FinalLayerType

logger = logging.getLogger(__name__)

class LaBoCBM(BaseCBM):
    """
    LaBo (Language in a Bottle) Concept Bottleneck Model
    
    Implements the complete LaBo pipeline:
    1. Submodular concept selection 
    2. CLIP-based concept encoding
    3. Unified final layer training
    
    Key differences from other CBMs:
    - Uses submodular optimization for concept selection
    - No explicit projection layer (uses CLIP directly)
    - Language prior initialization for final layer
    """
    
    def __init__(self,
                 backbone: Optional[nn.Module],  
                 num_classes: int,
                 clip_model_name: str = "ViT-L/14",
                 k_per_class: int = 50,
                 alpha: float = 1e7,
                 beta: float = 1.0,
                 device: str = "cuda"):
        
        super().__init__(backbone, 0, num_classes, device)
        
        self.clip_model_name = clip_model_name
        self.k_per_class = k_per_class
        self.alpha = alpha
        self.beta = beta
        
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        
        self.backbone = self.clip_model.visual
        
        self.projection_config = LaBoProjectionConfig(
            k_per_class=k_per_class,
            alpha=alpha,
            beta=beta,
            clip_model_name=clip_model_name,
            device=device
        )
        self.projection_trainer = LaBoConceptProjectionTrainer(self.projection_config)
        
        self.concept_embeddings = None
        self.selected_concepts_per_class = {}
        self.selection_metadata = {}
    
    def train_concept_layer(self,
                          dataset: Any,
                          concepts: List[str],
                          config: Dict[str, Any]) -> torch.Tensor:
        """
        Train LaBo concept projection using submodular selection and CLIP encoding
        
        Args:
            dataset: Training dataset
            concepts: Generated concepts from concept generation module
            config: Training configuration
            
        Returns:
            concept_activations: [N_samples, N_concepts] concept activations
        """
        
        logger.info(f"Training LaBo concept layer with {len(concepts)} input concepts")
        
        if hasattr(dataset, 'classes'):
            classes = dataset.classes
        else:
            classes = [f"class_{i}" for i in range(self.num_classes)]
        
        if 'k_per_class' in config:
            self.projection_config.k_per_class = config['k_per_class']
        if 'alpha' in config:
            self.projection_config.alpha = config['alpha']
        if 'beta' in config:
            self.projection_config.beta = config['beta']
        
        concept_activations, selected_concepts, metadata = self.projection_trainer.train_projection(
            concepts=concepts,
            dataset=dataset,
            classes=classes
        )
        
        self.concept_names = selected_concepts
        self.num_concepts = len(selected_concepts)
        self.selection_metadata = metadata
        
        self.concept_embeddings = self.projection_trainer._encode_concepts(selected_concepts)
        
        self.concept_layer = nn.Identity()
        
        logger.info(f"LaBo concept layer training complete: {self.num_concepts} concepts selected")
        
        return concept_activations
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get CLIP image features (LaBo uses CLIP as backbone)
        """
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = features / features.norm(dim=-1, keepdim=True)  
        return features
    
    def get_concept_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get concept activations using CLIP dot product
        """
        if self.concept_embeddings is None:
            raise ValueError("Model not trained yet - no concept embeddings available")
        
        image_features = self.get_backbone_features(x)
        
        concept_activations = torch.mm(image_features, self.concept_embeddings.T)
        
        return concept_activations
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LaBo CBM
        """
        if self.final_layer is None:
            raise ValueError("Final layer not trained yet")
        
        concept_activations = self.get_concept_activations(x)
        
    
        if self.concept_mean is not None and self.concept_std is not None:
            normalized_concepts = (concept_activations - self.concept_mean) / self.concept_std
        else:
            normalized_concepts = concept_activations
        
        logits = self.final_layer(normalized_concepts)
        
        return logits, concept_activations
    
    def train_with_language_priors(self,
                                 concept_activations: torch.Tensor,
                                 labels: torch.Tensor,
                                 classes: List[str],
                                 final_layer_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train final layer with LaBo's language prior initialization
        
        This is a key feature of LaBo: initializing the concept-class weight matrix
        based on which concepts were selected for which classes
        """
        
        if final_layer_config is None:
            final_layer_config = {}
        
        final_layer_config.setdefault('layer_type', FinalLayerType.DENSE_LINEAR)
        final_layer_config.setdefault('use_language_priors', True)
        
        result = self.train_final_layer(
            concept_activations=concept_activations,
            labels=labels,
            layer_type=final_layer_config['layer_type'],
            **{k: v for k, v in final_layer_config.items() if k != 'layer_type'}
        )
        
        if final_layer_config.get('use_language_priors', True):
            self._initialize_with_language_priors(classes)
        
        return result
    
    def _initialize_with_language_priors(self, classes: List[str]):
        """
        Initialize final layer weights with language priors
        
        From LaBo paper: W_{y,c} = 1 if c ∈ C_y, otherwise 0
        """
        
        if self.final_layer is None:
            return
        
        logger.info("Initializing final layer with LaBo language priors")
        
        with torch.no_grad():
            weights = self.final_layer.weight.data  # [num_classes, num_concepts]
            
            weights.zero_()
            
            concept_idx = 0
            for class_idx, class_name in enumerate(classes):
                if class_name in self.selected_concepts_per_class:
                    class_concepts = self.selected_concepts_per_class[class_name]
                    num_class_concepts = len(class_concepts)
                    
                    weights[class_idx, concept_idx:concept_idx + num_class_concepts] = 1.0
                    concept_idx += num_class_concepts
        
        logger.info("Language prior initialization complete")
    
    def get_concept_importance_per_class(self) -> Dict[str, Dict[str, float]]:
        """
        Get concept importance scores organized by class
        
        Returns:
            Dictionary mapping class names to concept importance scores
        """
        if self.final_layer is None:
            raise ValueError("Model not trained yet")
        
        importance_per_class = {}
        
        weights = self.final_layer.weight.data.cpu()  # [num_classes, num_concepts]
        
        concept_idx = 0
        for class_idx, class_name in enumerate(self.selected_concepts_per_class.keys()):
            class_concepts = self.selected_concepts_per_class[class_name]
            num_class_concepts = len(class_concepts)
            
            class_weights = weights[class_idx, concept_idx:concept_idx + num_class_concepts]
            
            class_importance = {}
            for i, concept in enumerate(class_concepts):
                class_importance[concept] = abs(class_weights[i].item())
            
            importance_per_class[class_name] = class_importance
            concept_idx += num_class_concepts
        
        return importance_per_class
    
    def get_discriminability_scores(self) -> Dict[str, float]:
        """
        Get discriminability scores for selected concepts
        """
        discriminability = {}
        
        for class_name, metadata in self.selection_metadata.get('selection_metadata', {}).items():
            if class_name in self.selected_concepts_per_class:
                concepts = self.selected_concepts_per_class[class_name]
                scores = metadata.get('selection_scores', [])
                
                for concept, score in zip(concepts, scores):
                    discriminability[concept] = score
        
        return discriminability
    
    def save_model(self, save_path: str):
        """Save LaBo CBM model with all components"""
        
        super().save_model(save_path)
        
        import os
        
        labo_data = {
            'concept_embeddings': self.concept_embeddings,
            'selected_concepts_per_class': self.selected_concepts_per_class,
            'selection_metadata': self.selection_metadata,
            'projection_config': self.projection_config,
            'clip_model_name': self.clip_model_name,
            'k_per_class': self.k_per_class,
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        torch.save(labo_data, os.path.join(save_path, 'labo_specific.pt'))
        
        logger.info(f"Saved LaBo CBM model to {save_path}")
    
    def load_model(self, load_path: str, device: str = None):
        """Load LaBo CBM model with all components"""
        
        super().load_model(load_path, device)
        
        import os
        
        labo_path = os.path.join(load_path, 'labo_specific.pt')
        if os.path.exists(labo_path):
            labo_data = torch.load(labo_path, map_location=device or self.device)
            
            self.concept_embeddings = labo_data['concept_embeddings']
            self.selected_concepts_per_class = labo_data['selected_concepts_per_class']
            self.selection_metadata = labo_data['selection_metadata']
            self.projection_config = labo_data['projection_config']
            self.clip_model_name = labo_data['clip_model_name']
            self.k_per_class = labo_data['k_per_class']
            self.alpha = labo_data['alpha']
            self.beta = labo_data['beta']
            
            logger.info(f"Loaded LaBo-specific components from {labo_path}")

def create_labo_cbm_few_shot(num_classes: int, 
                            k_per_class: int = 25,
                            clip_model: str = "ViT-L/14",
                            device: str = "cuda") -> LaBoCBM:
    """
    Create LaBo CBM optimized for few-shot learning
    
    Uses smaller k_per_class and higher discriminability weight
    """
    return LaBoCBM(
        backbone=None,
        num_classes=num_classes,
        clip_model_name=clip_model,
        k_per_class=k_per_class,
        alpha=1e7,     
        beta=1.0,
        device=device
    )

def create_labo_cbm_full_data(num_classes: int,
                             k_per_class: int = 50,
                             clip_model: str = "ViT-L/14", 
                             device: str = "cuda") -> LaBoCBM:
    """
    Create LaBo CBM optimized for full data scenarios
    
    Uses larger k_per_class and balanced weights
    """
    return LaBoCBM(
        backbone=None,
        num_classes=num_classes,
        clip_model_name=clip_model,
        k_per_class=k_per_class,
        alpha=1e7,
        beta=5.0,      
        device=device
    )

class LaBoTrainingPipeline:
    """
    Complete LaBo training pipeline that integrates with the unified CBM system
    """
    
    def __init__(self, 
                 num_classes: int,
                 clip_model_name: str = "ViT-L/14",
                 device: str = "cuda"):
        self.num_classes = num_classes
        self.clip_model_name = clip_model_name
        self.device = device
    
    def train_complete_labo(self,
                           dataset: Any,
                           concepts: List[str],
                           classes: List[str],
                           config: Optional[Dict[str, Any]] = None) -> Tuple[LaBoCBM, Dict[str, Any]]:
        """
        Complete LaBo training pipeline
        
        Args:
            dataset: Training dataset
            concepts: Generated concepts from concept generation module
            classes: List of class names
            config: Training configuration
            
        Returns:
            model: Trained LaBo CBM
            results: Training results and metadata
        """
        
        default_config = {
            'k_per_class': 50,
            'alpha': 1e7,
            'beta': 1.0,
            'final_layer_type': FinalLayerType.DENSE_LINEAR,
            'use_language_priors': True,
            'learning_rate': 1e-4,
            'max_epochs': 100
        }
        
        if config is not None:
            default_config.update(config)
        
        logger.info("Starting complete LaBo training pipeline")
        
        model = LaBoCBM(
            backbone=None,
            num_classes=self.num_classes,
            clip_model_name=self.clip_model_name,
            k_per_class=default_config['k_per_class'],
            alpha=default_config['alpha'],
            beta=default_config['beta'],
            device=self.device
        )
        
        concept_activations = model.train_concept_layer(
            dataset=dataset,
            concepts=concepts,
            config=default_config
        )
        
        if hasattr(dataset, 'targets'):
            labels = torch.tensor(dataset.targets, device=self.device)
        elif hasattr(dataset, 'labels'):
            labels = torch.tensor(dataset.labels, device=self.device)
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")
        
        final_layer_config = {
            'layer_type': default_config['final_layer_type'],
            'use_language_priors': default_config['use_language_priors'],
            'learning_rate': default_config['learning_rate'],
            'max_epochs': default_config['max_epochs']
        }
        
        final_result = model.train_with_language_priors(
            concept_activations=concept_activations,
            labels=labels,
            classes=classes,
            final_layer_config=final_layer_config
        )
        
        results = {
            'model': model,
            'concept_activations': concept_activations,
            'final_layer_result': final_result,
            'selection_metadata': model.selection_metadata,
            'total_concepts_selected': model.num_concepts,
            'concepts_per_class': {name: len(concepts) for name, concepts in model.selected_concepts_per_class.items()},
            'config': default_config
        }
        
        logger.info(f"LaBo training complete: {model.num_concepts} concepts, {final_result['sparsity_stats']['sparsity_percentage']:.2%} final layer sparsity")
        
        return model, results

def get_labo_config(num_concepts: int, num_classes: int, **kwargs) -> Dict[str, Any]:
    """Get recommended configuration for LaBo CBM"""
    
    defaults = {
        'layer_type': FinalLayerType.DENSE_LINEAR,
        'k_per_class': 50,
        'alpha': 1e7,
        'beta': 1.0,
        'learning_rate': 1e-4,
        'max_epochs': 100,
        'normalize_concepts': False,  
        'use_language_priors': True,
        'weight_decay': 0.0
    }
    defaults.update(kwargs)
    
    return defaults

if __name__ == "__main__":
    print("LaBo CBM Implementation")
    print("======================")
    
    model = create_labo_cbm_few_shot(num_classes=10, k_per_class=25)
    
    example_concepts = [
        "red color", "blue color", "round shape", "square shape",
        "furry texture", "smooth surface", "large size", "small size",
        "metallic appearance", "natural texture"
    ] * 50 
    
    print(f"Created LaBo CBM with {len(example_concepts)} input concepts")
    print(f"Model will select up to {model.k_per_class} concepts per class")
    print(f"Submodular weights: alpha={model.alpha}, beta={model.beta}")
    
    pipeline = LaBoTrainingPipeline(num_classes=10)
    print("\nLaBo training pipeline ready for:")
    print("1. Submodular concept selection")
    print("2. CLIP-based concept encoding") 
    print("3. Language prior initialization")
    print("4. Unified final layer training")
