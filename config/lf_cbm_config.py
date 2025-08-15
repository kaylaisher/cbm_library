"""
Label-Free CBM specific configuration
"""

from dataclasses import dataclass
from typing import List
from .base_config import CBMConfig


@dataclass
class LFCBMConfig(CBMConfig):
    """Configuration for Label-Free Concept Bottleneck Models"""
    
    # Label-Free CBM Specific Parameters
    concept_set_size: int = 1000
    concept_filtering: bool = True
    concept_similarity_threshold: float = 0.85
    max_concept_length: int = 30
    
    # Concept Generation
    concept_source: str = "gpt"  # "gpt", "conceptnet", "manual"
    gpt_temperature: float = 0.7
    conceptnet_relations: List[str] = None
    conceptnet_limit: int = 100
    
    # Sparse Training
    sparsity_lambda: float = 1e-3
    sparsity_warmup_epochs: int = 10
    final_layer_sparsity: bool = True
    target_sparsity: float = 0.1
    
    # Concept Layer Training
    concept_lr: float = 0.01
    concept_epochs: int = 50
    concept_batch_size: int = 256
    
    # Final Layer Training  
    final_lr: float = 0.001
    final_epochs: int = 100
    final_batch_size: int = 64
    
    def __post_init__(self):
        """Set default values that need initialization"""
        if self.conceptnet_relations is None:
            self.conceptnet_relations = ['RelatedTo', 'IsA', 'PartOf', 'HasProperty']
    
    def validate(self) -> List[str]:
        """Extended validation for LF-CBM parameters"""
        issues = super().validate()

        if self.max_concept_length <= 0:
            issues.append("max_concept_length must be positive")

        if self.concept_batch_size <= 0:
            issues.append("concept_batch_size must be positive")

        if self.final_batch_size <= 0:
            issues.append("final_batch_size must be positive")

        # Training hyperparams sanity
        if self.concept_lr <= 0:
            issues.append("concept_lr must be positive")
        elif self.concept_lr > 1.0:
            issues.append("concept_lr is unusually high (>1.0)")
        if self.final_lr <= 0:
            issues.append("final_lr must be positive")
        elif self.final_lr > 1.0:
            issues.append("final_lr is unusually high (>1.0)")
        if not isinstance(self.concept_filtering, bool):
            issues.append("concept_filtering must be a boolean")
        if not isinstance(self.sparsity_warmup_epochs, int) or self.sparsity_warmup_epochs <= 0:
            issues.append("sparsity_warmup_epochs must be a positive integer")
        if not isinstance(self.conceptnet_limit, int) or self.conceptnet_limit <= 0:
            issues.append("conceptnet_limit must be a positive integer")
        if not (0 <= self.gpt_temperature <= 1):
            issues.append("gpt_temperature must be between 0 and 1")

        # LF-CBM specific validations
        if self.concept_set_size <= self.num_concepts:
            issues.append("concept_set_size should be larger than num_concepts")
        
        if not (0 <= self.concept_similarity_threshold <= 1):
            issues.append("concept_similarity_threshold must be between 0 and 1")
        
        if self.concept_source not in ['gpt', 'conceptnet', 'manual']:
            issues.append("concept_source must be 'gpt', 'conceptnet', or 'manual'")
        
        if not (0 <= self.target_sparsity <= 1):
            issues.append("target_sparsity must be between 0 and 1")
        
        if self.sparsity_lambda < 0:
            issues.append("sparsity_lambda must be non-negative")
        
        return issues
    
    @classmethod
    def get_dataset_preset(cls, dataset: str) -> 'LFCBMConfig':
        """Get preset configuration for specific datasets"""
        
        if dataset == "cifar10":
            return cls(
                dataset="cifar10",
                num_concepts=100,
                num_classes=10,
                backbone="resnet18",
                batch_size=128,
                learning_rate=0.001,
                concept_set_size=500,
                sparsity_lambda=1e-3,
                max_epochs=50
            )
        
        elif dataset == "cifar100":
            return cls(
                dataset="cifar100", 
                num_concepts=200,
                num_classes=100,
                backbone="resnet50",
                batch_size=64,
                learning_rate=0.0005,
                concept_set_size=1000,
                sparsity_lambda=5e-4,
                max_epochs=100
            )
        
        elif dataset == "cub":
            return cls(
                dataset="cub",
                num_concepts=200,
                num_classes=200,
                backbone="resnet50",
                batch_size=32,
                learning_rate=0.0001,
                concept_set_size=2000,
                sparsity_lambda=1e-4,
                max_epochs=200,
                concept_source="gpt"
            )
        
        elif dataset == "imagenet":
            return cls(
                dataset="imagenet",
                num_concepts=500,
                num_classes=1000,
                backbone="resnet101",
                batch_size=16,
                learning_rate=0.00005,
                concept_set_size=5000,
                sparsity_lambda=1e-5,
                max_epochs=100,
                concept_source="gpt"
            )
        
        elif dataset == "places365":
            return cls(
                dataset="places365",
                num_concepts=300,
                num_classes=365,
                backbone="resnet50",
                batch_size=32,
                learning_rate=0.0001,
                concept_set_size=3000,
                sparsity_lambda=5e-5,
                max_epochs=150
            )
        
        else:
            # Default configuration
            return cls(dataset=dataset)
    
    def __repr__(self) -> str:
        return (f"LFCBMConfig(dataset={self.dataset}, concepts={self.num_concepts}, "
                f"source={self.concept_source}, sparsity={self.target_sparsity})")

