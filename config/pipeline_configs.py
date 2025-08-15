# cbm_library/config/pipeline_configs.py
"""
Complete pipeline configuration system for all CBM methods
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
import torch.nn as nn

from .base_config import CBMConfig
from .lf_cbm_config import LFCBMConfig
from ..training.final_layer import FinalLayerConfig, get_label_free_cbm_config, get_dense_cbm_config
from ..utils.logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)

@dataclass
class BackboneConfig:
    """Configuration for backbone networks"""
    architecture: str = "resnet18"  # resnet18, resnet50, vit_b_16, etc.
    pretrained: bool = True
    freeze_layers: int = 0  # Number of layers to freeze (0 = no freezing)
    input_size: int = 224
    dropout: float = 0.0
    
    @classmethod
    def for_dataset(cls, dataset: str) -> 'BackboneConfig':
        """Get recommended backbone config for dataset"""
        dataset_configs = {
            'cifar10': cls(architecture='resnet18', input_size=32),
            'cifar100': cls(architecture='resnet50', input_size=32),
            'cub': cls(architecture='resnet50', input_size=224),
            'imagenet': cls(architecture='resnet101', input_size=224),
            'places365': cls(architecture='resnet50', input_size=224),
        }
        return dataset_configs.get(dataset, cls())
    
    def create_backbone(self) -> nn.Module:
        """Create backbone network from config"""
        from torchvision import models
        
        if self.architecture == 'resnet18':
            backbone = models.resnet18(pretrained=self.pretrained)
        elif self.architecture == 'resnet50':
            backbone = models.resnet50(pretrained=self.pretrained)
        elif self.architecture == 'resnet101':
            backbone = models.resnet101(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {self.architecture}")

        # Remove final classification layer and flatten to vectors
        backbone = nn.Sequential(
            *list(backbone.children())[:-1],
            nn.Flatten(1)
        )

        # Apply freezing if specified
        if self.freeze_layers > 0:
            for i, child in enumerate(backbone.children()):
                if i < self.freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False
        
        return backbone

@dataclass
class CBLConfig:
    """Base configuration for Concept Bottleneck Layer training"""
    learning_rate: float = 0.001
    batch_size: int = 256
    max_steps: int = 1000
    device: str = "cuda"

@dataclass
class LabelFreeCBLConfig(CBLConfig):
    """Configuration for Label-Free CBM concept layer training"""
    # Concept filtering
    clip_cutoff: float = 0.25
    interpretability_cutoff: float = 0.45
    
    # Projection layer training
    proj_steps: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 256
    
    # Similarity function
    similarity_function: str = "cosine_cubed"  # "cosine", "cosine_cubed", "dot_product"
    cos_power: int = 3  # Power for cosine similarity (3 for cos³)
    
    # CLIP model
    clip_model: str = "ViT-B/32"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class VLGCBLConfig(CBLConfig):
    """Configuration for VLG-CBM concept layer training"""
    vlm_model: str = "clip"  # Vision-language model to use
    concept_generation_method: str = "automatic"
    concept_refinement: bool = True
    refinement_iterations: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CBLLMCBLConfig(CBLConfig):
    """Configuration for CB-LLM concept layer training"""
    llm_model: str = "gpt-3.5-turbo"
    concept_generation_prompts: List[str] = None
    concept_validation: bool = True
    max_concept_length: int = 20
    
    def __post_init__(self):
        if self.concept_generation_prompts is None:
            self.concept_generation_prompts = [
                "Generate visual concepts for {dataset}",
                "List important features for {dataset} classification"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LaBoCBLConfig(CBLConfig):
    """Configuration for LaBo-CBM concept layer training"""
    bottleneck_type: str = "linear"  # "linear", "nonlinear"
    concept_sparsity: float = 0.1
    regularization_weight: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PipelineConfig:
    """Complete pipeline configuration for CBM training"""
    method: str  # "label_free", "vlg", "cb_llm", "labo"
    backbone_config: BackboneConfig
    cbl_config: CBLConfig  # Method-specific
    final_config: FinalLayerConfig  # From UnifiedFinalTrainer
    
    # Dataset and general settings
    dataset: str = "cifar10"
    num_concepts: int = 100
    num_classes: int = 10
    device: str = "cuda"
    seed: int = 42
    
    # Training settings
    batch_size: int = 64
    save_dir: str = "./saved_models"
    log_dir: str = "./logs"
    experiment_name: str = "cbm_experiment"
    
    @classmethod
    def for_label_free_cbm(cls, dataset: str, num_concepts: int = None, 
                          num_classes: int = None, **kwargs) -> 'PipelineConfig':
        """Create pipeline config for Label-Free CBM"""
        
        # Dataset-specific defaults
        dataset_defaults = {
            'cifar10': {'num_concepts': 100, 'num_classes': 10},
            'cifar100': {'num_concepts': 200, 'num_classes': 100},
            'cub': {'num_concepts': 200, 'num_classes': 200},
            'imagenet': {'num_concepts': 500, 'num_classes': 1000},
            'places365': {'num_concepts': 300, 'num_classes': 365},
        }
        
        defaults = dataset_defaults.get(dataset, {'num_concepts': 100, 'num_classes': 10})
        if num_concepts is None:
            num_concepts = defaults['num_concepts']
        if num_classes is None:
            num_classes = defaults['num_classes']
        
        return cls(
            method="label_free",
            backbone_config=BackboneConfig.for_dataset(dataset),
            cbl_config=LabelFreeCBLConfig(
                proj_steps=kwargs.get('proj_steps', 1000),
                learning_rate=kwargs.get('cbl_lr', 0.001),
                clip_cutoff=kwargs.get('clip_cutoff', 0.25),
                interpretability_cutoff=kwargs.get('interpretability_cutoff', 0.45),
                similarity_function=kwargs.get('similarity_function', 'cosine_cubed')
            ),
            final_config=get_label_free_cbm_config(
                num_concepts=num_concepts,
                num_classes=num_classes,
                device=kwargs.get('device', 'cuda'),
                sparsity_lambda=kwargs.get('sparsity_lambda', 0.0007),
                normalize_concepts=kwargs.get('normalize_concepts', True),
                saga_batch_size=kwargs.get('saga_batch_size', 256)
            ),
            dataset=dataset,
            num_concepts=num_concepts,
            num_classes=num_classes,
            device=kwargs.get('device', 'cuda'),
            batch_size=kwargs.get('batch_size', 64),
            save_dir=kwargs.get('save_dir', f'./saved_models/{dataset}_label_free'),
            experiment_name=kwargs.get('experiment_name', f'label_free_{dataset}')
        )
    
    @classmethod
    def for_vlg_cbm(cls, dataset: str, num_concepts: int = None, 
                   num_classes: int = None, **kwargs) -> 'PipelineConfig':
        """Create pipeline config for VLG-CBM"""
        
        # Use similar defaults as Label-Free CBM
        dataset_defaults = {
            'cifar10': {'num_concepts': 100, 'num_classes': 10},
            'cifar100': {'num_concepts': 200, 'num_classes': 100},
            'cub': {'num_concepts': 200, 'num_classes': 200},
        }
        
        defaults = dataset_defaults.get(dataset, {'num_concepts': 100, 'num_classes': 10})
        if num_concepts is None:
            num_concepts = defaults['num_concepts']
        if num_classes is None:
            num_classes = defaults['num_classes']
        
        return cls(
            method="vlg",
            backbone_config=BackboneConfig.for_dataset(dataset),
            cbl_config=VLGCBLConfig(
                learning_rate=kwargs.get('cbl_lr', 0.001),
                batch_size=kwargs.get('cbl_batch_size', 256),
                vlm_model=kwargs.get('vlm_model', 'clip'),
                concept_refinement=kwargs.get('concept_refinement', True)
            ),
            final_config=get_label_free_cbm_config(  # VLG often uses similar final layer
                num_concepts=num_concepts,
                num_classes=num_classes,
                device=kwargs.get('device', 'cuda'),
                sparsity_lambda=kwargs.get('sparsity_lambda', 0.0007)
            ),
            dataset=dataset,
            num_concepts=num_concepts,
            num_classes=num_classes,
            device=kwargs.get('device', 'cuda'),
            experiment_name=kwargs.get('experiment_name', f'vlg_{dataset}')
        )
    
    @classmethod
    def for_cb_llm(cls, dataset: str, num_concepts: int = None, 
                  num_classes: int = None, **kwargs) -> 'PipelineConfig':
        """Create pipeline config for CB-LLM"""
        
        dataset_defaults = {
            'cifar10': {'num_concepts': 100, 'num_classes': 10},
            'cifar100': {'num_concepts': 200, 'num_classes': 100},
            'cub': {'num_concepts': 200, 'num_classes': 200},
        }
        
        defaults = dataset_defaults.get(dataset, {'num_concepts': 100, 'num_classes': 10})
        if num_concepts is None:
            num_concepts = defaults['num_concepts']
        if num_classes is None:
            num_classes = defaults['num_classes']
        
        return cls(
            method="cb_llm",
            backbone_config=BackboneConfig.for_dataset(dataset),
            cbl_config=CBLLMCBLConfig(
                learning_rate=kwargs.get('cbl_lr', 0.001),
                llm_model=kwargs.get('llm_model', 'gpt-3.5-turbo'),
                concept_validation=kwargs.get('concept_validation', True)
            ),
            final_config=get_dense_cbm_config(  # CB-LLM often uses dense final layer
                num_concepts=num_concepts,
                num_classes=num_classes,
                device=kwargs.get('device', 'cuda'),
                learning_rate=kwargs.get('final_lr', 0.001),
                normalize_concepts=kwargs.get('normalize_concepts', True)
            ),
            dataset=dataset,
            num_concepts=num_concepts,
            num_classes=num_classes,
            device=kwargs.get('device', 'cuda'),
            experiment_name=kwargs.get('experiment_name', f'cb_llm_{dataset}')
        )
    
    @classmethod
    def for_labo_cbm(cls, dataset: str, num_concepts: int = None, 
                    num_classes: int = None, **kwargs) -> 'PipelineConfig':
        """Create pipeline config for LaBo-CBM"""
        
        dataset_defaults = {
            'cifar10': {'num_concepts': 100, 'num_classes': 10},
            'cifar100': {'num_concepts': 200, 'num_classes': 100},
        }
        
        defaults = dataset_defaults.get(dataset, {'num_concepts': 100, 'num_classes': 10})
        if num_concepts is None:
            num_concepts = defaults['num_concepts']
        if num_classes is None:
            num_classes = defaults['num_classes']
        
        # LaBo uses sparse linear final layer
        from ..training.final_layer import get_labo_config
        
        return cls(
            method="labo",
            backbone_config=BackboneConfig.for_dataset(dataset),
            cbl_config=LaBoCBLConfig(
                learning_rate=kwargs.get('cbl_lr', 0.001),
                bottleneck_type=kwargs.get('bottleneck_type', 'linear'),
                concept_sparsity=kwargs.get('concept_sparsity', 0.1)
            ),
            final_config=get_labo_config(
                num_concepts=num_concepts,
                num_classes=num_classes,
                device=kwargs.get('device', 'cuda'),
                target_sparsity_per_class=kwargs.get('target_sparsity_per_class', 25)
            ),
            dataset=dataset,
            num_concepts=num_concepts,
            num_classes=num_classes,
            device=kwargs.get('device', 'cuda'),
            experiment_name=kwargs.get('experiment_name', f'labo_{dataset}')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'method': self.method,
            'backbone_config': asdict(self.backbone_config),
            'cbl_config': self.cbl_config.to_dict(),
            'final_config': self.final_config.to_dict(),
            'dataset': self.dataset,
            'num_concepts': self.num_concepts,
            'num_classes': self.num_classes,
            'device': self.device,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir,
            'experiment_name': self.experiment_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary"""
        # Reconstruct nested configs
        backbone_config = BackboneConfig(**config_dict['backbone_config'])
        
        # Reconstruct CBL config based on method
        method = config_dict['method']
        cbl_dict = config_dict['cbl_config']
        
        if method == 'label_free':
            cbl_config = LabelFreeCBLConfig(**cbl_dict)
        elif method == 'vlg':
            cbl_config = VLGCBLConfig(**cbl_dict)
        elif method == 'cb_llm':
            cbl_config = CBLLMCBLConfig(**cbl_dict)
        elif method == 'labo':
            cbl_config = LaBoCBLConfig(**cbl_dict)
        else:
            cbl_config = CBLConfig(**cbl_dict)
        
        # Reconstruct final config
        from ..training.final_layer import FinalLayerConfig
        final_config = FinalLayerConfig.from_dict(config_dict['final_config'])
        
        return cls(
            method=method,
            backbone_config=backbone_config,
            cbl_config=cbl_config,
            final_config=final_config,
            dataset=config_dict['dataset'],
            num_concepts=config_dict['num_concepts'],
            num_classes=config_dict['num_classes'],
            device=config_dict['device'],
            seed=config_dict['seed'],
            batch_size=config_dict['batch_size'],
            save_dir=config_dict['save_dir'],
            log_dir=config_dict['log_dir'],
            experiment_name=config_dict['experiment_name']
        )
    
    def save(self, filepath: str):
        """Save pipeline config to file"""
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"💾 Pipeline config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """Load pipeline config from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"📂 Pipeline config loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """Validate complete pipeline configuration"""
        issues = []
        
        # Validate method
        supported_methods = ['label_free', 'vlg', 'cb_llm', 'labo']
        if self.method not in supported_methods:
            issues.append(f"Method must be one of {supported_methods}")
        
        # Validate dimensions
        if self.num_concepts <= 0:
            issues.append("num_concepts must be positive")
        
        if self.num_classes <= 0:
            issues.append("num_classes must be positive")
        
        # Validate CBL config consistency
        if hasattr(self.cbl_config, 'device') and self.cbl_config.device != self.device:
            issues.append("CBL config device must match pipeline device")
        
        # Validate final config consistency
        if self.final_config.num_concepts != self.num_concepts:
            issues.append("Final config num_concepts must match pipeline num_concepts")
        
        if self.final_config.num_classes != self.num_classes:
            issues.append("Final config num_classes must match pipeline num_classes")
        
        return issues
    
    def __repr__(self) -> str:
        return (f"PipelineConfig(method={self.method}, dataset={self.dataset}, "
                f"concepts={self.num_concepts}, classes={self.num_classes})")

# Convenience functions for quick config creation
def create_label_free_config(dataset: str, **kwargs) -> PipelineConfig:
    """Quick creation of Label-Free CBM config"""
    return PipelineConfig.for_label_free_cbm(dataset, **kwargs)

def create_vlg_config(dataset: str, **kwargs) -> PipelineConfig:
    """Quick creation of VLG-CBM config"""
    return PipelineConfig.for_vlg_cbm(dataset, **kwargs)

def create_cb_llm_config(dataset: str, **kwargs) -> PipelineConfig:
    """Quick creation of CB-LLM config"""
    return PipelineConfig.for_cb_llm(dataset, **kwargs)

def create_labo_config(dataset: str, **kwargs) -> PipelineConfig:
    """Quick creation of LaBo-CBM config"""
    return PipelineConfig.for_labo_cbm(dataset, **kwargs)

# Usage examples
def usage_examples():
    """Examples of how to use the pipeline configs"""
    
    # Example 1: Label-Free CBM for CIFAR-10
    config = create_label_free_config(
        dataset='cifar10',
        clip_cutoff=0.25,
        interpretability_cutoff=0.45,
        sparsity_lambda=0.0007
    )
    
    # Example 2: Load from existing LF-CBM config
    lf_config = LFCBMConfig.get_dataset_preset('cub')
    pipeline_config = PipelineConfig.for_label_free_cbm(
        dataset='cub',
        num_concepts=lf_config.num_concepts,
        sparsity_lambda=lf_config.sparsity_lambda
    )
    
    # Example 3: Save and load
    config.save('./configs/my_experiment.json')
    loaded_config = PipelineConfig.load('./configs/my_experiment.json')
    
    # Example 4: Validation
    issues = config.validate()
    if issues:
        print(f"Configuration issues: {issues}")
    
    print("✅ Pipeline configuration examples ready!")

if __name__ == "__main__":
    usage_examples()