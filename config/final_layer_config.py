"""
Final Layer configuration dataclass
Separated from training logic for clarity.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import torch
from enum import Enum


class FinalLayerType(Enum):
    SPARSE_GLM = "sparse_glm"      # Label-free CBM (GLM-SAGA)
    DENSE_LINEAR = "dense_linear"  # Standard dense training
    SPARSE_LINEAR = "sparse_linear"  # Dense + top-k masking
    ELASTIC_NET = "elastic_net"    # Alias to dense (can add real EN later)


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

    # GLM-SAGA specific
    glm_step_size: float = 0.1
    glm_alpha: float = 0.99
    glm_max_iters: int = 1000
    glm_epsilon: float = 1.0
    saga_batch_size: int = 256

    # Standard optimization
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 100
    weight_decay: float = 0.0

    # Normalization
    normalize_concepts: bool = True
    concept_mean: Optional[torch.Tensor] = None
    concept_std: Optional[torch.Tensor] = None

    # Device
    device: str = "cuda"

    def to_dict(self):
        out = asdict(self)
        out["layer_type"] = self.layer_type.value
        # do not serialize tensors
        out["concept_mean"] = None
        out["concept_std"] = None
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        d = dict(data)
        d["layer_type"] = FinalLayerType(d["layer_type"])
        return cls(**d)
