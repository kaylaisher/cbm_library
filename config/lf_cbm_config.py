"""
Label-Free CBM Configuration (repo-style, single source of truth)

"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class LabelFreeCBMConfig:
    # ===== Dataset & Model =====
    dataset: str = "cifar10"
    concept_set: Optional[str] = None          # path to concept set file (if used)
    backbone: str = "clip_RN50"                # e.g., clip_RN50, resnet18_cub, resnet18_places
    clip_name: str = "ViT-B/16"
    device: str = "cuda"

    # Model architecture (left None to be set by training/data pipeline)
    num_concepts: Optional[int] = None         # determined after concept filtering
    num_classes: Optional[int] = None          # inferred from dataset loader

    # ===== Training Hyperparameters =====
    batch_size: int = 512
    saga_batch_size: int = 256                 # final layer (GLM-SAGA)
    proj_batch_size: int = 50000               # projection training batch “budget”

    learning_rate: float = 0.001               # kept for compatibility
    proj_steps: int = 1000                     # projection training steps
    n_iters: int = 1000                        # GLM-SAGA iterations
    max_epochs: int = 100
    weight_decay: float = 1e-4

    # ===== Label-Free specifics =====
    clip_cutoff: float = 0.25                  # low CLIP activations pruned
    interpretability_cutoff: float = 0.45      # low concept similarity pruned
    lam: float = 0.0007                        # sparsity (higher => sparser)

    feature_layer: str = "layer4"
    pool_mode: str = "avg"
    similarity_function: str = "cos_similarity_cubed"
    cos_power: int = 3
    min_norm: float = 1e-3

    # ===== Directories =====
    activation_dir: str = "saved_activations"
    save_dir: str = "saved_models"
    concept_dir: str = "data/concept_sets"
    data_dir: str = "data"
    log_dir: str = "logs"

    # ===== GLM-SAGA params =====
    saga_step_size: float = 0.1
    saga_alpha: float = 0.99
    saga_epsilon: float = 1.0
    saga_k: int = 1
    do_zero: bool = False

    # ===== CLIP =====
    clip_batch_size: int = 200

    # ===== Logging / Debug =====
    verbose: bool = False
    save_activations: bool = True
    save_clip_features: bool = True

    # ===== Eval / Early stopping knobs =====
    validation_split: float = 0.1
    eval_batch_size: int = 256
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 1e-4

    # ===== Experiment meta =====
    seed: int = 42
    experiment_name: str = "label_free_cbm_experiment"
    save_frequency: int = 10
    eval_frequency: int = 5

    # ----- Serialization -----
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str) -> None:
        cfg = self.to_dict()
        cfg["config_version"] = "1.0"
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LabelFreeCBMConfig":
        with open(filepath, "r") as f:
            cfg = json.load(f)
        cfg.pop("config_version", None)
        return cls(**cfg)

    # ----- Convenience (optional) -----
    def summary(self) -> str:
        return (
            "Label-Free CBM Config\n"
            f"- dataset: {self.dataset}\n"
            f"- backbone: {self.backbone} | clip: {self.clip_name}\n"
            f"- num_classes: {self.num_classes} | num_concepts: {self.num_concepts}\n"
            f"- proj_steps: {self.proj_steps} | n_iters: {self.n_iters} | lam: {self.lam}\n"
            f"- dirs: save={self.save_dir}, acts={self.activation_dir}, logs={self.log_dir}\n"
        )
