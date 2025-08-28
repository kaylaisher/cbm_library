from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import torch

@dataclass
class VLGCBMConfig:
    # ---- dataset & IO ----
    dataset: str = "cifar10"
    save_dir: str = "saved_models"
    load_dir: Optional[str] = None
    dataset_dir: str = "/kayla/dataset"
    annotation_dir: Optional[str] = "/kayla/Annotations"
    concept_set: Optional[str] = None
    filter_set: Optional[str] = None
    val_split: float = 0.2

    # ---- device & runtime ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4
    data_parallel: bool = False
    visualize_concepts: bool = False

    # ---- backbone ----
    backbone: str = "clip_RN50"
    feature_layer: str = "layer4"
    use_clip_penultimate: bool = True

    # ---- CBL ----
    cbl_batch_size: int = 32
    cbl_epochs: int = 20
    cbl_lr: float = 5e-4
    cbl_weight_decay: float = 1e-5
    cbl_hidden_layers: int = 0
    cbl_confidence_threshold: float = 0.15
    crop_to_concept_prob: float = 0.5
    cbl_finetune: bool = False
    cbl_optimizer: str = "adam"
    cbl_scheduler: Optional[str] = None
    cbl_bb_lr_rate: float = 0.1
    cbl_loss_type: str = "bce"
    cbl_pos_weight: Optional[float] = None
    cbl_auto_weight: bool = False
    cbl_twoway_tp: bool = False
    allones_concept: bool = False

    # ---- Final layer ----
    dense: bool = False
    saga_batch_size: int = 512
    saga_step_size: float = 0.1
    saga_n_iters: int = 2000
    saga_lam: float = 7e-4
    saga_alpha: float = 0.99
    saga_epsilon: float = 1.0

    # ---- Optional dense head ----
    dense_lr: float = 3e-4
    dense_batch_size: int = 512
    dense_epochs: int = 2000
    dense_weight_decay: float = 0.0

    skip_concept_filter: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
