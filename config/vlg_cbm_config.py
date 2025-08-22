# cbm_library/config/vlg_cbm_config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict
import os
import json
import datetime

@dataclass
class VLGCBMConfig:
    # ---- run id / paths ----
    dataset: str = "cifar10"
    num_classes: int = 10
    data_dir: str = "/kayla/dataset"
    annotations_dir: str = "cbm_library/data/annotations"
    save_dir: str = "/kayla/saved_models"
    run_name: Optional[str] = None
    device: str = "cuda"

    # ---- backbone / features ----
    backbone: str = "resnet50"            # {"resnet50","clip_visual"}
    feature_layer: str = "layer4"         # for torchvision backbones
    feature_pool: str = "avg"             # {"avg","max"}
    feature_dim: Optional[int] = None     # set automatically if None

    # ---- CLIP (vision-language guidance) ----
    clip_model: str = "ViT-B/32"
    clip_device: Optional[str] = None     # default: same as device
    prompt_template: str = "a photo of {concept}"

    # ---- concept vocabulary & filtering ----
    concept_list_path: Optional[str] = None     # txt file, one concept per line
    extra_concepts: List[str] = field(default_factory=list)
    min_pos_per_concept: int = 5
    clip_topk: int = 5
    clip_cutoff: float = 0.15                  # drop concepts below this mean@topk

    # ---- projection (features -> concepts) ----
    proj_hidden_dim: Optional[int] = None      # None => linear
    proj_lr: float = 3e-4
    proj_weight_decay: float = 1e-4
    proj_epochs: int = 20
    proj_early_stop_patience: int = 5
    cosine_tau: float = 0.07                   # softmax temperature for CLIP sims

    # ---- concept normalization ----
    eps: float = 1e-6

    # ---- final layer (concepts -> classes) ----
    final_type: str = "saga"                   # {"saga","dense"}
    dense_lr: float = 5e-4
    dense_weight_decay: float = 0.0
    dense_epochs: int = 40
    saga_step_size: float = 0.1
    saga_n_iters: int = 2000
    saga_lam: float = 7e-4
    max_sparsity: float = 0.95

    # ---- dataloader / misc ----
    batch_size: int = 256
    num_workers: int = min(8, os.cpu_count() or 2)
    seed: int = 42
    log_every: int = 50

    def finalize(self) -> "VLGCBMConfig":
        """Compute derived fields and defaults."""
        if self.run_name is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"vlg_cbm_{self.dataset}_{ts}"
        if self.clip_device is None:
            self.clip_device = self.device
        os.makedirs(self.save_dir, exist_ok=True)
        return self

    # Utilities
    def to_dict(self) -> Dict: return asdict(self)
    def dump_json(self, out_path: str) -> None:
        with open(out_path, "w") as f: json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_overrides(cls, **kw) -> "VLGCBMConfig":
        cfg = cls(**kw)
        return cfg.finalize()
