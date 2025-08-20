# cbm_library/scripts/lf_cbm_train.py
"""
Label‑Free CBM paper reproduction runner (no CLI, single entrypoint)

- Reads concept sets from cbm_library/concepts/main/outputs/label_free/{dataset}_filtered.txt
- Supports datasets: cifar10, cifar100, cub, imagenet, places365
- Uses your merged LabelFreeCBM + UnifiedFinalTrainer stack
- Saves final layer artifacts compatible with your repo format

USAGE (example):
    python -m cbm_library.scripts.lf_cbm_train cifar10 2>&1 | tee cbm_library/logs/lf_cbm_$(date +%Y%m%d_%H%M%S).log


Notes:
- CIFAR10/100 will auto-download to cfg.data_dir if missing.
- CUB / ImageNet / Places365 generally require you to place data under cfg.data_dir.
  Adjust the folder structure accordingly.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Library imports (do not change your codebase)
from cbm_library.config import LabelFreeCBMConfig
from cbm_library.models import LabelFreeCBM, read_concepts_file
from cbm_library.models.final_layer import (
    UnifiedFinalTrainer,
    get_label_free_cbm_config,
)
from cbm_library.utils.logging import setup_enhanced_logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TORCHVISION_DISABLE_PROGRESS_BARS"] = "1"

from tqdm import tqdm, trange

# ----------------------------
# Dataset utilities (minimal)
# ----------------------------

_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "cub": 200,
    "imagenet": 1000,
    "places365": 365,
}

def _concepts_path(dataset: str) -> str:
    base = Path(__file__).resolve().parents[2] / "cbm_library" / "concepts" / "main" / "outputs" / "label_free"
    return str(base / f"{dataset}_filtered.txt")

def _ensure_concepts(dataset: str) -> List[str]:
    cpath = _concepts_path(dataset)
    if not os.path.exists(cpath):
        raise FileNotFoundError(
            f"Concept file not found for dataset '{dataset}': {cpath}\n"
            f"Make sure you have the filtered concepts under concepts/main/outputs/label_free/"
        )
    return read_concepts_file(cpath)

def _make_dataset(dataset: str, root: str, split: str, clip_preprocess: Optional[transforms.Compose]) -> torch.utils.data.Dataset:
    """
    IMPORTANT:
    - We set transform = clip_preprocess so the model's CLIP backbone can consume tensors directly.
    - The model also re-preprocesses in its CLIP-image path; this is tolerated here to avoid changing your model code.
      (For strict reproduction, you may later refactor the model to avoid double-preprocessing.)
    """
    tfm = clip_preprocess  # see note above

    if dataset == "cifar10":
        is_train = split == "train"
        return datasets.CIFAR10(root=root, train=is_train, transform=tfm, download=True)
    elif dataset == "cifar100":
        is_train = split == "train"
        return datasets.CIFAR100(root=root, train=is_train, transform=tfm, download=True)
    elif dataset == "cub":
        # torchvision.datasets.CUB available in newer torchvision; requires manual download sometimes.
        is_train = split == "train"
        try:
            return datasets.CUB(root=root, train=is_train, transform=tfm, download=True)
        except TypeError:
            # older torchvision versions use 'download' only in constructor without split arg names
            return datasets.CUB(root=root, train=is_train, transform=tfm, download=True)  # type: ignore
    elif dataset == "imagenet":
        # Expect standard ImageNet structure under root/ILSVRC/Data/...
        # For quick repro, we map to root/imagenet/{train,val}
        sub = "train" if split == "train" else "val"
        path = os.path.join(root, "imagenet", sub)
        return datasets.ImageFolder(path, transform=tfm)
    elif dataset == "places365":
        # Places365 (standard) under root/places365/{train,val}
        sub = "train" if split == "train" else "val"
        path = os.path.join(root, "places365", sub)
        return datasets.ImageFolder(path, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ----------------------------
# Training pipeline
# ----------------------------

def train_label_free(
    dataset_name: str,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    End-to-end training that mirrors the Label-Free CBM paper flow:
      1) Build backbone (CLIP RN50 visual)
      2) Load dataset (train split)
      3) Read filtered concept set
      4) Train concept layer: top-5 filter → cos^3 projection → interpretability cutoff
      5) Train final layer with GLM-SAGA (sparse)
      6) Save artifacts to cfg.save_dir/lf_cbm_{dataset}
    """
    logger = setup_enhanced_logging("lf_cbm_train")

    # ---- config
    cfg = LabelFreeCBMConfig(dataset=dataset_name)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    if dataset_name not in _NUM_CLASSES:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from: {list(_NUM_CLASSES.keys())}")

    cfg.num_classes = _NUM_CLASSES[dataset_name]
    exp_name = f"lf_cbm_{dataset_name}"
    save_dir = Path(cfg.save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(cfg.summary())
    logger.info("⚙️ Regularization and training parameters:")
    logger.info(f"  sparsity_lambda:        {cfg.lam}")
    logger.info(f"  target_sparsity:        {getattr(cfg, 'target_sparsity', None)}")
    logger.info(f"  clip_cutoff:            {cfg.clip_cutoff}")
    logger.info(f"  interpretability_cutoff:{cfg.interpretability_cutoff}")
    logger.info(f"  proj_steps:             {cfg.proj_steps}")
    logger.info(f"  proj_batch_size:        {cfg.proj_batch_size}")
    logger.info(f"  saga_batch_size:        {cfg.saga_batch_size}")
    logger.info(f"  glm_step_size:          {cfg.saga_step_size}")
    logger.info(f"  glm_alpha:              {cfg.saga_alpha}")
    logger.info(f"  glm_epsilon:            {cfg.saga_epsilon}")
    logger.info(f"  glm_max_iters:          {cfg.n_iters}")
    logger.info(f"  learning_rate:          {cfg.learning_rate}")
    logger.info(f"  max_epochs:             {cfg.max_epochs}")
    logger.info(f"  weight_decay:           {cfg.weight_decay}")

    # ---- concepts
    concepts = _ensure_concepts(dataset_name)
    logger.info(f"Loaded {len(concepts)} concepts from file")

    # ---- backbone & CLIP preprocess
    # Build a temporary CLIP model to get its preprocess; BUT our model also exposes it after init.
    # We'll build the model first then reuse its clip_preprocess to construct datasets.
    backbone = LabelFreeCBM.build_backbone(device=cfg.device)
    model = LabelFreeCBM(
        backbone=backbone,
        num_concepts=max(len(concepts), 10),  # placeholder; real count set after filtering
        num_classes=cfg.num_classes,
        device=cfg.device,
        clip_name=cfg.clip_name,
    )

    # ---- dataset (train only, as the original paper primarily fits on train and validates internally)
    train_ds = _make_dataset(dataset_name, root=cfg.data_dir, split="train", clip_preprocess=model.clip_preprocess)

    # ---- concept layer (returns concept activations for full train set)
    cbl_cfg = dict(
        val_frac=cfg.validation_split,
        clip_cutoff=cfg.clip_cutoff,
        interpretability_cutoff=cfg.interpretability_cutoff,
        proj_steps=cfg.proj_steps,
        proj_batch_size=cfg.proj_batch_size,
        learning_rate=cfg.learning_rate,
        normalize_concepts=True,
        standardize_activations=False,
        log_every_n_steps=50,
        min_concepts_kept=10,
    )
    logger.info("Training concept layer (cos^3 projection with filters)…")
    concept_acts = model.train_concept_layer(train_ds, concepts, cbl_cfg)  # [N, C']

    # ---- labels tensor
    # We need labels for GLM-SAGA training. Pull from dataset deterministically.
    targets: List[int] = []
    loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    for _, y in loader:
        targets.extend([int(t) for t in y])
    labels = torch.tensor(targets, dtype=torch.long, device=cfg.device)
    
    concept_acts = concept_acts.detach().float().cpu()
    labels = labels.cpu()

    # ---- final layer with GLM‑SAGA sparse training
    trainer = UnifiedFinalTrainer()
    fl_cfg = get_label_free_cbm_config(
        num_concepts=concept_acts.shape[1],
        num_classes=cfg.num_classes,
        cfg=cfg,
        device='cpu',
        # you can override here e.g. glm_max_iters=2000
    )
    logger.info("Training final layer (GLM‑SAGA sparse)…")
    result = trainer.train(
        concept_activations=concept_acts,
        labels=labels,
        config=fl_cfg,
        validation_data=None,
        progress_callback=None,
    )

    # attach trained final layer back to the model for immediate inference use
    final_layer = trainer.create_final_layer(fl_cfg, result)
    model.final_layer = final_layer
    model.concept_mean = result.get("concept_mean")
    model.concept_std = result.get("concept_std")

    # ---- save artifacts (classifier weights & stats)
    trainer.save_training_result(result, str(save_dir))

    # save a lightweight model bundle for quick reuse (projection + final layer)
    torch.save(
        {
            "state_dict": {
                "concept_layer.weight": model.concept_layer.weight.detach().cpu(),
                "final_layer.weight": model.final_layer.weight.detach().cpu(),
                "final_layer.bias": model.final_layer.bias.detach().cpu(),
            },
            "concept_names": getattr(model, "concept_names", []),
            "feature_dim": getattr(model, "feature_dim", None),
            "num_classes": cfg.num_classes,
            "dataset": dataset_name,
            "clip_name": cfg.clip_name,
            "concept_mean": result.get("concept_mean"),
            "concept_std": result.get("concept_std"),
        },
        save_dir / "lf_cbm_minimal_bundle.pt",
    )
    logger.info(f"✅ Saved model bundle to: {save_dir/'lf_cbm_minimal_bundle.pt'}")

    return {
        "save_dir": str(save_dir),
        "num_concepts_final": int(concept_acts.shape[1]),
        "num_classes": cfg.num_classes,
        "dataset": dataset_name,
    }


# ----------------------------
# Script entry
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m cbm_library.scripts.lf_cbm_train <dataset>")
        print("Supported datasets: cifar10 | cifar100 | cub | imagenet | places365")
        sys.exit(1)

    dataset = sys.argv[1].lower().strip()
    _ = train_label_free(dataset)


if __name__ == "__main__":
    main()
