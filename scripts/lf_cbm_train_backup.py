"""
Label‑Free CBM paper reproduction runner (no CLI, single entrypoint)

- Reads concept sets from cbm_library/concepts/main/outputs/label_free/{dataset}_filtered.txt
- Supports datasets: cifar10, cifar100, cub, imagenet, places365
- Uses your merged LabelFreeCBM + UnifiedFinalTrainer stack
- Saves final layer artifacts compatible with your repo format

USAGE (example):
    python -m cbm_library.scripts.lf_cbm_train cifar10 2>&1 | tee cbm_library/logs/lf_cbm_$(date +%Y%m%d_%H%M%S).log

"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import datetime


import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

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
    tfm = clip_preprocess

    if dataset == "cifar10":
        is_train = split == "train"
        return datasets.CIFAR10(root=root, train=is_train, transform=tfm, download=True)
    
    elif dataset == "cifar100":
        is_train = split == "train"
        return datasets.CIFAR100(root=root, train=is_train, transform=tfm, download=True)
    
    elif dataset == "cub":
        is_train = split == "train"
        try:
            return datasets.CUB(root=root, train=is_train, transform=tfm, download=True)
        except TypeError:
            return datasets.CUB(root=root, train=is_train, transform=tfm, download=True) 
    
    elif dataset == "imagenet":
        sub = "train" if split == "train" else ("val" if split == "val" else "val" if split == "test" else split)
        path = os.path.join(root, "imagenet", sub)
        return datasets.ImageFolder(path, transform=tfm)
    
    elif dataset == "places365":
        sub = "train" if split == "train" else ("val" if split in ("val", "test") else split)
        path = os.path.join(root, "places365", sub)
        return datasets.ImageFolder(path, transform=tfm)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ----------------------------
# Training pipeline
# ----------------------------

def train_label_free(
    dataset_name: str,
    cfg_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    logger = setup_enhanced_logging("lf_cbm_train")

    # ---- config
    cfg = LabelFreeCBMConfig(dataset=dataset_name)
    cfg.data_dir = "/kayla/dataset"
    # defensive defaults for fields used below
    if not hasattr(cfg, "num_workers"):
        cfg.num_workers = min(8, (os.cpu_count() or 2))
    if not hasattr(cfg, "batch_size"):
        cfg.batch_size = 256
    cfg.save_dir = getattr(cfg, "save_dir", "/kayla/saved_models")

    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    if dataset_name not in _NUM_CLASSES:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from: {list(_NUM_CLASSES.keys())}")

    # --- ensure num_classes is set (defensive) ---
    num_classes = getattr(cfg, "num_classes", None)
    if not isinstance(num_classes, int) or num_classes <= 0:
        num_classes = _NUM_CLASSES[dataset_name]
        setattr(cfg, "num_classes", num_classes)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"lf_cbm_{dataset_name}_{timestamp}"
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
    backbone = LabelFreeCBM.build_backbone(device=cfg.device)
    model = LabelFreeCBM(
        backbone=backbone,
        num_concepts=max(len(concepts), 10),  # placeholder; real count set after filtering
        num_classes=cfg.num_classes,
        device=cfg.device,
        clip_name=cfg.clip_name,
    )

    # ---- dataset (CIFAR10 special split)
    '''
    if dataset_name == "cifar10":
        train_full = datasets.CIFAR10(root=cfg.data_dir, train=True,  download=True, transform=model.clip_preprocess)
        test_set   = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=model.clip_preprocess)

        assert len(train_full) == 50_000, f"Unexpected CIFAR-10 train size: {len(train_full)}"
        labels_np = np.array(train_full.targets)
        num_classes = 10
        val_per_class = 500
        rng = np.random.default_rng(getattr(cfg, "seed", 42))

        val_idx, train_idx = [], []
        for c in range(num_classes):
            idx_c = np.where(labels_np == c)[0]
            rng.shuffle(idx_c)
            val_idx.extend(idx_c[:val_per_class])
            train_idx.extend(idx_c[val_per_class:])

        train_ds = Subset(train_full, train_idx)   # 45,000
        val_ds   = Subset(train_full, val_idx)     # 5,000

        pin_mem = bool(getattr(cfg, "pin_memory", True)) and torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=pin_mem)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=pin_mem)
        test_loader  = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=pin_mem)

        if hasattr(cfg, "validation_split"):
            cfg.validation_split = 0.0
    else:
        train_ds = _make_dataset(dataset_name, root=cfg.data_dir, split="train", clip_preprocess=model.clip_preprocess)
    '''
        # ---- dataset (CIFAR10 special split)
    if dataset_name == "cifar10":
        train_full = datasets.CIFAR10(root=cfg.data_dir, train=True,  download=True, transform=model.clip_preprocess)
        test_set   = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=model.clip_preprocess)

        assert len(train_full) == 50_000, f"Unexpected CIFAR-10 train size: {len(train_full)}"
        labels_np = np.array(train_full.targets)
        num_classes = 10
        val_per_class = 500
        rng = np.random.default_rng(getattr(cfg, "seed", 42))

        val_idx, train_idx = [], []
        for c in range(num_classes):
            idx_c = np.where(labels_np == c)[0]
            rng.shuffle(idx_c)
            val_idx.extend(idx_c[:val_per_class])
            train_idx.extend(idx_c[val_per_class:])

        train_ds = Subset(train_full, train_idx)   # 45,000
        val_ds   = Subset(train_full, val_idx)     # 5,000

        # --- log sizes ---
        logger.info(f"[DATA] train size: {len(train_ds)}")
        logger.info(f"[DATA] val size:   {len(val_ds)}")
        logger.info(f"[DATA] test size:  {len(test_set)}")

        pin_mem = bool(getattr(cfg, "pin_memory", True)) and torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=pin_mem)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=pin_mem)
        test_loader  = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=pin_mem)

        if hasattr(cfg, "validation_split"):
            cfg.validation_split = 0.0
    else:
        train_ds = _make_dataset(dataset_name, root=cfg.data_dir,
                                 split="train", clip_preprocess=model.clip_preprocess)
        test_ds  = _make_dataset(dataset_name, root=cfg.data_dir,
                                 split="test" if dataset_name in ("cifar100", "cub") else "val",
                                 clip_preprocess=model.clip_preprocess)

        # --- log sizes ---
        logger.info(f"[DATA] train size: {len(train_ds)}")
        logger.info(f"[DATA] test size:  {len(test_ds)}")

    # ---- concept layer
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

    # ---- labels tensor for train
    targets: List[int] = []
    loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=min(4, cfg.num_workers))
    for _, y in loader:
        targets.extend([int(t) for t in y])
    labels = torch.tensor(targets, dtype=torch.long, device=cfg.device)

    concept_acts = concept_acts.detach().float().cpu()
    labels = labels.cpu()

    # ---- (ONLY for CIFAR-10) compute VAL concept activations/labels
    val_acts, val_labels = None, None
    if dataset_name == "cifar10":
        model.eval()
        v_acts, v_lbls = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device, non_blocking=("cuda" in str(cfg.device)))

                # --- backbone features (INLINE robust extraction; no helpers) ---
                feats = None
                if hasattr(model, "get_backbone_features"):
                    out = model.get_backbone_features(xb)
                elif hasattr(model, "backbone"):
                    out = model.backbone(xb)
                else:
                    out = xb  # extreme fallback (should not happen)

                if isinstance(out, torch.Tensor):
                    feats = out
                elif isinstance(out, (list, tuple)) and len(out) > 0:
                    for item in out:
                        if isinstance(item, torch.Tensor):
                            feats = item
                            break
                elif isinstance(out, dict):
                    for k in ("feats", "features", "penultimate", "pool", "last_hidden_state", "x"):
                        v = out.get(k, None)
                        if isinstance(v, torch.Tensor):
                            feats = v
                            break

                if feats is None:
                    raise ValueError(f"Backbone returned no tensor features (type={type(out)}). "
                                     "Adapt feature extraction to your encoder.")

                if feats.ndim > 2:
                    feats = torch.flatten(feats, 1)

                # --- concept projection ---
                if hasattr(model, "proj_layer"):
                    acts_val = model.proj_layer(feats)
                elif hasattr(model, "concept_layer"):
                    feats = feats.to(model.concept_layer.weight.dtype)
                    acts_val = model.concept_layer(feats)
                else:
                    raise RuntimeError("No projection/concept layer found on model.")

                v_acts.append(acts_val.detach().cpu())
                v_lbls.append(yb.detach().cpu())

        val_acts = torch.cat(v_acts, 0).float()
        val_labels = torch.cat(v_lbls, 0).long()

    # ---- final layer with GLM‑SAGA sparse training
    trainer = UnifiedFinalTrainer()
    fl_cfg = get_label_free_cbm_config(
        num_concepts=concept_acts.shape[1],
        num_classes=cfg.num_classes,
        cfg=cfg,
        device='cpu',
    )
    logger.info("Training final layer (GLM‑SAGA sparse)…")

    val_pair = (val_acts, val_labels) if dataset_name == "cifar10" else None
    result = trainer.train(
        concept_activations=concept_acts,
        labels=labels,
        config=fl_cfg,
        validation_data=val_pair,
        progress_callback=None,
    )

    # attach trained final layer back to the model for immediate inference use
    final_layer = trainer.create_final_layer(fl_cfg, result)
    model.final_layer = final_layer
    model.concept_mean = result.get("concept_mean")
    model.concept_std = result.get("concept_std")

    # ---- save artifacts
    trainer.save_training_result(result, str(save_dir))

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

    # ----------------------------
    # ANEC export
    # ----------------------------
    anec_dir = save_dir / "anec_export"
    anec_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    model.eval()

    # TRAIN μ/σ
    mu = result["concept_mean"].detach().cpu().float().view(1, -1)
    sigma = result["concept_std"].detach().cpu().float().clamp_min(1e-6).view(1, -1)

    # (A) export TRAIN (normalized)
    train_acts_norm = (concept_acts.float() - mu) / sigma
    torch.save(train_acts_norm, anec_dir / "train_activations.pt")
    torch.save(labels.long().cpu(), anec_dir / "train_labels.pt")

    # (B) also export VAL for CIFAR-10
    if dataset_name == "cifar10" and val_acts is not None:
        Xn_val = (val_acts.float() - mu) / sigma
        torch.save(Xn_val, anec_dir / "val_activations.pt")
        torch.save(val_labels.long().cpu(), anec_dir / "val_labels.pt")

    # (C) export TEST/VAL splits for other datasets as applicable
    maybe_splits = []
    if dataset_name in ("cifar10", "cifar100", "cub"):
        maybe_splits.append("test")
    if dataset_name in ("imagenet", "places365"):
        maybe_splits.append("val")

    for split in maybe_splits:
        ds = _make_dataset(dataset_name, root=cfg.data_dir, split=split, clip_preprocess=model.clip_preprocess)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=min(4, cfg.num_workers))
        acts_list, lbl_list = [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=("cuda" in str(device)))

                # --- backbone features (INLINE robust extraction; no helpers) ---
                feats = None
                if hasattr(model, "get_backbone_features"):
                    out = model.get_backbone_features(xb)
                elif hasattr(model, "backbone"):
                    out = model.backbone(xb)
                else:
                    out = xb

                if isinstance(out, torch.Tensor):
                    feats = out
                elif isinstance(out, (list, tuple)) and len(out) > 0:
                    for item in out:
                        if isinstance(item, torch.Tensor):
                            feats = item
                            break
                elif isinstance(out, dict):
                    for k in ("feats", "features", "penultimate", "pool", "last_hidden_state", "x"):
                        v = out.get(k, None)
                        if isinstance(v, torch.Tensor):
                            feats = v
                            break

                if feats is None:
                    raise ValueError(f"Backbone returned no tensor features (type={type(out)}). "
                                     "Adapt feature extraction to your encoder.")

                if feats.ndim > 2:
                    feats = torch.flatten(feats, 1)

                # --- concept projection ---
                if hasattr(model, "proj_layer"):
                    acts = model.proj_layer(feats)
                elif hasattr(model, "concept_layer"):
                    feats = feats.to(model.concept_layer.weight.dtype)
                    acts = model.concept_layer(feats)
                else:
                    raise RuntimeError("No projection/concept layer found on model.")

                acts_list.append(acts.detach().cpu())
                lbl_list.append(yb.detach().cpu())

        X = torch.cat(acts_list, 0).float()
        y = torch.cat(lbl_list, 0).long()
        Xn = (X - mu) / sigma

        if split == "test":
            torch.save(Xn, anec_dir / "test_activations.pt")
            torch.save(y,  anec_dir / "test_labels.pt")
        elif split == "val":
            torch.save(Xn, anec_dir / "val_activations.pt")
            torch.save(y,  anec_dir / "val_labels.pt")

    logger.info(f"📦 ANEC export ready at: {anec_dir}")

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
