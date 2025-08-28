"""
USAGE (example):
    python -m cbm_library.scripts.lf_cbm_train cifar100 
"""

from __future__ import annotations

import os
import sys
import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from cbm_library.config import LabelFreeCBMConfig
from cbm_library.models import LabelFreeCBM, read_concepts_file
from cbm_library.models.final_layer import UnifiedFinalTrainer, get_label_free_cbm_config
from cbm_library.utils.logging import setup_enhanced_logging
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TORCHVISION_DISABLE_PROGRESS_BARS"] = "1"


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

def _make_dataset(
    dataset: str,
    root: str,
    split: str,
    clip_preprocess: Optional[transforms.Compose],
) -> torch.utils.data.Dataset:
   
    tfm = clip_preprocess
    name = dataset.lower()
    sp = split.lower()

    if name == "cifar10":
        is_train = sp == "train"
        return datasets.CIFAR10(root=root, train=is_train, transform=tfm, download=True)

    elif name == "cifar100":
        is_train = sp == "train"
        return datasets.CIFAR100(root=root, train=is_train, transform=tfm, download=True)

    elif name in {"cub", "cub200", "cub_200_2011"}:
        cub_split = "train" if sp == "train" else "test"
        data_dir = os.path.join(root, "CUB", cub_split)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"CUB split not found at: {data_dir}")
        return datasets.ImageFolder(data_dir, transform=tfm)

    elif name == "imagenet":
        sub = "train" if sp == "train" else ("val" if sp in {"val", "test"} else sp)
        path = os.path.join(root, "imagenet", sub)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"ImageNet split not found at: {path}")
        return datasets.ImageFolder(path, transform=tfm)

    elif name == "places365":
        split_tv = "train-standard" if sp == "train" else ("val" if sp in {"val", "test"} else sp)
        return datasets.Places365(root=root, split=split_tv, small=True, transform=tfm, download=False)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


class RemappedImageFolder(torch.utils.data.Dataset):
    def __init__(self, base_ds, raw_to_contig: Dict[int, int]):
        self.base = base_ds
        self.raw_to_contig = raw_to_contig
        self._keep = []
        for i in range(len(base_ds)):
            _, y = base_ds[i]
            if int(y) in raw_to_contig:
                self._keep.append(i)

    def __len__(self):
        return len(self._keep)

    def __getitem__(self, idx):
        j = self._keep[idx]
        x, y = self.base[j]
        return x, torch.tensor(self.raw_to_contig[int(y)], dtype=torch.long)

def _build_raw_to_contig_from_ds(ds, num_classes: int) -> Dict[int, int]:
    
    classes = getattr(ds, "classes", None)
    class_to_idx = getattr(ds, "class_to_idx", None)

    if classes is not None and class_to_idx is not None:
        classes_sorted = sorted(classes)
        if len(classes_sorted) < num_classes:
            raise RuntimeError(
                f"Dataset exposes only {len(classes_sorted)} classes but cfg.num_classes={num_classes}"
            )
        kept = classes_sorted[:num_classes]
        raw_ids = [class_to_idx[c] for c in kept]
        return {raw: i for i, raw in enumerate(raw_ids)}

    seen = set()
    probe = min(4096, len(ds))
    for i in range(probe):
        _, y = ds[i]
        seen.add(int(y))
    seen_sorted = sorted(seen)
    if len(seen_sorted) < num_classes:
        raise RuntimeError(
            f"Dataset shows {len(seen_sorted)} distinct labels in probe but cfg.num_classes={num_classes}"
        )
    kept = seen_sorted[:num_classes]
    return {raw: i for i, raw in enumerate(kept)}

def _wrap_with_remap_if_needed(ds, num_classes: int, tag: str):
   
    ys = []
    for i in range(min(2048, len(ds))):
        _, y = ds[i]
        ys.append(int(y))
    y_min = min(ys) if ys else 0
    y_max = max(ys) if ys else -1
    ok_range = (y_min >= 0) and (y_max < num_classes)

    if ok_range:
        return ds

    raw_to_contig = _build_raw_to_contig_from_ds(ds, num_classes)
    wrapped = RemappedImageFolder(ds, raw_to_contig)

    ys2 = []
    for i in range(min(2048, len(wrapped))):
        _, y = wrapped[i]
        ys2.append(int(y))
    y2_min = min(ys2) if ys2 else 0
    y2_max = max(ys2) if ys2 else -1

    print(f"[{tag}] remapped labels: before(min={y_min}, max={y_max}) → after(min={y2_min}, max={y2_max}); "
          f"kept {len(wrapped)}/{len(ds)} samples")
    return wrapped


def train_label_free(
    dataset_name: str,
    cfg_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    logger = setup_enhanced_logging("lf_cbm_train")

    cfg = LabelFreeCBMConfig(dataset=dataset_name)
    cfg.data_dir = "/path/to/your/dataset"
    cfg.save_dir = getattr(cfg, "save_dir", "/path/to/your/saved_models")
    cfg.num_workers = getattr(cfg, "num_workers", min(8, (os.cpu_count() or 2)))
    cfg.batch_size = getattr(cfg, "batch_size", 256)

    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    seed_env = os.getenv("LF_SEED")
    if seed_env is not None:
        try:
            cfg.seed = int(seed_env)
        except ValueError:
            cfg.seed = getattr(cfg, "seed", 42)
    else:
        cfg.seed = getattr(cfg, "seed", 42)


    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"[repro] seed={cfg.seed}")

    if dataset_name not in _NUM_CLASSES:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from: {list(_NUM_CLASSES.keys())}")

    if not isinstance(getattr(cfg, "num_classes", None), int) or cfg.num_classes <= 0:
        cfg.num_classes = _NUM_CLASSES[dataset_name]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = Path(cfg.save_dir)
    save_dir = save_root / f"lf_cbm_{dataset_name}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    env_cache = os.getenv("LF_CACHE_DIR", "").strip() or None
    if env_cache:
        cache_root = Path(env_cache)
    elif getattr(cfg, "feature_cache_dir", None):
        cache_root = Path(cfg.feature_cache_dir)
    else:
        cache_root = save_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cfg.feature_cache_dir = str(cache_root)

    use_cache_env = os.getenv("LF_USE_CACHE")
    cfg.use_cache = (use_cache_env.strip().lower() in ("1", "true", "yes", "y", "on")) if use_cache_env is not None else bool(getattr(cfg, "use_cache", True))

    # Top-k & patience knobs
    cfg.topk_k = int(os.getenv("LF_TOPK_K", str(getattr(cfg, "topk_k", 5))))
    cfg.proj_patience = int(os.getenv("LF_PROJ_PATIENCE", str(getattr(cfg, "proj_patience", 100))))

    logger.info(f"[paths] save_dir={save_dir} | cache_dir={cfg.feature_cache_dir} | use_cache={cfg.use_cache} | seed={cfg.seed} | topk_k={cfg.topk_k} | proj_patience={cfg.proj_patience}")
    logger.info(cfg.summary())

    logger.info("⚙️ Regularization and training parameters:")
    logger.info(f"  sparsity_lambda:        {cfg.lam}")
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
    logger.info(f"  topk_k:                 {cfg.topk_k}")
    logger.info(f"  proj_patience:          {cfg.proj_patience}")
    logger.info(f"  feature_cache_dir:      {cache_root}")

    concepts = _ensure_concepts(dataset_name)
    logger.info(f"Loaded {len(concepts)} concepts from file")

    backbone = LabelFreeCBM.build_backbone(device=cfg.device, clip_name=cfg.clip_name)
    model = LabelFreeCBM(
        backbone=backbone,
        num_concepts=max(len(concepts), 10), 
        num_classes=cfg.num_classes,
        device=cfg.device,
        clip_name=cfg.clip_name,
    )

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

        logger.info(f"[DATA] train size: {len(train_ds)}")
        logger.info(f"[DATA] test size:  {len(test_ds)}")

        train_ds = _wrap_with_remap_if_needed(train_ds, num_classes=cfg.num_classes, tag="train")
        test_ds  = _wrap_with_remap_if_needed(test_ds,  num_classes=cfg.num_classes, tag="test")

    # ---- concept layer (cos^3 per paper; handled inside model)
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
        feature_cache_dir=cfg.feature_cache_dir,
        use_cache=bool(cfg.use_cache),
        cache_split="train",
        topk_k=int(cfg.topk_k),
        proj_patience=int(cfg.proj_patience),
    )
    logger.info("Training concept layer (cos^3 projection with filters)…")
    concept_acts = model.train_concept_layer(train_ds, concepts, cbl_cfg)  # [N, C']

    kept = getattr(model, "kept_concepts_", None) or getattr(model, "concept_names", [])
    kept_path = save_dir / "kept_concepts.txt"
    with open(kept_path, "w", encoding="utf-8") as f:
        for name in kept:
            f.write(str(name).strip() + "\n")
    logger.info(f"Saved kept concept names → {kept_path}")

    targets: List[int] = []
    loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=min(4, cfg.num_workers))
    for _, y in loader:
        targets.extend([int(t) for t in y])
    labels = torch.tensor(targets, dtype=torch.long, device=cfg.device)

    concept_acts = concept_acts.detach().float().cpu()
    labels = labels.cpu()

    # ---- (ONLY for CIFAR-10) compute VAL concept activations/labels (cache-aware)
    val_acts, val_labels = None, None
    if dataset_name == "cifar10":
        feats_val = model._extract_dataset_features(
            dataset=val_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            cache_dir=cfg.feature_cache_dir,
            split="val",
            use_cache=getattr(cfg, "use_cache", True),
        )

        v_lbls = []
        for _, yb in val_loader:
            v_lbls.append(yb.detach().cpu())
        val_labels = torch.cat(v_lbls, 0).long()

        def _project_in_chunks(feats, chunk=8192):
            outs = []
            with torch.no_grad():
                for i in range(0, feats.size(0), chunk):
                    f = feats[i:i+chunk].to(model.concept_layer.weight.dtype).to(cfg.device, non_blocking=("cuda" in str(cfg.device)))
                    a = model.concept_layer(f)
                    outs.append(a.detach().cpu())
            return torch.cat(outs, 0)

        val_acts = _project_in_chunks(feats_val)

    # ---- final layer with GLM-SAGA sparse training
    trainer = UnifiedFinalTrainer()
    fl_cfg = get_label_free_cbm_config(
        num_concepts=concept_acts.shape[1],
        num_classes=cfg.num_classes,
        cfg=cfg,
        device='cpu', 
    )
    logger.info("Training final layer (GLM-SAGA sparse)…")

    val_pair = (val_acts, val_labels) if dataset_name == "cifar10" else None
    result = trainer.train(
        concept_activations=concept_acts,
        labels=labels,
        config=fl_cfg,
        validation_data=val_pair,
        progress_callback=None,
    )

    final_layer = trainer.create_final_layer(fl_cfg, result)
    model.final_layer = final_layer
    model.concept_mean = result.get("concept_mean")
    model.concept_std = result.get("concept_std")

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
    logger.info(f" Saved model bundle to: {save_dir/'lf_cbm_minimal_bundle.pt'}")

    # ----------------------------
    # ANEC export
    # ----------------------------
    anec_dir = save_dir / "anec_export"
    anec_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # TRAIN μ/σ
    mu = result["concept_mean"].detach().cpu().float().view(1, -1)
    sigma = result["concept_std"].detach().cpu().float().clamp_min(1e-6).view(1, -1)

    # (A) export TRAIN (normalized)
    train_acts_norm = (concept_acts.float() - mu) / sigma
    torch.save(train_acts_norm, anec_dir / "train_activations.pt")
    torch.save(labels.long().cpu(), anec_dir / "train_labels.pt")

    # (B) export VAL for CIFAR-10
    if dataset_name == "cifar10" and val_acts is not None:
        Xn_val = (val_acts.float() - mu) / sigma
        torch.save(Xn_val, anec_dir / "val_activations.pt")
        torch.save(val_labels.long().cpu(), anec_dir / "val_labels.pt")

    # (C) export TEST/VAL for other datasets
    maybe_splits = []
    if dataset_name in ("cifar10", "cifar100", "cub"):
        maybe_splits.append("test")
    if dataset_name in ("imagenet", "places365"):
        maybe_splits.append("val")

    for split in maybe_splits:
        ds = _make_dataset(dataset_name, root=cfg.data_dir, split=split, clip_preprocess=model.clip_preprocess)
        ds = _wrap_with_remap_if_needed(ds, num_classes=cfg.num_classes, tag=f"{split}")
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=min(4, cfg.num_workers))

        feats_split = model._extract_dataset_features(
            dataset=ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            cache_dir=cfg.feature_cache_dir,
            split=split,
            use_cache=getattr(cfg, "use_cache", True),
        )

        lbl_list = []
        for _, yb in loader:
            lbl_list.append(yb.detach().cpu())
        y_split = torch.cat(lbl_list, 0).long()

        def _project_in_chunks(feats, chunk=8192):
            outs = []
            with torch.no_grad():
                for i in range(0, feats.size(0), chunk):
                    f = feats[i:i+chunk].to(model.concept_layer.weight.dtype).to(cfg.device, non_blocking=("cuda" in str(cfg.device)))
                    a = model.concept_layer(f)
                    outs.append(a.detach().cpu())
            return torch.cat(outs, 0)

        acts_split = _project_in_chunks(feats_split)

        Xn = (acts_split.float() - mu) / sigma
        torch.save(Xn, anec_dir / f"{split}_activations.pt")
        torch.save(y_split, anec_dir / f"{split}_labels.pt")

    logger.info(f" ANEC export ready at: {anec_dir}")

    return {
        "save_dir": str(save_dir),
        "num_concepts_final": int(concept_acts.shape[1]),
        "num_classes": cfg.num_classes,
        "dataset": dataset_name,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m cbm_library.scripts.lf_cbm_train <dataset>")
        print("Supported datasets: cifar10 | cifar100 | cub | imagenet | places365")
        sys.exit(1)
    dataset = sys.argv[1].lower().strip()
    _ = train_label_free(dataset)

if __name__ == "__main__":
    main()
