# cbm_library/scripts/vlg_cbm_train.py
"""
Minimal runner for VLG-CBM.

Usage:
    python -m cbm_library.scripts.vlg_cbm_train cifar10

Notes:
- CIFAR10/100 will auto-download under cfg.data_dir if missing.
- Replace the simple dataset block with your repo's loaders when ready.
"""
from __future__ import annotations
import sys
import os
from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cbm_library.config.vlg_cbm_config import VLGCBMConfig
from cbm_library.models.vlg_cbm import VLGCBM

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

def _make_loaders(dataset: str, data_root: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # CLIP-ish preprocessing; swap with your exact transforms if needed
    tfm_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    if dataset == "cifar10":
        tr = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
        te = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_eval)
        # simple val split
        val_size = 5000
        tr_set, va_set = torch.utils.data.random_split(tr, [len(tr) - val_size, val_size],
                                                       generator=torch.Generator().manual_seed(42))
    elif dataset == "cifar100":
        tr = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tfm_train)
        te = datasets.CIFAR100(root=data_root, train=False, download=True, transform=tfm_eval)
        val_size = 5000
        tr_set, va_set = torch.utils.data.random_split(tr, [len(tr) - val_size, val_size],
                                                       generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dl_tr = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(te,      batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_tr, dl_va, dl_te

def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    if dataset not in _NUM_CLASSES:
        raise SystemExit(f"Dataset '{dataset}' not supported. Choose from {list(_NUM_CLASSES)}")

    # ---- config (all settings live here) ----
    cfg = VLGCBMConfig.from_overrides(
        dataset=dataset,
        num_classes=_NUM_CLASSES[dataset],
        data_dir="/kayla/dataset",
        annotations_dir="cbm_library/data/annotations",
        save_dir="/kayla/saved_models",
        backbone="resnet50",          # or "clip_visual"
        feature_layer="layer4",
        feature_pool="avg",
        clip_model="ViT-B/32",
        concept_list_path=f"cbm_library/concepts/main/outputs/label_free/{dataset}_filtered.txt",
        clip_topk=5,
        clip_cutoff=0.15,
        proj_epochs=20,
        final_type="saga",            # swap to "dense" if you haven’t wired SAGA yet
        batch_size=256,
        num_workers=min(8, os.cpu_count() or 2),
    )

    # ---- data ----
    dl_tr, dl_va, dl_te = _make_loaders(cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers)

    # ---- model ----
    model = VLGCBM(cfg).to(cfg.device)

    # ---- train / eval ----
    results = model.run_full_pipeline(dl_tr, dl_va, dl_te)
    print("Results:", results)

if __name__ == "__main__":
    main()
