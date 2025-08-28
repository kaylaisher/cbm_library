from __future__ import annotations

import os
import json
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


BACKBONE_ENCODING_DIMENSION = {"resnet18": 512}
BACKBONE_VISUALIZATION_TARGET_LAYER = {"resnet18": "layer4"}

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DATASET_FOLDER    = os.environ.get("DATASET_FOLDER", "/kayla/dataset")
ANNOTATION_FOLDER = os.environ.get("ANNOTATION_FOLDER", "/kayla/Annotations")


IMAGENET_ROOT = f"{DATASET_FOLDER}/imagenet"
CUB_ROOT      = f"{DATASET_FOLDER}/CUB"
PLACES_ROOT   = f"{DATASET_FOLDER}/places365" 

def _safe_classes_from_dataset(ctor, root: str) -> List[str]:
    if hasattr(ctor, "classes"):
        return list(getattr(ctor, "classes"))

    try:
        ds = ctor(root=root, download=False)
        return list(getattr(ds, "classes"))
    except TypeError:
       
        try:
            ds = ctor(root=root, train=True, download=False)
            return list(getattr(ds, "classes"))
        except Exception as e:
            raise RuntimeError(f"Could not obtain classes from {ctor.__name__}: {e}")

def _norm_split(split: str) -> str:
    sp = (split or "").lower()
    if sp in {"train", "tr", "training"}:
        return "train"
    if sp in {"val", "valid", "validation"}:
        return "val"
    if sp in {"test", "eval"}:
        return "test"
    return "train"

def _class_names_from_dir(p: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])
    except FileNotFoundError:
        return []

def get_target_model(backbone_name: str, device: str = "cuda"):
    
    if backbone_name != "resnet18":
        raise NotImplementedError("Minimal utils supports only resnet18 backbone")

    model = models.resnet18(weights=None).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return model, preprocess

def make_dataset(
    dataset: str,
    split: str,
    preprocess: Optional[transforms.Compose] = None,
    *,
    cifar_root: str = "/directory/to/cifar",
    cub_root: str = "/directory/to/CUB",
    imagenet_root: str = "/directory/to/imagenet",
    places_root: str = "/directory/to/places365",
    places_small: bool = True
    ) -> Dataset:
    
    ds = (dataset or "").lower()
    sp = _norm_split(split)

    if ds in {"cifar10", "cifar-10"}:
        return datasets.CIFAR10(root=cifar_root, train=(sp == "train"),
                                transform=preprocess, download=True)

    if ds in {"cifar100", "cifar-100"}:
        return datasets.CIFAR100(root=cifar_root, train=(sp == "train"),
                                 transform=preprocess, download=True)

    if ds in {"places365", "places"}:
        split_name = "train-standard" if sp == "train" else "val"
        return datasets.Places365(root=places_root,
                                  split=split_name,
                                  small=places_small,
                                  transform=preprocess,
                                  download=False)

    if ds in {"cub", "cub200", "cub-200", "cub_200_2011"}:
        sub = "train" if sp == "train" else "test"
        cub_split_root = os.path.join(cub_root, sub)
        return datasets.ImageFolder(root=cub_split_root, transform=preprocess)

    if ds in {"imagenet", "imagenet1k", "imagenet-1k"}:
        sub = "train" if sp == "train" else "val"
        imagenet_split_root = os.path.join(imagenet_root, sub)
        return datasets.ImageFolder(root=imagenet_split_root, transform=preprocess)

    raise NotImplementedError(f"{dataset} not supported")

def get_classes(
    dataset: str,
    *,
    cifar_root: str = "/directory/to/cifar",
    cub_root: str = "/directory/to/CUB",
    imagenet_root: str = "/directory/to/imagenet",
    places_root: str = "/directory/to/places365",
    places_small: bool = False
) -> List[str]:
    ds = (dataset or "").lower()

    # CIFAR10/100
    if ds in {"cifar10", "cifar-10"}:
        return list(datasets.CIFAR10.classes)
    if ds in {"cifar100", "cifar-100"}:
        return list(datasets.CIFAR100.classes)

    
    if ds in {"places365", "places"}:
        try:
            tmp = datasets.Places365(root=PLACES_ROOT, split="val",
                                     small=places_small, download=False)
            return list(tmp.classes)
        except Exception:
            return [str(i) for i in range(365)]

    if ds in {"cub", "cub200", "cub-200", "cub_200_2011"}:
        names = _class_names_from_dir(os.path.join(CUB_ROOT, "train")) \
             or _class_names_from_dir(os.path.join(CUB_ROOT, "test"))
        if not names:
            raise FileNotFoundError(f"CUB class folders not found under {CUB_ROOT}/{{train|test}}/")
        return names

    if ds in {"imagenet", "imagenet1k", "imagenet-1k"}:
        names = _class_names_from_dir(os.path.join(IMAGENET_ROOT, "train")) \
             or _class_names_from_dir(os.path.join(IMAGENET_ROOT, "val"))
        if not names:
            raise FileNotFoundError(f"ImageNet class folders not found under {IMAGENET_ROOT}/{{train|val}}/")
        return names

    raise NotImplementedError(f"{dataset} not supported")


def get_concepts(concept_file: Optional[str] = None, filter_file: Optional[str] = None) -> List[str]:
    if concept_file and os.path.exists(concept_file):
        with open(concept_file) as f:
            return [l.strip() for l in f if l.strip()]
    return ["has_fur", "has_wings", "is_vehicle"]

def load_concept_and_count(base_dir: str, filter_file: Optional[str] = None):
    path = os.path.join(base_dir, "concepts.txt")
    concepts = get_concepts(path, filter_file)
    counts = {c: 1 for c in concepts}
    return concepts, counts

def save_concept_count(concepts, counts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "concept_counts.json"), "w") as f:
        json.dump(counts, f, indent=2)

def save_filtered_concepts(filtered_concepts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "filtered_concepts.txt"), "w") as f:
        f.write("\n".join(filtered_concepts))

def format_concept(s: str) -> str:
    return s.strip().lower().replace(" ", "_")
