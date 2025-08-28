from __future__ import annotations
import os
import sys
import random
import datetime
from typing import Optional
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from cbm_library.config.vlg_cbm_config import VLGCBMConfig
from cbm_library.models.vlg_cbm import (
    Backbone,
    BackboneCLIP,
    ConceptLayer,
    NormalizationLayer,
    VLGCBM,
    per_class_accuracy,
    train_cbl,
    test_model,
    validate_cbl,
    train_sparse_final,
    train_dense_final,
)
from cbm_library.models.final_layer import run_final_layer_pipeline
from cbm_library.utils import data_utils
from cbm_library.utils.concept_dataset import (
    get_concept_dataloader,
    get_filtered_concepts_and_counts,
    get_final_layer_dataset,
)
from cbm_library.utils.loss import get_loss
from cbm_library.utils.model_utils import write_parameters_tensorboard, display_top_activated_images



def _dump_feature_loader(loader, out_x_path: str, out_y_path: str):
   
   
    xs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            # Support (x, y) or (x, y, *extras)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise RuntimeError("Unexpected batch format from concept feature loader.")
            xs.append(x.detach().cpu())
            ys.append(y.detach().cpu())
    X = torch.cat(xs, dim=0) if xs else torch.empty(0)
    Y = torch.cat(ys, dim=0) if ys else torch.empty(0)
    torch.save(X, out_x_path)
    torch.save(Y, out_y_path)
    print(f"[DUMP] Saved {out_x_path} shape={tuple(X.shape)}; {out_y_path} shape={tuple(Y.shape)}")


def _dump_from_images(backbone, cbl, normalization_layer, image_loader, device: str,
                      out_x_path: str, out_y_path: str):
    
    
    xs, ys = [], []
    backbone.eval()
    cbl.eval()
    with torch.no_grad():
        for batch in image_loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Unexpected batch format from image loader.")
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)

            feats = cbl(backbone(images))         
            if normalization_layer is not None:    
                feats = normalization_layer(feats)

            xs.append(feats.detach().cpu())
            ys.append(labels.detach().cpu())

    X = torch.cat(xs, dim=0) if xs else torch.empty(0)
    Y = torch.cat(ys, dim=0) if ys else torch.empty(0)
    torch.save(X, out_x_path)
    torch.save(Y, out_y_path)
    print(f"[DUMP] Saved {out_x_path} shape={tuple(X.shape)}; {out_y_path} shape={tuple(Y.shape)}")


# -----------------------
# Progress print helpers
# -----------------------

def dense_progress(epoch: int, metrics: dict):
    loss = float(metrics.get("loss", 0.0))
    vacc = float(metrics.get("val_accuracy", 0.0))
    print(f"[FINAL] epoch={epoch:04d} loss={loss:.6f} val_accuracy={vacc:.4f}")

def glm_progress(iter_or_epoch: int, metrics: dict):
    lam = metrics.get("lambda")
    loss = metrics.get("loss")
    if lam is not None and loss is not None:
        print(f"[FINAL] GLM step={iter_or_epoch:05d} lam={float(lam):.6g} loss={float(loss):.6f}")


def train_vlg_cbm(cfg: VLGCBMConfig) -> str:
    run_dir = os.path.join(
        cfg.save_dir,
        f"vlg_cbm_{cfg.dataset}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
    )
    while os.path.exists(run_dir):
        run_dir += "-1"
    os.makedirs(run_dir, exist_ok=True)

    print(f"[SETUP] Saving run to {run_dir}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    classes = data_utils.get_classes(cfg.dataset)

    if cfg.backbone.startswith("clip_"):
        backbone = BackboneCLIP(cfg.backbone, use_penultimate=cfg.use_clip_penultimate, device=cfg.device)
    else:
        backbone = Backbone(cfg.backbone, cfg.feature_layer, cfg.device)

    print(f"[CONCEPTS] annotation_dir={os.path.abspath(cfg.annotation_dir) if cfg.annotation_dir else 'None'}")
    if cfg.load_dir is None:
        if cfg.skip_concept_filter:
            print("[CONCEPTS] Skipping concept filtering (using prefiltered set)")
            concepts, concept_counts = data_utils.load_concept_and_count(
                os.path.dirname(cfg.concept_set) if cfg.concept_set else run_dir,
                filter_file=cfg.filter_set,
            )
        else:
            print("[CONCEPTS] Filtering concepts using annotations")
            raw_concepts = data_utils.get_concepts(cfg.concept_set, cfg.filter_set)
            concepts, concept_counts, filtered_concepts = get_filtered_concepts_and_counts(
                cfg.dataset,
                raw_concepts,
                preprocess=backbone.preprocess,
                val_split=cfg.val_split,
                batch_size=cfg.cbl_batch_size,
                num_workers=cfg.num_workers,
                confidence_threshold=cfg.cbl_confidence_threshold,
                label_dir=cfg.annotation_dir,
                use_allones=cfg.allones_concept,
                seed=cfg.seed,
            )
            data_utils.save_concept_count(concepts, concept_counts, run_dir)
            data_utils.save_filtered_concepts(filtered_concepts, run_dir)
    else:
        print(f"[CONCEPTS] Loading concepts from {cfg.load_dir}")
        concepts, concept_counts = data_utils.load_concept_and_count(cfg.load_dir, filter_file=cfg.filter_set)

    with open(os.path.join(run_dir, "concepts.txt"), "w") as f:
        if len(concepts) > 0:
            f.write(concepts[0])
            for c in concepts[1:]:
                f.write("\n" + c)

    tb_writer = SummaryWriter(run_dir)

    augmented_train_cbl_loader = get_concept_dataloader(
        cfg.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=cfg.val_split,
        batch_size=cfg.cbl_batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        confidence_threshold=cfg.cbl_confidence_threshold,
        crop_to_concept_prob=cfg.crop_to_concept_prob,
        label_dir=cfg.annotation_dir,
        use_allones=cfg.allones_concept,
        seed=cfg.seed,
    )
    train_cbl_loader = get_concept_dataloader(
        cfg.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=cfg.val_split,
        batch_size=cfg.cbl_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        confidence_threshold=cfg.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=cfg.annotation_dir,
        use_allones=cfg.allones_concept,
        seed=cfg.seed,
    )
    val_cbl_loader = get_concept_dataloader(
        cfg.dataset,
        "val",
        concepts,
        preprocess=backbone.preprocess,
        val_split=cfg.val_split,
        batch_size=cfg.cbl_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        confidence_threshold=cfg.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=cfg.annotation_dir,
        use_allones=cfg.allones_concept,
        seed=cfg.seed,
    )
    test_cbl_loader = get_concept_dataloader(
        cfg.dataset,
        "test",
        concepts,
        preprocess=backbone.preprocess,
        val_split=None,
        batch_size=cfg.cbl_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        confidence_threshold=cfg.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=cfg.annotation_dir,
        use_allones=cfg.allones_concept,
        seed=cfg.seed,
    )

    loss_fn = get_loss(
        "bce",
        len(concepts),
        len(train_cbl_loader.dataset),
        concept_counts,
        pos_weight=None,
        auto_weight=None,
        twoway_tp=False,
        device=cfg.device,
    )

    if cfg.load_dir is None:
        print("[CBL] Training Concept Bottleneck Layer (CBL)")
        cbl = ConceptLayer(backbone.output_dim, len(concepts), num_hidden=cfg.cbl_hidden_layers, device=cfg.device)
        cbl, backbone = train_cbl(
            backbone,
            cbl,
            augmented_train_cbl_loader,
            val_cbl_loader,
            cfg.cbl_epochs,
            loss_fn=loss_fn,
            lr=cfg.cbl_lr,
            weight_decay=cfg.cbl_weight_decay,
            concepts=concepts,
            tb_writer=tb_writer,
            device=cfg.device,
            finetune=cfg.cbl_finetune,
            optimizer=cfg.cbl_optimizer,
            scheduler=cfg.cbl_scheduler,
            backbone_lr=cfg.cbl_lr * cfg.cbl_bb_lr_rate,
            data_parallel=cfg.data_parallel,
        )
    else:
        print(f"[CBL] Loading CBL/backbone from {cfg.load_dir}")
        cbl = ConceptLayer.from_pretrained(cfg.load_dir, cfg=cfg, device=cfg.device)
        if cfg.backbone.startswith("clip_"):
            backbone = BackboneCLIP.from_pretrained(
                cfg.load_dir,
                backbone_name=cfg.backbone,
                use_penultimate=getattr(cfg, "use_clip_penultimate", True),
                device=cfg.device,
            )
        else:
            backbone = Backbone.from_pretrained(
                cfg.load_dir,
                backbone_name=cfg.backbone,
                feature_layer=cfg.feature_layer,
                device=cfg.device,
            )

    cbl.save_model(run_dir)
    if cfg.cbl_finetune:
        backbone.save_model(run_dir)

    train_concept_loader, val_concept_loader, normalization_layer = get_final_layer_dataset(
        backbone,
        cbl,
        train_cbl_loader,
        val_cbl_loader,
        run_dir,
        load_dir=cfg.load_dir,
        batch_size=cfg.saga_batch_size,
        device=cfg.device,
    )

    ys = []
    for _, _, yb in train_concept_loader:
        ys.append(yb)
    ys = torch.cat(ys).to(torch.long)
    uniq = torch.unique(ys)
    mapping = None
    if len(uniq) == len(classes) and (uniq.min().item() != 0 or uniq.max().item() != len(classes) - 1):
        mapping = {int(v): i for i, v in enumerate(torch.sort(uniq).values.tolist())}
        def remap(loader):
            for xb, cb, yb in loader:
                yb = torch.tensor([mapping[int(v)] for v in yb.tolist()],
                                  dtype=torch.long, device=yb.device)
                yield xb, cb, yb
        train_concept_loader = remap(train_concept_loader)
        val_concept_loader   = remap(val_concept_loader)

    # ---- Dump train/val/test activations + labels ----
    _dump_feature_loader(
        train_concept_loader,
        os.path.join(run_dir, "train_activations.pt"),
        os.path.join(run_dir, "train_labels.pt"),
    )
    _dump_feature_loader(
        val_concept_loader,
        os.path.join(run_dir, "val_activations.pt"),
        os.path.join(run_dir, "val_labels.pt"),
    )
    _dump_from_images(
        backbone,
        cbl,
        normalization_layer,
        test_cbl_loader,
        cfg.device,
        os.path.join(run_dir, "test_activations.pt"),
        os.path.join(run_dir, "test_labels.pt"),
    )

    # ---- Unified final-layer training (delegated) ----
    final_layer, result = run_final_layer_pipeline(
        train_loader=train_concept_loader,
        val_loader=val_concept_loader,
        num_concepts=len(concepts),
        num_classes=len(classes),
        cfg=cfg,
        save_dir=run_dir, 
        progress_callback=dense_progress if cfg.dense else glm_progress,
    )

    model = VLGCBM(backbone, cbl, normalization_layer, final_layer, device=cfg.device)
    model.save(run_dir)

    test_accuracy = model.evaluate_top1(test_cbl_loader)
    print(f"[TEST] Top-1 accuracy: {test_accuracy:.4f}")

    per_class = per_class_accuracy(model, test_cbl_loader, classes, device=cfg.device)
    glm_meta = result.get("glm_metadata", None) or {}
    W_g = result["weight"]
    nnz = (W_g.abs() > 1e-5).sum().item()
    total = W_g.numel()
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write("=== Metrics Summary ===\n")
        f.write(f"Test Top-1 Accuracy: {test_accuracy:.6f}\n")
        f.write(f"Lambda: {float(glm_meta.get('lambda', -1.0))}\n")
        f.write(f"Alpha: {float(glm_meta.get('alpha', -1.0))}\n")
        f.write(f"Time: {float(glm_meta.get('time', 0.0))}\n")
        f.write(f"Non-zero weights: {nnz}\n")
        f.write(f"Total weights: {total}\n")
        f.write(f"Percent non-zero: {nnz/total if total else 0.0:.6f}\n")
        f.write("\n=== Per-class Accuracy (%) ===\n")
        for k, v in per_class.get("Per class accuracy", {}).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nOverall Accuracy (%): {per_class.get('Overall accuracy','N/A')}\n")
        f.write(f"Datapoints: {per_class.get('Datapoints','N/A')}\n")

    write_parameters_tensorboard(tb_writer, cfg.to_dict(), test_accuracy * 100.0, (nnz / total) * 100.0)

    # ---- Visualization (optional) ----
    if cfg.visualize_concepts:
        target_layer = data_utils.BACKBONE_VISUALIZATION_TARGET_LAYER[cfg.backbone]
        vis_dir = os.path.join(run_dir, "concept_visualization")
        os.makedirs(vis_dir, exist_ok=True)

        cbl_with_backbone = nn.Sequential(backbone, cbl).to(cfg.device)
        import numpy as _np
        concepts_logits = []
        from tqdm import tqdm as _tqdm
        for (images_tensor, _, _) in _tqdm(test_cbl_loader):
            images_tensor = images_tensor.to(cfg.device)
            with torch.no_grad():
                concepts_logits.append(cbl_with_backbone(images_tensor).detach().cpu().numpy())
        concepts_logits = _np.concatenate(concepts_logits, axis=0)

        for concept_idx, concept in enumerate(concepts):
            fig = display_top_activated_images(
                concept_idx,
                concepts_logits,
                cbl_with_backbone,
                target_layer,
                test_cbl_loader.dataset,
                transform=backbone.preprocess,
                device=cfg.device,
                k=10,
            )
            fig.savefig(os.path.join(vis_dir, f"{concept}.png"))

    return run_dir


def parse_args() -> VLGCBMConfig:
    parser = argparse.ArgumentParser(description="VLG-CBM training (JSON-free)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name (e.g., cifar10, cifar100, cub, imagenet, places365)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="Directory where outputs/checkpoints are saved",
    )
    args = parser.parse_args()
    return VLGCBMConfig(
        dataset=args.dataset,
        save_dir=args.save_dir,
    )

def main():
    cfg = parse_args()
    print(" vlg_cbm loaded")
    print(f" dataset={getattr(cfg,'dataset','?')} backbone={cfg.backbone} device={cfg.device}")
    print(f" feature_layer={cfg.feature_layer} use_clip_penultimate={cfg.use_clip_penultimate}")
    print(f" save_dir={cfg.save_dir}")
    print(f" annotation_dir={os.path.abspath(cfg.annotation_dir) if cfg.annotation_dir else 'None'}")
    if cfg.annotation_dir and not os.path.isdir(cfg.annotation_dir):
        print(f"[WARN] annotation_dir does not exist: {cfg.annotation_dir}")
    print(f" cbl: layers={cfg.cbl_hidden_layers} bs={cfg.cbl_batch_size} epochs={cfg.cbl_epochs} lr={cfg.cbl_lr} wd={cfg.cbl_weight_decay}")
    print(f" final: dense={cfg.dense} saga_bs={cfg.saga_batch_size} step={cfg.saga_step_size} lam={cfg.saga_lam} iters={cfg.saga_n_iters}")
    run_dir = train_vlg_cbm(cfg)
    print(f" Run saved to: {run_dir}")

if __name__ == "__main__":
    main()
