"""
Unified Final Layer Training Module with Original LF-CBM Integration
"""

import os
import json
from cbm_library.config.final_layer_config import FinalLayerConfig, FinalLayerType
from enum import Enum
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logging import setup_enhanced_logging
from cbm_library.config import LabelFreeCBMConfig

logger = setup_enhanced_logging(__name__)


# ---- single facade that holds all methods ----
class FinalLayerMethod:
    def __init__(self):
        # patch softmax dim default for glm_saga
        import torch.nn.functional as F
        _orig = F.softmax

        def _softmax_with_dim(x, dim=None, *args, **kwargs):
            if dim is None:
                dim = -1
            return _orig(x, dim=dim, *args, **kwargs)

        F.softmax = _softmax_with_dim

        # optional GLM-SAGA
        try:
            from glm_saga.elasticnet import glm_saga, IndexedTensorDataset
            self._glm_saga = glm_saga
            self._IndexedTensorDataset = IndexedTensorDataset
            self._glm_ready = True
            logger.info("✅ GLM-SAGA available")
        except ImportError:
            self._glm_saga = None
            self._IndexedTensorDataset = None
            self._glm_ready = False
            logger.warning("❌ GLM-SAGA not available. Install: pip install glm-saga")

    # ---------- public API ----------
    def train(
        self,
        concept_activations: torch.Tensor,
        labels: torch.Tensor,
        config: FinalLayerConfig,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        progress_callback: Optional[callable] = None,
        ) -> Dict[str, Any]:
        """Dispatch to the chosen method."""
        if config.layer_type == FinalLayerType.SPARSE_GLM:
            return self._train_sparse_glm(concept_activations, labels, config, validation_data, progress_callback)
        elif config.layer_type in (FinalLayerType.DENSE_LINEAR, FinalLayerType.ELASTIC_NET):
            return self._train_dense(concept_activations, labels, config, validation_data, progress_callback)
        elif config.layer_type == FinalLayerType.SPARSE_LINEAR:
            dense = self._train_dense(concept_activations, labels, config, validation_data, progress_callback)
            return self._apply_topk_sparsity(dense, config)
        else:
            raise ValueError(f"Unsupported layer type: {config.layer_type}")

    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        return nn.Linear(config.num_concepts, config.num_classes)

    # ---------- private impls ----------
    def _normalize(self, X: torch.Tensor, use_norm: bool):
        if use_norm:
            m = X.mean(dim=0, keepdim=True)
            s = X.std(dim=0, keepdim=True).clamp_min(1e-6)
            return (X - m) / s, m.squeeze(0), s.squeeze(0)
        return X, torch.zeros(X.size(1)), torch.ones(X.size(1))

    def _train_dense(
        self,
        concept_acts: torch.Tensor,
        labels: torch.Tensor,
        cfg: FinalLayerConfig,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]],
        progress_callback: Optional[callable],
        ):
        logger.info("🚀 Training dense/elastic-net linear head")
        X, mu, sigma = self._normalize(concept_acts, cfg.normalize_concepts)

        model = nn.Linear(cfg.num_concepts, cfg.num_classes).to(cfg.device)
        opt = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        ce = nn.CrossEntropyLoss()

        loader = DataLoader(TensorDataset(X, labels.long()), batch_size=cfg.batch_size, shuffle=True)

        train_losses, val_accs = [], []
        for e in range(cfg.max_epochs):
            model.train()
            total = 0.0
            for xb, yb in loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                opt.zero_grad()
                loss = ce(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item()
            avg = total / max(len(loader), 1)
            train_losses.append(avg)

            vacc = 0.0
            if validation_data is not None:
                vX, vy = validation_data
                if cfg.normalize_concepts:
                    vX = (vX - mu) / sigma
                model.eval()
                with torch.no_grad():
                    pred = model(vX.to(cfg.device)).argmax(1)
                    vacc = (pred == vy.to(cfg.device)).float().mean().item()
            val_accs.append(vacc)

            if progress_callback and e % 10 == 0:
                progress_callback(e, {"loss": avg, "val_accuracy": vacc})
            if e % 20 == 0:
                logger.info(f"[dense] epoch {e} loss={avg:.4f} val_acc={vacc:.4f}")

        return {
            "weight": model.weight.detach().cpu(),
            "bias": model.bias.detach().cpu(),
            "concept_mean": mu,
            "concept_std": sigma,
            "training_metrics": {
                "train_losses": train_losses,
                "val_accuracies": val_accs,
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
            },
            "sparsity_stats": {
                "non_zero_weights": model.weight.numel(),
                "total_weights": model.weight.numel(),
                "sparsity_percentage": 1.0,
                "sparsity_per_class": model.weight.shape[0],
            },
        }

    def _train_sparse_glm(
        self,
        concept_acts: torch.Tensor,
        labels: torch.Tensor,
        cfg: FinalLayerConfig,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]],
        progress_callback: Optional[callable],
        ):
        if not self._glm_ready:
            raise ImportError("GLM-SAGA not available. `pip install glm-saga`")

        logger.info("🚀 Training GLM-SAGA sparse head")
        X, mu, sigma = self._normalize(concept_acts, cfg.normalize_concepts)

        idx_ds = self._IndexedTensorDataset(X, labels.long())
        train_loader = DataLoader(idx_ds, batch_size=cfg.saga_batch_size, shuffle=True)

        val_loader = None
        if validation_data is not None:
            vX, vy = validation_data
            if cfg.normalize_concepts:
                vX = (vX - mu) / sigma
            val_ds = self._IndexedTensorDataset(vX, vy.long())
            val_loader = DataLoader(val_ds, batch_size=cfg.saga_batch_size, shuffle=False)

        linear = nn.Linear(cfg.num_concepts, cfg.num_classes).to(cfg.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()

        meta = {"max_reg": {"nongrouped": cfg.sparsity_lambda}}
        out = self._glm_saga(
            linear,
            train_loader,
            cfg.glm_step_size,
            cfg.glm_max_iters,
            cfg.glm_alpha,
            epsilon=cfg.glm_epsilon,
            k=1,
            val_loader=val_loader,
            do_zero=False,
            metadata=meta,
            n_ex=len(concept_acts),
            n_classes=cfg.num_classes,
        )

        path = out.get("path", [])
        if not path:
            raise RuntimeError("GLM-SAGA returned empty path")

        best = min(path, key=lambda r: r.get("loss", float("inf")))
        W_g, b_g = best["weight"], best["bias"]

        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()

        if progress_callback:
            progress_callback(cfg.glm_max_iters, {"loss": float(best.get("loss", 0.0)), "lambda": float(best.get("lam", cfg.sparsity_lambda))})

        return {
            "weight": W_g,
            "bias": b_g,
            "concept_mean": mu,
            "concept_std": sigma,
            "training_metrics": best.get("metrics", {}),
            "sparsity_stats": {
                "non_zero_weights": nnz,
                "total_weights": total,
                "sparsity_percentage": nnz / total,
                "sparsity_per_class": nnz / cfg.num_classes,
            },
            "glm_metadata": {
                "lambda": float(best.get("lam", cfg.sparsity_lambda)),
                "alpha": float(best.get("alpha", cfg.glm_alpha)),
                "final_loss": float(best.get("loss", 0.0)),
                "time": float(best.get("time", 0.0)),
            },
        }

    def _apply_topk_sparsity(self, dense_result: Dict[str, Any], cfg: FinalLayerConfig):
        W = dense_result["weight"]
        if cfg.sparsity_percentage is not None:
            k = int(W.numel() * cfg.sparsity_percentage)
        else:
            k = cfg.target_sparsity_per_class * cfg.num_classes

        flat = W.flatten()
        _, idx = torch.topk(flat.abs(), k)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        mask = mask.view_as(W)

        W_sparse = W * mask.float()
        nnz = mask.sum().item()

        dense_result["weight"] = W_sparse
        dense_result["sparsity_stats"] = {
            "non_zero_weights": nnz,
            "total_weights": W.numel(),
            "sparsity_percentage": nnz / W.numel(),
            "sparsity_per_class": nnz / cfg.num_classes,
        }
        return dense_result


# ---- thin trainer wrapper (kept for compatibility) ----
class UnifiedFinalTrainer:
    def __init__(self):
        self._method = FinalLayerMethod()

    def train(
        self,
        concept_activations: torch.Tensor,
        labels: torch.Tensor,
        config: FinalLayerConfig,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        if config.num_concepts == 0:
            config.num_concepts = concept_activations.shape[1]
        if config.num_classes == 0:
            config.num_classes = int(labels.max().item()) + 1

        logger.info(f"🎯 Training {config.layer_type.value} final layer")
        result = self._method.train(concept_activations, labels, config, validation_data, progress_callback)
        result["config"] = config
        result["method"] = config.layer_type.value
        return result

    def create_final_layer(self, config: FinalLayerConfig, training_result: Dict[str, Any]) -> nn.Module:
        layer = self._method.create_layer(config)
        layer.weight.data = training_result["weight"].to(config.device)
        layer.bias.data = training_result["bias"].to(config.device)
        return layer

    def save_training_result(self, result: Dict[str, Any], save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(result["weight"], os.path.join(save_path, "W_g.pt"))
        torch.save(result["bias"], os.path.join(save_path, "b_g.pt"))
        torch.save(result["concept_mean"], os.path.join(save_path, "proj_mean.pt"))
        torch.save(result["concept_std"], os.path.join(save_path, "proj_std.pt"))

        meta = {
            "config": result["config"].to_dict(),
            "method": result["method"],
            "training_metrics": result.get("training_metrics", {}),
            "sparsity_stats": result.get("sparsity_stats", {}),
        }
        if "glm_metadata" in result:
            meta["glm_metadata"] = result["glm_metadata"]

        if result["method"] == "sparse_glm" and "glm_metadata" in result:
            glm = result["glm_metadata"]
            legacy = {
                "lam": glm.get("lambda"),
                "alpha": glm.get("alpha"),
                "time": glm.get("time", 0.0),
                "metrics": result.get("training_metrics", {}),
                "sparsity": {
                    "Non-zero weights": meta["sparsity_stats"].get("non_zero_weights", 0),
                    "Total weights": meta["sparsity_stats"].get("total_weights", 0),
                    "Percentage non-zero": meta["sparsity_stats"].get("sparsity_percentage", 0.0),
                },
            }
            with open(os.path.join(save_path, "metrics.txt"), "w") as f:
                json.dump(legacy, f, indent=2)

        with open(os.path.join(save_path, "final_layer_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"💾 Training results saved to {save_path}")

    def load_training_result(self, load_path: str, device: str = "cuda") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["weight"] = torch.load(os.path.join(load_path, "W_g.pt"), map_location=device)
        out["bias"] = torch.load(os.path.join(load_path, "b_g.pt"), map_location=device)
        out["concept_mean"] = torch.load(os.path.join(load_path, "proj_mean.pt"), map_location=device)
        out["concept_std"] = torch.load(os.path.join(load_path, "proj_std.pt"), map_location=device)

        meta_path = os.path.join(load_path, "final_layer_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            out["config"] = FinalLayerConfig.from_dict(meta["config"])
            out.update(meta)
        else:
            logger.warning("Enhanced metadata not found; using tensors only")

        logger.info(f"📂 Loaded final layer from {load_path}")
        return out


# ---- mapper: global config → trainer config (single source of truth) ----
def get_label_free_cbm_config(
    num_concepts: int,
    num_classes: int,
    cfg: LabelFreeCBMConfig,
    **overrides,
) -> FinalLayerConfig:
    mapped = {
        "layer_type": FinalLayerType.SPARSE_GLM,
        "num_concepts": num_concepts,
        "num_classes": num_classes,
        "sparsity_lambda": cfg.lam,
        "glm_step_size": cfg.saga_step_size,
        "glm_alpha": cfg.saga_alpha,
        "glm_epsilon": cfg.saga_epsilon,
        "glm_max_iters": cfg.n_iters,
        "saga_batch_size": cfg.saga_batch_size,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "weight_decay": cfg.weight_decay,
        "normalize_concepts": True,
        "device": cfg.device,
    }
    mapped.update(overrides)
    return FinalLayerConfig(**mapped)
