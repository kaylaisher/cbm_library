"""
This module implements a final layer training pipeline for machine learning models, 
supporting dense and sparse linear layers. 
It provides utilities for training, saving, and loading final layers, as well as compatibility with LF-CBM and VLG-CBM configurations.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, Tuple, Iterable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cbm_library.config.final_layer_config import FinalLayerConfig, FinalLayerType
from cbm_library.config import LabelFreeCBMConfig  
from cbm_library.config.vlg_cbm_config import VLGCBMConfig
from cbm_library.utils.logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)


class FinalLayer(nn.Linear):
    """
    A subclass of torch.nn.Linear that represents the final layer of a model.
    """
    def __init__(self, in_features: int, out_features: int, device: str = "cuda"):
        super().__init__(in_features, out_features, bias=True)
        self.to(device)

    def export_state(self) -> Dict[str, torch.Tensor]:
        '''Exports the layer's weights and biases as a dictionary.'''
        return {
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
        }

    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "final.pt"))

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cuda"):
        '''Loads a pretrained model from the specified directory.'''
        state_path = os.path.join(load_dir, "final.pt")
        if os.path.exists(state_path):
            sd = torch.load(state_path, map_location=device)
            in_features = sd["weight"].shape[1]
            out_features = sd["weight"].shape[0]
            model = cls(in_features, out_features, device=device)
            model.load_state_dict(sd)
            return model
        # Legacy support
        W = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
        b = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)
        model = cls(W.shape[1], W.shape[0], device=device)
        with torch.no_grad():
            model.weight.copy_(W.to(device))
            model.bias.copy_(b.to(device))
        return model



class FinalLayerMethod:
    """
    Unified dense linear, sparse Top-K, and sparse GLM-SAGA training for final layer.
    Compatible with LF-CBM; also exposes a loader-based entrypoint for VLG-CBM.
    """
    def __init__(self):
        import torch.nn.functional as F
        _orig = F.softmax

        def _softmax_with_dim(x, dim=None, *args, **kwargs):
            if dim is None:
                dim = -1
            return _orig(x, dim=dim, *args, **kwargs)

        F.softmax = _softmax_with_dim

        self._glm_ready = False
        self._glm_saga = None
        self._IndexedTensorDataset = None
        try:
            from cbm_library.utils.glm_saga import glm_saga as _glm_saga  
            self._glm_saga = _glm_saga
            try:
                from glm_saga.elasticnet import IndexedTensorDataset as _Indexed  
                self._IndexedTensorDataset = _Indexed
            except Exception:
                class _IndexedTensorDataset(TensorDataset):
                    pass
                self._IndexedTensorDataset = _IndexedTensorDataset
            self._glm_ready = True
            logger.info(" GLM-SAGA (local wrapper) available")
        except Exception:
            try:
                from glm_saga.elasticnet import glm_saga as _glm_saga, IndexedTensorDataset as _Indexed  # type: ignore
                self._glm_saga = _glm_saga
                self._IndexedTensorDataset = _Indexed
                self._glm_ready = True
                logger.info(" GLM-SAGA (external) available")
            except Exception:
                logger.warning(" GLM-SAGA not available. Run: pip install glm-saga")

    def train(
        self,
        concept_activations: torch.Tensor,
        labels: torch.Tensor,
        config: FinalLayerConfig,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        '''Trains the final layer based on the specified configuration.'''
        if config.num_concepts == 0:
            config.num_concepts = int(concept_activations.shape[1])
            
        if config.num_classes == 0:
            config.num_classes = int(labels.max().item()) + 1

        if config.layer_type == FinalLayerType.SPARSE_GLM:                                                                    #Internal method for training sparse GLM-SAGA layers.
            return self._train_sparse_glm(concept_activations, labels, config, validation_data, progress_callback)
            
        elif config.layer_type in (FinalLayerType.DENSE_LINEAR, FinalLayerType.ELASTIC_NET, FinalLayerType.SPARSE_LINEAR):    #Internal method for training dense/elastic-net layers.
            dense = self._train_dense(concept_activations, labels, config, validation_data, progress_callback)
            if config.layer_type == FinalLayerType.SPARSE_LINEAR:                                                             #Applies Top-K sparsity to a dense layer's weights.
                return self._apply_topk_sparsity(dense, config)
            return dense
            
        else:
            raise ValueError(f"Unsupported layer type: {config.layer_type}")

    def create_layer(self, config: FinalLayerConfig) -> nn.Module:
        '''Creates a FinalLayer instance based on the configuration.'''
        return FinalLayer(config.num_concepts, config.num_classes, device=config.device)

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
        logger.info(" Training dense/elastic-net linear head")
        X, mu, sigma = self._normalize(concept_acts, cfg.normalize_concepts)

        model = nn.Linear(cfg.num_concepts, cfg.num_classes).to(cfg.device)

        opt = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        ce = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X, labels.long()), batch_size=cfg.batch_size, shuffle=True)

        train_losses: list[float] = []
        val_accs: list[float] = []

        for e in range(cfg.max_epochs):
            model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                logits = model(xb)
                loss = ce(logits, yb)
                # Optional: L1 penalty if cfg has attribute l1 (keep backward-compat)
                l1 = getattr(cfg, "l1", 0.0)
                if l1 and l1 > 0:
                    loss = loss + l1 * model.weight.abs().sum() / model.weight.numel()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            avg = total / max(len(loader), 1)

            vacc = 0.0
            if validation_data is not None:
                vX, vy = validation_data
                if cfg.normalize_concepts:
                    vX = (vX - mu) / sigma
                model.eval()
                with torch.no_grad():
                    pred = model(vX.to(cfg.device)).argmax(1)
                    vacc = (pred == vy.to(cfg.device)).float().mean().item()

            train_losses.append(avg)
            val_accs.append(vacc)

            if progress_callback and (e % 10 == 0):
                progress_callback(e, {"loss": avg, "val_accuracy": vacc})
            if e % 20 == 0:
                logger.info(f"[dense] epoch {e} loss={avg:.4f} val_acc={vacc:.4f}")

        W = model.weight.detach().cpu()
        nnz = (W.abs() > 1e-12).sum().item()
        total = W.numel()
        return {
            "weight": W,
            "bias": model.bias.detach().cpu(),
            "concept_mean": mu,
            "concept_std": sigma,
            "training_metrics": {
                "train_losses": train_losses,
                "val_accuracies": val_accs,
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
            },
            "sparsity_stats": {
                "non_zero_weights": nnz,
                "total_weights": total,
                "sparsity_percentage": nnz / max(total, 1),
                "sparsity_per_class": nnz / max(cfg.num_classes, 1),
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
        logger.info(" Training GLM-SAGA sparse head")

        X, mu, sigma = self._normalize(concept_acts, cfg.normalize_concepts)
        idx_ds = self._IndexedTensorDataset(X, labels.long())
        train_loader = DataLoader(idx_ds, batch_size=cfg.saga_batch_size, shuffle=True)

        val_loader = None
        vX = vy = None
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

        def on_iter(step_idx: int, info: Dict[str, Any]):
            parts = [f"[{step_idx}/{cfg.glm_max_iters}]"]
            loss = info.get("loss")
            lam = info.get("lam", cfg.sparsity_lambda)
            nnz = info.get("nnz")
            tot = info.get("tot")
            val_acc = info.get("val_acc")
            val_loss = info.get("val_loss")

            if loss is not None:
                parts.append(f"loss={float(loss):.6f}")
            parts.append(f"λ={float(lam):g}")
            if nnz is not None and tot is not None:
                parts.append(f"nnz={int(nnz)}/{int(tot)}")
            if val_acc is not None:
                parts.append(f"val_acc={float(val_acc):.4f}")
            if val_loss is not None:
                parts.append(f"val_loss={float(val_loss):.6f}")
            logger.info("v " + "  ".join(parts))

            if progress_callback:
                payload = dict(info)
                payload.setdefault("lambda", float(lam))
                progress_callback(step_idx, payload)

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
            iter_callback=on_iter,
            callback_every=1,
        )

        path = out.get("path", [])
        if not path:
            raise RuntimeError("GLM-SAGA returned empty path")

        logger.info(f" GLM-SAGA progress: {len(path)} checkpoints")
        for i, rec in enumerate(path, start=1):
            loss = float(rec.get("loss", float("nan")))
            lam = float(rec.get("lam", cfg.sparsity_lambda))
            alpha = float(rec.get("alpha", cfg.glm_alpha))
            t = float(rec.get("time", 0.0))
            w = rec.get("weight", None)

            if isinstance(w, torch.Tensor):
                nnz_i = int((w.abs() > 1e-5).sum().item())
                tot_i = int(w.numel())
            else:
                nnz_i = None
                tot_i = None

            m = rec.get("metrics", {})
            val_acc = m.get("val_acc", None)
            val_loss = m.get("val_loss", None)

            parts = [f"[{i:04d}]", f"loss={loss:.6f}", f"λ={lam:g}", f"α={alpha:g}", f"t={t:.3f}s"]
            if nnz_i is not None and tot_i is not None:
                parts.append(f"nnz={nnz_i}/{tot_i}")
            if val_acc is not None:
                parts.append(f"val_acc={float(val_acc):.4f}")
            if val_loss is not None:
                parts.append(f"val_loss={float(val_loss):.6f}")
            logger.info("  " + "  ".join(parts))

            if progress_callback:
                progress_payload = {
                    "loss": loss,
                    "lambda": lam,
                    "alpha": alpha,
                    "time": t,
                    "nonzeros": nnz_i,
                    "total": tot_i,
                }
                if val_acc is not None:
                    progress_payload["val_acc"] = float(val_acc)
                if val_loss is not None:
                    progress_payload["val_loss"] = float(val_loss)
                progress_callback(i, progress_payload)

        best = min(path, key=lambda r: r.get("loss", float("inf")))
        W_g, b_g = best["weight"], best["bias"]
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()

        if progress_callback:
            progress_callback(
                cfg.glm_max_iters,
                {"loss": float(best.get("loss", 0.0)), "lambda": float(best.get("lam", cfg.sparsity_lambda))},
            )

        return {
            "weight": W_g,
            "bias": b_g,
            "concept_mean": mu,
            "concept_std": sigma,
            "training_metrics": best.get("metrics", {}),
            "sparsity_stats": {
                "non_zero_weights": nnz,
                "total_weights": total,
                "sparsity_percentage": nnz / max(total, 1),
                "sparsity_per_class": nnz / max(cfg.num_classes, 1),
            },
            "glm_metadata": {
                "lambda": float(best.get("lam", cfg.sparsity_lambda)),
                "alpha": float(best.get("alpha", cfg.glm_alpha)),
                "final_loss": float(best.get("loss", 0.0)),
                "time": float(best.get("time", 0.0)),
            },
            "train_concept_features": X.detach().cpu(),
            "train_concept_labels": labels.detach().cpu(),
            "val_concept_features": vX.detach().cpu() if vX is not None else None,
            "val_concept_labels": vy.detach().cpu() if vy is not None else None,
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
        W_sparse = (flat * mask.float()).view_as(W)
        nnz = mask.sum().item()
        dense_result["weight"] = W_sparse
        dense_result["sparsity_stats"] = {
            "non_zero_weights": nnz,
            "total_weights": W.numel(),
            "sparsity_percentage": nnz / max(W.numel(), 1),
            "sparsity_per_class": nnz / max(cfg.num_classes, 1),
        }
        return dense_result

    @staticmethod
    def _stack_features_labels(loader: Iterable):
        """Accepts loaders that yield (X, y) or (X, y, *). Returns CPU tensors."""
        feats, labels = [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                X, y = batch[0], batch[1]
            else:
                raise RuntimeError("Expected loader to yield (features, labels, [*])")
            feats.append(X.detach().cpu())
            labels.append(y.detach().cpu())
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    def train_vlg_cbm_from_loaders(
        self,
        train_loader,
        val_loader,
        config: FinalLayerConfig,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        '''Trains the final layer using data loaders.'''
        Xtr, ytr = self._stack_features_labels(train_loader)
        val_tuple = None
        if val_loader is not None:
            Xva, yva = self._stack_features_labels(val_loader)
            val_tuple = (Xva, yva)
        return self.train(Xtr, ytr, config, val_tuple, progress_callback)


class UnifiedFinalTrainer:
    """
    A unified interface for training and managing final layers.
    """
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
        '''Trains the final layer using FinalLayerMethod.'''
        
        logger.info(f" Training {config.layer_type.value} final layer")

        def _progress_proxy(step_idx: int, payload: Dict[str, Any]):
            parts = [f"[step={step_idx}]"]
            for k, v in payload.items():
                try:
                    parts.append(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}")
                except Exception:
                    parts.append(f"{k}={v}")
            logger.info("v " + "  ".join(parts))
            if progress_callback is not None:
                try:
                    progress_callback(step_idx, payload)
                except Exception as e:
                    logger.warning(f"progress_callback raised: {e}")

        out = self._method.train(
            concept_activations=concept_activations,
            labels=labels,
            config=config,
            validation_data=validation_data,
            progress_callback=_progress_proxy if progress_callback is not None else None,
        )
        out["config"] = config
        out["method"] = config.layer_type.value
        return out

    def create_final_layer(self, config: FinalLayerConfig, training_result: Dict[str, Any]) -> nn.Module:
        '''Creates a final layer from training results.'''
        layer = self._method.create_layer(config)
        layer.weight.data = training_result["weight"].to(config.device)
        layer.bias.data = training_result["bias"].to(config.device)
        return layer

    def save_training_result(self, result: Dict[str, Any], save_path: str):
        '''Loads training results from the specified directory.'''
        
        os.makedirs(save_path, exist_ok=True)
        torch.save(result["weight"], os.path.join(save_path, "W_g.pt"))
        torch.save(result["bias"], os.path.join(save_path, "b_g.pt"))
        torch.save(result["concept_mean"], os.path.join(save_path, "proj_mean.pt"))
        torch.save(result["concept_std"], os.path.join(save_path, "proj_std.pt"))

        for k in ("train_concept_features", "train_concept_labels", "val_concept_features", "val_concept_labels"):
            if result.get(k) is not None:
                torch.save(result[k], os.path.join(save_path, f"{k}.pt"))

        meta = {
            "config": result["config"].to_dict() if hasattr(result["config"], "to_dict") else vars(result["config"]),
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
        logger.info(f" Training results saved to {save_path}")

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
        logger.info(f" Loaded final layer from {load_path}")
        return out


def get_label_free_cbm_config(
    num_concepts: int,
    num_classes: int,
    cfg: LabelFreeCBMConfig,
    **overrides,
) -> FinalLayerConfig:
    """Map LF-CBM config → FinalLayerConfig (kept for parity)."""
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


def make_fl_config_from_vlg(
    num_concepts: int,
    num_classes: int,
    cfg: VLGCBMConfig,
) -> FinalLayerConfig:
    """Map VLG runner config → FinalLayerConfig."""
    fl_type = FinalLayerType.DENSE_LINEAR if getattr(cfg, "dense", False) else FinalLayerType.SPARSE_GLM
    return FinalLayerConfig(
        layer_type=fl_type,
        num_concepts=num_concepts,
        num_classes=num_classes,
        # GLM-SAGA knobs
        sparsity_lambda=cfg.saga_lam,
        glm_step_size=cfg.saga_step_size,
        glm_alpha=0.99,
        glm_max_iters=cfg.saga_n_iters,
        glm_epsilon=1.0,
        saga_batch_size=cfg.saga_batch_size,
        # Dense knobs
        learning_rate=cfg.dense_lr,
        batch_size=cfg.saga_batch_size,
        max_epochs=cfg.saga_n_iters,
        weight_decay=0.0,
        normalize_concepts=True,
        device=cfg.device,
    )


def train_vlg_cbm_from_loaders(
    train_loader,
    val_loader,
    config: FinalLayerConfig,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Trains the final layer using data loaders."""
    
    return FinalLayerMethod().train_vlg_cbm_from_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        progress_callback=progress_callback,
    )


def run_final_layer_pipeline(
    train_loader,
    val_loader,
    num_concepts: int,
    num_classes: int,
    cfg: VLGCBMConfig,
    *,
    save_dir: str | None = None,
    progress_callback: callable | None = None,
):
    """
    Trains the final layer from dataloaders and returns (final_layer_module, result_dict).
    Also saves artifacts to save_dir if provided.
    """
    # 1) Build config
    fl_cfg = make_fl_config_from_vlg(num_concepts, num_classes, cfg)

    # 2) Train
    method = FinalLayerMethod()
    result = method.train_vlg_cbm_from_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        config=fl_cfg,
        progress_callback=progress_callback,
    )
    result["config"] = fl_cfg
    result["method"] = fl_cfg.layer_type.value

    # 3) Materialize layer
    final_layer = method.create_layer(fl_cfg)
    final_layer.weight.data = result["weight"].to(cfg.device)
    final_layer.bias.data = result["bias"].to(cfg.device)

    # 4) Optional saving
    if save_dir is not None:
        final_layer.save_model(save_dir)  # final.pt
        UnifiedFinalTrainer().save_training_result(result, save_dir)  # legacy + metadata

    return final_layer, result


__all__ = [
    "FinalLayer",
    "FinalLayerMethod",
    "UnifiedFinalTrainer",
    "get_label_free_cbm_config",
    "make_fl_config_from_vlg",
    "train_vlg_cbm_from_loaders",
    "run_final_layer_pipeline",
]
