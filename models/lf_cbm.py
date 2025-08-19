from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import clip  


def _is_cuda(device) -> bool:
    """True if device is CUDA. Works for torch.device or string-like."""
    try:
        if hasattr(device, "type"):
            return str(device.type).lower().startswith("cuda")
        return "cuda" in str(device).lower()
    except Exception:
        return False


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Flatten to [N, D] if x has more than 2 dims."""
    return x.view(x.size(0), -1) if x.ndim > 2 else x

def cos_cubed_similarity_mean(proj: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Mean cos^3 similarity between projected activations and CLIP concept scores.
    proj: [B, C], Y: [B, C] (paired along first dim)
    """
    proj_norm = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    cos = (proj_norm * Y_norm).sum(dim=1)
    return (cos ** 3).mean()


def cos_cubed_similarity_per_concept(proj: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Per-concept cos^3 similarity on a batch; returns [C]."""
    # Column-wise cosine across the batch
    Pn = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)  # [B, C]
    Tn = Y / (Y.norm(dim=0, keepdim=True) + 1e-8)        # [B, C]
    cos_c = (Pn * Tn).sum(dim=0)                          # [C]
    return cos_c ** 3

class LabelFreeCBM(nn.Module):
    """
    Provides:
      - Backbone feature extraction (frozen CLIP RN50 visual)
      - Concept layer training (top-5 filter → cos^3 W-training → interpretability cutoff)
      - Final layer training (multinomial logistic in PyTorch)
      - Inference: forward(x) → logits
    """

    @staticmethod
    def build_backbone(device: str = "cuda") -> nn.Module:
        """Build CLIP RN50 visual encoder as backbone (frozen)."""
        dev = torch.device(device)
        model, _ = clip.load("RN50", device=dev)
        visual = model.visual.eval()
        for p in visual.parameters():
            p.requires_grad = False
        class CLIPRN50Backbone(nn.Module):
            def __init__(self, visual_encoder: nn.Module):
                super().__init__()
                self.visual = visual_encoder
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.visual(x)
        return CLIPRN50Backbone(visual).to(dev)

    def __init__(self, backbone: nn.Module, num_concepts: int, num_classes: int,
                 device: str = "cuda", clip_name: str = "RN50"):
        super().__init__()
        self.device = torch.device(device)

        if backbone is None or not isinstance(backbone, nn.Module):
            raise ValueError("Backbone must be a torch.nn.Module")
        self.backbone = backbone.to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        if not isinstance(num_concepts, int) or num_concepts <= 0:
            raise ValueError("num_concepts must be a positive int")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive int")

        self.num_concepts = num_concepts
        self.num_classes = num_classes

        self.concept_layer: Optional[nn.Linear] = None
        self.final_layer: Optional[nn.Linear] = None
        self.concept_names: List[str] = []

        self.register_buffer("concept_mean", None, persistent=False)
        self.register_buffer("concept_std", None, persistent=False)

        # CLIP encoders + preprocess/tokenize
        self.clip_model, self.clip_preprocess = clip.load(clip_name, device=self.device)
        self.clip_tokenize = clip.tokenize

        # Infer backbone feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            feat = _ensure_2d(self.backbone(dummy))
            self.feature_dim = int(feat.shape[1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feats = self.backbone(x)
        return _ensure_2d(feats)

    def get_concept_activations(self, features: torch.Tensor) -> torch.Tensor:
        if self.concept_layer is None:
            raise RuntimeError("Concept layer not initialized. Train the model first.")
        return self.concept_layer(features)

    def predict_from_concepts(self, concept_activations: torch.Tensor) -> torch.Tensor:
        if self.final_layer is None:
            raise RuntimeError("Final layer not initialized. Train the model first.")
        if self.concept_mean is not None and self.concept_std is not None:
            concept_activations = (concept_activations - self.concept_mean) / (self.concept_std + 1e-8)
        return self.final_layer(concept_activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        concepts = self.get_concept_activations(feats)
        logits = self.predict_from_concepts(concepts)
        return logits

    def _extract_dataset_features(self, dataset, batch_size: int = 64, num_workers: int = 4) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=_is_cuda(self.device))
        self.backbone.eval()
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(self.device, non_blocking=_is_cuda(self.device))
                z = self.backbone(x)
                feats.append(_ensure_2d(z).float().cpu())
        return torch.cat(feats, dim=0).to(self.device)

    def _extract_clip_image_features(self, dataset, batch_size: int = 64) -> torch.Tensor:
        from PIL import Image
        import numpy as np

        class _ClipView(torch.utils.data.Dataset):
            def __init__(self, base_ds, preprocess):
                self.ds = base_ds
                self.pre = preprocess
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                img, y = self.ds[idx]
                if isinstance(img, torch.Tensor):
                    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray((img * 255).astype(np.uint8))
                img = self.pre(img)
                return img, y

        view = _ClipView(dataset, self.clip_preprocess)
        dl = DataLoader(view, batch_size=batch_size, shuffle=False, pin_memory=_is_cuda(self.device))
        feats: List[torch.Tensor] = []
        self.clip_model.eval()
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(self.device, non_blocking=_is_cuda(self.device))
                f = self.clip_model.encode_image(x).to(dtype=torch.float32)
                f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
                feats.append(f.float().cpu())
        return torch.cat(feats, dim=0).to(self.device)

    def _extract_clip_concept_features(self, concepts: List[str]) -> torch.Tensor:
        tokens = self.clip_tokenize(concepts).to(self.device)
        with torch.no_grad():
            tf = self.clip_model.encode_text(tokens).to(dtype=torch.float32)
            tf = tf / (tf.norm(dim=1, keepdim=True) + 1e-8)
        return tf

    # ----------------- Concept filtering -----------------
    @staticmethod
    def filter_concepts_by_top5_clip(clip_scores: torch.Tensor, concepts: List[str], cutoff: float
                                     ) -> Tuple[List[str], torch.Tensor]:
        """Mean-of-top-5 across images per concept; keep if > cutoff.
        clip_scores: [N, C] (I @ T^T)
        Returns: (kept_concepts, kept_indices)
        """
        k = min(5, clip_scores.size(0))
        topk_mean = torch.topk(clip_scores, dim=0, k=k)[0].mean(dim=0)  # [C]
        keep_mask = topk_mean > cutoff
        kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        kept = [c for c, m in zip(concepts, keep_mask.tolist()) if m]
        return kept, kept_idx

    # ----------------- Training: projection W -----------------
    def _train_projection_layer(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,  # [N, C]
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        """Learn W [C, D] s.t. proj = X @ W^T aligns with CLIP pseudo-labels under cos^3.
        Adds a tiny L2 penalty to stabilize.
        """
        Xtr = _ensure_2d(X_train).to(self.device)
        Xva = _ensure_2d(X_val).to(self.device)
        Ytr = Y_train.to(self.device)
        Yva = Y_val.to(self.device)

        N, D = Xtr.shape
        C = Ytr.shape[1]

        steps = int(config.get("proj_steps", 1000))
        bs = min(int(config.get("proj_batch_size", 256)), N)
        lr = float(config.get("learning_rate", 1e-3))
        std_acts = bool(config.get("standardize_activations", False))
        log_every = int(config.get("log_every_n_steps", 50))

        if bool(config.get("normalize_concepts", True)):
            Ytr = Ytr / (Ytr.norm(dim=0, keepdim=True) + 1e-8)
            Yva = Yva / (Yva.norm(dim=0, keepdim=True) + 1e-8)

        W = torch.empty(C, D, device=self.device, requires_grad=True)
        nn.init.xavier_uniform_(W)
        opt = optim.Adam([W], lr=lr)

        best_val = float("inf")
        best_W = None

        for step in range(steps):
            if bs < N:
                idx = torch.randint(0, N, (bs,), device=self.device)
                Xb, Yb = Xtr[idx], Ytr[idx]
            else:
                Xb, Yb = Xtr, Ytr

            proj = Xb @ W.T  # [B, C]
            if std_acts:
                proj = (proj - proj.mean(dim=0, keepdim=True)) / (proj.std(dim=0, keepdim=True) + 1e-8)

            sim = cos_cubed_similarity_mean(proj, Yb)
            loss = -sim + 1e-4 * (W ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                vproj = Xva @ W.T
                if std_acts:
                    vproj = (vproj - vproj.mean(dim=0, keepdim=True)) / (vproj.std(dim=0, keepdim=True) + 1e-8)
                vloss = -cos_cubed_similarity_mean(vproj, Yva)

            if vloss.item() < best_val:
                best_val = float(vloss.item())
                best_W = W.detach().clone()

            if (step % log_every) == 0:
                print(f"[W-train] step {step}/{steps} loss={loss.item():.6e} val={vloss.item():.6e}")

        if best_W is None:
            best_W = W.detach().clone()
        print(f"[W-train] best val loss: {best_val:.6f}")
        return best_W

    # ----------------- Public training API -----------------
    def train_concept_layer(self, dataset, concepts: List[str], config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Top-5 filter → train W under cos^3 → interpretability cutoff → set concept layer."""
        cfg = {
            "val_frac": 0.1,
            "clip_cutoff": 0.25,
            "interpretability_cutoff": 0.45,
            "proj_steps": 1000,
            "proj_batch_size": 256,
            "learning_rate": 1e-3,
            "normalize_concepts": True,
            "standardize_activations": False,
            "log_every_n_steps": 50,
            "min_concepts_kept": 10,
        }
        if config:
            cfg.update(config)

        # Extract features and CLIP pseudo labels
        X_full = self._extract_dataset_features(dataset)  # [N, D]
        I_full = self._extract_clip_image_features(dataset)  # [N, K]
        T_full = self._extract_clip_concept_features(concepts)  # [C, K]
        Y_full = I_full @ T_full.T  # [N, C]

        # Split train/val
        N = X_full.size(0)
        nval = max(1, int(N * float(cfg["val_frac"])) )
        X_val, X_tr = X_full[:nval], X_full[nval:]
        Y_val, Y_tr = Y_full[:nval], Y_full[nval:]

        # Top-5 CLIP filter
        kept_concepts, kept_idx = self.filter_concepts_by_top5_clip(Y_tr, concepts, float(cfg["clip_cutoff"]))
        if kept_idx.numel() == 0:
            # Keep top-k by same metric if nothing passes
            k = min(max(5, int(cfg["min_concepts_kept"])) , Y_tr.shape[1])
            top5 = torch.topk(Y_tr, dim=0, k=min(5, Y_tr.size(0)))[0].mean(dim=0)
            kept_idx = torch.topk(top5, k=k, largest=True).indices
            kept_concepts = [concepts[i] for i in kept_idx.tolist()]
        Y_tr = Y_tr[:, kept_idx]
        Y_val = Y_val[:, kept_idx]

        # Train projection W
        W = self._train_projection_layer(X_tr, Y_tr, X_val, Y_val, cfg)  # [C_kept, D]

        # Interpretability cutoff on validation
        with torch.no_grad():
            vproj = X_val @ W.T
            sim_per = cos_cubed_similarity_per_concept(vproj, Y_val)
            keep_mask = sim_per >= float(cfg["interpretability_cutoff"])
        if keep_mask.sum().item() == 0:
            # Keep top-k if nothing passes
            k = min(int(cfg["min_concepts_kept"]), int(sim_per.numel()))
            topk_idx = torch.topk(sim_per, k=k, largest=True).indices
            mask = torch.zeros_like(sim_per, dtype=torch.bool)
            mask[topk_idx] = True
            keep_mask = mask

        W_final = W[keep_mask]
        final_concepts = [c for c, m in zip(kept_concepts, keep_mask.tolist()) if m]

        # Install concept layer
        self.concept_layer = nn.Linear(self.feature_dim, W_final.size(0), bias=False).to(self.device)
        with torch.no_grad():
            self.concept_layer.weight.copy_(W_final)
        self.concept_names = final_concepts

        # Return activations on full set (useful for final layer training)
        with torch.no_grad():
            concept_acts = self.concept_layer(X_full)  # [N, C_final]
        return concept_acts

    def train_final_layer(self, concept_activations: torch.Tensor, labels: torch.Tensor,
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train a multinomial logistic regression on concept activations.
        This is a dense PyTorch implementation. Replace with GLM-SAGA to match the repo exactly.
        Returns a dict with learned stats and references.
        """
        cfg = {
            "max_epochs": 100,
            "batch_size": 256,
            "lr": 1e-2,
            "weight_decay": 0.0,  # L2
            "l1_lambda": 0.0,      # optional L1 penalty if you want elastic-net-ish
            "log_every": 10,
        }
        if config:
            cfg.update(config)

        X = concept_activations.to(self.device)
        y = labels.to(self.device).long()

        # Standardize concept activations
        with torch.no_grad():
            mu = X.mean(dim=0, keepdim=True)
            sigma = X.std(dim=0, keepdim=True)
            sigma = torch.where(sigma < 1e-8, torch.ones_like(sigma), sigma)
            Xs = (X - mu) / sigma
            self.concept_mean = mu
            self.concept_std = sigma

        num_samples, num_concepts = Xs.shape
        num_classes = int(y.max().item()) + 1

        self.final_layer = nn.Linear(num_concepts, num_classes, bias=True).to(self.device)
        opt = optim.Adam(self.final_layer.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        ce = nn.CrossEntropyLoss()

        dl = DataLoader(torch.utils.data.TensorDataset(Xs, y), batch_size=int(cfg["batch_size"]), shuffle=True,
                        pin_memory=_is_cuda(self.device))

        for epoch in range(int(cfg["max_epochs"])):
            running = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=_is_cuda(self.device))
                yb = yb.to(self.device, non_blocking=_is_cuda(self.device))
                logits = self.final_layer(xb)
                loss = ce(logits, yb)
                l1 = float(cfg["l1_lambda"]) * self.final_layer.weight.abs().mean()
                loss = loss + l1

                opt.zero_grad()
                loss.backward()
                opt.step()
                running += float(loss.item())

            if (epoch % int(cfg["log_every"])) == 0:
                print(f"[final] epoch {epoch}/{cfg['max_epochs']} loss={running/len(dl):.4f}")

        with torch.no_grad():
            logits = self.final_layer(Xs)
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        print(f"[final] train acc: {acc:.4f}")

        return {
            "train_accuracy": acc,
            "num_concepts": num_concepts,
            "num_classes": num_classes,
        }

def read_concepts_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
