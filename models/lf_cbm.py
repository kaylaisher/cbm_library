"""
This module implements the Label-Free Concept Bottleneck Model (LF-CBM), 
which is a neural network architecture designed to learn interpretable concepts without requiring labeled data. 
It uses CLIP as a backbone for feature extraction and supports training concept layers and final layers with various configurations.
"""

from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
import clip  

def _is_cuda(device) -> bool:
    """Check if the given device string or object refers to a CUDA-capable device."""
    try:
        if hasattr(device, "type"):
            return str(device.type).lower().startswith("cuda")
        return "cuda" in str(device).lower()
    except Exception:
        return False


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape [N, D]; flattens if more than 2 dimensions."""
    return x.view(x.size(0), -1) if x.ndim > 2 else x

def _zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """Apply z-score normalization along a given dimension with numerical stability."""
    mu = x.mean(dim=dim, keepdim=True)
    sd = x.std(dim=dim, keepdim=True)
    return (x - mu) / (sd + eps)

@torch.no_grad()
def cos_cubed_cols(proj: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the cosine similarity between the cube of the normalized columns of two tensors."""
    
    Pz = _zscore(proj, dim=0)
    Yz = _zscore(Y,    dim=0)
    P3 = Pz ** 3
    Y3 = Yz ** 3
    # L2 normalize columns and take column-wise cosine
    Pn = P3 / (P3.norm(dim=0, keepdim=True) + 1e-8)
    Yn = Y3 / (Y3.norm(dim=0, keepdim=True) + 1e-8)
    cos_c = (Pn * Yn).sum(dim=0)  # [C], each in [-1,1]
    return cos_c.mean(), cos_c
    # cos_c.mean(): A scalar tensor representing the mean cosine similarity between the cubed, normalized columns of the input tensors proj and Y. This value is a single number in the range [-1, 1].
    # cos_c: A tensor of shape [C], where C is the number of columns in the input tensors. Each element in this tensor represents the cosine similarity for the corresponding column between proj and Y. Each value is in the range [-1, 1].

def read_concepts_file(path: str) -> List[str]:
    """Reads a list of concepts from a text file."""
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

class LabelFreeCBM(nn.Module):
    """
    The main class for implementing the Label-Free Concept Bottleneck Model.
    """
    @staticmethod
    def build_backbone(device: str = "cuda", clip_name: str = "RN50") -> nn.Module:
        """Load and freeze a CLIP visual backbone model for feature extraction."""
        dev = torch.device(device)
        model, _ = clip.load(clip_name, device=dev)
        visual = model.visual.eval()                                                             # The input tensor x is moved to the determined device and data type, then passed through the visual encoder.
        for p in visual.parameters():
            p.requires_grad = False

        class CLIPVisualBackbone(nn.Module):
            def __init__(self, visual_encoder: nn.Module):
                super().__init__()
                self.visual = visual_encoder
            def forward(self, x: torch.Tensor) -> torch.Tensor:                                   # If the visual encoder has a conv1 layer with weights, it retrieves the weights (w) and uses their device and data type for the input tensor.
                if hasattr(self.visual, "conv1") and hasattr(self.visual.conv1, "weight"):        # If conv1 is not present, it iterates through the modules of the visual encoder to find the first Conv2d layer and retrieves its weights. 
                    w = self.visual.conv1.weight
                else:
                    w = None
                    for m in self.visual.modules():
                        if isinstance(m, torch.nn.Conv2d):
                            w = m.weight
                            break
                    if w is None:
                        p = next(self.visual.parameters())
                        return self.visual(x.to(device=p.device, dtype=p.dtype))
                return self.visual(x.to(device=w.device, dtype=w.dtype))


        return CLIPVisualBackbone(visual).to(dev)

    def __init__(self, backbone: nn.Module, num_concepts: int, num_classes: int,
                 device: str = "cuda", clip_name: str = "RN50"):
        """Initialize CBM with backbone, number of concepts/classes, and CLIP models."""
        super().__init__()
        self.device = torch.device(device)
        self.clip_name = clip_name

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
        self.kept_concepts_: Optional[List[str]] = None

        self.register_buffer("concept_mean", None, persistent=False)
        self.register_buffer("concept_std", None, persistent=False)

        self.clip_model, self.clip_preprocess = clip.load(clip_name, device=self.device)
        self.clip_tokenize = clip.tokenize

        self.feature_dim: Optional[int] = None
        

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images using the backbone."""
        
        _bdtype = next(self.backbone.parameters()).dtype                                              # Retrieves the data type (dtype) of the parameters in the backbone model.
        x = x.to(self.device, dtype=_bdtype)                                                          # Moves the input tensor x to the device and data type of the backbone.
        feats = self.backbone(x)                                                                      # Passes the input tensor through the backbone to extract features.
        feats = _ensure_2d(feats)                                                                     # Flattens the extracted features to a 2D tensor of shape [N, D] (if necessary), where N is the batch size and D is the feature dimension.
        return feats                                                                                  # Returns the processed feature tensor.

    def get_concept_activations(self, features: torch.Tensor) -> torch.Tensor:
        """Project features into concept space using the trained concept layer."""
        
        if self.concept_layer is None:                                                                # If the concept_layer is not initialized (i.e., None), it raises a RuntimeError 
            raise RuntimeError("Concept layer not initialized. Train the model first.")               # Converts the input features tensor to the same data type as the weights of the concept_layer. Passes the processed features through the concept_layer to compute the activations.
        return self.concept_layer(features.to(self.concept_layer.weight.dtype))                       # Returns the output of the concept_layer, which represents the concept activations.

    def predict_from_concepts(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """Predict class logits from concept activations using the final layer."""
        
        if self.final_layer is None:
            raise RuntimeError("Final layer not initialized. Train the model first.")
        if self.concept_mean is not None and self.concept_std is not None:
            concept_activations = (concept_activations - self.concept_mean) / (self.concept_std + 1e-8)
        return self.final_layer(concept_activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: extract features → compute concepts → predict logits."""
        
        feats = self.extract_features(x)
        concepts = self.get_concept_activations(feats)
        logits = self.predict_from_concepts(concepts)
        return logits

    def _extract_dataset_features(
        self,
        dataset,
        batch_size: int = 64,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Extract and optionally cache backbone features for a dataset."""
        
        cache_path = None
        mode = getattr(self, "_feat_mode", getattr(self, "backbone_feature_mode", "final"))
        if cache_dir and split:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"backbone_{split}_{mode}.pt")
            if use_cache and os.path.exists(cache_path):
                return torch.load(cache_path, map_location=self.device)
    
        feats: List[torch.Tensor] = []
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=_is_cuda(self.device))
        self.backbone.eval()
        with torch.no_grad():
            for x, _ in dl:
                _bdtype = next(self.backbone.parameters()).dtype
                x = x.to(self.device, dtype=_bdtype, non_blocking=_is_cuda(self.device))
                z = self.backbone(x)
                feats.append(_ensure_2d(z).float().cpu())
        feats = torch.cat(feats, dim=0)
        if getattr(self, "feature_dim", None) in (None, 0):
            self.feature_dim = int(feats.shape[1])
        if cache_path:
            torch.save(feats, cache_path)
        return feats       

    def _extract_clip_image_features(
        self,
        dataset,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        use_cache: bool = True,
        ) -> torch.Tensor:
        """Extract and normalize CLIP image features for a dataset, with caching support."""

        cache_path = None
        if cache_dir and split:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"clip_images_{split}.pt")
            if use_cache and os.path.exists(cache_path):
                return torch.load(cache_path, map_location=self.device)
    
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
                _cdtype = next(self.clip_model.parameters()).dtype
                x = x.to(self.device, dtype=_cdtype, non_blocking=_is_cuda(self.device))
                f = self.clip_model.encode_image(x).to(dtype=torch.float32)
                f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
                feats.append(f.float().cpu())

        feats = torch.cat(feats, dim=0)
        if cache_path:
            torch.save(feats, cache_path)
        return feats        
    
    def _extract_clip_concept_features(
        self,
        concepts: List[str],
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        use_cache: bool = True,
        ) -> torch.Tensor:
        """Extract CLIP text embeddings for a list of concepts, with caching support."""
            
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            split_tag = split or "concepts"
            clip_tag = getattr(self, "clip_name", getattr(self, "clip_model_name", "clip"))
            cache_path = os.path.join(cache_dir, f"cliptxt_{split_tag}_{clip_tag}.pt")
            if use_cache and os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")

        tokens = self.clip_tokenize(concepts).to(self.device)
        with torch.no_grad():
            tf = self.clip_model.encode_text(tokens).to(dtype=torch.float32)
            tf = tf / (tf.norm(dim=1, keepdim=True) + 1e-8)

        tf_cpu = tf.detach().cpu()       

        if cache_path:
            torch.save(tf_cpu, cache_path)

        return tf_cpu

    @staticmethod
    def filter_concepts_by_topk_clip(
        clip_scores: torch.Tensor, concepts: List[str], cutoff: float, k: int = 5
    ) -> Tuple[List[str], torch.Tensor]:   
        """Mean-of-top-5 across images per concept; keep if > cutoff.
        clip_scores: [N, C] (I @ T^T)
        Returns: (kept_concepts, kept_indices)
        """
        k = max(1, min(int(k), clip_scores.size(0)))
        topk_mean = torch.topk(clip_scores, dim=0, k=k)[0].mean(dim=0)  # [C]
        keep_mask = topk_mean > cutoff
        kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        kept = [c for c, m in zip(concepts, keep_mask.tolist()) if m]
        return kept, kept_idx
    filter_concepts_by_top5_clip = filter_concepts_by_topk_clip

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
        log_every = int(config.get("log_every_n_steps", 50))

        W = torch.empty(C, D, device=self.device, requires_grad=True)
        nn.init.xavier_uniform_(W)
        opt = optim.Adam([W], lr=lr)

        best_val = float("inf")
        best_W = None
        patience = int(config.get("proj_patience", 100))
        stall = 0

        with trange(steps, desc="Concept projection (cos^3)", leave=True) as pbar:
            for step in pbar:
                if bs < N:
                    idx = torch.randint(0, N, (bs,), device=self.device)
                    Xb, Yb = Xtr[idx], Ytr[idx]
                else:
                    Xb, Yb = Xtr, Ytr
                
                proj = Xb @ W.T                  # [B, C]
                sim_mean, _ = cos_cubed_cols(proj, Yb)
                # minimize negative similarity + tiny L2 on W (stabilizer)
                loss = -sim_mean + 1e-4 * (W ** 2).mean()                

                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    vproj = Xva @ W.T
                    vmean, _ = cos_cubed_cols(vproj, Yva)
                    vloss = -vmean

                if vloss.item() < best_val:
                    best_val = float(vloss.item())
                    best_W = W.detach().clone()
                    stall = 0
                else:
                    stall += 1
                    if stall >= patience:
                        break

                if (step % log_every) == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4e}", val=f"{vloss.item():.4e}")

        if best_W is None:
            best_W = W.detach().clone()
        tqdm.write(f"[W-train] best val loss: {best_val:.6f}")
        return best_W

    def train_concept_layer(self, dataset, concepts: List[str], config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Top-5 filter → train W under cos^3 → interpretability cutoff → set concept layer."""
        cfg = {
            "feature_cache_dir": None,
            "use_cache": True,
            "cache_split": "full",
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
            "topk_k": 5,
            "proj_patience": 100,
        }
        if config:
            cfg.update(config)

      
        cache_root = cfg.get("feature_cache_dir")
        split_tag = str(cfg.get("cache_split") or "full")
        use_cache = bool(cfg.get("use_cache", True))
        X_full = self._extract_dataset_features(
            dataset, cache_dir=cache_root, split=f"{split_tag}", use_cache=use_cache
        )  # [N, D]
       
        if getattr(self, "feature_dim", None) in (None, 0):
            self.feature_dim = int(X_full.shape[1])
        I_full = self._extract_clip_image_features(
            dataset, cache_dir=cache_root, split=f"{split_tag}", use_cache=use_cache
        )  # [N, K]
        T_full = self._extract_clip_concept_features(
            concepts, cache_dir=cache_root, split="concepts", use_cache=use_cache
        )  # [C, K]
        Y_full = I_full @ T_full.T  # [N, C]

        # Split train/val
        N = X_full.size(0)
        nval = max(1, int(N * float(cfg["val_frac"])) )
        X_val, X_tr = X_full[:nval], X_full[nval:]
        Y_val, Y_tr = Y_full[:nval], Y_full[nval:]

        # Top-5 CLIP filter
        kept_concepts, kept_idx = self.filter_concepts_by_topk_clip(
            Y_tr, concepts, float(cfg["clip_cutoff"]), k=int(cfg.get("topk_k", 5))
        )
        if kept_idx.numel() == 0:
            k = min(max(5, int(cfg["min_concepts_kept"])) , Y_tr.shape[1])
            top5 = torch.topk(Y_tr, dim=0, k=min(5, Y_tr.size(0)))[0].mean(dim=0)
            kept_idx = torch.topk(top5, k=k, largest=True).indices
            kept_concepts = [concepts[i] for i in kept_idx.tolist()]
        Y_tr = Y_tr[:, kept_idx]
        Y_val = Y_val[:, kept_idx]

        # Train projection W
        W = self._train_projection_layer(X_tr, Y_tr, X_val, Y_val, cfg)  # [C_kept, D]
        
        with torch.no_grad():
            vproj = (X_val.to(self.device)) @ W.T
            _, sim_per = cos_cubed_cols(vproj, Y_val.to(self.device))
            keep_mask = sim_per >= float(cfg["interpretability_cutoff"])  # paper: manual cutoff; common=0.45
        
        if keep_mask.sum().item() == 0:
            # Keep top-k if nothing passes
            k = min(int(cfg["min_concepts_kept"]), int(sim_per.numel()))
            topk_idx = torch.topk(sim_per, k=k, largest=True).indices
            mask = torch.zeros_like(sim_per, dtype=torch.bool)
            mask[topk_idx] = True
            keep_mask = mask

        W_final = W[keep_mask]
        final_concepts = [c for c, m in zip(kept_concepts, keep_mask.tolist()) if m]

        self.concept_layer = nn.Linear(self.feature_dim, W_final.size(0), bias=False).to(self.device)
        with torch.no_grad():
            self.concept_layer.weight.copy_(W_final)
        self.concept_names = final_concepts
        self.kept_concepts_ = list(final_concepts)        

        with torch.no_grad():
            concept_acts = self.concept_layer(X_full.to(self.device)).detach().cpu()  
        return concept_acts

    def train_final_layer(self, concept_activations: torch.Tensor, labels: torch.Tensor,
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train interpretable final classifier layer on concept activations using GLM-SAGA or fallback."""
        
        cfg = {
            "max_epochs": 100,
            "batch_size": 256,
            "lr": 1e-2,
            "weight_decay": 0.0,      # elastic-net L2 part if using fallback
            "l1_lambda": 0.0,         # elastic-net L1 part if using fallback
            "target_nonzeros_per_class": 30,  # paper: ~25–35 per class
            "use_glm_saga": True,
            "log_every": 10,
        }
        if config:
            cfg.update(config)

        X = concept_activations.to(self.device)
        y = labels.to(self.device).long()

        with torch.no_grad():
            mu = X.mean(dim=0, keepdim=True)
            sigma = X.std(dim=0, keepdim=True)
            sigma = torch.where(sigma < 1e-8, torch.ones_like(sigma), sigma)
            Xs = (X - mu) / sigma
            self.concept_mean = mu
            self.concept_std = sigma

        num_samples, num_concepts = Xs.shape
        num_classes = int(y.max().item()) + 1
        
        # Try GLM-SAGA first (paper), else fall back to Adam+prune to target k nonzeros/class
        self.final_layer = nn.Linear(num_concepts, num_classes, bias=True).to(self.device)
        used_glm = False
        if bool(cfg.get("use_glm_saga", True)):
            try:
                from glm_saga import multiclass_logistic_enet 
                Wb, b = multiclass_logistic_enet(
                    Xs.cpu().numpy(), y.cpu().numpy(),
                    l1=float(cfg.get("l1_lambda", 0.0)),
                    l2=float(cfg.get("weight_decay", 0.0)),
                    max_epochs=int(cfg["max_epochs"]),
                    k_target=int(cfg["target_nonzeros_per_class"])
                )
                with torch.no_grad():
                    self.final_layer.weight.copy_(torch.from_numpy(Wb).to(self.final_layer.weight.dtype).to(self.device))
                    self.final_layer.bias.copy_(torch.from_numpy(b).to(self.final_layer.bias.dtype).to(self.device))
                used_glm = True
            except Exception as e:
                print(f"[final] GLM-SAGA unavailable, falling back to Adam+prune: {e}")

        if not used_glm:
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
                    loss = loss + float(cfg["l1_lambda"]) * self.final_layer.weight.abs().mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    running += float(loss.item())
                if (epoch % int(cfg["log_every"])) == 0:
                    print(f"[final] epoch {epoch}/{cfg['max_epochs']} loss={running/len(dl):.4f}")
            # Enforce ~k nonzeros/class by hard-pruning smallest |w| per column
            with torch.no_grad():
                W = self.final_layer.weight.data  # [num_classes, num_concepts]
                k = int(cfg["target_nonzeros_per_class"])
                for c in range(W.size(0)):
                    mag = W[c].abs()
                    if (mag > 0).sum() > k:
                        thr = torch.topk(mag, k, largest=True).values.min()
                        W[c] *= (mag >= thr).float()

        with torch.no_grad():
            logits = self.final_layer(Xs)
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        print(f"[final] train acc: {acc:.4f}")

        return {
            "train_accuracy": acc,
            "num_concepts": num_concepts,
            "num_classes": num_classes,
            "used_glm_saga": used_glm,
        }

