# cbm_library/models/vlg_cbm.py
from __future__ import annotations
from dataclasses import replace
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import os
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import clip  # OpenAI CLIP
except Exception:
    clip = None  # allow import without CLIP installed (you'll wire it up)

from cbm_library.config.vlg_cbm_config import VLGCBMConfig

# ---------------------------
# VLGCBM (MAIN METHODS FIRST)
# ---------------------------

class VLGCBM(nn.Module):
    """
    Vision-Language-Guided CBM
    Pipeline:
      1) Build backbone + CLIP, read concepts
      2) (Optional) CLIP-based prefilter of concepts
      3) Learn projection W_c: features -> concepts via cosine/sim alignment
      4) Cache concept activations; compute normalization stats (train)
      5) Train final head (dense or sparse)
      6) Evaluate & export artifacts
    """
    def __init__(self, cfg: VLGCBMConfig):
        super().__init__()
        self.cfg = cfg.finalize()
        self.device = torch.device(self.cfg.device)

        # populated during build()
        self.backbone = None
        self.feature_dim = self.cfg.feature_dim
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None

        # concepts
        self.concepts: List[str] = []
        self.num_concepts: int = 0
        self.text_embeds = None  # (C, E)

        # learnable modules
        self.proj: nn.Module = None         # features -> concept logits
        self.final_head: nn.Linear = None   # concepts -> classes

        # normalization
        self.concept_mean = None
        self.concept_std = None

    # ---- Orchestrator ----
    @torch.no_grad()
    def run_full_pipeline(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        work_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the whole VLG-CBM pipeline.
        Returns metrics and paths to saved artifacts.
        """
        work_dir = work_dir or os.path.join(self.cfg.save_dir, self.cfg.run_name)
        os.makedirs(work_dir, exist_ok=True)
        (Path(work_dir) / "args.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))

        # 1) Models + concepts
        self._build_backbone_and_clip()
        self._load_concepts()
        self._encode_concept_texts()

        # 2) Optional prefilter
        if self.cfg.clip_cutoff is not None:
            self._prefilter_concepts(train_loader)

        # 3) Learn projection (features -> concepts)
        self._init_projection()
        self._fit_projection(train_loader, val_loader)

        # 4) Cache concept activations + normalization
        train_Z, train_y = self.compute_concept_activations(train_loader)
        val_Z,   val_y   = self.compute_concept_activations(val_loader)
        self._fit_normalization(train_Z)
        train_Zn = self.normalize_concepts(train_Z)
        val_Zn   = self.normalize_concepts(val_Z)

        # 5) Train final head
        self._init_final_head()
        if self.cfg.final_type == "dense":
            self._fit_final_dense(train_Zn, train_y, val_Zn, val_y)
        else:
            self._fit_final_saga(train_Zn, train_y, val_Zn, val_y)

        # 6) Evaluate & export
        results = {"val_acc": float(self._eval_matrix(val_Zn, val_y))}
        if test_loader is not None:
            test_Z, test_y = self.compute_concept_activations(test_loader)
            test_Zn = self.normalize_concepts(test_Z)
            results["test_acc"] = float(self._eval_matrix(test_Zn, test_y))

        artifacts = self._export_artifacts(work_dir)
        results.update(artifacts)
        return results

    # ---- Public utility: compute activations from a loader ----
    @torch.no_grad()
    def compute_concept_activations(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        feats, labels = [], []
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            fb = self._extract_features(xb)                               # (B, D)
            zb = self.proj(fb)                                            # (B, C)
            feats.append(zb.detach().cpu())
            labels.append(yb.detach().cpu())
        Z = torch.cat(feats, dim=0)       # (N, C)
        y = torch.cat(labels, dim=0)      # (N,)
        return Z, y

    # ---------------------------
    # IMPLEMENTATION (HELPERS)
    # ---------------------------

    # -- build blocks --
    def _build_backbone_and_clip(self):
        if self.cfg.backbone == "clip_visual":
            assert clip is not None, "pip install git+https://github.com/openai/CLIP.git"
            self.clip_model, self.clip_preprocess = clip.load(self.cfg.clip_model, device=self.cfg.device)
            self.backbone = self.clip_model.visual.eval()
            self.feature_dim = self.feature_dim or self._infer_clip_visual_dim()
        else:
            # Torchvision backbone; you can swap this for your repo util
            import torchvision.models as tvm
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            m.eval().to(self.device)

            # register forward hook to grab the configured layer output
            self.feature_dim = self.feature_dim or 2048
            layer = dict(m.named_modules())[self.cfg.feature_layer]
            buffer = {"out": None}
            def hook(_, __, out): buffer["out"] = out
            layer.register_forward_hook(hook)
            self.backbone = _ForwardHookBackbone(m, buffer, pool=self.cfg.feature_pool).to(self.device)

        # CLIP text encoder (even if backbone isn't CLIP)
        if clip is not None:
            self.clip_model_text = self.clip_model if self.clip_model is not None else clip.load(self.cfg.clip_model, device=self.cfg.clip_device)[0]
            self.clip_tokenize = clip.tokenize
            self.clip_model_text.eval()

    def _load_concepts(self):
        concepts = []
        if self.cfg.concept_list_path and os.path.isfile(self.cfg.concept_list_path):
            with open(self.cfg.concept_list_path) as f:
                concepts += [ln.strip() for ln in f if ln.strip()]
        concepts += [c for c in self.cfg.extra_concepts if c.strip()]
        if not concepts:
            raise ValueError("No concepts provided. Set cfg.concept_list_path or cfg.extra_concepts.")
        self.concepts = concepts
        self.num_concepts = len(concepts)

    @torch.no_grad()
    def _encode_concept_texts(self):
        assert clip is not None, "CLIP required for VLG guidance."
        toks = self.clip_tokenize([self.cfg.prompt_template.format(concept=c) for c in self.concepts]).to(self.cfg.clip_device)
        txt = self.clip_model_text.encode_text(toks)
        self.text_embeds = F.normalize(txt.float(), dim=-1)  # (C, E)

    # -- CLIP prefilter (optional) --
    @torch.no_grad()
    def _prefilter_concepts(self, train_loader: DataLoader):
        sims = []
        kept_mask = torch.ones(self.num_concepts, dtype=torch.bool)
        for xb, _ in train_loader:
            xb = xb.to(self.cfg.clip_device)
            if self.cfg.backbone == "clip_visual" and self.clip_model is not None:
                img_emb = self.clip_model.encode_image(xb)
            else:
                # always compute CLIP image embedding using text model’s visual pair
                img_emb = clip.load(self.cfg.clip_model, device=self.cfg.clip_device)[0].encode_image(xb)  # lightweight reload for correctness
            img_emb = F.normalize(img_emb.float(), dim=-1)        # (B, E)
            s = img_emb @ self.text_embeds.T                      # (B, C)
            sims.append(s.cpu())
        S = torch.cat(sims, dim=0)                                # (N, C)

        # keep concepts whose top-k mean >= cutoff
        topk = S.topk(k=self.cfg.clip_topk, dim=0).values
        means = topk.mean(dim=0)                                  # (C,)
        kept_mask &= (means >= self.cfg.clip_cutoff)

        # apply mask
        if kept_mask.sum().item() == 0:
            # fallback: keep everything to avoid empty set
            return
        self._apply_concept_mask(kept_mask)

    # -- projection W_c (features -> concepts) --
    def _init_projection(self):
        D, C = self.feature_dim, self.num_concepts
        if self.cfg.proj_hidden_dim:
            self.proj = nn.Sequential(
                nn.Linear(D, self.cfg.proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg.proj_hidden_dim, C),
            ).to(self.device)
        else:
            self.proj = nn.Linear(D, C).to(self.device)

    def _fit_projection(self, train_loader: DataLoader, val_loader: DataLoader):
        opt = torch.optim.AdamW(self.proj.parameters(), lr=self.cfg.proj_lr, weight_decay=self.cfg.proj_weight_decay)
        best_state, best_val = None, float("inf")
        patience = self.cfg.proj_early_stop_patience
        stale = 0

        for epoch in range(self.cfg.proj_epochs):
            self.train()
            tr_loss = 0.0
            for i, (xb, _) in enumerate(train_loader):
                xb = xb.to(self.device, non_blocking=True)
                fb = self._extract_features(xb)                       # (B, D)
                zb = self.proj(fb)                                    # (B, C)
                with torch.no_grad():
                    img_emb = self._clip_image_embed(xb)              # (B, E)
                    target = self._clip_similarity_targets(img_emb)   # (B, C)
                loss = _cosine_alignment_loss(zb, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tr_loss += float(loss.item())

            # val
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    fb = self._extract_features(xb)
                    zb = self.proj(fb)
                    img_emb = self._clip_image_embed(xb)
                    target = self._clip_similarity_targets(img_emb)
                    val_loss += float(_cosine_alignment_loss(zb, target).item())

            if val_loss < best_val:
                best_val, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.proj.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            self.proj.load_state_dict(best_state)

    # -- normalization over concepts --
    def _fit_normalization(self, Z_train: torch.Tensor) -> None:
        mean = Z_train.mean(dim=0)
        std = Z_train.std(dim=0).clamp_min(self.cfg.eps)
        self.concept_mean = mean
        self.concept_std = std

    @torch.no_grad()
    def normalize_concepts(self, Z: torch.Tensor) -> torch.Tensor:
        return (Z - self.concept_mean) / self.concept_std

    # -- final layer (dense) --
    def _init_final_head(self):
        self.final_head = nn.Linear(self.num_concepts, self.cfg.num_classes).to(self.device)

    def _fit_final_dense(
        self, Ztr: torch.Tensor, ytr: torch.Tensor, Zva: torch.Tensor, yva: torch.Tensor
    ):
        ds_tr = torch.utils.data.TensorDataset(Ztr, ytr)
        dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(self.final_head.parameters(), lr=self.cfg.dense_lr, weight_decay=self.cfg.dense_weight_decay)
        best, best_state, stale = -1.0, None, 0

        for epoch in range(self.cfg.dense_epochs):
            self.final_head.train()
            for Zb, yb in dl_tr:
                Zb = Zb.to(self.device); yb = yb.to(self.device)
                logits = self.final_head(Zb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            # val
            acc = self._eval_matrix(Zva, yva)
            if acc > best:
                best, stale = acc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.final_head.state_dict().items()}
            else:
                stale += 1
                if stale >= 10:
                    break

        if best_state is not None:
            self.final_head.load_state_dict(best_state)

    # -- final layer (sparse SAGA) --
    def _fit_final_saga(
        self, Ztr: torch.Tensor, ytr: torch.Tensor, Zva: torch.Tensor, yva: torch.Tensor
    ):
        # Placeholder: plug in your GLM-SAGA implementation here.
        # For now, fall back to dense; swap later with your glm_saga() function.
        self._fit_final_dense(Ztr, ytr, Zva, yva)

    @torch.no_grad()
    def _eval_matrix(self, Z: torch.Tensor, y: torch.Tensor) -> float:
        self.final_head.eval()
        logits = self.final_head(Z.to(self.device))
        pred = logits.argmax(dim=1).cpu()
        return (pred == y).float().mean().item()

    # -- exporting --
    def _export_artifacts(self, work_dir: str) -> Dict[str, Any]:
        paths = {
            "W_c": os.path.join(work_dir, "W_c.pt"),
            "norm": os.path.join(work_dir, "concept_norm.json"),
            "concepts": os.path.join(work_dir, "concepts.txt"),
            "W_g": os.path.join(work_dir, "W_g.pt"),
        }
        # projection (W_c) / or full module
        torch.save(self.proj.state_dict(), paths["W_c"])
        # normalization
        with open(paths["norm"], "w") as f:
            json.dump({"mean": self.concept_mean.tolist(), "std": self.concept_std.tolist()}, f)
        # concepts
        Path(paths["concepts"]).write_text("\n".join(self.concepts))
        # final head
        torch.save(self.final_head.state_dict(), paths["W_g"])
        return paths

    # ---------------------------
    # LOWER-LEVEL HELPERS
    # ---------------------------

    @torch.no_grad()
    def _extract_features(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Returns pooled backbone features (B, D)
        """
        if isinstance(self.backbone, _ForwardHookBackbone):
            return self.backbone(xb)
        elif self.cfg.backbone == "clip_visual":
            feats = self.backbone(xb)              # raw visual transformer output
            if feats.ndim == 3:                    # (B, T, D) -> CLS token
                feats = feats[:, 0]
            return feats.float()
        else:
            raise RuntimeError("Backbone not initialized")

    @torch.no_grad()
    def _clip_image_embed(self, xb: torch.Tensor) -> torch.Tensor:
        assert clip is not None
        # Use a CLIP image encoder (paired with text embeds) for guidance
        if self.cfg.backbone == "clip_visual" and self.clip_model is not None:
            img = self.clip_model.encode_image(xb.to(self.cfg.clip_device))
        else:
            model, _ = clip.load(self.cfg.clip_model, device=self.cfg.clip_device)
            img = model.encode_image(xb.to(self.cfg.clip_device))
        return F.normalize(img.float(), dim=-1)  # (B, E)

    @torch.no_grad()
    def _clip_similarity_targets(self, img_emb: torch.Tensor) -> torch.Tensor:
        # soft targets: (B, C), normalized cosine similarities with temperature
        logits = (img_emb @ self.text_embeds.T) / self.cfg.cosine_tau
        probs = logits.softmax(dim=-1)
        return probs

    def _apply_concept_mask(self, mask: torch.Tensor):
        # mask: (C,)
        new_concepts = [c for c, k in zip(self.concepts, mask) if bool(k)]
        self.concepts = new_concepts
        self.num_concepts = len(new_concepts)
        if self.text_embeds is not None:
            self.text_embeds = self.text_embeds[mask]

    def _infer_clip_visual_dim(self) -> int:
        # heuristic dims for popular CLIP models
        table = {"ViT-B/32": 768, "ViT-B/16": 768, "ViT-L/14": 1024, "RN50": 1024, "RN101": 512}
        return table.get(self.cfg.clip_model, 768)


class _ForwardHookBackbone(nn.Module):
    """Wrap a torchvision model and expose pooled feature map from a hooked layer."""
    def __init__(self, backbone: nn.Module, buffer: Dict[str, torch.Tensor], pool: str = "avg"):
        super().__init__()
        self.backbone = backbone
        self.buffer = buffer
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.backbone(x)
        fm = self.buffer["out"]  # (B, C, H, W)
        if fm.ndim == 4:
            if self.pool == "avg":
                fm = fm.mean(dim=(2, 3))
            else:
                fm = fm.amax(dim=(2, 3))
        return fm.float()


def _cosine_alignment_loss(pred_logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    Aligns projected concept logits with CLIP similarity soft-targets.
    Both are normalized and compared via MSE on probabilities.
    """
    pred = pred_logits / pred_logits.std(dim=0, keepdim=True).clamp_min(1e-6)
    pred = pred.softmax(dim=-1)
    loss = F.mse_loss(pred, soft_targets)
    return loss
