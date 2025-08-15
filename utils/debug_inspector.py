# cbm_library/utils/debug_inspector.py
from __future__ import annotations
import os
import time
import json
import functools
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
import logging


class CBMDebugInspector:
    """
    A lightweight debugging/inspection utility for CBM pipelines.

    Features:
    - File + console logging with consistent formatting
    - Action/function tracing via @inspector.trace("name")
    - Save + preview tensor embeddings (image/text/concept)
    - Report loaded classes/concepts and concept filtering results
    - Explain matrices: image features (X), concept features (C), activations (S)
    - Explain weights: concept projection Wc and final layer (Wg, bg)
    - cosine^3 helper (Label-Free CBM default)
    """

    def __init__(
        self,
        log_dir: str | Path = None,
        artifacts_dir: str | Path = None,
        logger_name: str = "cbm_debug",
        log_filename: str = "cbm_debug.log",
        log_level: int = logging.INFO,
        overwrite_log: bool = True,
    ):
        # Directories (fall back to env -> defaults)
        self.log_dir = Path(log_dir or os.getenv("CBM_DEBUG_DIR", "/kayla/logs"))
        self.artifacts_dir = Path(artifacts_dir or os.getenv("CBM_ARTIFACTS_DIR", "/kayla/artifacts"))
        (self.log_dir).mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "embeddings").mkdir(parents=True, exist_ok=True)

        self.log_path = self.log_dir / log_filename

        # Logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self._configure_handlers(self.logger, self.log_path, log_level, overwrite_log)

        # Convenience handle
        self.log = self.logger

    # -------------------------- Logging & Tracing --------------------------- #
    @staticmethod
    def _configure_handlers(logger: logging.Logger, log_path: Path, level: int, overwrite: bool):
        # Clear existing handlers (avoid duplication if re-imported)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File
        fh = logging.FileHandler(log_path, mode=("w" if overwrite else "a"), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def trace(self, name: Optional[str] = None):
        """Decorator to trace start/end of any function."""
        def deco(fn):
            tag = name or fn.__name__
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.time()
                self.log.info(f"▶ {tag}() START")
                out = fn(*args, **kwargs)
                dt = (time.time() - t0) * 1000
                self.log.info(f"⏹ {tag}() END — {dt:.1f} ms")
                return out
            return wrapper
        return deco

    # --------------------------- Tensor Preview ---------------------------- #
    @staticmethod
    def _preview_array(x, max_rows=3, max_cols=6):
        if isinstance(x, torch.Tensor):
            arr = x.detach().float().cpu().numpy()
        else:
            arr = np.asarray(x)
        if arr.ndim >= 2:
            r = arr[:max_rows, :max_cols]
        else:
            r = arr[:max_rows]
        return {"dtype": str(arr.dtype), "shape": list(arr.shape), "preview": r.tolist()}

    # ---------------------- Save & Inspect Embeddings ---------------------- #
    def save_and_preview_embeddings(self, arr: torch.Tensor, kind: str, split: str) -> Path:
        """
        Save embeddings to disk and print where & what they look like.
        kind ∈ {"image","text","concept"}; split ∈ {"train","val","all"}.
        """
        out_dir = self.artifacts_dir / "embeddings" / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{kind}_{split}.pt"
        torch.save(arr.detach().cpu(), path)
        info = self._preview_array(arr)
        self.log.info(f"[{kind.upper()} EMBEDDINGS] saved → {path}")
        self.log.info(f"[{kind.upper()} EMBEDDINGS] shape={info['shape']} dtype={info['dtype']} preview={info['preview']}")
        return path

    # ---------------------- Classes & Concepts Reports --------------------- #
    def report_classes_and_concepts(self, classes: List[str], concepts: List[str], title: str = "LOAD"):
        self.log.info(f"[{title}] classes({len(classes)}): {classes}")
        self.log.info(
            f"[{title}] concepts({len(concepts)}) sample: "
            f"{concepts[:20]}{' ...' if len(concepts) > 20 else ''}"
        )

    def report_concept_filtering(self, original: List[str], filtered: List[str], reason: str = "similarity threshold"):
        removed = [c for c in original if c not in set(filtered)]
        self.log.info(f"[CONCEPTS] loaded={len(original)} kept={len(filtered)} removed={len(removed)} reason='{reason}'")
        self.log.info(f"[CONCEPTS] kept (first 10): {filtered[:10]}")
        self.log.info(f"[CONCEPTS] removed (first 10): {removed[:10]}")

    # -------------------- Matrix & Weight Explanations --------------------- #
    def explain_concept_matrix(
        self,
        X_img: torch.Tensor,       # N × D image features
        C_txt: torch.Tensor,       # K × D concept/text features
        S: Optional[torch.Tensor] = None  # N × K concept activations
    ):
        self.log.info("[CONCEPT MATRIX] X_img (image features) → rows=images, cols=feature dims")
        self.log.info(f"X_img info: {self._preview_array(X_img)}")
        self.log.info("[CONCEPT MATRIX] C_txt (concept/text features) → rows=concepts, cols=feature dims")
        self.log.info(f"C_txt info: {self._preview_array(C_txt)}")
        if S is not None:
            self.save_and_preview_embeddings(S, "concept", "all")
            self.log.info("S (N×K) often S = (cosine(X_img, C_txt))^3 in Label‑Free CBM.")

    def explain_weights(
        self,
        Wc: torch.Tensor,                  # K × D (or D × K; we auto-fix)
        Wg: torch.Tensor,                  # C × K
        bg: Optional[torch.Tensor],        # C
        concept_names: List[str],
        class_names: List[str],
        topk: int = 5,
    ):
        # Normalize Wc to K × D if needed
        Wc_view = Wc
        if Wc.dim() == 2 and Wc.shape[0] < Wc.shape[1]:
            Wc_view = Wc.T
        self.log.info(f"[WEIGHTS] Concept bottleneck Wc (K×D) info: {self._preview_array(Wc_view)}")

        with torch.no_grad():
            k_dims = min(topk, Wc_view.shape[1])
            k_rows = min(topk, Wc_view.shape[0])
            topdims = torch.topk(Wc_view.abs(), k=k_dims, dim=1).indices[:k_rows]
        for i, idxs in enumerate(topdims):
            cname = concept_names[i] if i < len(concept_names) else f"concept_{i}"
            self.log.info(f"[Wc TOPDIMS] {cname}: dims {idxs.tolist()} (largest |weight|)")

        self.log.info(f"[WEIGHTS] Final layer Wg (classes×concepts) info: {self._preview_array(Wg)}")
        if bg is not None:
            self.log.info(f"[WEIGHTS] Final layer bias bg (classes,) info: {self._preview_array(bg)}")

        with torch.no_grad():
            tk = min(topk, Wg.shape[1])
            cls_top = torch.topk(Wg.abs(), k=tk, dim=1)
        for c in range(min(topk, Wg.shape[0])):
            cname = class_names[c] if c < len(class_names) else f"class_{c}"
            top_ids = cls_top.indices[c].tolist()
            top_concepts = [
                concept_names[i] if i < len(concept_names) else f"concept_{i}" for i in top_ids
            ]
            self.log.info(f"[Wg TOPCONCEPTS] {cname}: {top_concepts}")

        self.log.info(
            "[MEANING]\n"
            "• Wc projects image features (D) into concept activations (K). Each row of Wc shows which feature dims drive a concept.\n"
            "• Wg maps concept activations (K) to class logits (C). Large |Wg[c, k]| ⇒ concept k strongly influences class c (sign = direction).\n"
            "• bg is a per‑class bias term, shifting logits independent of concepts."
        )

    # ----------------------------- Math utils ------------------------------ #
    @staticmethod
    def cosine_cubed(X_img: torch.Tensor, C_txt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        X = torch.nn.functional.normalize(X_img, dim=1)
        C = torch.nn.functional.normalize(C_txt, dim=1)
        S = (X @ C.T).clamp(-1, 1)
        return S * S * S
