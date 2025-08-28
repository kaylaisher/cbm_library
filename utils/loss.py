import torch
import torch.nn as nn
from typing import Iterable, Optional, Union

Number = Union[int, float]

def _pos_weight_tensor(
    n_concepts: int,
    device,
    *,
    pos_weight: Optional[Union[Number, Iterable[Number], torch.Tensor]] = None,
    auto_weight: bool = False,
    n_train_examples: Optional[int] = None,
    concept_counts: Optional[Union[Iterable[Number], torch.Tensor]] = None,
):
    
    if auto_weight:
        if n_train_examples is None or concept_counts is None:
            raise ValueError("auto_weight=True requires n_train_examples and concept_counts")
        c = torch.as_tensor(concept_counts, dtype=torch.float32, device=device)
        if c.numel() != n_concepts:
            raise ValueError(f"concept_counts length {c.numel()} != n_concepts {n_concepts}")
        N_pos = c.clamp(min=0.0)
        N_neg = float(n_train_examples) - N_pos
        eps = 1e-6
        w = N_neg / (N_pos + eps)         
        w = torch.where(torch.isfinite(w), w, torch.full_like(w, 1.0))
        return w.clamp(min=eps)

    
    if pos_weight is None:
        return torch.ones(n_concepts, device=device, dtype=torch.float32)

    if isinstance(pos_weight, torch.Tensor):
        if pos_weight.numel() == 1:
            return pos_weight.to(device=device, dtype=torch.float32).repeat(n_concepts)
        if pos_weight.numel() != n_concepts:
            raise ValueError(f"pos_weight length {pos_weight.numel()} != n_concepts {n_concepts}")
        return pos_weight.to(device=device, dtype=torch.float32)

    if isinstance(pos_weight, (list, tuple)):
        if len(pos_weight) != n_concepts:
            raise ValueError(f"pos_weight length {len(pos_weight)} != n_concepts {n_concepts}")
        return torch.tensor(pos_weight, device=device, dtype=torch.float32)

    
    return torch.full((n_concepts,), float(pos_weight), device=device, dtype=torch.float32)


def get_loss(
    loss_type,
    n_concepts,
    n_train_examples,
    concept_counts,
    pos_weight,
    auto_weight,
    twoway_tp,   
    device,
):
    loss_type = (loss_type or "bce").lower()

    if loss_type == "bce":
        pos_w = _pos_weight_tensor(
            n_concepts=n_concepts,
            device=device,
            pos_weight=pos_weight,
            auto_weight=bool(auto_weight),
            n_train_examples=n_train_examples,
            concept_counts=concept_counts,
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="mean")

    return nn.BCEWithLogitsLoss(reduction="mean")
