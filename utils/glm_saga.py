import time
from typing import Optional, Callable, Dict, Tuple
import torch
import torch.nn.functional as F

def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        Xb, _, yb = batch[:3]
    else:
        Xb, yb = batch
    return Xb, yb

def _label_guard(yb: torch.Tensor, num_classes: int, where: str = "") -> torch.Tensor:
    if yb.dtype != torch.long:
        yb = yb.long()
    if yb.numel() == 0:
        return yb
    y_min = int(yb.min().item())
    y_max = int(yb.max().item())
    assert y_min >= 0 and y_max < num_classes, (
        f"[Label range error{(' @'+where) if where else ''}] "
        f"min={y_min}, max={y_max}, expected in [0,{num_classes-1}]"
    )
    return yb


@torch.no_grad()
def _quick_eval(linear, loader, max_batches: int = 5):
    if loader is None:
        return None, None
    linear.eval()
    tot_loss, tot_correct, tot_examples = 0.0, 0, 0
    C = getattr(linear, "out_features", None)
    for i, batch in enumerate(loader, start=1):
        Xb, yb = _unpack_batch(batch)

        dev = next(linear.parameters()).device
        Xb = Xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        if C is None:
            C = linear.out_features
        yb = _label_guard(yb, C, where="eval")

        logits = linear(Xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        preds = logits.argmax(dim=1)
        tot_loss += float(loss)
        tot_correct += int((preds == yb).sum())
        tot_examples += int(yb.numel())
        if i >= max_batches:
            break
    if tot_examples == 0:
        return None, None
    avg_loss = tot_loss / tot_examples
    acc = tot_correct / tot_examples
    return avg_loss, acc

def glm_saga(
    linear,
    train_loader,
    step_size,
    n_iters,
    alpha,
    epsilon,
    k,
    val_loader,
    do_zero,
    metadata,
    n_ex,
    n_classes,
    verbose=False,
    iter_callback: Optional[Callable[[int, Dict], None]] = None,
    callback_every: int = 10,
):
    
    start = time.time()
    path = []

    lam = 0.0
    try:
        lam = float(metadata.get("max_reg", {}).get("nongrouped", 0.0))
    except Exception:
        pass

    C_head = getattr(linear, "out_features", None)
    if isinstance(n_classes, int) and C_head is not None and n_classes != C_head:
        print(f"[glm_saga] WARNING: n_classes={n_classes} != linear.out_features={C_head}. Using {C_head} for label checks.")
    C = C_head if C_head is not None else int(n_classes)

    total_params = int(linear.weight.numel())
    step = 0
    train_iter = iter(train_loader)
    dev = next(linear.parameters()).device

    while step < int(n_iters):
        step += 1
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        Xb, yb = _unpack_batch(batch)

        with torch.no_grad():
            Xb = Xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            yb = _label_guard(yb, C, where="train")

            logits = linear(Xb)
            ce = F.cross_entropy(logits, yb, reduction="mean")


            l1 = linear.weight.abs().sum()
            l2 = (linear.weight**2).sum()
            disp_loss = float(ce + lam * (alpha * l1 + (1.0 - alpha) * 0.5 * l2))

        if (step % max(1, int(callback_every))) == 0 or step == n_iters:
            with torch.no_grad():
                nnz = int((linear.weight.abs() > 1e-5).sum().item())
                val_loss, val_acc = _quick_eval(linear, val_loader, max_batches=5)
                info = {
                    "loss": float(ce),
                    "lam": lam,
                    "alpha": float(alpha),
                    "time": float(time.time() - start),
                    "nnz": nnz,
                    "tot": total_params,
                }
                if val_loss is not None:
                    info["val_loss"] = float(val_loss)
                if val_acc is not None:
                    info["val_acc"] = float(val_acc)

                if iter_callback is not None:
                    try:
                        iter_callback(step, info)
                    except Exception as e:
                        print(f"[glm_saga] iter_callback error: {e}")

                path.append({
                    "loss": float(ce),
                    "lam": lam,
                    "alpha": float(alpha),
                    "time": float(time.time() - start),
                    "weight": linear.weight.data.clone().cpu(),
                    "bias": linear.bias.data.clone().cpu(),
                    "metrics": {
                        "val_loss": float(val_loss) if val_loss is not None else None,
                        "val_acc": float(val_acc) if val_acc is not None else None,
                    },
                })

    return {
        "path": path if path else [{
            "weight": linear.weight.data.clone().cpu(),
            "bias": linear.bias.data.clone().cpu(),
            "lr": step_size, "lam": lam, "alpha": float(alpha),
            "time": float(time.time() - start),
            "metrics": {"val_acc": None, "val_loss": None},
        }],
        "weight": linear.weight.data.clone().cpu(),
        "bias": linear.bias.data.clone().cpu(),
    }
