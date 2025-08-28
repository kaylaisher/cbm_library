from __future__ import annotations
import os, json, operator
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from cbm_library.utils import data_utils
from cbm_library.utils.glm_saga import glm_saga
from cbm_library.utils.logging import setup_enhanced_logging
from cbm_library.models.final_layer import FinalLayer  
logger = setup_enhanced_logging("vlg_cbm")


class Backbone(nn.Module):
    def __init__(self, backbone_name: str, feature_layer: str, device: str = "cuda"):
        super().__init__()
        target_model, target_preprocess = data_utils.get_target_model(backbone_name, device)

        self._feature_vals = {}
        self._hook_handle = None

        def hook(module, _inp, output):
            self._feature_vals[output.device] = output

        submod = operator.attrgetter(feature_layer)(target_model)
        self._hook_handle = submod.register_forward_hook(hook)

        self.backbone = target_model
        self.preprocess = target_preprocess

        self.feature_layer = feature_layer
        try:
            with torch.no_grad():
                pdev = next(target_model.parameters()).device
                dummy = torch.randn(1, 3, 224, 224, device=pdev)
                _ = target_model(dummy)               
                feat = self._feature_vals.get(pdev)
                if feat is None:
                    raise RuntimeError(f"Failed to probe feature_layer='{feature_layer}'")
                self.output_dim = feat.shape[1] if feat.ndim == 4 else feat.view(1, -1).shape[1]
        except Exception as e:
            raise RuntimeError(f"Could not infer output_dim via hook for layer='{feature_layer}': {e}")
    
    def forward(self, x: torch.Tensor):
        _ = self.backbone(x)  
        dev = x.device
        feat = self._feature_vals.get(dev)
        if feat is None:
            raise RuntimeError(
                f"Hook did not run; check feature_layer='{getattr(self, 'feature_layer', '?')}' and model forward."
            )
        # Support both 4D maps (N,C,H,W) and already-pooled 2D features (N,C)
        if feat.ndim == 4:
            return feat.mean(dim=(2, 3))   # GAP → [N, C]
        if feat.ndim == 2:
            return feat                    # already [N, C]
        raise RuntimeError(f"Unsupported feature tensor shape: {tuple(feat.shape)}")
    

    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))
    
    @classmethod
    def from_pretrained(
        cls,
        load_dir: str,
        *,
        backbone_name: str,
        feature_layer: str,
        device: str = "cuda",
    ):
        model = cls(backbone_name, feature_layer, device)
        state = torch.load(os.path.join(load_dir, "backbone.pt"), map_location=device)
        model.backbone.load_state_dict(state)
        return model

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

class BackboneCLIP(nn.Module):
    def __init__(self, backbone_name: str, use_penultimate=True, device="cuda"):
        super().__init__()
        import clip
        model, preprocess = clip.load(backbone_name[5:], device=device)
        visual = model.visual
        self.preprocess = preprocess

        if use_penultimate:
            N = visual.attnpool.c_proj.in_features
            identity = nn.Linear(N, N, bias=True, device=device)
            nn.init.zeros_(identity.bias)
            with torch.no_grad():
                identity.weight.copy_(torch.eye(N, device=device))
            visual.attnpool.c_proj = identity
            self.output_dim = N
        else:
            if hasattr(visual.attnpool, "c_proj") and hasattr(visual.attnpool.c_proj, "out_features"):
                self.output_dim = int(visual.attnpool.c_proj.out_features)
            elif hasattr(visual, "output_dim"):
                self.output_dim = int(visual.output_dim)
            else:
                self.output_dim = int(visual.attnpool.c_proj.in_features)

        self.backbone = visual.float()

    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone_clip.pt"))

    @classmethod
    def from_pretrained(cls, load_dir: str, backbone_name: str, use_penultimate: bool = True, device: str = "cuda"):
        model = cls(backbone_name, use_penultimate=use_penultimate, device=device)
        path = os.path.join(load_dir, "backbone_clip.pt")
        if os.path.exists(path):
            sd = torch.load(path, map_location=device)
            model.backbone.load_state_dict(sd, strict=False)
        return model


class ConceptLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_hidden: int = 0, bias: bool = True, device: str = "cuda"):
        super().__init__()
        layers = [nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(num_hidden):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(out_features, out_features, bias=bias))
        self.model = nn.Sequential(*layers).to(device)
        self.out_features = out_features

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))
  
    
    @classmethod
    def from_pretrained(cls, load_dir: str, cfg, device: str = "cuda"):
        if cfg.backbone.startswith("clip_"):
            tmp_backbone = BackboneCLIP(cfg.backbone,
                                        getattr(cfg, "use_clip_penultimate", True),
                                        device=device)
        else:
            tmp_backbone = Backbone(cfg.backbone, cfg.feature_layer, device=device)
        encoder_dim = int(tmp_backbone.output_dim)

        concepts_path = os.path.join(load_dir, "concepts.txt")
        concepts = [c for c in data_utils.get_concepts(concepts_path) if str(c).strip()]
        num_concepts = len(concepts)

        model = cls(encoder_dim, num_concepts,
                    num_hidden=getattr(cfg, "cbl_hidden_layers", 0),
                    device=device)
        model.load_state_dict(torch.load(os.path.join(load_dir, "cbl.pt"), map_location=device))
        return model


class NormalizationLayer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cuda"):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device).clamp_min(1e-6)

    def forward(self, x):
        return (x - self.mean) / self.std

    def save_model(self, save_dir):
        torch.save(self.mean, os.path.join(save_dir, "train_concept_features_mean.pt"))
        torch.save(self.std, os.path.join(save_dir, "train_concept_features_std.pt"))

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cuda"):
        mean = torch.load(os.path.join(load_dir, "train_concept_features_mean.pt"), map_location=device)
        std = torch.load(os.path.join(load_dir, "train_concept_features_std.pt"), map_location=device)
        return cls(mean, std, device=device)

class VLGCBM(nn.Module):
    def __init__(self, backbone: nn.Module, cbl: nn.Module, normalization: nn.Module, final_layer: nn.Module, device: str = "cuda"):
        super().__init__()
        self.backbone = backbone
        self.cbl = cbl
        self.normalization = normalization
        self.final_layer = final_layer
        self.device = device
        self.to(device)
        self.feature_vals = {}

    def forward(self, x):
        emb = self.backbone(x)
        concepts = self.cbl(emb)
        norm_c = self.normalization(concepts)
        logits = self.final_layer(norm_c)
        return logits, norm_c


    @classmethod
    def from_config(cls, cfg) -> "VLGCBM":
        """
        Build VLGCBM from a VLGCBMConfig-like object with fields:
          - backbone (e.g., "resnet50" or "clip_ViT-B/32" style as "clip_ViT-B/32")
          - feature_layer (e.g., "layer4") [for non-CLIP]
          - use_clip_penultimate (bool)
          - device (e.g., "cuda")
          - cbl_hidden_layers (int)
          - num_concepts (int), num_classes (int)  [these should be filled by your script after reading concepts/classes]
        """
        device = getattr(cfg, "device", "cuda")

        if cfg.backbone.startswith("clip_"):
            backbone = BackboneCLIP(cfg.backbone, use_penultimate=getattr(cfg, "use_clip_penultimate", True), device=device)
            encoder_dim = backbone.output_dim
        else:
            backbone = Backbone(cfg.backbone, cfg.feature_layer, device=device)
            encoder_dim = backbone.output_dim

        cbl = ConceptLayer(encoder_dim, cfg.num_concepts, num_hidden=getattr(cfg, "cbl_hidden_layers", 0), device=device)
        # default to identity normalization; your training pipeline should set real stats later
        norm = NormalizationLayer(torch.zeros(cfg.num_concepts), torch.ones(cfg.num_concepts), device=device)
        final = FinalLayer(cfg.num_concepts, cfg.num_classes, device=device)
        return cls(backbone, cbl, norm, final, device=device)

    def save(self, save_dir: str, args_dict: dict | None = None):
        os.makedirs(save_dir, exist_ok=True)
        if args_dict is not None:
            with open(os.path.join(save_dir, "args.txt"), "w") as f:
                json.dump(args_dict, f, indent=2)
        if hasattr(self.backbone, "save_model"):
            self.backbone.save_model(save_dir)
        self.cbl.save_model(save_dir)
        self.normalization.save_model(save_dir)
        self.final_layer.save_model(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, "vlg_cbm_full.pt"))

   
    @classmethod
    def from_pretrained(cls, load_dir: str, cfg, device: str = "cuda"):
        if cfg.backbone.startswith("clip_"):
            backbone = BackboneCLIP.from_pretrained(load_dir, cfg.backbone,
                                                    getattr(cfg, "use_clip_penultimate", False), device)
        else:
            backbone = Backbone.from_pretrained(load_dir, device, backbone_name=cfg.backbone,
                                                feature_layer=cfg.feature_layer)
        cbl = ConceptLayer.from_pretrained(load_dir, cfg, device)
        normalization = NormalizationLayer.from_pretrained(load_dir, device)
        final_layer = FinalLayer.from_pretrained(load_dir, device)
        model = cls(backbone, cbl, normalization, final_layer, device)
        ckpt_path = os.path.join(load_dir, "vlg_cbm_full.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model


    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        logits, _ = self.forward(x.to(self.device))
        return logits.argmax(dim=1)

    @torch.inference_mode()
    def encode_concepts(self, x: torch.Tensor):
        _, concepts = self.forward(x.to(self.device))
        return concepts

    @torch.inference_mode()
    def evaluate_top1(self, loader: DataLoader) -> float:
        self.eval()
        correct, total = 0, 0
        for images, _, targets in tqdm(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            logits, _ = self.forward(images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
        return correct / max(total, 1)


# -----------------
# Training + Eval
# -----------------

def per_class_accuracy(model: nn.Module, loader: DataLoader, classes: list[str], device: str = "cuda"):
    correct = torch.zeros(len(classes)).to(device)
    total = torch.zeros(len(classes)).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for features, _, targets in tqdm(loader):
            features = features.to(device)
            targets = targets.to(device)
            logits, _ = model(features)
            preds = logits.argmax(dim=1)
            for pred, target in zip(preds, targets):
                total[target] += 1
                if pred == target:
                    correct[target] += 1
   
    den = torch.where(total == 0, torch.ones_like(total), total)
    per_class_acc = (correct / den) * 100.0

    overall_den = total.sum().clamp_min(1)
    overall_acc = (correct.sum() / overall_den) * 100.0

    return {
        "Per class accuracy": {classes[i]: f"{per_class_acc[i]:.2f}" for i in range(len(classes))},
        "Overall accuracy": f"{overall_acc:.2f}",
        "Datapoints": int(total.sum().item()),
    }



def test_model(loader: DataLoader, backbone: Backbone, cbl: ConceptLayer, normalization: NormalizationLayer, final_layer: FinalLayer, device="cuda"):
    acc_sum = 0
    with torch.no_grad():
        for features, _, targets in tqdm(loader):
            features, targets = features.to(device), targets.to(device)
            emb = backbone(features)
            concept_logits = cbl(emb)
            concept_probs = normalization(concept_logits)
            logits = final_layer(concept_probs)
            preds = logits.argmax(dim=1)
            acc_sum += (preds == targets).sum().item()
    return acc_sum / len(loader.dataset)

def train_cbl(backbone: Backbone, cbl: ConceptLayer, train_loader, val_loader, epochs: int,
              loss_fn, lr=1e-3, weight_decay=1e-5, concepts=None, tb_writer=None, device="cuda",
              finetune=False, optimizer="sgd", scheduler=None, backbone_lr=1e-3,
              data_parallel=False):

    if optimizer == "sgd":
        opt = torch.optim.SGD(cbl.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        opt = torch.optim.Adam(cbl.parameters(), lr=lr, weight_decay=weight_decay)
    if finetune:
        opt.add_param_group({"params": backbone.parameters(), "lr": backbone_lr})
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = None

    logger.info(" [CBL] start training")
    best_val, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(epochs):
        cbl.train()
        running = 0.0
        for features, concept_targets, _ in train_loader:
            features = features.to(device)
            concept_targets = concept_targets.to(device)

            if finetune:
                emb = backbone(features)
            else:
                with torch.no_grad():
                    emb = backbone(features)

            logits = cbl(emb)
            loss = loss_fn(logits, concept_targets)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        val_loss = validate_cbl(backbone, cbl, val_loader, loss_fn, device)

        logger.info(f"[CBL] epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val, best_state, best_epoch = val_loss, cbl.state_dict(), epoch
            msg = f"[CBL] best_val={best_val:.6f} epoch={best_epoch:04d} "
            if hasattr(logger, "success"):
                logger.success(msg)   
            else:
                logger.info(msg)     

        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
        if sched: sched.step()

    if best_state: cbl.load_state_dict(best_state)
    logger.info(" [CBL] done")
    return cbl, backbone


def validate_cbl(backbone: Backbone, cbl: ConceptLayer, val_loader, loss_fn, device="cuda"):
    val_loss = 0
    cbl.eval()
    with torch.no_grad():
        for features, concept_targets, _ in tqdm(val_loader):
            features, concept_targets = features.to(device), concept_targets.to(device)
            emb = backbone(features)
            logits = cbl(emb)
            val_loss += loss_fn(logits, concept_targets).item()
    return val_loss/len(val_loader)


def train_sparse_final(linear: nn.Linear, train_loader, val_loader, n_iters, lam, step_size=0.1, device="cuda"):
    num_classes = linear.weight.shape[0]
    linear.weight.data.zero_(); linear.bias.data.zero_()
    metadata = {"max_reg": {"nongrouped": lam}}
    return glm_saga(linear, train_loader, step_size, n_iters, 0.99, epsilon=1, k=1, val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(train_loader.dataset), n_classes=num_classes, verbose=True)


def train_dense_final(model: nn.Linear, train_loader, val_loader, n_iters, lr=1e-3, device="cuda"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    ce = nn.CrossEntropyLoss()
    for epoch in range(n_iters):
        for inputs, targets, _ in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = ce(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return {"path": [{"weight": model.weight, "bias": model.bias, "lr": lr, "lam": -1, "alpha": -1, "time": -1, "metrics": {}}]}
