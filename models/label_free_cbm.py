import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import clip
import numpy as np
from .base_cbm import BaseCBM

class LabelFreeCBM(BaseCBM):
    """
    Label-free CBM implementation
    Based on: "Label-free Concept Bottleneck Models" (ICLR 2023)
    """
    
    def __init__(self, backbone: nn.Module, num_concepts: int, num_classes: int, device: str = "cuda"):
        super().__init__(backbone, num_concepts, num_classes, device)
        
        # Load CLIP model for concept activation
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Concept projection layer
        self.projection_layer = None
        
    def train_concept_layer(self, dataset, concepts: List[str], config: Dict[str, Any]) -> torch.Tensor:
        """
        Train Label-free CBM concept projection layer using cos cubed similarity
        """
        logger.info(f"Training Label-free CBM concept layer with {len(concepts)} concepts")
        
        # Extract backbone features and CLIP features
        backbone_features = self._extract_backbone_features(dataset)
        clip_features = self._extract_clip_features(concepts)
        
        # Initialize projection weights W_c
        d0 = backbone_features.shape[1]  # Backbone feature dimension
        M = len(concepts)  # Number of concepts
        
        W_c = torch.randn(M, d0, requires_grad=True, device=self.device)
        optimizer = optim.Adam([W_c], lr=config.get('learning_rate', 0.001))
        
        # Training loop
        for step in range(config.get('max_steps', 1000)):
            optimizer.zero_grad()
            
            # Forward pass: project features
            projected_features = torch.mm(W_c, backbone_features.T)  # M x N
            
            # Compute cos cubed similarity loss
            loss = self._compute_cos_cubed_loss(projected_features, clip_features)
            
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}")
        
        # Create concept layer
        self.concept_layer = nn.Linear(d0, M, bias=False).to(self.device)
        self.concept_layer.weight.data = W_c.detach()
        
        # Store concept names
        self.concept_names = concepts
        self.num_concepts = len(concepts)
        
        # Compute concept activations for the dataset
        with torch.no_grad():
            concept_activations = self.concept_layer(backbone_features)
        
        logger.info("Label-free CBM concept layer training complete")
        return concept_activations
    
    def _extract_backbone_features(self, dataset):
        """Extract features using the backbone model"""
        features = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        
        self.backbone.eval()
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                feat = self.get_backbone_features(batch_x)
                features.append(feat)
        
        return torch.cat(features, dim=0)
    
    def _extract_clip_features(self, concepts: List[str]):
        """Extract CLIP text features for concepts"""
        text_tokens = clip.tokenize(concepts).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def _compute_cos_cubed_loss(self, projected_features, clip_features):
        """
        Compute cos cubed similarity loss as in Label-free CBM paper
        """
        # Normalize features
        proj_norm = self._normalize_features(projected_features)
        clip_norm = self._normalize_features(clip_features)
        
        # Cube the normalized features
        proj_cubed = torch.sign(proj_norm) * torch.abs(proj_norm) ** 3
        clip_cubed = torch.sign(clip_norm) * torch.abs(clip_norm) ** 3
        
        # Compute cosine similarity
        similarity = torch.sum(proj_cubed * clip_cubed.T, dim=1) / (
            torch.norm(proj_cubed, dim=1) * torch.norm(clip_cubed.T, dim=0)
        )
        
        # Return negative similarity as loss
        return -torch.mean(similarity)
    
    def _normalize_features(self, features):
        """Normalize features to unit norm"""
        return features / (torch.norm(features, dim=-1, keepdim=True) + 1e-8)