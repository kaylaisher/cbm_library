# cbm_library/models/label_free_cbm.py - WORKING VERSION
"""
Minimal working Label-Free CBM for training reproduction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import clip
import numpy as np
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from .base_cbm import BaseCBM
from ..training.final_layer import UnifiedFinalTrainer, get_label_free_cbm_config
from ..utils.logging import setup_enhanced_logging

logger = setup_enhanced_logging(__name__)


class LabelFreeCBM(BaseCBM):
    """Minimal working Label-Free CBM implementation"""
    
    def __init__(self, backbone: nn.Module, num_concepts: int, num_classes: int, 
                 device: str = "cuda", config=None):
        super().__init__(backbone, num_concepts, num_classes, device, config)
        
        # Load CLIP for concept similarity
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Get backbone feature dimension
        self.feature_dim = self._get_feature_dimension()
        
        logger.info(f"LabelFreeCBM initialized: {self.feature_dim} -> {num_concepts} -> {num_classes}")
    
    def _get_feature_dimension(self) -> int:
        """Determine backbone feature dimension"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            features = self.extract_features(dummy_input)
            return features.shape[1]
    
    def train_concept_layer(self, dataset, concepts: List[str], config: Dict[str, Any]) -> torch.Tensor:
        """
        Train Label-Free CBM concept layer using cos³ similarity
        
        Args:
            dataset: PyTorch dataset
            concepts: List of concept strings
            config: Training configuration
            
        Returns:
            Concept activations tensor [N, num_concepts]
        """
        logger.info(f"Training concept layer with {len(concepts)} concepts")
        
        # Extract backbone features from dataset
        logger.info("Extracting backbone features...")
        backbone_features = self._extract_dataset_features(dataset)
        
        # Extract CLIP text features for concepts
        logger.info("Extracting CLIP concept features...")
        clip_features = self._extract_clip_concept_features(concepts)
        
        # Train projection layer W_c using cos³ similarity
        logger.info("Training projection layer...")
        W_c = self._train_projection_layer(backbone_features, clip_features, config)
        
        # Create and store concept layer
        self.concept_layer = nn.Linear(self.feature_dim, len(concepts), bias=False).to(self.device)
        self.concept_layer.weight.data = W_c
        
        # Store concept names
        self.concept_names = concepts
        
        # Compute concept activations for the dataset
        with torch.no_grad():
            concept_activations = self.concept_layer(backbone_features)
        
        logger.info(f"Concept layer training complete. Activations shape: {concept_activations.shape}")
        return concept_activations
    
    def _extract_dataset_features(self, dataset) -> torch.Tensor:
        """Extract backbone features from entire dataset"""
        features = []
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        
        self.backbone.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, _) in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    logger.debug(f"Processing batch {batch_idx}/{len(dataloader)}")
                
                batch_x = batch_x.to(self.device)
                batch_features = self.extract_features(batch_x, normalize=False)
                features.append(batch_features.cpu())
        
        all_features = torch.cat(features, dim=0).to(self.device)
        logger.info(f"Extracted features shape: {all_features.shape}")
        return all_features
    
    def _extract_clip_concept_features(self, concepts: List[str]) -> torch.Tensor:
        """Extract CLIP text features for concepts"""
        text_tokens = clip.tokenize(concepts).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logger.info(f"CLIP concept features shape: {text_features.shape}")
        return text_features
    
    def _train_projection_layer(self, backbone_features: torch.Tensor, 
                               clip_features: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """Train projection layer W_c using cos³ similarity loss"""
        
        d0 = backbone_features.shape[1]  # Backbone feature dimension
        M = clip_features.shape[0]       # Number of concepts
        
        # Initialize projection weights
        W_c = torch.randn(M, d0, requires_grad=True, device=self.device)
        
        # Setup optimizer
        optimizer = optim.Adam([W_c], lr=config.get('learning_rate', 0.001))
        
        # Training parameters
        max_steps = config.get('proj_steps', 1000)
        log_interval = max(max_steps // 10, 1)
        
        logger.info(f"Training projection: {M} concepts x {d0} features for {max_steps} steps")
        
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # Project backbone features: [M, N] = [M, d0] @ [d0, N]
            projected_features = torch.mm(W_c, backbone_features.T)
            
            # Compute cos³ similarity loss
            loss = self._compute_cos_cubed_loss(projected_features, clip_features)
            
            loss.backward()
            optimizer.step()
            
            if step % log_interval == 0:
                logger.info(f"Step {step}/{max_steps}: Loss = {loss.item():.6f}")
        
        logger.info(f"Projection training completed. Final loss: {loss.item():.6f}")
        return W_c.detach()
    
    def _compute_cos_cubed_loss(self, projected_features: torch.Tensor, 
                               clip_features: torch.Tensor) -> torch.Tensor:
        """Compute cos³ similarity loss as in Label-Free CBM paper"""
        
        # Normalize projected features: [M, N]
        proj_norm = self._normalize_features(projected_features)
        
        # Normalize clip features: [M, clip_dim] 
        clip_norm = self._normalize_features(clip_features)
        
        # Cube the normalized features
        proj_cubed = torch.sign(proj_norm) * torch.abs(proj_norm) ** 3
        clip_cubed = torch.sign(clip_norm) * torch.abs(clip_norm) ** 3
        
        # Compute cosine similarity between cubed features
        # We need to compute similarity between proj_cubed[i] and clip_cubed[i] for each concept i
        similarities = []
        for i in range(projected_features.shape[0]):  # For each concept
            proj_concept = proj_cubed[i]  # [N] - projected features for concept i
            clip_concept = clip_cubed[i]  # [clip_dim] - CLIP features for concept i
            
            # For each data point, compute similarity with the concept
            proj_concept_norm = proj_concept / (torch.norm(proj_concept) + 1e-8)
            clip_concept_norm = clip_concept / (torch.norm(clip_concept) + 1e-8)
            
            # Average similarity across all data points for this concept
            sim = torch.mean(proj_concept_norm) * torch.mean(clip_concept_norm)
            similarities.append(sim)
        
        # Return negative mean similarity as loss
        total_similarity = torch.stack(similarities).mean()
        return -total_similarity
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to unit norm"""
        return features / (torch.norm(features, dim=-1, keepdim=True) + 1e-8)
    
    def complete_training(self, dataset, concepts: List[str], 
                         concept_config: Dict[str, Any], final_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete end-to-end training: concept layer + final layer
        
        Args:
            dataset: Training dataset  
            concepts: List of concept strings
            concept_config: Config for concept layer training
            final_config: Config for final layer training
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting complete Label-Free CBM training")
        
        # Step 1: Train concept layer
        concept_activations = self.train_concept_layer(dataset, concepts, concept_config)
        
        # Step 2: Get labels from dataset
        labels = []
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        for _, batch_labels in dataloader:
            labels.append(batch_labels)
        labels = torch.cat(labels, dim=0).to(self.device)
        
        # Step 3: Train final layer using your UnifiedFinalTrainer
        logger.info("Training final layer...")
        trainer = UnifiedFinalTrainer()
        
        # Use your existing config system
        final_layer_config = get_label_free_cbm_config(
            num_concepts=len(concepts),
            num_classes=self.num_classes,
            device=self.device,
            **final_config
        )
        
        final_result = trainer.train(concept_activations, labels, final_layer_config)
        
        # Step 4: Create and store final layer
        self.final_layer = trainer.create_final_layer(final_layer_config, final_result)
        self.concept_mean = final_result['concept_mean']
        self.concept_std = final_result['concept_std']
        
        # Update training state
        self.is_trained = True
        
        logger.info("✅ Complete Label-Free CBM training finished")
        
        return {
            'concept_activations': concept_activations,
            'final_layer_result': final_result,
            'num_concepts': len(concepts),
            'concept_names': concepts
        }
    
    def forward(self, x: torch.Tensor, return_concepts: bool = False):
        """Forward pass matching original CBM format"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call complete_training() first.")
        
        # Extract features
        features = self.extract_features(x, normalize=False)
        
        # Get concept activations
        concepts = self.concept_layer(features)
        
        # Normalize concepts
        if self.concept_mean is not None and self.concept_std is not None:
            concepts_norm = (concepts - self.concept_mean) / (self.concept_std + 1e-8)
        else:
            concepts_norm = concepts
        
        # Final prediction
        logits = self.final_layer(concepts_norm)
        
        if return_concepts:
            return logits, concepts_norm
        return logits
    
    def save_model(self, save_dir: str):
        """Save model in format compatible with original repository"""
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save weights in original format
        torch.save(self.concept_layer.weight.data, os.path.join(save_dir, 'W_c.pt'))
        torch.save(self.final_layer.weight.data, os.path.join(save_dir, 'W_g.pt'))
        torch.save(self.final_layer.bias.data, os.path.join(save_dir, 'b_g.pt'))
        torch.save(self.concept_mean, os.path.join(save_dir, 'proj_mean.pt'))
        torch.save(self.concept_std, os.path.join(save_dir, 'proj_std.pt'))
        
        # Save concept names
        with open(os.path.join(save_dir, 'concepts.txt'), 'w') as f:
            for concept in self.concept_names:
                f.write(concept + '\n')
        
        # Save args in original format
        args_dict = {
            'backbone': 'resnet18',  # Adjust based on actual backbone
            'dataset': 'custom',
            'num_concepts': self.num_concepts,
            'num_classes': self.num_classes
        }
        
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")


# Usage example for reproduction
def reproduce_training_example():
    """Example of how to reproduce training with the minimal implementation"""
    
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Create backbone (example with ResNet18)
    from torchvision import models
    backbone = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC layer
    
    # 3. Create CBM
    cbm = LabelFreeCBM(
        backbone=backbone,
        num_concepts=50,
        num_classes=10,
        device=device
    )
    
    # 4. Prepare concepts (example)
    concepts = [
        "red color", "blue color", "green color", "round shape", "square shape",
        "small size", "large size", "metallic texture", "smooth surface", "rough texture"
        # ... add more concepts
    ]
    
    # 5. Training configs
    concept_config = {
        'learning_rate': 0.001,
        'proj_steps': 1000
    }
    
    final_config = {
        'sparsity_lambda': 0.0007,
        'normalize_concepts': True
    }
    
    # 6. Train (assuming you have a dataset)
    # dataset = YourDataset()  # Implement your dataset
    # result = cbm.complete_training(dataset, concepts, concept_config, final_config)
    
    # 7. Save model
    # cbm.save_model("./saved_models/test_cbm")
    
    print("✅ Training reproduction setup complete")

if __name__ == "__main__":
    reproduce_training_example()