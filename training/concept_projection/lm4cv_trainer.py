"""
LM4CV Learning-to-Search Concept Projection Trainer
Implements dictionary learning and attribute selection from ICCV 2023 paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import clip

logger = logging.getLogger(__name__)

@dataclass
class LM4CVProjectionConfig:
    """Configuration for LM4CV concept projection training"""
    
    # Attribute selection parameters
    num_attributes: int = 32           # K - number of attributes to select
    clip_model_name: str = "ViT-B/32"
    
    # Learning parameters
    learning_rate: float = 0.01
    max_epochs: int = 5000
    batch_size: int = 4096
    early_stopping: bool = True
    patience: int = 100
    
    # Regularization parameters
    mahalanobis_lambda: float = 0.01   # λ for Mahalanobis regularization
    division_power: float = 1.0        # Control strength of Mahalanobis constraints
    
    # Initialization
    reinit: bool = True                # Initialize with image feature weights
    
    # Device
    device: str = "cuda"

class MahalanobisRegularizer:
    """
    Mahalanobis distance regularization for LM4CV
    Encourages learned embeddings to stay in CLIP text embedding space
    """
    
    def __init__(self, attribute_embeddings: torch.Tensor, division_power: float = 1.0):
        """
        Args:
            attribute_embeddings: [N, D] CLIP embeddings of all candidate attributes
            division_power: Control regularization strength
        """
        self.division_power = division_power
        
        # Compute statistics of attribute embedding distribution
        self.mu = attribute_embeddings.mean(dim=0)  # [D]
        
        # Compute covariance matrix
        centered = attribute_embeddings - self.mu
        self.cov = torch.mm(centered.T, centered) / (attribute_embeddings.shape[0] - 1)
        
        # Add small regularization for numerical stability
        self.cov_inv = torch.inverse(self.cov + 1e-6 * torch.eye(self.cov.shape[0], device=self.cov.device))
        
    def compute_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance loss for embeddings
        
        Args:
            embeddings: [K, D] learned dictionary embeddings
            
        Returns:
            Mahalanobis distance loss
        """
        
        # Center embeddings
        centered_emb = embeddings - self.mu.unsqueeze(0)  # [K, D]
        
        # Compute Mahalanobis distance for each embedding
        # D_mah = sqrt((x - μ)^T Σ^(-1) (x - μ))
        maha_distances = []
        
        for i in range(embeddings.shape[0]):
            centered_vec = centered_emb[i:i+1]  # [1, D]
            
            # Compute (x - μ)^T Σ^(-1) (x - μ)
            quad_form = torch.mm(
                torch.mm(centered_vec, self.cov_inv),
                centered_vec.T
            ).squeeze()
            
            maha_distances.append(torch.sqrt(quad_form + 1e-8))
        
        maha_distance = torch.stack(maha_distances).mean()
        
        # Apply division power to control regularization strength
        return maha_distance / self.division_power

class LM4CVLearningToSearch:
    """
    LM4CV Learning-to-Search Dictionary Optimization
    
    Implements the core algorithm from "Learning Concise and Descriptive Attributes for Visual Recognition"
    """
    
    def __init__(self, config: LM4CVProjectionConfig):
        self.config = config
        self.device = config.device
        
        # Load CLIP model
        self.clip_model, _ = clip.load(config.clip_model_name, device=config.device)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def train_dictionary(self, 
                        candidate_attributes: List[str],
                        images: torch.Tensor,
                        labels: torch.Tensor,
                        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Train learnable dictionary E to approximate best K attributes
        
        Args:
            candidate_attributes: List of candidate attribute strings
            images: [N, 3, H, W] input images
            labels: [N] class labels
            validation_data: Optional (val_images, val_labels) for early stopping
            
        Returns:
            E: Learned dictionary [K, D]
        """
        
        logger.info(f"Training LM4CV dictionary to select {self.config.num_attributes} attributes from {len(candidate_attributes)} candidates")
        
        # Get CLIP embeddings for all candidate attributes
        attribute_embeddings = self._encode_attributes(candidate_attributes)  # [N, D]
        N, D = attribute_embeddings.shape
        K = self.config.num_attributes
        
        # Initialize learnable dictionary E ∈ R^(K×D)
        E = self._initialize_dictionary(attribute_embeddings, K, D)
        
        # Setup optimization
        optimizer = optim.Adam([E], lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8)
        
        # Setup regularization
        maha_regularizer = MahalanobisRegularizer(
            attribute_embeddings, 
            division_power=self.config.division_power
        )
        
        # Setup data loaders
        train_loader = self._create_dataloader(images, labels, shuffle=True)
        val_loader = None
        if validation_data is not None:
            val_images, val_labels = validation_data
            val_loader = self._create_dataloader(val_images, val_labels, shuffle=False)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.config.max_epochs):
            # Training step
            train_loss, train_acc = self._train_epoch(
                E, optimizer, train_loader, maha_regularizer
            )
            train_losses.append(train_loss)
            
            # Validation step
            if val_loader is not None:
                val_acc = self._validate_epoch(E, val_loader)
                val_accuracies.append(val_acc)
                scheduler.step(val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_E = E.clone()
                else:
                    patience_counter += 1
                    
                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    E = best_E
                    break
                    
                if epoch % 100 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            else:
                if epoch % 100 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        logger.info(f"Dictionary training complete.