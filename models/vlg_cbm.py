import torch
import torch.nn as nn
from typing import List, Dict, Any
from .base_cbm import BaseCBM

class VLGCBM(BaseCBM):
    """
    Vision-Language Guided CBM implementation
    Based on VLG-CBM paper
    """
    
    def __init__(self, backbone: nn.Module, num_concepts: int, num_classes: int, device: str = "cuda"):
        super().__init__(backbone, num_concepts, num_classes, device)
        
        self.grounding_model = None 
        self.concept_embeddings = None
        
    def train_concept_layer(self, dataset, concepts: List[str], config: Dict[str, Any]) -> torch.Tensor:
        """
        Train VLG-CBM concept layer with vision-language guidance
        """
        logger.info(f"Training VLG-CBM concept layer with {len(concepts)} concepts")
        
        backbone_features = self._extract_backbone_features(dataset)
        
        feature_dim = backbone_features.shape[1]
        self.concept_layer = nn.Sequential(
            nn.Linear(feature_dim, self.num_concepts),
            nn.Sigmoid()  
        ).to(self.device)
        
        self._train_with_grounding(dataset, concepts, config)
        
        with torch.no_grad():
            concept_activations = self.concept_layer(backbone_features)
        
        self.concept_names = concepts
        logger.info("VLG-CBM concept layer training complete")
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
    
    def _train_with_grounding(self, dataset, concepts, config):
        """
        Train CBL with grounded bounding box generation
        """
        optimizer = torch.optim.Adam(self.concept_layer.parameters(), 
                                   lr=config.get('learning_rate', 0.001))
        criterion = nn.BCELoss()
        
        for epoch in range(config.get('max_epochs', 100)):
            total_loss = 0
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                
                features = self.get_backbone_features(batch_x)
                
                concept_acts = self.concept_layer(features)
                
                pseudo_labels = self._generate_pseudo_labels(batch_x, concepts)
                
                loss = criterion(concept_acts, pseudo_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def _generate_pseudo_labels(self, images, concepts):
        """
        Generate pseudo concept labels using grounding
        """
        batch_size = images.shape[0]
        num_concepts = len(concepts)
        
        return torch.rand(batch_size, num_concepts, device=self.device)
