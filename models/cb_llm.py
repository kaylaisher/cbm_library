import torch
import torch.nn as nn
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from .base_cbm import BaseCBM

class CBLLM(BaseCBM):
    """
    Concept Bottleneck Large Language Model implementation
    Based on CB-LLM paper for text classification
    """
    
    def __init__(self, backbone: nn.Module, num_concepts: int, num_classes: int, device: str = "cuda"):
        super().__init__(backbone, num_concepts, num_classes, device)
        
        # CB-LLM specific components
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.concept_scores = None
        
    def train_concept_layer(self, dataset, concepts: List[str], config: Dict[str, Any]) -> torch.Tensor:
        """
        Train CB-LLM concept layer with automatic concept scoring
        """
        logger.info(f"Training CB-LLM concept layer with {len(concepts)} concepts")
        
        # Extract sentence embeddings from text data
        text_embeddings = self._extract_text_embeddings(dataset)
        
        # Automatic concept scoring
        concept_scores = self._compute_concept_scores(dataset, concepts)
        
        # Create concept bottleneck layer
        embedding_dim = text_embeddings.shape[1]
        self.concept_layer = nn.Sequential(
            nn.Linear(embedding_dim, self.num_concepts),
            nn.ReLU()
        ).to(self.device)
        
        # Train CBL
        self._train_cbl(text_embeddings, concept_scores, config)
        
        # Compute concept activations
        with torch.no_grad():
            concept_activations = self.concept_layer(text_embeddings)
        
        self.concept_names = concepts
        logger.info("CB-LLM concept layer training complete")
        return concept_activations
    
    def _extract_text_embeddings(self, dataset):
        """Extract sentence embeddings from text data"""
        texts = []
        
        # Extract text from dataset
        for i in range(len(dataset)):
            text, _ = dataset[i]
            if isinstance(text, str):
                texts.append(text)
            else:
                # Handle tokenized text
                texts.append(" ".join(text))
        
        # Get sentence embeddings
        embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)
        return embeddings.to(self.device)
    
    def _compute_concept_scores(self, dataset, concepts):
        """
        Automatic concept scoring with sentence embedding model
        """
        # Get concept embeddings
        concept_embeddings = self.sentence_model.encode(concepts, convert_to_tensor=True)
        concept_embeddings = concept_embeddings.to(self.device)
        
        # Compute similarity scores between text and concepts
        text_embeddings = self._extract_text_embeddings(dataset)
        
        # Compute cosine similarity
        similarity_scores = torch.mm(text_embeddings, concept_embeddings.T)
        
        # Apply sigmoid to get concept scores
        concept_scores = torch.sigmoid(similarity_scores)
        
        return concept_scores
    
    def _train_cbl(self, text_embeddings, concept_scores, config):
        """Train the Concept Bottleneck Layer"""
        optimizer = torch.optim.Adam(self.concept_layer.parameters(), 
                                   lr=config.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        
        for epoch in range(config.get('max_epochs', 100)):
            optimizer.zero_grad()
            
            # Forward pass
            predicted_concepts = self.concept_layer(text_embeddings)
            
            # Loss: predict concept scores
            loss = criterion(predicted_concepts, concept_scores)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def get_backbone_features(self, x):
        """Override for text data"""
        if isinstance(x, list) and isinstance(x[0], str):
            # Handle list of strings
            return self.sentence_model.encode(x, convert_to_tensor=True).to(self.device)
        else:
            # Handle tokenized input
            return self.backbone(x)