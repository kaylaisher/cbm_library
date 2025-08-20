class CBLTrainerFactory:
    @staticmethod
    def create(method: str) -> CBLTrainer:
        if method == "label_free":
            return LabelFreeCBLTrainer()
        elif method == "vlg":
            return VLGCBLTrainer()
        elif method == "cb_llm":
            return CBLLMTrainer()
        else:
            raise ValueError(f"Unknown method: {method}")

class LabelFreeCBLTrainer(CBLTrainer):
    """Cos³ similarity training for Label-Free CBM"""
    
    def train(self, backbone_features, concepts, config):
        # Extract CLIP features for concepts
        clip_features = self._extract_clip_features(concepts)
        
        # Initialize projection weights W_c
        W_c = self._initialize_projection(backbone_features.shape[1], len(concepts))
        
        # Train with cos³ similarity loss
        W_c = self._train_projection(backbone_features, clip_features, W_c, config)
        
        # Create concept layer and compute activations
        concept_layer = nn.Linear(backbone_features.shape[1], len(concepts), bias=False)
        concept_layer.weight.data = W_c
        
        concept_activations = concept_layer(backbone_features)
        return concept_activations, concept_layer

class VLGCBLTrainer(CBLTrainer):
    """Multi-label classification training for VLG-CBM"""
    
    def train(self, backbone_features, concept_annotations, config):
        # Use concept annotations (not just names)
        # Train with BCE loss for multi-label prediction
        pass

class CBLLMTrainer(CBLTrainer):
    """Sentence embedding training for CB-LLM"""
    
    def train(self, text_features, concepts, config):
        # Use sentence embeddings instead of visual features
        # Automatic Concept Correction (ACC)
        pass