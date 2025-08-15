class LM4CVCBM(BaseCBM):
    """
    LM4CV CBM Model - Learning Concise and Descriptive Attributes
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 num_classes: int,
                 num_attributes: int = 32,
                 clip_model_name: str = "ViT-B/32",
                 device: str = "cuda"):
        
        super().__init__(backbone, num_attributes, num_classes, device)
        
        # LM4CV-specific parameters
        self.num_attributes = num_attributes
        self.clip_model_name = clip_model_name
        
        # Load CLIP model for semantic projection
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        
        # LM4CV uses CLIP as backbone
        self.backbone = self.clip_model.visual
        
        # Learning-to-search trainer
        self.projection_config = LM4CVProjectionConfig(
            num_attributes=num_attributes,
            clip_model_name=clip_model_name,
            device=device
        )
        self.lm4cv_trainer = LM4CVLearningToSearch(self.projection_config)
        
        # Store selected attributes and their embeddings
        self.selected_attributes = []
        self.selected_embeddings = None
    
    def train_concept_layer(self,
                          dataset: Any,
                          concepts: List[str],
                          config: Dict[str, Any]) -> torch.Tensor:
        """
        Train LM4CV concept projection using learning-to-search
        """
        
        logger.info(f"Training LM4CV concept layer with {len(concepts)} candidate attributes")
        
        # Update configuration
        if 'num_attributes' in config:
            self.projection_config.num_attributes = config['num_attributes']
            self.num_attributes = config['num_attributes']
        
        # Extract images and labels
        images = self._extract_images(dataset)
        labels = torch.tensor(dataset.targets, device=self.device)
        
        # Train learnable dictionary
        learned_dictionary = self.lm4cv_trainer.train_dictionary(
            candidate_attributes=concepts,
            images=images,
            labels=labels
        )
        
        # Select K attributes using nearest neighbor search
        self.selected_attributes = self.lm4cv_trainer.select_attributes(
            E=learned_dictionary,
            candidate_attributes=concepts
        )
        
        # Update model attributes
        self.concept_names = self.selected_attributes
        self.num_concepts = len(self.selected_attributes)
        
        # Encode selected attributes
        self.selected_embeddings = self._encode_concepts(self.selected_attributes)
        
        # Create dummy concept layer (LM4CV doesn't need explicit projection)
        self.concept_layer = nn.Identity()
        
        # Compute concept activations for all images using semantic projection
        concept_activations = self._compute_concept_activations(images)
        
        logger.info(f"LM4CV concept layer training complete: {self.num_concepts} attributes selected")
        
        return concept_activations
    
    def _compute_concept_activations(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute concept activations using CLIP semantic projection
        A[i,j] = cos(V_i, T_j) where V=image features, T=attribute features
        """
        
        if self.selected_embeddings is None:
            raise ValueError("No attributes selected yet")
        
        # Get CLIP image features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarities (semantic projection)
        concept_activations = torch.mm(image_features, self.selected_embeddings.T)
        
        return concept_activations
    
    def get_concept_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get concept activations for input images"""
        return self._compute_concept_activations(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LM4CV CBM"""
        
        if self.final_layer is None:
            raise ValueError("Final layer not trained yet")
        
        # Get concept activations via semantic projection
        concept_activations = self.get_concept_activations(x)
        
        # LM4CV doesn't normalize concept activations (different from other CBMs)
        if self.concept_mean is not None and self.concept_std is not None:
            normalized_concepts = (concept_activations - self.concept_mean) / self.concept_std
        else:
            normalized_concepts = concept_activations
        
        # Final classification
        logits = self.final_layer(normalized_concepts)
        
        return logits, concept_activations