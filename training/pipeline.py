class CBMTrainingPipeline:
    """Unified pipeline: Concepts → Trained CBM"""
    
    def __init__(self, method: str, config: CBMConfig):
        self.method = method
        self.config = config
        
        # Pipeline stages
        self.feature_extractor = FeatureExtractor()
        self.cbl_trainer = CBLTrainerFactory.create(method)
        self.final_trainer = UnifiedFinalTrainer()
        
    def train(self, dataset, concepts: List[str]) -> TrainedCBM:
        # Stage 1: Extract backbone features
        features = self.feature_extractor.extract(dataset, self.config.backbone)
        
        # Stage 2: Train Concept Bottleneck Layer (method-specific)
        concept_activations = self.cbl_trainer.train(
            features, concepts, self.config.cbl_config
        )
        
        # Stage 3: Train Final Layer (unified)
        final_result = self.final_trainer.train(
            concept_activations, dataset.labels, self.config.final_config
        )
        
        # Stage 4: Assemble trained model
        return self._create_trained_model(final_result)