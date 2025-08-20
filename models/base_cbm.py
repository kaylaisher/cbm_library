"""
Enhanced Base CBM with all improvements - COMPLETED:
- Better Error Handling
- Modular Design  
- Enhanced Logging
- Configuration Management
- Early Stopping
- Validation
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import time
import warnings
from pathlib import Path
import json
import copy
import numpy as np

# Import enhanced utilities
from ..utils.logging import setup_enhanced_logging, TrainingProgress, MetricsLogger
from ..utils.validation import ModelValidator, ValidationResult, ComprehensiveValidator
from ..utils.safety import safe_training_context, memory_management_context, SafetyMonitor, CheckpointManager
from ..training.early_stopping import EarlyStopping, ValidationConfig
from ..config.config_manager import ConfigManager, CBMBaseConfig

logger = setup_enhanced_logging(__name__)


class BaseCBM(nn.Module, ABC):
    """
    Enhanced Abstract base class for all Concept Bottleneck Models
    
    Features:
    - 🛡️ Better Error Handling: Comprehensive validation and graceful failure recovery
    - 🏗️ Modular Design: Separated concerns with reusable components
    - 📝 Enhanced Logging: Colored logging with progress tracking and metrics
    - ⚙️ Configuration Management: Flexible config with validation and history
    - ⏹️ Early Stopping: Advanced early stopping with multiple metrics
    - 🔍 Validation: Comprehensive model and data validation
    """
    
    def __init__(self, 
                 backbone: nn.Module,
                 num_concepts: int,
                 num_classes: int,
                 device: str = "cuda",
                 config: Optional[CBMBaseConfig] = None):
        super().__init__()
        
        # Enhanced logging setup
        self.logger = setup_enhanced_logging(f"CBM_{self.__class__.__name__}")
        self.logger.info(f"🏗️ Initializing {self.__class__.__name__}")
        
        # Validate and store core parameters
        self.backbone = self._validate_backbone(backbone)
        self.num_concepts = self._validate_positive_int(num_concepts, "num_concepts")
        self.num_classes = self._validate_positive_int(num_classes, "num_classes")
        self.device = device
        
        # Enhanced configuration management
        base_config = config or CBMBaseConfig(
            num_concepts=num_concepts,
            num_classes=num_classes,
            device=device
        )
        self.config_manager = ConfigManager(base_config.to_dict())
        
        # Validate initial configuration
        config_issues = base_config.validate()
        if config_issues:
            self.logger.warning(f"⚠️ Configuration issues: {config_issues}")
        
        # Core CBM components
        self.concept_layer = None  # Concept Bottleneck Layer (CBL)
        self.final_layer = None    # Final classification layer
        self.concept_mean = None   # Normalization parameters
        self.concept_std = None
        
        # Enhanced state management
        self.concept_names: List[str] = []
        self.is_trained = False
        self.training_progress = TrainingProgress()
        self.training_history = []
        
        # Validation and safety components
        self.model_validator = ModelValidator()
        self.safety_monitor = SafetyMonitor()
        self.checkpoint_manager = None
        
        # Training enhancement components
        self.early_stopping = None
        self.metrics_logger = None
        
        # Performance tracking
        self._training_start_time = None
        self._last_validation_time = None
        
        self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        self.logger.info(f"📊 Concepts: {num_concepts}, Classes: {num_classes}, Device: {device}")
    
    # =================== VALIDATION METHODS ===================
    
    def _validate_backbone(self, backbone: nn.Module) -> nn.Module:
        """Validate backbone model with comprehensive checks"""
        if backbone is None:
            raise ValueError("Backbone cannot be None")
        
        if not isinstance(backbone, nn.Module):
            raise TypeError("Backbone must be a PyTorch nn.Module")
        
        # Check if backbone has parameters
        backbone_params = sum(p.numel() for p in backbone.parameters())
        if backbone_params == 0:
            self.logger.warning("⚠️ Backbone has no parameters")
        
        # Move to device
        backbone = backbone.to(self.device)
        
        # Set to evaluation mode for feature extraction
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        
        self.logger.debug(f"✅ Backbone validated: {type(backbone).__name__} with {backbone_params:,} parameters")
        return backbone
    
    def _validate_positive_int(self, value: int, name: str) -> int:
        """Validate positive integer parameters"""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return value
    
    def _validate_final_layer_inputs(self, concept_activations: torch.Tensor, labels: torch.Tensor) -> None:
        """Validate inputs for final layer training"""
        if not isinstance(concept_activations, torch.Tensor):
            raise TypeError("concept_activations must be a torch.Tensor")
        
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor")
        
        if concept_activations.size(0) != labels.size(0):
            raise ValueError(
                f"Batch size mismatch: concept_activations ({concept_activations.size(0)}) "
                f"!= labels ({labels.size(0)})"
            )
        
        if concept_activations.size(1) != self.num_concepts:
            raise ValueError(
                f"Concept dimension mismatch: expected {self.num_concepts}, "
                f"got {concept_activations.size(1)}"
            )
        
        # Check for NaN or infinite values
        if torch.any(torch.isnan(concept_activations)):
            raise ValueError("concept_activations contains NaN values")
        
        if torch.any(torch.isinf(concept_activations)):
            raise ValueError("concept_activations contains infinite values")
        
        if torch.any(torch.isnan(labels)):
            raise ValueError("labels contains NaN values")
        
        # Check label range
        unique_labels = torch.unique(labels)
        if torch.min(unique_labels) < 0 or torch.max(unique_labels) >= self.num_classes:
            raise ValueError(
                f"Labels out of range [0, {self.num_classes-1}]: "
                f"found range [{torch.min(unique_labels)}, {torch.max(unique_labels)}]"
            )
    
    def validate_model_state(self) -> ValidationResult:
        """Comprehensive model validation with detailed reporting"""
        try:
            result = self.model_validator.validate_model_components(self)
            
            # Add CBM-specific validations
            if len(self.concept_names) != self.num_concepts:
                result.add_warning(
                    f"Concept names count ({len(self.concept_names)}) != num_concepts ({self.num_concepts})"
                )
            
            # Check training state
            if not self.is_trained:
                result.add_warning("Model is not fully trained")
            
            # Check device consistency
            devices = set()
            for name, module in [('backbone', self.backbone), ('concept_layer', self.concept_layer), ('final_layer', self.final_layer)]:
                if module is not None:
                    try:
                        device = str(next(module.parameters()).device)
                        devices.add(device)
                    except StopIteration:
                        result.add_warning(f"{name} has no parameters")
            
            if len(devices) > 1:
                result.add_issue(f"Model components on different devices: {devices}")
            
            # Add model info
            result.add_info('num_concepts', self.num_concepts)
            result.add_info('num_classes', self.num_classes)
            result.add_info('is_trained', self.is_trained)
            result.add_info('has_concept_names', len(self.concept_names) > 0)
            result.add_info('devices', list(devices))
            result.add_info('training_history_length', len(self.training_history))
            
            return result
            
        except Exception as e:
            result = ValidationResult(valid=False, issues=[f"Validation error: {e}"], warnings=[], info={})
            return result

    def train_with_pipeline(self, dataset, concepts: List[str]):
        """Use unified pipeline for training"""
        pipeline = CBMTrainingPipeline(
            method=self.method_name,
            config=self.config_manager.config
        )
        
        # Run the pipeline
        result = pipeline.train(dataset, concepts)
        
        # Update model state
        self.concept_layer = result.concept_layer
        self.final_layer = result.final_layer
        self.concept_mean = result.concept_mean
        self.concept_std = result.concept_std
        self.is_trained = True
        
        return result    
    
    # =================== ABSTRACT METHODS ===================
    
    @abstractmethod
    def train_concept_layer(self, 
                          dataset: Any, 
                          concepts: List[str],
                          config: Dict[str, Any]) -> torch.Tensor:
        """
        Train the concept bottleneck layer with enhanced error handling
        
        Args:
            dataset: Training dataset
            concepts: List of concept strings
            config: Training configuration
        
        Returns:
            Concept activations tensor [N, num_concepts]
        """
        pass
    
    # =================== TRAINING METHODS ===================
    
    def setup_training(self, 
                      experiment_name: str = None,
                      checkpoint_dir: str = None,
                      enable_early_stopping: bool = True,
                      early_stopping_config: Dict[str, Any] = None) -> None:
        """
        Setup enhanced training components
        
        Args:
            experiment_name: Name for logging and checkpoints
            checkpoint_dir: Directory for saving checkpoints
            enable_early_stopping: Whether to enable early stopping
            early_stopping_config: Configuration for early stopping
        """
        experiment_name = experiment_name or f"{self.__class__.__name__}_{int(time.time())}"
        
        # Setup metrics logger
        log_dir = self.config_manager.get('log_dir', './logs')
        self.metrics_logger = MetricsLogger(log_dir, experiment_name)
        
        # Setup checkpoint manager
        checkpoint_dir = checkpoint_dir or f"{log_dir}/checkpoints"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            max_checkpoints=self.config_manager.get('max_checkpoints', 5),
            save_frequency=self.config_manager.get('save_frequency', 10)
        )
        
        # Setup early stopping
        if enable_early_stopping:
            early_stopping_config = early_stopping_config or {}
            val_config = ValidationConfig(
                patience=early_stopping_config.get('patience', self.config_manager.get('patience', 50)),
                min_delta=early_stopping_config.get('min_delta', self.config_manager.get('min_delta', 1e-4)),
                metric_name=early_stopping_config.get('metric_name', 'loss'),
                minimize=early_stopping_config.get('minimize', True),
                **early_stopping_config
            )
            self.early_stopping = EarlyStopping(val_config)
            self.logger.info(f"🛡️ Early stopping enabled with patience={val_config.patience}")
        
        # Log model info
        model_info = {
            'model_class': self.__class__.__name__,
            'num_concepts': self.num_concepts,
            'num_classes': self.num_classes,
            'device': self.device,
            'config': self.config_manager.config
        }
        self.metrics_logger.log_model_info(model_info)
        
        self.logger.info(f"🚀 Training setup complete for {experiment_name}")
    
    def train_final_layer(self, 
                         concept_activations: torch.Tensor,
                         labels: torch.Tensor,
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced final layer training with comprehensive error handling and monitoring
        """
        with safe_training_context("Final Layer Training"):
            # Validate inputs
            self._validate_final_layer_inputs(concept_activations, labels)
            
            # Update configuration
            self.config_manager.update(**config)
            
            # Start training monitoring
            self._training_start_time = time.time()
            
            # Import here to avoid circular imports
            from ..training.final_layer import UnifiedFinalTrainer, get_label_free_cbm_config
            
            trainer = UnifiedFinalTrainer()
            
            # Create enhanced configuration
            final_config = get_label_free_cbm_config(
                num_concepts=self.num_concepts,
                num_classes=self.num_classes,
                device=self.device,
                **config
            )
            
            try:
                # Monitor training with safety checks
                def progress_callback(step: int, metrics: Dict[str, float]):
                    # Safety monitoring
                    safety_status = self.safety_monitor.check_safety(
                        metrics.get('loss', float('inf')), 
                        self.final_layer
                    )
                    
                    if not safety_status['safe']:
                        self.logger.error(f"❌ Safety issues detected: {safety_status['critical_issues']}")
                        raise RuntimeError(f"Training stopped due to safety issues: {safety_status['critical_issues']}")
                    
                    # Log warnings
                    for warning in safety_status['warnings']:
                        self.logger.warning(f"⚠️ {warning}")
                    
                    # Update progress
                    self.training_progress.update(
                        step=step,
                        elapsed_time=time.time() - self._training_start_time,
                        **metrics
                    )
                    
                    # Log metrics
                    if self.metrics_logger:
                        self.metrics_logger.log_metrics(step, metrics, 'final_layer_training')
                    
                    # Early stopping check
                    if self.early_stopping:
                        model_state = {'final_layer': self.final_layer.state_dict() if self.final_layer else None}
                        should_stop = self.early_stopping(metrics.get('loss', float('inf')), model_state, step)
                        if should_stop:
                            raise StopIteration("Early stopping triggered")
                    
                    # Checkpoint saving
                    if self.checkpoint_manager and self.checkpoint_manager.should_save_checkpoint(step):
                        checkpoint_state = {
                            'concept_activations': concept_activations,
                            'labels': labels,
                            'config': config,
                            'training_progress': self.training_progress.to_dict()
                        }
                        self.checkpoint_manager.save_checkpoint(step, checkpoint_state, metrics=metrics)
                
                # Train with enhanced monitoring
                with memory_management_context(log_memory_usage=True):
                    result = trainer.train(
                        concept_activations, 
                        labels, 
                        final_config,
                        progress_callback=progress_callback
                    )
                
                # Create and store final layer
                self.final_layer = trainer.create_final_layer(final_config, result)
                self.concept_mean = result['concept_mean']
                self.concept_std = result['concept_std']
                
                # Update training state
                self.is_trained = True
                self.training_history.append({
                    'timestamp': time.time(),
                    'stage': 'final_layer',
                    'result': result,
                    'config': config
                })
                
                # Restore best model if early stopping was used
                if self.early_stopping and self.early_stopping.stopped:
                    best_state = self.early_stopping.restore_best_model()
                    if best_state and 'final_layer' in best_state:
                        self.final_layer.load_state_dict(best_state['final_layer'])
                        self.logger.info("🔄 Restored best final layer weights")
                
                # Log success metrics
                training_time = time.time() - self._training_start_time
                self.logger.info(f"✅ Final layer training completed in {training_time:.2f}s")
                self.logger.info(f"📊 Non-zero weights: {result['sparsity_stats']['non_zero_weights']}")
                self.logger.info(f"🎯 Sparsity: {result['sparsity_stats']['sparsity_per_class']:.1f} weights/class")
                
                # Final validation
                validation_result = self.validate_model_state()
                if not validation_result.valid:
                    self.logger.warning(f"⚠️ Post-training validation issues: {validation_result.issues}")
                
                return result
                
            except StopIteration as e:
                self.logger.info(f"🛑 Training stopped: {e}")
                # Return partial results if available
                if hasattr(trainer, 'partial_results'):
                    return trainer.partial_results
                raise
            except Exception as e:
                self.logger.error(f"❌ Final layer training failed: {e}")
                # Attempt to save emergency checkpoint
                if self.checkpoint_manager:
                    emergency_state = {
                        'error': str(e),
                        'concept_activations': concept_activations,
                        'labels': labels,
                        'config': config
                    }
                    self.checkpoint_manager.save_emergency_checkpoint(emergency_state)
                raise
    
    # =================== INFERENCE METHODS ===================
    
    def extract_features(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extract features from backbone with enhanced error handling
        
        Args:
            x: Input tensor [batch_size, ...]
            normalize: Whether to normalize features
        
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if x.size(0) == 0:
            raise ValueError("Empty batch provided")
        
        try:
            with torch.no_grad():
                # Move to correct device
                x = x.to(self.device)
                
                # Extract features
                features = self.backbone(x)
                
                # Validate output
                if torch.any(torch.isnan(features)):
                    raise RuntimeError("Backbone produced NaN features")
                
                if torch.any(torch.isinf(features)):
                    raise RuntimeError("Backbone produced infinite features")
                
                # Optional normalization
                if normalize:
                    features = nn.functional.normalize(features, dim=-1)
                
                return features
                
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            raise RuntimeError(f"Feature extraction failed: {e}") from e
    
    def get_concept_activations(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get concept activations from features with validation
        
        Args:
            features: Feature tensor [batch_size, feature_dim]
        
        Returns:
            Concept activations [batch_size, num_concepts]
        """
        if self.concept_layer is None:
            raise RuntimeError("Concept layer not initialized. Train the model first.")
        
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")
        
        try:
            with torch.no_grad():
                # Move to correct device
                features = features.to(self.device)
                
                # Get concept activations
                concept_activations = self.concept_layer(features)
                
                # Normalize if parameters are available
                if self.concept_mean is not None and self.concept_std is not None:
                    concept_activations = (concept_activations - self.concept_mean) / (self.concept_std + 1e-8)
                
                return concept_activations
                
        except Exception as e:
            self.logger.error(f"❌ Concept activation extraction failed: {e}")
            raise RuntimeError(f"Concept activation extraction failed: {e}") from e
    
    def predict_from_concepts(self, concept_activations: torch.Tensor) -> torch.Tensor:
        """
        Make predictions from concept activations with validation
        
        Args:
            concept_activations: Concept activations [batch_size, num_concepts]
        
        Returns:
            Predictions [batch_size, num_classes]
        """
        if self.final_layer is None:
            raise RuntimeError("Final layer not initialized. Train the model first.")
        
        if not isinstance(concept_activations, torch.Tensor):
            raise TypeError("Concept activations must be a torch.Tensor")
        
        if concept_activations.size(-1) != self.num_concepts:
            raise ValueError(
                f"Concept dimension mismatch: expected {self.num_concepts}, "
                f"got {concept_activations.size(-1)}"
            )
        
        try:
            with torch.no_grad():
                # Move to correct device
                concept_activations = concept_activations.to(self.device)
                
                # Get predictions
                predictions = self.final_layer(concept_activations)
                
                return predictions
                
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def forward(self, x: torch.Tensor, return_concepts: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Enhanced forward pass with comprehensive error handling
        
        Args:
            x: Input tensor [batch_size, ...]
            return_concepts: Whether to return concept activations
        
        Returns:
            Predictions [batch_size, num_classes]
            or (predictions, concepts) if return_concepts=True
        """
        if not self.is_trained:
            self.logger.warning("⚠️ Model may not be fully trained")
        
        try:
            # Extract features
            features = self.extract_features(x, normalize=True)
            
            # Get concept activations
            concepts = self.get_concept_activations(features)
            
            # Get predictions
            predictions = self.predict_from_concepts(concepts)
            
            if return_concepts:
                return predictions, concepts
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Forward pass failed: {e}")
            raise
    
    # =================== ANALYSIS AND INTERPRETATION METHODS ===================
    
    def analyze_concept_importance(self, concept_activations: torch.Tensor, 
                                 labels: torch.Tensor,
                                 method: str = 'correlation') -> Dict[str, Any]:
        """
        Enhanced concept importance analysis
        
        Args:
            concept_activations: Concept activations [N, num_concepts]
            labels: Ground truth labels [N]
            method: Analysis method ('correlation', 'mutual_info', 'shap')
        
        Returns:
            Dictionary with importance scores and statistics
        """
        if not isinstance(concept_activations, torch.Tensor):
            raise TypeError("concept_activations must be a torch.Tensor")
        
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor")
        
        if concept_activations.size(0) != labels.size(0):
            raise ValueError("Batch size mismatch between activations and labels")
        
        # Convert to numpy for analysis
        activations_np = concept_activations.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        importance_scores = {}
        statistics = {}
        
        try:
            if method == 'correlation':
                # Correlation-based importance
                correlations = []
                for i in range(self.num_concepts):
                    corr = np.corrcoef(activations_np[:, i], labels_np)[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                
                importance_scores['correlation'] = np.array(correlations)
                
            elif method == 'mutual_info':
                # Mutual information-based importance
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(activations_np, labels_np, random_state=42)
                importance_scores['mutual_info'] = mi_scores
                
            elif method == 'variance':
                # Variance-based importance
                variances = np.var(activations_np, axis=0)
                importance_scores['variance'] = variances
                
            else:
                raise ValueError(f"Unknown importance method: {method}")
            
            # Calculate statistics
            scores = importance_scores[method]
            statistics = {
                'mean_importance': np.mean(scores),
                'std_importance': np.std(scores),
                'max_importance': np.max(scores),
                'min_importance': np.min(scores),
                'top_concepts': np.argsort(scores)[-10:].tolist(),  # Top 10
                'bottom_concepts': np.argsort(scores)[:10].tolist()  # Bottom 10
            }
            
            # Add concept names if available
            if len(self.concept_names) == self.num_concepts:
                statistics['top_concept_names'] = [self.concept_names[i] for i in statistics['top_concepts']]
                statistics['bottom_concept_names'] = [self.concept_names[i] for i in statistics['bottom_concepts']]
            
            self.logger.info(f"✅ Concept importance analysis completed using {method}")
            self.logger.info(f"📊 Mean importance: {statistics['mean_importance']:.4f}")
            
            return {
                'method': method,
                'importance_scores': importance_scores,
                'statistics': statistics,
                'concept_names': self.concept_names.copy() if self.concept_names else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Concept importance analysis failed: {e}")
            raise RuntimeError(f"Concept importance analysis failed: {e}") from e
    
    def get_concept_statistics(self, concept_activations: torch.Tensor) -> Dict[str, Any]:
        """
        Get comprehensive statistics about concept activations
        
        Args:
            concept_activations: Concept activations [N, num_concepts]
        
        Returns:
            Dictionary with detailed statistics
        """
        if not isinstance(concept_activations, torch.Tensor):
            raise TypeError("concept_activations must be a torch.Tensor")
        
        activations_np = concept_activations.detach().cpu().numpy()
        
        stats = {
            'shape': activations_np.shape,
            'mean_per_concept': np.mean(activations_np, axis=0).tolist(),
            'std_per_concept': np.std(activations_np, axis=0).tolist(),
            'min_per_concept': np.min(activations_np, axis=0).tolist(),
            'max_per_concept': np.max(activations_np, axis=0).tolist(),
            'sparsity_per_concept': (np.sum(activations_np == 0, axis=0) / activations_np.shape[0]).tolist(),
            'global_stats': {
                'global_mean': float(np.mean(activations_np)),
                'global_std': float(np.std(activations_np)),
                'global_min': float(np.min(activations_np)),
                'global_max': float(np.max(activations_np)),
                'global_sparsity': float(np.sum(activations_np == 0) / activations_np.size),
                'nan_count': int(np.sum(np.isnan(activations_np))),
                'inf_count': int(np.sum(np.isinf(activations_np)))
            }
        }
        
        # Add concept names if available
        if len(self.concept_names) == self.num_concepts:
            stats['concept_names'] = self.concept_names.copy()
        
        return stats
    
    # =================== MODEL STATE MANAGEMENT ===================
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'model_class': self.__class__.__name__,
            'num_concepts': self.num_concepts,
            'num_classes': self.num_classes,
            'device': self.device,
            'is_trained': self.is_trained,
            'has_concept_names': len(self.concept_names) > 0,
            'num_concept_names': len(self.concept_names),
            'training_history_length': len(self.training_history),
            'config': self.config_manager.config.copy(),
            'training_progress': self.training_progress.to_dict() if hasattr(self.training_progress, 'to_dict') else None,
            'validation_state': self.validate_model_state().to_dict() if hasattr(self.validate_model_state(), 'to_dict') else None,
            'component_states': {
                'backbone': 'initialized' if self.backbone is not None else 'missing',
                'concept_layer': 'initialized' if self.concept_layer is not None else 'missing',
                'final_layer': 'initialized' if self.final_layer is not None else 'missing',
                'concept_mean': 'computed' if self.concept_mean is not None else 'missing',
                'concept_std': 'computed' if self.concept_std is not None else 'missing',
            }
        }
    
    def save_model_state(self, filepath: str) -> None:
        """
        Enhanced model state saving with comprehensive error handling
        
        Args:
            filepath: Path to save model state
        """
        try:
            # Validate model state before saving
            validation_result = self.validate_model_state()
            if not validation_result.valid:
                self.logger.warning(f"⚠️ Saving model with validation issues: {validation_result.issues}")
            
            # Prepare state dictionary
            state = {
                'model_class': self.__class__.__name__,
                'model_state_dict': self.state_dict(),
                'num_concepts': self.num_concepts,
                'num_classes': self.num_classes,
                'device': self.device,
                'concept_names': self.concept_names.copy(),
                'is_trained': self.is_trained,
                'concept_mean': self.concept_mean.detach().cpu() if self.concept_mean is not None else None,
                'concept_std': self.concept_std.detach().cpu() if self.concept_std is not None else None,
                'config': self.config_manager.config.copy(),
                'training_history': copy.deepcopy(self.training_history),
                'training_progress': self.training_progress.to_dict() if hasattr(self.training_progress, 'to_dict') else None,
                'timestamp': time.time(),
                'validation_result': validation_result.to_dict() if hasattr(validation_result, 'to_dict') else str(validation_result)
            }
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save state
            torch.save(state, filepath)
            
            self.logger.info(f"💾 Model state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model state: {e}")
            raise RuntimeError(f"Failed to save model state: {e}") from e
    
    def load_model_state(self, filepath: str, strict: bool = True) -> None:
        """
        Enhanced model state loading with comprehensive validation
        
        Args:
            filepath: Path to load model state from
            strict: Whether to enforce strict loading
        """
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model state file not found: {filepath}")
            
            # Load state
            state = torch.load(filepath, map_location=self.device)
            
            # Validate loaded state
            required_keys = ['model_state_dict', 'num_concepts', 'num_classes']
            missing_keys = [key for key in required_keys if key not in state]
            if missing_keys:
                raise ValueError(f"Missing required keys in saved state: {missing_keys}")
            
            # Check compatibility
            if state['num_concepts'] != self.num_concepts:
                if strict:
                    raise ValueError(
                        f"Concept count mismatch: current={self.num_concepts}, "
                        f"saved={state['num_concepts']}"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ Concept count mismatch: current={self.num_concepts}, "
                        f"saved={state['num_concepts']}"
                    )
            
            if state['num_classes'] != self.num_classes:
                if strict:
                    raise ValueError(
                        f"Class count mismatch: current={self.num_classes}, "
                        f"saved={state['num_classes']}"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ Class count mismatch: current={self.num_classes}, "
                        f"saved={state['num_classes']}"
                    )
            
            # Load model state dict
            try:
                self.load_state_dict(state['model_state_dict'], strict=strict)
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to load model state dict: {e}") from e
                else:
                    self.logger.warning(f"⚠️ Partial state loading failed: {e}")
            
            # Restore additional attributes
            if 'concept_names' in state:
                self.concept_names = state['concept_names']
            
            if 'is_trained' in state:
                self.is_trained = state['is_trained']
            
            if 'concept_mean' in state and state['concept_mean'] is not None:
                self.concept_mean = state['concept_mean'].to(self.device)
            
            if 'concept_std' in state and state['concept_std'] is not None:
                self.concept_std = state['concept_std'].to(self.device)
            
            if 'config' in state:
                self.config_manager.update(**state['config'])
            
            if 'training_history' in state:
                self.training_history = state['training_history']
            
            # Validate loaded model
            validation_result = self.validate_model_state()
            if not validation_result.valid:
                self.logger.warning(f"⚠️ Loaded model has validation issues: {validation_result.issues}")
            
            self.logger.info(f"📥 Model state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model state: {e}")
            raise RuntimeError(f"Failed to load model state: {e}") from e
    
    # =================== UTILITY METHODS ===================
    
    def to(self, device: str) -> 'BaseCBM':
        """Enhanced device movement with state updates"""
        try:
            # Move base module
            super().to(device)
            
            # Update device attribute
            self.device = device
            
            # Move concept normalization parameters
            if self.concept_mean is not None:
                self.concept_mean = self.concept_mean.to(device)
            
            if self.concept_std is not None:
                self.concept_std = self.concept_std.to(device)
            
            # Update config
            self.config_manager.update(device=device)
            
            self.logger.info(f"📱 Model moved to device: {device}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"❌ Failed to move model to device {device}: {e}")
            raise RuntimeError(f"Failed to move model to device {device}: {e}") from e
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        memory_info = {}
        
        try:
            # Model parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            # Memory usage by component
            components = {
                'backbone': self.backbone,
                'concept_layer': self.concept_layer,
                'final_layer': self.final_layer
            }
            
            component_params = {}
            component_memory = {}
            
            for name, module in components.items():
                if module is not None:
                    params = sum(p.numel() for p in module.parameters())
                    memory_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 * 1024)
                    component_params[name] = params
                    component_memory[name] = memory_mb
                else:
                    component_params[name] = 0
                    component_memory[name] = 0.0
            
            memory_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'component_parameters': component_params,
                'component_memory_mb': component_memory,
                'total_memory_mb': sum(component_memory.values()),
                'device': self.device
            }
            
            # Add GPU memory info if CUDA is available
            if torch.cuda.is_available() and 'cuda' in self.device:
                memory_info['gpu_memory'] = {
                    'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                    'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
                }
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory usage: {e}")
            return {'error': str(e)}
    
    def reset_training_state(self) -> None:
        """Reset training state while preserving model architecture"""
        try:
            # Reset training components
            self.concept_layer = None
            self.final_layer = None
            self.concept_mean = None
            self.concept_std = None
            
            # Reset state flags
            self.is_trained = False
            
            # Clear training history
            self.training_history.clear()
            self.training_progress = TrainingProgress()
            
            # Reset training components
            self.early_stopping = None
            self.metrics_logger = None
            self.checkpoint_manager = None
            
            # Reset timing
            self._training_start_time = None
            self._last_validation_time = None
            
            self.logger.info("🔄 Training state reset successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reset training state: {e}")
            raise RuntimeError(f"Failed to reset training state: {e}") from e
    
    def cleanup(self) -> None:
        """Cleanup resources and temporary files"""
        try:
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available() and 'cuda' in self.device:
                torch.cuda.empty_cache()
            
            # Close loggers and files
            if hasattr(self.metrics_logger, 'close'):
                self.metrics_logger.close()
            
            # Clear large tensors
            if self.concept_mean is not None:
                del self.concept_mean
                self.concept_mean = None
            
            if self.concept_std is not None:
                del self.concept_std
                self.concept_std = None
            
            self.logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup encountered issues: {e}")
    
    def __del__(self):
        """Destructor with safe cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Avoid errors during destruction
    
    def __repr__(self) -> str:
        """Enhanced string representation"""
        status = "✅ Trained" if self.is_trained else "⚠️ Not Trained"
        concept_info = f"concepts: {len(self.concept_names)}" if self.concept_names else f"concepts: {self.num_concepts}"
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  {status}\n"
            f"  {concept_info}, classes: {self.num_classes}\n"
            f"  device: {self.device}\n"
            f"  backbone: {type(self.backbone).__name__ if self.backbone else 'None'}\n"
            f"  concept_layer: {'✓' if self.concept_layer else '✗'}\n"
            f"  final_layer: {'✓' if self.final_layer else '✗'}\n"
            f")"
        )
    
    # =================== CONTEXT MANAGERS ===================
    
    def training_mode_context(self, enable_training: bool = True):
        """Context manager for training mode"""
        class TrainingContext:
            def __init__(self, model, enable_training):
                self.model = model
                self.enable_training = enable_training
                self.original_training_states = {}
            
            def __enter__(self):
                # Store original states
                for name, module in [
                    ('backbone', self.model.backbone),
                    ('concept_layer', self.model.concept_layer),
                    ('final_layer', self.model.final_layer)
                ]:
                    if module is not None:
                        self.original_training_states[name] = module.training
                        module.train(self.enable_training)
                
                return self.model
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore original states
                for name, module in [
                    ('backbone', self.model.backbone),
                    ('concept_layer', self.model.concept_layer),
                    ('final_layer', self.model.final_layer)
                ]:
                    if module is not None and name in self.original_training_states:
                        module.train(self.original_training_states[name])
        
        return TrainingContext(self, enable_training)
    
    def evaluation_context(self):
        """Context manager for evaluation mode"""
        return self.training_mode_context(enable_training=False)
    
    # =================== EXPORT METHODS ===================
    
    def export_config(self, filepath: str = None) -> Dict[str, Any]:
        """Export current configuration"""
        config = {
            'model_class': self.__class__.__name__,
            'architecture': {
                'num_concepts': self.num_concepts,
                'num_classes': self.num_classes,
                'backbone_type': type(self.backbone).__name__ if self.backbone else None
            },
            'training_config': self.config_manager.config.copy(),
            'concept_names': self.concept_names.copy(),
            'training_status': {
                'is_trained': self.is_trained,
                'training_history_length': len(self.training_history)
            },
            'export_timestamp': time.time()
        }
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"📄 Configuration exported to {filepath}")
        
        return config
    
    def export_concept_names(self, filepath: str) -> None:
        """Export concept names to file"""
        if not self.concept_names:
            raise ValueError("No concept names to export")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            for i, name in enumerate(self.concept_names):
                f.write(f"{i}\t{name}\n")
        
        self.logger.info(f"📝 Concept names exported to {filepath}")
    
    def import_concept_names(self, filepath: str) -> None:
        """Import concept names from file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Concept names file not found: {filepath}")
        
        concept_names = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    concept_names.append(parts[1])
                else:
                    concept_names.append(parts[0])
        
        if len(concept_names) != self.num_concepts:
            raise ValueError(
                f"Concept count mismatch: expected {self.num_concepts}, "
                f"found {len(concept_names)}"
            )
        
        self.concept_names = concept_names
        self.logger.info(f"📥 Concept names imported from {filepath}")


# =================== ENHANCED CBM FACTORY ===================

class CBMFactory:
    """Factory class for creating enhanced CBM instances with proper configuration"""
    
    @staticmethod
    def create_cbm(cbm_type: str, 
                   backbone: nn.Module,
                   num_concepts: int,
                   num_classes: int,
                   device: str = "cuda",
                   config: Dict[str, Any] = None) -> BaseCBM:
        """
        Create CBM instance with enhanced configuration
        
        Args:
            cbm_type: Type of CBM to create
            backbone: Backbone model
            num_concepts: Number of concepts
            num_classes: Number of classes
            device: Device to use
            config: Additional configuration
        
        Returns:
            Configured CBM instance
        """
        # Dynamic import based on CBM type
        if cbm_type.lower() == 'lfcbm':
            from .label_free_cbm import LabelFreeCBM
            cbm_class = LabelFreeCBM
        elif cbm_type.lower() == 'pcbm':
            from .post_hoc_cbm import PostHocCBM
            cbm_class = PostHocCBM
        elif cbm_type.lower() == 'hybrid':
            from .hybrid_cbm import HybridCBM
            cbm_class = HybridCBM
        else:
            raise ValueError(f"Unknown CBM type: {cbm_type}")
        
        # Create configuration
        base_config = CBMBaseConfig(
            num_concepts=num_concepts,
            num_classes=num_classes,
            device=device,
            **(config or {})
        )
        
        # Validate configuration
        config_issues = base_config.validate()
        if config_issues:
            logger.warning(f"⚠️ Configuration issues: {config_issues}")
        
        # Create CBM instance
        cbm = cbm_class(
            backbone=backbone,
            num_concepts=num_concepts,
            num_classes=num_classes,
            device=device,
            config=base_config
        )
        
        logger.info(f"🏭 Created {cbm_type} CBM with {num_concepts} concepts and {num_classes} classes")
        
        return cbm


# =================== ENHANCED UTILITIES ===================

def validate_cbm_compatibility(cbm1: BaseCBM, cbm2: BaseCBM) -> ValidationResult:
    """Validate compatibility between two CBM models"""
    result = ValidationResult(valid=True, issues=[], warnings=[], info={})
    
    # Check basic compatibility
    if cbm1.num_concepts != cbm2.num_concepts:
        result.add_issue(f"Concept count mismatch: {cbm1.num_concepts} vs {cbm2.num_concepts}")
    
    if cbm1.num_classes != cbm2.num_classes:
        result.add_issue(f"Class count mismatch: {cbm1.num_classes} vs {cbm2.num_classes}")
    
    if cbm1.device != cbm2.device:
        result.add_warning(f"Device mismatch: {cbm1.device} vs {cbm2.device}")
    
    # Check concept names compatibility
    if cbm1.concept_names and cbm2.concept_names:
        if cbm1.concept_names != cbm2.concept_names:
            result.add_warning("Concept names differ between models")
    
    # Check training status
    if cbm1.is_trained != cbm2.is_trained:
        result.add_warning(f"Training status mismatch: {cbm1.is_trained} vs {cbm2.is_trained}")
    
    return result


def compare_cbm_performance(cbm1: BaseCBM, 
                          cbm2: BaseCBM, 
                          test_data: torch.Tensor,
                          test_labels: torch.Tensor) -> Dict[str, Any]:
    """Compare performance between two CBM models"""
    if not validate_cbm_compatibility(cbm1, cbm2).valid:
        raise ValueError("CBM models are not compatible for comparison")
    
    results = {}
    
    with torch.no_grad():
        # Get predictions from both models
        pred1 = cbm1(test_data)
        pred2 = cbm2(test_data)
        
        # Calculate accuracy
        acc1 = (pred1.argmax(dim=1) == test_labels).float().mean().item()
        acc2 = (pred2.argmax(dim=1) == test_labels).float().mean().item()
        
        results['accuracy'] = {'cbm1': acc1, 'cbm2': acc2, 'difference': acc2 - acc1}
        
        # Calculate confidence scores
        conf1 = torch.softmax(pred1, dim=1).max(dim=1)[0].mean().item()
        conf2 = torch.softmax(pred2, dim=1).max(dim=1)[0].mean().item()
        
        results['confidence'] = {'cbm1': conf1, 'cbm2': conf2, 'difference': conf2 - conf1}
        
        # Agreement between models
        agreement = (pred1.argmax(dim=1) == pred2.argmax(dim=1)).float().mean().item()
        results['agreement'] = agreement
    
    return results