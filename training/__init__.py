"""
CBM Training Module
Provides unified training components for Concept Bottleneck Models (CBMs).
"""

# Logging
from ..utils.logging import setup_enhanced_logging
logger = setup_enhanced_logging(__name__)
logger.info("📦 CBM Training Module initialized.")

# Feature extraction utilities
from .feature_extraction import FeatureExtractor

# Final layer training
from .final_layer import (
    UnifiedFinalTrainer,
    FinalLayerConfig,
    FinalLayerType,
    get_label_free_cbm_config,
    get_dense_cbm_config
)

# Early stopping utilities
from .early_stopping import EarlyStopping, ValidationConfig

__all__ = [
    "FeatureExtractor",
    "UnifiedFinalTrainer", "FinalLayerConfig", "FinalLayerType",
    "get_label_free_cbm_config", "get_dense_cbm_config",
    "EarlyStopping", "ValidationConfig",
    "setup_enhanced_logging"
]
