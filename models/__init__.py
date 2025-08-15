"""
CBM Models Module
Provides implementations of various Concept Bottleneck Model (CBM) methods.
"""

# Logging
from ..utils.logging import setup_enhanced_logging
logger = setup_enhanced_logging(__name__)
logger.info("📦 CBM Models Module initialized.")

# Core CBM models
from .base_cbm import BaseCBM
from .label_free_cbm import LabelFreeCBM
from .vlg_cbm import VLGCBM
try:
    from .cb_llm import CBLLM
except Exception:
    CBLLM = None  # sentence-transformers not installed
from .labo_cbm import LaBoCBM
from .lm4cv_cbm import LM4CVCBM

# Factory and method registry
from .model_factory import CBMFactory, CBMMethod

# Enhanced utilities
from ..training.early_stopping import EarlyStopping
from ..config.config_manager import ConfigManager

__all__ = [
    'BaseCBM', 'LabelFreeCBM', 'VLGCBM', 'CBLLM', 'LaBoCBM', 'LM4CVCBM',
    'CBMFactory', 'CBMMethod',
    'setup_enhanced_logging', 'EarlyStopping', 'ConfigManager'
]
