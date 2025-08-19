"""
cbm_library
Lean package initializer: exports only the merged model APIs you actually use.
"""

__version__ = "0.1.0"

# Lightweight logging helper
try:
    from .utils.logging import setup_enhanced_logging  # noqa: F401
except Exception:
    def setup_enhanced_logging(name: str):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return logger

logger = setup_enhanced_logging(__name__)
logger.info("📦 cbm_library loaded")

# Export only the merged model (no optional subpackages, no star imports)
from .models import LabelFreeCBM, read_concepts_file  # <- make sure models/__init__.py points to your merged file

__all__ = [
    "__version__",
    "setup_enhanced_logging",
    "LabelFreeCBM",
    "read_concepts_file",
]
