__version__ = "0.1.0"

try:
    from .utils.logging import setup_enhanced_logging  
except Exception:
    def setup_enhanced_logging(name: str):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        return logger

logger = setup_enhanced_logging(__name__)
logger.info("=== cbm_library loaded ===")

from .models import LabelFreeCBM, read_concepts_file 

__all__ = [
    "__version__",
    "setup_enhanced_logging",
    "LabelFreeCBM",
    "read_concepts_file",
]
