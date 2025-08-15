
from ..utils.logging import setup_enhanced_logging
logger = setup_enhanced_logging(__name__)
logger.info("⚠️ `cbm_library.training.pipeline` is deprecated. Use LabelFreeCBM.complete_training().")

class CBMTrainingPipeline:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "`cbm_library.training.pipeline` is deprecated. "
            "Use LabelFreeCBM.complete_training() instead."
        )