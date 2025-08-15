"""
Concept generation integration layer.
Provides unified interface to the main concept generation module.
"""

from .unified_interface import UnifiedConceptInterface

# Convenience functions for direct usage
async def generate_concepts(dataset_name: str, 
                           cbm_method: str,
                           config_path: str = None,
                           **kwargs) -> Dict[str, Any]:
    """Quick concept generation with default settings."""
    interface = UnifiedConceptInterface(config_path)
    return await interface.generate_concepts_for_cbm(dataset_name, cbm_method, **kwargs)

def generate_concepts_sync(dataset_name: str, 
                          cbm_method: str,
                          config_path: str = None,
                          **kwargs) -> Dict[str, Any]:
    """Synchronous concept generation."""
    interface = UnifiedConceptInterface(config_path)
    return interface.generate_concepts_sync(dataset_name, cbm_method, **kwargs)

def run_interactive_concept_generation(config_path: str = None):
    """Launch interactive concept generation using your existing interface."""
    interface = UnifiedConceptInterface(config_path)
    return interface.run_interactive_mode()

# Export main interface
__all__ = [
    'UnifiedConceptInterface',
    'generate_concepts',
    'generate_concepts_sync', 
    'run_interactive_concept_generation'
]