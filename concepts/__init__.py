"""
Concept generation integration layer.
Provides unified interface to the main concept generation module.
"""

from typing import Dict, Any, Optional

# Try to import the interface, but degrade gracefully if optional deps are missing.
_uci_import_error: Optional[Exception] = None
try:
    from .unified_interface import UnifiedConceptInterface  # noqa: F401
except Exception as e:
    _uci_import_error = e

    class UnifiedConceptInterface:  # type: ignore
        """Fallback shim that raises a clear error only when actually used."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "UnifiedConceptInterface is unavailable because an optional dependency "
                "failed to import (likely the LLM querier). Install/patch it, or add a shim. "
                f"Original import error: {_uci_import_error}"
            )

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

__all__ = [
    'UnifiedConceptInterface',
    'generate_concepts',
    'generate_concepts_sync',
    'run_interactive_concept_generation',
]
