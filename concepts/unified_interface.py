"""
Unified interface that wraps your existing querier classes.
Preserves all functionality while providing standardized access.
"""

from pathlib import Path
import sys
import asyncio
from typing import Optional, Dict, Any, List

# -----------------------------------------------------------------------------
# Repo root discovery and path setup
# -----------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent

def _find_repo_root(start: Path) -> Path:
    """
    Walk up a couple levels looking for a folder named 'llm_query_module'.
    Falls back to the starting directory if not found.
    """
    for p in (start, start.parent, start.parent.parent):
        if (p / "llm_query_module").is_dir():
            return p
    return start

_REPO_ROOT = _find_repo_root(_CURRENT_DIR)

def _ensure_on_path(p: Path):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Ensure project root and llm_query_module (folder style) are importable
_ensure_on_path(_REPO_ROOT)
_ensure_on_path(_REPO_ROOT / "llm_query_module")

# -----------------------------------------------------------------------------
# Resilient imports for queriers (package or folder style). We DO NOT fail import
# time; instead we create small factories that raise only when used.
# -----------------------------------------------------------------------------
_last_import_err: Optional[Exception] = None

def _resolve_class(pkg_path: str, fallback_mod: str, cls_name: str):
    """
    Try to import a class from a package path first (e.g., 'llm_query_module.cb_llm_querier:CBLLMQuerier'),
    then fallback to a plain module in the llm_query_module folder.
    Returns a factory that instantiates the class, or raises a clear ImportError at call time.
    """
    global _last_import_err

    # Attempt package-style import
    try:
        module_path, _, class_name = pkg_path.partition(":")
        if not class_name:
            module_path, class_name = pkg_path, cls_name
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        return lambda *a, **kw: cls(*a, **kw)
    except Exception as e1:
        _last_import_err = e1

    # Attempt folder-style import (module sitting in llm_query_module/)
    try:
        mod = __import__(fallback_mod, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        return lambda *a, **kw: cls(*a, **kw)
    except Exception as e2:
        _last_import_err = e2

        # Return a factory that raises when called
        def _raiser(*_a, **_kw):
            raise ImportError(
                f"Required class '{cls_name}' not found. "
                f"Tried '{pkg_path}' and module '{fallback_mod}'. "
                f"Ensure 'llm_query_module' is installed or on PYTHONPATH. "
                f"Last import error: {_last_import_err}"
            )
        return _raiser

# Factories for each querier (created at import-time but errors deferred)
_make_cb_llm_querier      = _resolve_class("llm_query_module.cb_llm_querier:CBLLMQuerier",
                                           "cb_llm_querier", "CBLLMQuerier")
_make_label_free_querier  = _resolve_class("llm_query_module.label_free_querier:LabelFreeQuerier",
                                           "label_free_querier", "LabelFreeQuerier")
_make_labo_querier        = _resolve_class("llm_query_module.labo_querier:LaBoQuerier",
                                           "labo_querier", "LaBoQuerier")
_make_lm4cv_querier       = _resolve_class("llm_query_module.lm4cv_querier:LM4CVQuerier",
                                           "lm4cv_querier", "LM4CVQuerier")
_make_async_interface     = _resolve_class("llm_query_module.async_main_interface:AsyncLLMQueryInterface",
                                           "async_main_interface", "AsyncLLMQueryInterface")

# -----------------------------------------------------------------------------
# UnifiedConceptInterface (lazy instantiation, cache, async APIs)
# -----------------------------------------------------------------------------
class UnifiedConceptInterface:
    """
    Unified interface wrapping your existing concept generation queriers.
    Preserves async capabilities and multi-method support.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize using your existing configuration system.

        Args:
            config_path: Path to your query_config.yaml file
        """
        # Resolve default config if none provided
        if config_path is None:
            cand1 = _REPO_ROOT / "concepts" / "main" / "config" / "query_config.yaml"
            cand2 = _REPO_ROOT / "llm_query_module" / "config" / "query_config.yaml"
            if cand1.is_file():
                config_path = str(cand1)
            elif cand2.is_file():
                config_path = str(cand2)
            else:
                raise FileNotFoundError(
                    "Could not locate query_config.yaml in expected paths:\n"
                    f" - {cand1}\n - {cand2}\n"
                    "Set config_path explicitly when constructing UnifiedConceptInterface."
                )

        self.config_path = config_path
        self._concept_cache: Dict[str, Dict[str, Any]] = {}

        # Lazy-initialized members
        self._async_interface = None
        self._cb_llm_querier = None
        self._label_free_querier = None
        self._labo_querier = None
        self._lm4cv_querier = None

    # ----- Lazy getters -------------------------------------------------------
    def _get_async_interface(self):
        if self._async_interface is None:
            self._async_interface = _make_async_interface(self.config_path)
        return self._async_interface

    def _get_cb_llm(self):
        if self._cb_llm_querier is None:
            self._cb_llm_querier = _make_cb_llm_querier(self.config_path)
        return self._cb_llm_querier

    def _get_label_free(self):
        if self._label_free_querier is None:
            self._label_free_querier = _make_label_free_querier(self.config_path, enable_detailed_logging=True)
        return self._label_free_querier

    def _get_labo(self):
        if self._labo_querier is None:
            self._labo_querier = _make_labo_querier(self.config_path)
        return self._labo_querier

    def _get_lm4cv(self):
        if self._lm4cv_querier is None:
            self._lm4cv_querier = _make_lm4cv_querier(self.config_path)
        return self._lm4cv_querier

    # ----- Public API ---------------------------------------------------------
    async def generate_concepts_for_cbm(
        self,
        dataset_name: str,
        cbm_method: str,
        num_concepts: int = 50,
        force_regenerate: bool = False,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate concepts using your existing queriers.

        Args:
            dataset_name: 'cifar10', 'cifar100', 'cub', 'places365', 'imagenet'
            cbm_method: 'label_free', 'labo', 'lm4cv', 'cb_llm'
            num_concepts: desired concepts count (may be advisory)
            force_regenerate: bypass cache if True
            **generation_kwargs: forwarded to specific querier

        Returns:
            Dict[str, Any]: formatted concepts ready for CBM training
        """
        cache_key = f"{dataset_name}_{cbm_method}_{num_concepts}"
        if not force_regenerate and cache_key in self._concept_cache:
            return self._concept_cache[cache_key]

        m = cbm_method.lower()
        if m in ("label_free", "lf_cbm"):
            concepts_data = await self._generate_label_free_concepts(dataset_name, **generation_kwargs)
        elif m == "labo":
            concepts_data = await self._generate_labo_concepts(dataset_name, **generation_kwargs)
        elif m == "lm4cv":
            concepts_data = await self._generate_lm4cv_concepts(dataset_name, **generation_kwargs)
        elif m == "cb_llm":
            concepts_data = await self._generate_cb_llm_concepts(dataset_name, **generation_kwargs)
        else:
            raise ValueError(f"Unsupported CBM method: {cbm_method}")

        formatted = self._format_concept_output(concepts_data, dataset_name, cbm_method)
        self._concept_cache[cache_key] = formatted
        return formatted

    async def _generate_label_free_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate Label-Free CBM concepts using your existing querier."""
        class_names = self._get_async_interface().get_dataset_classes(dataset_name)
        q = self._get_label_free()
        concepts = await q.generate_concepts(class_names, dataset_name)
        filtered_concepts = await q.apply_filtering(concepts, dataset_name)
        return {
            "concepts": filtered_concepts,
            "raw_concepts": concepts,
            "method": "label_free",
            "dataset": dataset_name,
        }

    async def _generate_labo_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate LaBo concepts using your existing querier."""
        class_names = self._get_async_interface().get_dataset_classes(dataset_name)
        q = self._get_labo()
        class2concepts = await q.generate_concepts(class_names, dataset_name)
        k_per_class = kwargs.get("k_per_class", 25)
        selected_concepts = q.submodular_selection(class2concepts, dataset_name, k_per_class)
        return {
            "concepts": selected_concepts,
            "raw_concepts": class2concepts,
            "method": "labo",
            "dataset": dataset_name,
        }

    async def _generate_lm4cv_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate LM4CV attributes using your existing querier."""
        class_names = self._get_async_interface().get_dataset_classes(dataset_name)
        q = self._get_lm4cv()
        attributes, cls2attributes = await q.generate_attributes(class_names, dataset_name)
        return {
            "concepts": attributes,
            "class_mappings": cls2attributes,
            "method": "lm4cv",
            "dataset": dataset_name,
        }

    async def _generate_cb_llm_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate CB-LLM concepts using your existing querier."""
        q = self._get_cb_llm()
        concepts = await q.generate_concepts(dataset_name)
        return {
            "concepts": concepts,
            "method": "cb_llm",
            "dataset": dataset_name,
        }

    def _format_concept_output(
        self,
        raw_concepts: Dict[str, Any],
        dataset_name: str,
        cbm_method: str
    ) -> Dict[str, Any]:
        """
        Format your concept output into a standardized format.
        """
        method = raw_concepts.get("method", cbm_method).lower()

        if method == "label_free":
            return {
                "concept_names": raw_concepts.get("concepts", []),
                "concept_embeddings": None,
                "concept_annotations": None,
                "metadata": {
                    "dataset": dataset_name,
                    "method": cbm_method,
                    "total_concepts": len(raw_concepts.get("concepts", [])),
                    "source": "main_concept_module",
                },
            }

        if method == "labo":
            all_concepts: List[str] = []
            for class_concepts in raw_concepts.get("concepts", {}).values():
                all_concepts.extend(class_concepts)
            return {
                "concept_names": list(dict.fromkeys(all_concepts)),  # de-dup, preserve order
                "class_based_concepts": raw_concepts.get("concepts", {}),
                "submodular_scores": raw_concepts.get("submodular_scores"),
                "metadata": {
                    "dataset": dataset_name,
                    "method": cbm_method,
                    "total_concepts": len(all_concepts),
                    "source": "main_concept_module",
                },
            }

        if method == "lm4cv":
            return {
                "concept_names": raw_concepts.get("concepts", []),
                "class_mappings": raw_concepts.get("class_mappings", {}),
                "metadata": {
                    "dataset": dataset_name,
                    "method": cbm_method,
                    "total_attributes": len(raw_concepts.get("concepts", [])),
                    "source": "main_concept_module",
                },
            }

        if method == "cb_llm":
            all_concepts: List[str] = []
            for class_concepts in raw_concepts.get("concepts", {}).values():
                all_concepts.extend(class_concepts)
            return {
                "concept_names": list(dict.fromkeys(all_concepts)),
                "class_based_concepts": raw_concepts.get("concepts", {}),
                "metadata": {
                    "dataset": dataset_name,
                    "method": cbm_method,
                    "total_concepts": len(all_concepts),
                    "source": "main_concept_module",
                },
            }

        raise ValueError(f"Unknown method: {method}")

    # ----- Convenience sync wrappers / metadata -------------------------------
    def generate_concepts_sync(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for async concept generation."""
        return asyncio.run(self.generate_concepts_for_cbm(*args, **kwargs))

    def get_supported_datasets(self) -> List[str]:
        """Get datasets supported by your module."""
        return self._get_async_interface().get_available_datasets()

    def get_supported_cbm_methods(self) -> List[str]:
        """Get CBM methods supported by your module."""
        return ["label_free", "labo", "lm4cv", "cb_llm"]

    def run_interactive_mode(self):
        """Launch your interactive concept generation UI."""
        return asyncio.run(self._get_async_interface().main_menu())
