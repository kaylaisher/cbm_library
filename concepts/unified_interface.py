"""
Unified interface that wraps your existing querier classes.
Preserves all functionality while providing standardized access.
"""

import sys
import os
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

current_dir = Path(__file__).parent
your_module_path = current_dir / "main" / "src"
sys.path.insert(0, str(your_module_path))

from main.cb_llm_querier import CBLLMQuerier
from main.label_free_querier import LabelFreeQuerier
from main.labo_querier import LaBoQuerier
from main.lm4cv_querier import LM4CVQuerier
from main.async_main_interface import AsyncLLMQueryInterface

class UnifiedConceptInterface:
    """
    Unified interface wrapping your existing concept generation queriers.
    Preserves all your async capabilities and multi-method support.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize using your existing configuration system.
        
        Args:
            config_path: Path to your query_config.yaml file
        """
        if config_path is None:
            config_path = "concepts/main/config/query_config.yaml"

        self.config_path = config_path
        
        self.cb_llm_querier = CBLLMQuerier(config_path, enable_detailed_logging=True)
        self.label_free_querier = LabelFreeQuerier(config_path, enable_detailed_logging=True)
        self.labo_querier = LaBoQuerier(config_path)
        self.lm4cv_querier = LM4CVQuerier(config_path)
        self.async_interface = AsyncLLMQueryInterface(config_path)
        
        self._concept_cache = {}
        
    async def generate_concepts_for_cbm(self,
                                       dataset_name: str,
                                       cbm_method: str,
                                       num_concepts: int = 50,
                                       force_regenerate: bool = False,
                                       **generation_kwargs) -> Dict[str, Any]:
        """
        Generate concepts using your existing queriers.
        
        Args:
            dataset_name: Dataset identifier ('cifar10', 'cifar100', 'cub', 'places365', 'imagenet')
            cbm_method: CBM method ('label_free', 'labo', 'lm4cv', 'cb_llm')
            num_concepts: Number of concepts to generate
            force_regenerate: Skip cache and regenerate concepts
            **generation_kwargs: Additional arguments for your generation method
            
        Returns:
            dict: Formatted concepts ready for CBM training
        """
        cache_key = f"{dataset_name}_{cbm_method}_{num_concepts}"
        
        if not force_regenerate and cache_key in self._concept_cache:
            return self._concept_cache[cache_key]
        
        if cbm_method.lower() == 'label_free' or cbm_method.lower() == 'lf_cbm':
            concepts_data = await self._generate_label_free_concepts(dataset_name, **generation_kwargs)
        elif cbm_method.lower() == 'labo':
            concepts_data = await self._generate_labo_concepts(dataset_name, **generation_kwargs)
        elif cbm_method.lower() == 'lm4cv':
            concepts_data = await self._generate_lm4cv_concepts(dataset_name, **generation_kwargs)
        elif cbm_method.lower() == 'cb_llm':
            concepts_data = await self._generate_cb_llm_concepts(dataset_name, **generation_kwargs)
        else:
            raise ValueError(f"Unsupported CBM method: {cbm_method}")
        
        formatted_output = self._format_concept_output(
            concepts_data, dataset_name, cbm_method
        )
        
        self._concept_cache[cache_key] = formatted_output
        
        return formatted_output
    
    async def _generate_label_free_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate Label-Free CBM concepts using your existing querier."""
        class_names = self.async_interface.get_dataset_classes(dataset_name)
        
        concepts = await self.label_free_querier.generate_concepts(class_names, dataset_name)
        filtered_concepts = await self.label_free_querier.apply_filtering(concepts, dataset_name)
        
        return {
            'concepts': filtered_concepts,
            'raw_concepts': concepts,
            'method': 'label_free',
            'dataset': dataset_name
        }
    
    async def _generate_labo_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate LaBo concepts using your existing querier."""
        class_names = self.async_interface.get_dataset_classes(dataset_name)
        
        class2concepts = await self.labo_querier.generate_concepts(class_names, dataset_name)
        
        k_per_class = kwargs.get('k_per_class', 25)
        selected_concepts = self.labo_querier.submodular_selection(class2concepts, dataset_name, k_per_class)
        
        return {
            'concepts': selected_concepts,
            'raw_concepts': class2concepts,
            'method': 'labo',
            'dataset': dataset_name
        }
    
    async def _generate_lm4cv_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate LM4CV attributes using your existing querier."""
        class_names = self.async_interface.get_dataset_classes(dataset_name)
        
        attributes, cls2attributes = await self.lm4cv_querier.generate_attributes(class_names, dataset_name)
        
        return {
            'concepts': attributes,
            'class_mappings': cls2attributes,
            'method': 'lm4cv',
            'dataset': dataset_name
        }
    
    async def _generate_cb_llm_concepts(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Generate CB-LLM concepts using your existing querier."""
        concepts = await self.cb_llm_querier.generate_concepts(dataset_name)
        
        return {
            'concepts': concepts,
            'method': 'cb_llm',
            'dataset': dataset_name
        }
    
    def _format_concept_output(self, 
                              raw_concepts: Dict[str, Any],
                              dataset_name: str,
                              cbm_method: str) -> Dict[str, Any]:
        """
        Format your concept output into a standardized format.
        """
        method = raw_concepts.get('method', cbm_method)
        
        if method == 'label_free':
            return {
                'concept_names': raw_concepts.get('concepts', []),
                'concept_embeddings': None,
                'concept_annotations': None,
                'metadata': {
                    'dataset': dataset_name,
                    'method': cbm_method,
                    'total_concepts': len(raw_concepts.get('concepts', [])),
                    'source': 'main_concept_module'
                }
            }
        elif method == 'labo':
            all_concepts = []
            for class_concepts in raw_concepts.get('concepts', {}).values():
                all_concepts.extend(class_concepts)
            
            return {
                'concept_names': list(set(all_concepts)), 
                'class_based_concepts': raw_concepts.get('concepts', {}),
                'submodular_scores': raw_concepts.get('submodular_scores'),
                'metadata': {
                    'dataset': dataset_name,
                    'method': cbm_method,
                    'total_concepts': len(all_concepts),
                    'source': 'main_concept_module'
                }
            }
        elif method == 'lm4cv':
            return {
                'concept_names': raw_concepts.get('concepts', []),
                'class_mappings': raw_concepts.get('class_mappings', {}),
                'metadata': {
                    'dataset': dataset_name,
                    'method': cbm_method,
                    'total_attributes': len(raw_concepts.get('concepts', [])),
                    'source': 'main_concept_module'
                }
            }
        elif method == 'cb_llm':
            all_concepts = []
            for class_concepts in raw_concepts.get('concepts', {}).values():
                all_concepts.extend(class_concepts)
            
            return {
                'concept_names': list(set(all_concepts)),
                'class_based_concepts': raw_concepts.get('concepts', {}),
                'metadata': {
                    'dataset': dataset_name,
                    'method': cbm_method,
                    'total_concepts': len(all_concepts),
                    'source': 'main_concept_module'
                }
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generate_concepts_sync(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for async concept generation."""
        return asyncio.run(self.generate_concepts_for_cbm(*args, **kwargs))
    
    def get_supported_datasets(self) -> List[str]:
        """Get datasets supported by your module."""
        return self.async_interface.get_available_datasets()
    
    def get_supported_cbm_methods(self) -> List[str]:
        """Get CBM methods supported by your module."""
        return ['label_free', 'labo', 'lm4cv', 'cb_llm']
    
    def run_interactive_mode(self):
        """Launch your interactive concept generation UI."""
        return asyncio.run(self.async_interface.main_menu())
