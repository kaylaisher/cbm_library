import json
import os
import re
from typing import List, Dict, Set
from utils.llm_clients import LLMClient
import asyncio
from collections import Counter


class LaBoQuerier:
    def __init__(self, config_path: str):
        self.llm_client = LLMClient(config_path)
        
        # Original LaBo prompts - keeping these unchanged
        self.prompts = [
            "describe what the {class_name} looks like:",
            "describe the appearance of the {class_name}:",
            "describe the color of the {class_name}:",
            "describe the pattern of the {class_name}:",
            "describe the shape of the {class_name}:"
        ]
        
        # CHANGED: Relaxed instruction - less restrictive
        self.instruction = """Provide descriptive concepts about this object. 
        Focus on visual characteristics, physical properties, and observable features.
        only give me words or phrases that describe the object, not actions or behaviors.
        don't give sentences, just concepts.
        """
        
        # CHANGED: Relaxed filter configuration
        self.filter_config = {
            'min_length': 2,
            'max_length': 50,  # Increased from 25
            'blacklist': {
                # Reduced blacklist - only the most generic terms
                'thing', 'object', 'item', 'stuff', 'something', 'anything',
                'a', 'an', 'the'
            },
            'min_word_length': 1,  # Reduced from 2
        }

    async def generate_concepts(self, class_names: List[str], dataset_name: str) -> Dict[str, List[str]]:
        """Generate concepts using LaBo's methodology with relaxed extraction"""
        print("\n🍾 Generating LaBo concepts (parallel async)...")
        class2concepts = {}

        async def query_prompts_for_class(class_name: str) -> tuple[str, List[str]]:
            """Generate concepts for a single class using all prompts"""
            all_concepts = []
            for prompt_template in self.prompts:
                prompt = prompt_template.format(class_name=class_name)
                prompt = prompt + self.instruction
                
                # Generate 10 responses per prompt as in original LaBo
                tasks = [self.llm_client.query(prompt) for _ in range(10)]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for response in responses:
                    if isinstance(response, Exception):
                        print(f"⚠️ Error querying for {class_name}: {response}")
                        continue
                    # Use relaxed parsing
                    concepts = self._parse_response_to_concepts(response)
                    # Apply minimal filtering
                    concepts = self._filter_concepts(concepts)
                    all_concepts.extend(concepts)
            
            return class_name, all_concepts

        # Run all classes in parallel
        tasks = [query_prompts_for_class(class_name) for class_name in class_names]
        results = await asyncio.gather(*tasks)

        # CHANGED: Process results with minimal cleaning
        for class_name, raw_concepts in results:
            # Remove class name references and apply final filtering
            cleaned = []
            for concept in raw_concepts:
                # Remove class name references
                clean_concept = self._remove_class_name(concept, class_name)
                
                # Apply final validation
                if clean_concept and self._is_valid_concept(clean_concept):
                    cleaned.append(clean_concept)
            
            # Apply final filtering
            cleaned = self._apply_final_filtering(cleaned)
            class2concepts[class_name] = cleaned
            print(f"  {class_name}: {len(cleaned)} concepts")

        # Save raw concepts
        self._save_class2concepts(class2concepts, dataset_name)
        return class2concepts

    def _parse_response_to_concepts(self, response: str) -> List[str]:
        """Relaxed parsing - accept more varied formats"""
        concepts = []
        
        # Split by lines and process each
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove basic formatting only
            line = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove numbering
            line = line.strip()
            
            if not line:
                continue
            
            # CHANGED: Use simpler extraction
            extracted = self._extract_concepts_from_line(line)
            concepts.extend(extracted)
        
        return concepts
    
    def _extract_concepts_from_line(self, line: str) -> List[str]:
        """CHANGED: More permissive extraction from a single line"""
        concepts = []
        
        # CHANGED: Accept longer lines and more complex phrases
        if ',' in line and len(line) > 50:  # Increased threshold from 30
            parts = [part.strip() for part in line.split(',')]
            for part in parts:
                cleaned = self._clean_concept(part)
                if cleaned:  # FIXED: was 'qed'
                    concepts.append(cleaned)
        else:
            # Process as single concept
            cleaned = self._clean_concept(line)
            if cleaned:  # FIXED: was 'qed'
                concepts.append(cleaned)
        
        return concepts
    
    def _clean_concept(self, concept: str) -> str:
        """CHANGED: Minimal cleaning - preserve more content"""
        if not concept:
            return ""
        
        # Remove quotes and extra whitespace
        concept = concept.strip('\'"')
        concept = re.sub(r'\s+', ' ', concept)
        
        # Remove common article words at the beginning
        concept = re.sub(r'^(a|an|the)\s+', '', concept, flags=re.IGNORECASE)
        
        # Remove trailing punctuation (except hyphens within words)
        concept = re.sub(r'[.!?;:]+', '', concept)
        
        # FIXED: Added missing return statement
        return concept.strip()

    def _is_valid_concept(self, concept: str) -> bool:
        """CHANGED: More permissive validation"""
        if not concept:
            return False
        
        # CHANGED: More permissive length check
        if len(concept) < self.filter_config['min_length'] or len(concept) > self.filter_config['max_length']:
            return False
        
        # CHANGED: Only check reduced blacklist
        concept_lower = concept.lower()
        if concept_lower in self.filter_config['blacklist']:
            return False
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', concept):
            return False
        
        return True
    
    def _filter_concepts(self, concepts: List[str]) -> List[str]:
        """Minimal filtering - preserve uniqueness only"""
        if not concepts:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        filtered = []
        
        for concept in concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen:
                seen.add(concept_lower)
                filtered.append(concept)
        
        return filtered

    def _remove_class_name(self, concept: str, class_name: str) -> str:
        """CHANGED: Light class name removal - only exact matches"""
        concept_lower = concept.lower()
        class_name_lower = class_name.lower()
        
        # CHANGED: Only remove exact class name matches
        concept_lower = re.sub(r'\b' + re.escape(class_name_lower) + r'\b', '', concept_lower)
        
        # Clean up extra spaces and return
        concept_cleaned = ' '.join(concept_lower.split())
        return concept_cleaned.strip()

    def _apply_final_filtering(self, concepts: List[str]) -> List[str]:
        """CHANGED: Relaxed final filtering"""
        if not concepts:
            return []
        
        # Count concept frequencies
        concept_counter = Counter(concept.lower() for concept in concepts)
        
        # Filter and sort
        filtered_concepts = []
        seen = set()
        
        for concept in concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen and self._is_valid_concept(concept):
                seen.add(concept_lower)
                filtered_concepts.append(concept)
        
        # Sort by length (shorter concepts first, they're usually better)
        filtered_concepts.sort(key=len)
        
        return filtered_concepts[:1000]  # CHANGED: Increased limit from 500

    def submodular_selection(self, class2concepts: Dict, dataset_name: str, k_per_class: int = 25) -> Dict:
        """Apply submodular selection as described in LaBo paper"""
        selected_concepts = {}
        
        print(f"\n🔍 Applying submodular selection (top {k_per_class} per class)...")
        
        for class_name, concepts in class2concepts.items():
            selected = self._relaxed_submodular_selection(concepts, k_per_class)
            selected_concepts[class_name] = selected
            print(f"  {class_name}: {len(selected)} selected")
        
        # Save selected concepts
        self._save_selected_concepts(selected_concepts, dataset_name)
        
        return selected_concepts

    def _relaxed_submodular_selection(self, concepts: List[str], k: int) -> List[str]:
        """More relaxed submodular selection"""
        scored_concepts = []
        
        for concept in concepts:
            score = 0
            words = concept.split()
            word_count = len(words)
            
            # More lenient length scoring
            if 1 <= word_count <= 3:
                score += 3
            elif word_count <= 5:
                score += 2
            else:
                score += 1
            
            # Bonus for visual/descriptive terms (less restrictive)
            visual_indicators = ['color', 'shape', 'size', 'texture', 'pattern', 'surface', 'material']
            if any(indicator in concept.lower() for indicator in visual_indicators):
                score += 1
            
            # Simple diversity scoring
            score += len(set(concept.lower().split())) * 0.1
            
            scored_concepts.append((score, len(concept), concept))
        
        # Sort by score (descending), then by length (ascending) for ties
        scored_concepts.sort(key=lambda x: (-x[0], x[1]))
        selected = [concept for _, _, concept in scored_concepts[:k]]
        
        return selected

    def _save_class2concepts(self, data: Dict, dataset_name: str):
        """Save raw concepts to JSON file"""
        filepath = f"outputs/labo/concepts/class2concepts_{dataset_name}.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {filepath}")

    def _save_selected_concepts(self, data: Dict, dataset_name: str):
        """Save selected concepts to JSON file"""
        filepath = f"outputs/labo/selected_concepts/{dataset_name.upper()}.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {filepath}")

    def apply_comprehensive_filtering(self, all_concepts: Dict, dataset_name: str) -> List[str]:
        """Relaxed comprehensive filtering"""
        print("🔍 Starting relaxed concept filtering process...")
        
        # Combine all concepts with frequency tracking
        concept_counter = Counter()
        
        for class_name, concepts in all_concepts.items():
            for concept in concepts:
                concept_counter[concept.lower()] += 1
        
        # Minimal filtering - keep most concepts
        filtered_concepts = []
        
        for concept, count in concept_counter.items():
            # Find original case version
            original_concept = None
            for class_name, concepts in all_concepts.items():
                for orig in concepts:
                    if orig.lower() == concept:
                        original_concept = orig
                        break
                if original_concept:
                    break
            
            # Apply only basic validation
            if original_concept and len(original_concept) >= 2:
                filtered_concepts.append(original_concept)
        
        # Sort by frequency then by length
        concept_freq = {concept: concept_counter[concept.lower()] for concept in filtered_concepts}
        filtered_concepts.sort(key=lambda x: (-concept_freq[x], len(x)))
        
        # Save filtered concepts
        output_path = f"outputs/labo/{dataset_name}_filtered.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
        
        print(f"✅ Saved {len(filtered_concepts)} filtered concepts to {output_path}")
        
        # FIXED: Return just the list, not a tuple
        return filtered_concepts