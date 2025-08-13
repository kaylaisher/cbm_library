
import json
import os
import re
from typing import List, Dict, Set
from utils.llm_clients import LLMClient
from utils.detailed_logger import DetailedLogger
import asyncio
import time
from collections import Counter


class LabelFreeQuerier:
    def __init__(self, config_path: str, enable_detailed_logging: bool = False):
        self.llm_client = LLMClient(config_path)
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = DetailedLogger() if enable_detailed_logging else None

        self.prompts = {
            "important": """List the most important visual features for recognizing a {class_name}. 
Provide only SHORT PHRASES or SINGLE WORDS (1-3 words each), one per line:
- Focus on physical characteristics, shapes, colors, textures
- Use format like: "round shape", "red color", "metal surface", "four legs"
- Avoid full sentences
- Provide exactly 10-12 features""",
            
            "around": """List objects, environments, or contexts commonly seen around a {class_name}.
Provide only SHORT PHRASES or SINGLE WORDS (1-3 words each), one per line:
- Use format like: "kitchen counter", "garden", "parking lot", "water"
- Avoid full sentences
- Provide exactly 10-12 features""",
            
            "superclass": """List categories and superclasses for {class_name}.
Provide only SHORT PHRASES or SINGLE WORDS (1-3 words each), one per line:
- Use format like: "vehicle", "living thing", "furniture", "tool"
- Avoid full sentences"""
        }
        
        self.concept_limits = {
            "important": 10, 
            "around": 10,    
            "superclass": None 
        }
        
        self.filter_config = {
            'min_length': 2,
            'max_length': 25,
            'blacklist': {
                'thing', 'object', 'item', 'stuff', 'something', 'anything',
                'various', 'different', 'many', 'some', 'all', 'most',
                'usually', 'often', 'typically', 'generally', 'commonly',
                'a', 'an', 'the', 'and', 'or', 'but', 'with', 'without'
            },
            'min_word_length': 2, 
        }
    
    async def generate_concepts(self, class_names: List[str], dataset_name: str) -> Dict:
        """Generate concepts using improved Label-Free CBM approach"""
        results = {}

        for prompt_type, prompt_template in self.prompts.items():
            header = f"\n Generating {prompt_type} concepts..."
            if self.logger:
                self.logger.log_and_print(header)
            else:
                print(header)
            concepts_dict = {}

            for class_name in class_names:
                prompt = prompt_template.format(class_name=class_name)

                if self.logger:
                    self.logger.log(f" Prompt for '{class_name}' [{prompt_type}]: {prompt}")
                print(f" Processing '{class_name}' [{prompt_type}]...")

                try:
                    response = await self.llm_client.query(prompt)

                    if self.logger:
                        self.logger.log(f"📨 Raw LLM response for '{class_name}' [{prompt_type}]: {response}")

                    concepts = self._parse_response_to_concepts(response)
                    concepts = self._filter_concepts(concepts)
                    
                    concepts_dict[class_name] = concepts

                    print(f"  {class_name}: {len(concepts)} concepts extracted")
                    if self.logger:
                        self.logger.log(f" Extracted concepts for '{class_name}': {concepts}")

                except Exception as e:
                    err_msg = f" Error querying class '{class_name}' [{prompt_type}]: {e}"
                    print(err_msg)
                    if self.logger: 
                        self.logger.log(err_msg)
                    concepts_dict[class_name] = []

                await asyncio.sleep(1.5)

            results[prompt_type] = concepts_dict
            
            output_dir = "outputs/label_free/gpt3_init"
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"gpt3_{dataset_name}_{prompt_type}.json")
            
            with open(json_path, 'w') as f:
                json.dump(concepts_dict, f, indent=2)

            msg = f" Saved {prompt_type} concepts to {json_path}"
            print(msg)
            if self.logger: 
                self.logger.log(msg)

        return results

    def _parse_response_to_concepts(self, response: str) -> List[str]:
        """Enhanced parsing to extract clean concept words/phrases"""
        concepts = []
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line = re.sub(r'^[-•*]\s*', '', line)  
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  
            line = line.strip()
            
            if not line:
                continue
            
            extracted = self._extract_concepts_from_line(line)
            concepts.extend(extracted)
        
        return concepts
    
    def _extract_concepts_from_line(self, line: str) -> List[str]:
        """Extract clean concepts from a single line"""
        concepts = []
        
        if ',' in line and len(line) > 30:
            parts = [part.strip() for part in line.split(',')]
            for part in parts:
                cleaned = self._clean_concept(part)
                if cleaned:
                    concepts.append(cleaned)
        else:

            cleaned = self._clean_concept(line)
            if cleaned:
                concepts.append(cleaned)
        
        return concepts
    
    def _clean_concept(self, concept: str) -> str:
        """Clean and normalize a single concept"""
        if not concept:
            return ""
        
        concept = concept.strip('\'"')
        concept = re.sub(r'\s+', ' ', concept)
        
        concept = re.sub(r'^(a|an|the)\s+', '', concept, flags=re.IGNORECASE)
        
        concept = re.sub(r'[.!?;:]+$', '', concept)
        
        if len(concept) > 25:
            words = concept.split()
            if len(words) > 4:
                meaningful_words = []
                for word in words[:4]:
                    if len(word) > 2 and word.lower() not in self.filter_config['blacklist']:
                        meaningful_words.append(word)
                    if len(meaningful_words) >= 3:
                        break
                concept = ' '.join(meaningful_words)
        
        concept = concept.strip()
        return concept if self._is_valid_concept(concept) else ""
    
    def _is_valid_concept(self, concept: str) -> bool:
        """Check if concept meets quality criteria"""
        if not concept:
            return False
        
        if len(concept) < self.filter_config['min_length'] or len(concept) > self.filter_config['max_length']:
            return False
        
        concept_lower = concept.lower()
        if concept_lower in self.filter_config['blacklist']:
            return False
        
        words = concept.split()
        if any(len(word) < self.filter_config['min_word_length'] for word in words if word.isalpha()):
            return False
        
        if not re.search(r'[a-zA-Z]', concept):
            return False
        
        generic_patterns = [
            r'^(some|any|many|various|different)\s',
            r'(usually|often|typically|generally|commonly)',
            r'^(used for|good for|helpful for)',
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, concept_lower):
                return False
        
        return True
    
    def _filter_concepts(self, concepts: List[str]) -> List[str]:
        """Apply additional filtering and deduplication"""
        if not concepts:
            return []
        
        seen = set()
        filtered = []
        
        for concept in concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen:
                seen.add(concept_lower)
                filtered.append(concept)
        
        return filtered
    
    async def apply_filtering(self, all_concepts: Dict, dataset_name: str) -> List[str]:
        """Apply comprehensive filtering to all concepts"""
        if self.logger:
            self.logger.log(" Starting concept filtering process...")
        
        concept_counter = Counter()
        
        for prompt_type in all_concepts:
            for class_name, concepts in all_concepts[prompt_type].items():
                for concept in concepts:
                    concept_counter[concept.lower()] += 1
        
        filtered_concepts = []
        
        for concept, count in concept_counter.items():
            original_concept = None
            for prompt_type in all_concepts:
                for class_name, concepts in all_concepts[prompt_type].items():
                    for orig in concepts:
                        if orig.lower() == concept:
                            original_concept = orig
                            break
                    if original_concept:
                        break
                if original_concept:
                    break
            
            if original_concept and self._is_valid_concept(original_concept):
                filtered_concepts.append(original_concept)
        
        filtered_concepts.sort(key=len)
        
        output_path = f"outputs/label_free/{dataset_name}_filtered.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
        
        print(f" Saved {len(filtered_concepts)} filtered concepts to {output_path}")
        if self.logger:
            self.logger.log(f" Filtering complete: {len(filtered_concepts)} concepts saved")
        
        return filtered_concepts
