import json
import os
import asyncio
import re
from typing import List, Dict, Tuple
from utils.llm_clients import LLMClient

class LM4CVQuerier:
    def __init__(self, config_path: str):
        self.llm_client = LLMClient(config_path)
        
        self.domain_mapping = {
            'cifar10': 'objects',
            'cifar100': 'objects', 
            'cub': 'birds',
            'cars': 'cars',
            'food': 'foods'
        }
        
        # Improved filtering configuration
        self.filter_config = {
            'min_length': 8,  # Minimum character length
            'max_length': 60,  # Maximum character length
            'brand_keywords': {
                'cars': ['acura', 'bmw', 'audi', 'mercedes', 'toyota', 'ford', 'chevrolet', 
                        'honda', 'nissan', 'volkswagen', 'volvo', 'lexus', 'cadillac'],
                'birds': ['sparrow', 'eagle', 'robin', 'cardinal'],  # specific species
                'default': []
            },
            'generic_terms': [
                'distinctive', 'unique', 'signature', 'characteristic', 'typical',
                'various', 'different', 'several', 'multiple', 'many'
            ],
            'measurement_patterns': [
                r"\d+\s*['\"]\s*(alloy\s+)?wheels?",  # "17" wheels"
                r'\d+\s*inch\s+(alloy\s+)?wheels?',    # "17 inch wheels"
                r'\d+\.\d+\s*[lL]\s+.*engine'          # "6.2L V8 engine"
            ]
        }

    
    async def generate_attributes(self, class_names: List[str], dataset_name: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Generate attributes using improved LM4CV strategy"""
        cls2attributes = {}
        all_attributes = set()
        
        domain_name = self.domain_mapping.get(dataset_name.lower(), 'objects')
        
        print(f"\n🔍 Generating LM4CV attributes for {dataset_name} ({domain_name})...")
        
        for class_name in class_names:
            class_attributes = []
            
            # Enhanced instance prompting with better examples
            instance_prompt = f"""make sure all the answer are visual features, it should be concise and descriptive. output example:
            <concepts>
            - four-limbed primate
            - black, grey, white, brown, or red-brown fur
            - wet and hairless nose with curved nostrils
            - long tail
            - large eyes
            - furry bodies
            - clawed hands and feet
            </concepts>
            """
            
            response = await self.llm_client.query(instance_prompt)
            attributes = self._parse_attributes(response)
            class_attributes.extend(attributes)
            
            # Domain-specific prompting with better context
            domain_prompt = f"""Q: What are useful visual features to distinguish {class_name} from other {domain_name} in a photo?"""
            
            response = await self.llm_client.query(domain_prompt)
            attributes = self._parse_attributes(response)
            class_attributes.extend(attributes)
            
            # Shape/form specific prompting for cars
            if domain_name == 'cars':
                shape_prompt = f""" What are the key body shape and design features of {class_name}?"""
                
                response = await self.llm_client.query(shape_prompt)
                attributes = self._parse_attributes(response)
                class_attributes.extend(attributes)
            
            # Clean and store
            cleaned_attributes = self._clean_attributes(class_attributes, class_name, dataset_name)
            cls2attributes[class_name] = cleaned_attributes
            all_attributes.update(cleaned_attributes)
            
            print(f"  {class_name}: {len(cleaned_attributes)} attributes")
            await asyncio.sleep(0.5)
        
        # Save outputs
        unified_attributes = sorted(list(all_attributes))
        self._save_attributes_txt(unified_attributes, dataset_name)
        self._save_cls2attributes_json(cls2attributes, dataset_name)
        
        return unified_attributes, cls2attributes
    
    def _parse_attributes(self, response: str) -> List[str]:
        """Parse response looking for bullet points and handle prompt leaks"""
        lines = response.split('\n')
        attributes = []
        
        for line in lines:
            line = line.strip()
            
            # Skip obvious prompt leaks
            if line.startswith(('Q:', 'A:')):
                continue
                
            if line.startswith(('- ', '• ', '* ')):
                clean_line = line.lstrip('- •* ').strip()
                if len(clean_line) > 5 and not self._is_prompt_leak(clean_line):
                    attributes.append(clean_line)
        
        return attributes
    
    def _is_prompt_leak(self, text: str) -> bool:
        """Detect if text is a leaked prompt or instruction"""
        leak_indicators = [
            'what are useful visual features',
            'distinguish',
            'in a photo',
            'there are several',
            'q:',
            'a:'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in leak_indicators)
    
    def _clean_attributes(self, attributes: List[str], class_name: str, dataset_name: str) -> List[str]:
        """Enhanced attribute cleaning with better filtering"""
        cleaned = []
        domain = self.domain_mapping.get(dataset_name.lower(), 'default')
        
        for attr in attributes:
            attr = attr.strip()
            
            # Length filter
            if not (self.filter_config['min_length'] <= len(attr) <= self.filter_config['max_length']):
                continue
            
            # Remove class name mentions (more sophisticated)
            if self._contains_class_name(attr, class_name):
                continue
            
            # Remove brand-specific terms
            if self._contains_brand_terms(attr, domain):
                continue
            
            # Remove specific measurements
            if self._contains_measurements(attr):
                continue
            
            # Remove overly generic terms
            if self._is_too_generic(attr):
                continue
            
            # Normalize format
            attr = self._normalize_attribute(attr)
            
            if attr and len(attr) >= self.filter_config['min_length']:
                cleaned.append(attr)
        
        # Remove duplicates and limit per class
        cleaned = list(dict.fromkeys(cleaned))  # Preserve order while removing duplicates
        return cleaned[:20]  # Increased limit
    
    def _contains_class_name(self, attr: str, class_name: str) -> bool:
        """Check if attribute contains class name or its parts"""
        attr_lower = attr.lower()
        class_lower = class_name.lower()
        
        # Check full class name
        if class_lower in attr_lower:
            return True
        
        # Check individual words in class name (but not common words)
        class_words = class_lower.split()
        common_words = {'sedan', 'coupe', 'suv', 'convertible', 'wagon', 'hatchback', 
                       'truck', 'van', 'cab', 'crew', 'extended', 'regular'}
        
        for word in class_words:
            if len(word) > 3 and word not in common_words and word in attr_lower:
                return True
        
        return False
    
    def _contains_brand_terms(self, attr: str, domain: str) -> bool:
        """Check if attribute contains brand-specific terms"""
        brand_keywords = self.filter_config['brand_keywords'].get(domain, [])
        attr_lower = attr.lower()
        
        return any(brand in attr_lower for brand in brand_keywords)
    
    def _contains_measurements(self, attr: str) -> bool:
        """Check if attribute contains specific measurements"""
        for pattern in self.filter_config['measurement_patterns']:
            if re.search(pattern, attr, re.IGNORECASE):
                return True
        return False
    
    def _is_too_generic(self, attr: str) -> bool:
        """Check if attribute is too generic"""
        attr_lower = attr.lower()
        
        # Check for generic terms
        for term in self.filter_config['generic_terms']:
            if term in attr_lower:
                return True
        
        # Check for overly vague descriptions
        vague_patterns = [
            r'^(nice|good|bad|great|beautiful|ugly)\s',
            r'^(big|small|large|little)\s+(car|vehicle|object)$',
            r'^(modern|old|new|vintage)\s+design$'
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, attr_lower):
                return True
        
        return False
    
    def _normalize_attribute(self, attr: str) -> str:
        """Normalize attribute format"""
        # Ensure lowercase start (unless proper noun)
        if attr and attr[0].isupper() and not self._starts_with_proper_noun(attr):
            attr = attr[0].lower() + attr[1:]
        
        # Remove trailing punctuation
        attr = re.sub(r'[.!?]+$', '', attr)
        
        # Normalize whitespace
        attr = re.sub(r'\s+', ' ', attr).strip()
        
        return attr
    
    def _starts_with_proper_noun(self, text: str) -> bool:
        """Check if text starts with a proper noun that should stay capitalized"""
        proper_nouns = ['LED', 'BMW', 'GPS', 'ABS', 'SUV', 'UV']
        return any(text.startswith(noun) for noun in proper_nouns)
    
    def _save_attributes_txt(self, attributes: List[str], dataset_name: str):
        """Save attributes list as TXT"""
        filepath = f"outputs/lm4cv/data/{dataset_name}/{dataset_name}_attributes.txt"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            for attr in attributes:
                f.write(f"{attr}\n")
        
        print(f"✅ Saved {len(attributes)} attributes to {filepath}")
    
    def _save_cls2attributes_json(self, data: Dict, dataset_name: str):
        """Save class-to-attributes mapping"""
        filepath = f"outputs/lm4cv/cls2attributes/{dataset_name}_cls2attributes.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved class mappings to {filepath}")
        
        # Also save a summary report
        self._save_summary_report(data, dataset_name)
    
    def _save_summary_report(self, data: Dict, dataset_name: str):
        """Save a summary report of generated attributes"""
        report_path = f"outputs/lm4cv/reports/{dataset_name}_summary.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        total_attributes = sum(len(attrs) for attrs in data.values())
        unique_attributes = len(set(attr for attrs in data.values() for attr in attrs))
        
        with open(report_path, 'w') as f:
            f.write(f"LM4CV Attribute Generation Summary - {dataset_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total classes: {len(data)}\n")
            f.write(f"Total attributes: {total_attributes}\n")
            f.write(f"Unique attributes: {unique_attributes}\n")
            f.write(f"Average attributes per class: {total_attributes/len(data):.1f}\n\n")
            
            f.write("Class breakdown:\n")
            for class_name, attrs in data.items():
                f.write(f"  {class_name}: {len(attrs)} attributes\n")
        
        print(f"📊 Saved summary report to {report_path}")