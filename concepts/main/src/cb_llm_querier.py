import json
import os
import re
from typing import List, Dict, Set
from utils.llm_clients import LLMClient
from utils.detailed_logger import DetailedLogger
import asyncio
import time
from collections import Counter


class CBLLMQuerier:
    def __init__(self, config_path: str, enable_detailed_logging: bool = False):
        self.llm_client = LLMClient(config_path)
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = DetailedLogger() if enable_detailed_logging else None

        # Add missing filter configuration
        self.filter_config = {
            'min_length': 3,
            'max_length': 100,
            'min_word_length': 2,
            'blacklist': {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'through',
                'against', 'within', 'without', 'toward', 'under', 'over',
                'example', 'feature', 'features', 'other', 'different', 'important'
            }
        }

        # Template-based prompts following the exact format from the research paper
        self.prompts = {
            "SST2": {
                "negative": """Here are some examples of key features that are often present in a negative movie rating. Each feature is shown between the tag <example></example>.
                            <example>Flat or one-dimensional characters</example>
                            <example>Uninteresting cinematography</example>
                            <example>Lack of tension-building scenes</example>
                            <example>Lack of emotional impact</example>
                            List 100 other different important features that are often present in a negative movie rating. Need to follow the template above, i.e. <example>features</example>.""",
                                            
                "positive": """Here are some examples of key features that are often present in a positive movie rating. Each feature is shown between the tag <example></example>.
                            <example>Engaging plot</example>
                            <example>Strong character development</example>
                            <example>Great humor</example>
                            <example>Clever narrative structure</example>
                            List 100 other different important features that are often present in a positive movie rating. Need to follow the template above, i.e. <example>features</example>."""
            },
            
            "YelpP": {
                "negative": """Here are some examples of key features that are often present in a negative Yelp review with lower star ratings (e.g., 1 or 2 stars). Each feature is shown between the tag <example></example>.
                            <example>Overpriced</example>
                            <example>Unappetizing food</example>
                            <example>Unprofessional service</example>
                            <example>Broken products</example>
                            The reviews fall into the following categories: Food, Automotive, Home Services, Entertainment, Medical, Hotels, Financial Services, Media, Parking, Clothing, Electronic devices, and Cleaning. List 100 other different important features that are often present in a negative Yelp review with lower star ratings (e.g., 1 or 2 stars). Need to follow the template above, i.e. <example>features</example>.""",
                                            
                "positive": """Here are some examples of key features that are often present in a positive Yelp review with higher star ratings (e.g., 4 or 5 stars). Each feature is shown between the tag <example></example>.
                            <example>Delicious food</example>
                            <example>Outstanding service</example>
                            <example>Great value for the price</example>
                            <example>High quality products</example>
                            The reviews fall into the following categories: Food, Automotive, Home Services, Entertainment, Medical, Hotels, Financial Services, Media, Parking, Clothing, Electronic devices, and Cleaning. List 100 other different important features that are often present in a positive Yelp review with higher star ratings (e.g., 4 or 5 stars). Need to follow the template above, i.e. <example>features</example>."""
            },
            
            "AGNews": {
                "world": """Here are some examples of key features that are often present in worldwide news. Each feature is shown between the tag <example></example>.
                            <example>Words related to country and place</example>
                            <example>Political stunts taken by governments</example>
                            <example>Global issues</example>
                            <example>Words related to war, conflict</example>
                            List 50 other important features that are often present in worldwide news. Need to follow the template above, i.e. <example>features</example>.""",
                                            
                "sports": """Here are some examples of key features that are often present in sport news. Each feature is shown between the tag <example></example>.
                            <example>Name of sports stars</example>
                            <example>Words related to game, competition</example>
                            <example>Ball games like baseball, basketball</example>
                            <example>Name of sport teams</example>
                            List 50 other important features that are often present in sport news. Need to follow the template above, i.e. <example>features</example>.""",
                
                "business": """Here are some examples of key features that are often present in business and financial news. Each feature is shown between the tag <example></example>.
                            <example>Words related to currency, money</example>
                            <example>The numerical amount of dollars</example>
                            <example>The symbol like $ or %</example>
                            <example>Words related to stock, Portfolio</example>
                            List 50 other important features that are often present in business and financial news. Need to follow the template above, i.e. <example>features</example>.""",
                                            
                "science/technology": """Here are some examples of key features that are often present in news related to science and technology. Each feature is shown between the tag <example></example>.
                            <example>Name of scientists or the word scientists</example>
                            <example>Words related to technical devices</example>
                            <example>Words related to universe, space, planet</example>
                            <example>Words related to computer</example>
                            List 50 other important features that are often present in news related to science and technology. Need to follow the template above, i.e. <example>features</example>."""
            },
            
            "DBpedia": {
                "company": """Here are some examples of key features that are often present when introducing a company. Each feature is shown between the tag <example></example>.
                            <example>The name of the company</example>
                            <example>The location of the company</example>
                            <example>The founding year of the company</example>
                            <example>Words related to organization, group</example>
                            List 30 other important features that are often present when introducing a company. Need to follow the template above, i.e. <example>features</example>.""",
                                            
                "educational_institution": """Here are some examples of key features that are often present when introducing an educational institution. Each feature is shown between the tag <example></example>.
                            <example>The name of the school</example>
                            <example>The location of the school</example>
                            <example>The founding year of the school</example>
                            <example>Words related to college, university</example>
                            List 30 other important features that are often present when introducing an educational institution. Need to follow the template above, i.e. <example>features</example>.""",
                
                "artist": """Here are some examples of key features that are often present when introducing an artist. Each feature is shown between the tag <example></example>.
                            <example>The artist's name</example>
                            <example>The artist's works</example>
                            <example>The artist's birth date</example>
                            <example>Words related to music, painting</example>
                            List 30 other important features that are often present when introducing an artist. Need to follow the template above, i.e. <example>features</example>.""",
                
                "athlete": """Here are some examples of key features that are often present when introducing an athlete or sports star. Each feature is shown between the tag <example></example>.
                            <example>The athlete's or sports star's name</example>
                            <example>The sport the athlete plays</example>
                            <example>The athlete's birth date</example>
                            <example>Words related to ball games, competition</example>
                            List 30 other important features that are often present when introducing an athlete or sports star. Need to follow the template above, i.e. <example>features</example>.""",
                
                "village": """Here are some examples of key features that are often present when introducing a village. Each feature is shown between the tag <example></example>.
                            <example>The name of the village</example>
                            <example>The population of the village</example>
                            <example>The census of the village</example>
                            <example>Words related to district, families</example>
                            List 30 other important features that are often present when introducing a village. Need to follow the template above, i.e. <example>features</example>.""",
                
                "animal": """Here are some examples of key features that are often present when introducing a kind of animal. Each feature is shown between the tag <example></example>.
                            <example>The species of the animal</example>
                            <example>The habitat of the animal</example>
                            <example>The type of the animal (bird, insect, moth)</example>
                            <example>Words related to genus, family</example>
                            List 30 other important features that are often present when introducing a kind of animal. Need to follow the template above, i.e. <example>features</example>.""",
                
                "plant": """Here are some examples of key features that are often present when introducing a kind of plant. Each feature is shown between the tag <example></example>.
                            <example>The name of the plant</example>
                            <example>The genus or family of plant</example>
                            <example>The place where the plant was found</example>
                            <example>Words related to grass, flower</example>
                            List 30 other important features that are often present when introducing a kind of plant. Need to follow the template above, i.e. <example>features</example>.""",
                
                "album": """Here are some examples of key features that are often present when introducing an album. Each feature is shown between the tag <example></example>.
                            <example>The name of the album</example>
                            <example>The type of music, instrument</example>
                            <example>The release date of the album</example>
                            <example>Words related to band, audio</example>
                            List 30 other important features that are often present when introducing an album. Need to follow the template above, i.e. <example>features</example>.""",
                
                "film": """Here are some examples of key features that are often present when introducing a film. Each feature is shown between the tag <example></example>.
                            <example>The name of the film</example>
                            <example>The maker or producer of the film</example>
                            <example>The type of the film (e.g. fiction, comedy, cartoon, animation)</example>
                            <example>Words related to TV, video</example>
                            List 30 other important features that are often present when introducing a film. Need to follow the template above, i.e. <example>features</example>.""",
                
                "written_work": """Here are some examples of key features that are often present when introducing a written work. Each feature is shown between the tag <example></example>.
                            <example>The name of the written work</example>
                            <example>The author of the film</example>
                            <example>The type of the written work (e.g. novel, manga, journal)</example>
                            <example>Words related to book</example>
                            List 30 other important features that are often present when introducing a written work. Need to follow the template above, i.e. <example>features</example>."""
            }
        }
        self.hint = """don't give any number, especially bullet points number like(1. concept, 2. concept), also don't give sentences, just phrases"""
    
    async def generate_concepts(self, dataset_name: str) -> Dict:
        if dataset_name not in self.prompts:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        concepts_dict = {}

        for class_name in self.prompts[dataset_name]:
            print(f"Processing '{class_name}'...")

            all_raw_concepts = []
            for part in range(2):  # Two prompts of 50 each
                prompt = (
                    "Do not include any numbering like '1.', '2.', etc. Just use <example> tags.\n"
                    + self.prompts[dataset_name][class_name]
                    + f"\nList 50 different features (Part {part + 1}).\n{self.hint}"
                )

                try:
                    response = await self.llm_client.query(prompt)
                    concepts = self._parse_response_to_concepts(response)
                    all_raw_concepts.extend(concepts)
                    await asyncio.sleep(1.5)
                except Exception as e:
                    print(f"Error on part {part + 1}: {e}")

            # Deduplicate raw concepts
            unique_concepts = list(dict.fromkeys(all_raw_concepts))
            filtered = self._filter_concepts(unique_concepts)
            concepts_dict[class_name] = filtered

            print(f"  → {class_name}: {len(filtered)} concepts (before: {len(unique_concepts)})")

        os.makedirs("outputs/cb_llm_concepts", exist_ok=True)
        json_path = f"outputs/cb_llm_concepts/cb_llm_{dataset_name}.json"
        with open(json_path, 'w') as f:
            json.dump(concepts_dict, f, indent=2)

        print(f"Saved concepts to {json_path}")
        return concepts_dict

    def _parse_response_to_concepts(self, response: str) -> List[str]:
        matches = re.findall(r'<example>(.*?)</example>', response, re.IGNORECASE | re.DOTALL)
        if matches:
            return [m.strip() for m in matches if m.strip()]
        lines = [re.sub(r'^[-•*]|\d+[\.\)]\s*', '', l).strip() for l in response.splitlines()]
        return [l for l in lines if l]

    def _clean_concept(self, concept: str) -> str:
        concept = re.sub(r'\s+', ' ', concept).strip('\'" ')
        concept = re.sub(r'[.!?;:]+$', '', concept)
        return concept

    def _filter_concepts(self, concepts: List[str]) -> List[str]:
        seen = set()
        filtered = []
        stats = {
            "total": len(concepts),
            "empty": 0, "duplicates": 0, "too_short": 0, "too_long": 0,
            "blacklist": 0, "short_word": 0, "no_alpha": 0, "generic_pattern": 0, "kept": 0
        }

        for concept in concepts:
            cleaned = self._clean_concept(concept)
            if not cleaned:
                stats["empty"] += 1
                continue

            clower = cleaned.lower()
            if clower in seen:
                stats["duplicates"] += 1
                continue
            seen.add(clower)

            if len(cleaned) < self.filter_config['min_length']:
                stats["too_short"] += 1
                continue
            if len(cleaned) > self.filter_config['max_length']:
                stats["too_long"] += 1
                continue
            if clower in self.filter_config['blacklist']:
                stats["blacklist"] += 1
                continue
            if any(len(w) < self.filter_config['min_word_length'] for w in cleaned.split() if w.isalpha()):
                stats["short_word"] += 1
                continue
            if not re.search(r'[a-zA-Z]', cleaned):
                stats["no_alpha"] += 1
                continue
            if any(re.search(p, clower) for p in [
                r'^(some|any|many|various|different)\s',
                r'(usually|often|typically|generally|commonly)',
                r'^(used for|good for|helpful for)',
            ]):
                stats["generic_pattern"] += 1
                continue

            filtered.append(cleaned)
            stats["kept"] += 1

        print("📊 Filter Loss Report:")
        for k, v in stats.items():
            print(f"  {k:>15}: {v}")
        return filtered

    async def generate_all_datasets(self) -> Dict[str, Dict[str, List[str]]]:
        all_results = {}
        for dataset in self.prompts:
            try:
                results = await self.generate_concepts(dataset)
                all_results[dataset] = results
            except Exception as e:
                print(f"Error generating {dataset}: {e}")
                all_results[dataset] = {}
        return all_results

    def export_for_cbm_benchmark(self, all_concepts: Dict[str, Dict[str, List[str]]],
                                output_path: str = "concepts.py") -> str:
        """Export flattened concepts per dataset: sst2 = [...] without class names"""
        with open(output_path, 'w') as f:
            for dataset_name, class_dict in all_concepts.items():
                flat_list = []
                for concept_list in class_dict.values():
                    flat_list.extend(concept_list)
                deduped = list(dict.fromkeys(flat_list))  # optional deduplication
                dataset_var = dataset_name.lower()
                f.write(f"{dataset_var} = {json.dumps(deduped, ensure_ascii=False)}\n\n")

        print(f"✅ Exported flat concepts to {output_path}")
        return output_path



# Example usage following CBM-benchmark-project pattern
async def main():
    """Main function demonstrating CB-LLM concept generation"""
    
    print("=== CB-LLM Concept Generator ===")
    
    # Initialize querier with config (adjust path as needed)
    config_path = "config/llm_config.json"  # Adjust to your config path
    querier = CBLLMQuerier(config_path, enable_detailed_logging=True)
    
    # Generate concepts for a specific dataset
    dataset_name = "SST2"  # Change as needed
    try:
        concepts = await querier.generate_concepts(dataset_name)
        print(f"\nGenerated concepts for {dataset_name}:")
        for class_name, concept_list in concepts.items():
            print(f"  {class_name}: {len(concept_list)} concepts")
            if concept_list:
                print(f"    Sample: {concept_list[:3]}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Optionally generate for all datasets
    # all_concepts = await querier.generate_all_datasets()
    # querier.export_for_cbm_benchmark(all_concepts, "outputs/cb_llm_concepts/concepts.py")


if __name__ == "__main__":
    asyncio.run(main())