import yaml
import os
import asyncio
import argparse
from pathlib import Path

import sys
sys.path.append('/kayla/llm_query_module/src') 

print(sys.path)

from label_free_querier import LabelFreeQuerier
from labo_querier import LaBoQuerier
from lm4cv_querier import LM4CVQuerier
from cb_llm_querier import CBLLMQuerier
from utils.detailed_logger import DetailedLogger

class AsyncLLMQueryInterface:
    def __init__(self, config_path: str = "config/query_config.yaml"):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.session_logger = DetailedLogger()

    def _load_classes_from_config(self, dataset_name: str) -> list:
        """Load class names from file specified in config"""
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        
        dataset_config = self.config['datasets'][dataset_name]
        
        if 'classes_file' not in dataset_config:
            raise ValueError(f"No classes_file specified for dataset {dataset_name}")
        
        classes_file = dataset_config['classes_file']
        filepath = Path(classes_file)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Classes file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        return classes

    def get_available_datasets(self):
        """Get list of available datasets from config"""
        return list(self.config['datasets'].keys())

    def get_dataset_classes(self, dataset_name: str):
        """Get classes for a specific dataset"""
        if dataset_name == "custom":
            return []
        try:
            return self._load_classes_from_config(dataset_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: Could not load classes for {dataset_name}: {e}")
            return []

    def get_dataset_info(self):
        """Get dataset information from user"""
        print("\n Available datasets:")
        
        available_datasets = list(self.config['datasets'].keys())
        
        for i, dataset in enumerate(available_datasets, 1):
            description = self.config['datasets'][dataset].get('description', 'No description')
            print(f"[{i}] {dataset} - {description}")
        print(f"[{len(available_datasets)+1}] Custom dataset")
        
        while True:
            choice = input(f"\nChoose dataset (1-{len(available_datasets)+1}): ").strip()
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_datasets):
                    dataset_name = available_datasets[choice_idx]
                    
                    try:
                        class_names = self._load_classes_from_config(dataset_name)
                        print(f" Loaded {len(class_names)} classes for {dataset_name}")
                        return dataset_name, class_names
                        
                    except (FileNotFoundError, ValueError) as e:
                        print(f" Error loading {dataset_name}: {e}")
                        continue  
                    
                elif choice_idx == len(available_datasets):
                    return self._fallback_to_custom()
                    
            except ValueError:
                pass
            
            print(" Invalid choice. Please try again.")

    def _fallback_to_custom(self):
        """Fallback to custom dataset input"""
        dataset_name = input("Enter dataset name: ").strip()
        class_names_input = input("Enter class names (comma-separated): ").strip()
        class_names = [name.strip() for name in class_names_input.split(',')]
        return dataset_name, class_names

    async def main_menu(self):
        """Interactive main menu"""
        while True:
            print("\n" + "="*50)
            print(" Async LLM Concept Query System")
            print("="*50)
            print("Choose a method:")
            print("[1] Label-Free CBM (3 JSON files + filtered TXT)")
            print("[2] LaBo CBM (2 JSON files: concepts + selected)")
            print("[3] LM4CV (TXT attributes + JSON class mapping)")
            print("[4] CB-LLM (Concept Bank generation)")
            print("[5] Run All Methods")
            print("[6] Show Configuration")
            print("[0] Exit")
            
            choice = input("\nEnter choice (0-6): ").strip()
            
            if choice == "0":
                print(" Goodbye!")
                break
            elif choice == "1":
                await self.run_label_free()
            elif choice == "2":
                await self.run_labo()
            elif choice == "3":
                await self.run_lm4cv()
            elif choice == "4":
                await self.run_cb_llm()
            elif choice == "5":
                await self.run_all_methods()
            elif choice == "6":
                self.show_configuration()
            else:
                print(" Invalid choice. Please try again.")

    def show_configuration(self):
        """Display current configuration"""
        print("\n Current Configuration:")
        print("-" * 30)
        
        print(" Available Datasets:")
        for dataset, config in self.config['datasets'].items():
            classes_file = config.get('classes_file', 'Not specified')
            description = config.get('description', 'No description')
            file_exists = "" if Path(classes_file).exists() else "❌"
            print(f"  {file_exists} {dataset}: {classes_file}")
            print(f"     {description}")
        
        print(f"\n LLM Configuration:")
        llm_config = self.config.get('llm', {})
        print(f"  Provider: {llm_config.get('provider', 'Not specified')}")
        print(f"  Model: {llm_config.get('model', 'Not specified')}")
        print(f"  Temperature: {llm_config.get('temperature', 'Not specified')}")
        
        print(f"\n Output Directory: {self.config.get('output', {}).get('base_dir', 'outputs')}")
    
    async def run_cb_llm(self):
        """Run CB-LLM pipeline"""
        print("\n CB-LLM Concept Bank Pipeline")
        
        print("\n CB-LLM supports these datasets:")
        cb_datasets = ["SST2", "YelpP", "AGNews", "DBpedia"]
        for i, dataset in enumerate(cb_datasets, 1):
            print(f"[{i}] {dataset}")
        print(f"[{len(cb_datasets)+1}] Generate all datasets")
        
        while True:
            choice = input(f"\nChoose dataset (1-{len(cb_datasets)+1}): ").strip()
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(cb_datasets):
                    dataset_name = cb_datasets[choice_idx]
                    break
                elif choice_idx == len(cb_datasets):
                    dataset_name = "all"
                    break
            except ValueError:
                pass
            
            print(" Invalid choice. Please try again.")
        
        querier = CBLLMQuerier(self.config_path, enable_detailed_logging=True)
        
        try:
            if dataset_name == "all":
                print(" Generating concepts for all CB-LLM datasets...")
                all_results = await querier.generate_all_datasets()
                
                export_path = "outputs/cb_llm_concepts/concepts.py"
                querier.export_for_cbm_benchmark(all_results, export_path)
                
                print(f"\n CB-LLM Complete for all datasets!")
                print(f" Files saved in outputs/cb_llm_concepts/")
                print(f' CBM-benchmark format exported to {export_path}")
                
            else:
                print(f" Generating concepts for {dataset_name}...")
                concepts = await querier.generate_concepts(dataset_name)
                
                # Apply filtering
                print( Applying filtering...")
                filtered_concepts = await querier.apply_filtering(concepts, dataset_name)
                
                print(f"\n CB-LLM Complete for {dataset_name}!")
                print(f" Generated {len(filtered_concepts)} filtered concepts")
                print(f" Files saved in outputs/cb_llm_concepts/")
                
        except Exception as e:
            print(f" Error in CB-LLM pipeline: {e}")
            return
        
        print(f" Detailed logs saved in query_logs/")
    
    async def run_label_free(self):
        """Run Label-Free CBM pipeline"""
        print("\n Label-Free CBM Pipeline")
        
        dataset_name, class_names = self.get_dataset_info()
        
        querier = LabelFreeQuerier(self.config_path, enable_detailed_logging=True)
        
        # Generate concepts
        print(" Generating concepts...")
        try:
            if asyncio.iscoroutinefunction(querier.generate_concepts):
                concepts = await querier.generate_concepts(class_names, dataset_name)
            else:
                concepts = querier.generate_concepts(class_names, dataset_name)
        except Exception as e:
            print(f" Error generating concepts: {e}")
            return
        
        # Apply filtering
        print(" Applying filtering...")
        try:
            if asyncio.iscoroutinefunction(querier.apply_filtering):
                filtered_concepts = await querier.apply_filtering(concepts, dataset_name)
            else:
                filtered_concepts = querier.apply_filtering(concepts, dataset_name)
        except Exception as e:
            print(f" Error applying filtering: {e}")
            return
        
        print(f"\n Label-Free CBM Complete!")
        print(f" Generated {len(filtered_concepts)} filtered concepts")
        print(f" Files saved in outputs/label_free/")
        print(f" Detailed logs saved in query_logs/")

        if hasattr(querier, 'logger') and querier.logger:
            querier.logger.create_summary_report()

    async def run_labo(self):
        """Run LaBo pipeline"""
        print("\n LaBo CBM Pipeline")
        
        dataset_name, class_names = self.get_dataset_info()
        
        querier = LaBoQuerier(self.config_path)
        
        print(" Generating concepts...")
        try:
            if asyncio.iscoroutinefunction(querier.generate_concepts):
                class2concepts = await querier.generate_concepts(class_names, dataset_name)
            else:
                class2concepts = querier.generate_concepts(class_names, dataset_name)
        except Exception as e:
            print(f" Error generating concepts: {e}")
            return
        
        print(" Performing submodular selection...")
        default_k = self.config.get('query_settings', {}).get('labo', {}).get('concepts_per_class', 25)
        try:
            k_per_class = int(input(f"Enter concepts per class (default {default_k}): ") or str(default_k))
        except ValueError:
            k_per_class = default_k
        
        try:
            if asyncio.iscoroutinefunction(querier.submodular_selection):
                selected_concepts = await querier.submodular_selection(class2concepts, dataset_name, k_per_class)
            else:
                selected_concepts = querier.submodular_selection(class2concepts, dataset_name, k_per_class)
        except Exception as e:
            print(f" Error in submodular selection: {e}")
            return
        
        if selected_concepts:
            total_selected = sum(len(concepts) for concepts in selected_concepts.values())
            print(f"\n LaBo CBM Complete!")
            print(f" Selected {total_selected} concepts total")
        else:
            print(f"\n LaBo CBM Complete!")
        print(f" Files saved in outputs/labo/")
    
    async def run_lm4cv(self):
        """Run LM4CV pipeline"""
        print("\n LM4CV Pipeline")
        
        dataset_name, class_names = self.get_dataset_info()
        
        querier = LM4CVQuerier(self.config_path)
        
        print(" Generating attributes...")
        try:
            if asyncio.iscoroutinefunction(querier.generate_attributes):
                attributes, cls2attributes = await querier.generate_attributes(class_names, dataset_name)
            else:
                attributes, cls2attributes = querier.generate_attributes(class_names, dataset_name)
        except Exception as e:
            print(f" Error generating attributes: {e}")
            return
        
        print(f"\n LM4CV Complete!")
        print(f" Generated {len(attributes)} unique attributes")
        print(f" Created mappings for {len(cls2attributes)} classes")
        print(f" Files saved in outputs/lm4cv/")
    
    async def run_all_methods(self):
        """Run all four methods"""
        print("\n Running All Methods")
        
        dataset_name, class_names = self.get_dataset_info()
        
        print("\n" + "="*30)
        await self.run_label_free_method(dataset_name, class_names)
        
        print("\n" + "="*30)
        await self.run_labo_method(dataset_name, class_names)
        
        print("\n" + "="*30)
        await self.run_lm4cv_method(dataset_name, class_names)
        
        print("\n" + "="*30)
        await self.run_cb_llm_method()
        
        print(f"\n All methods completed!")
    
    async def run_label_free_method(self, dataset_name, class_names):
        """Helper method for running label-free in batch mode"""
        print(" Running Label-Free CBM...")
        querier = LabelFreeQuerier(self.config_path)
        
        try:
            if asyncio.iscoroutinefunction(querier.generate_concepts):
                concepts = await querier.generate_concepts(class_names, dataset_name)
            else:
                concepts = querier.generate_concepts(class_names, dataset_name)
                
            if asyncio.iscoroutinefunction(querier.apply_filtering):
                filtered_concepts = await querier.apply_filtering(concepts, dataset_name)
            else:
                filtered_concepts = querier.apply_filtering(concepts, dataset_name)
                
            print(f"Label-Free: {len(filtered_concepts)} concepts")
        except Exception as e:
            print(f" Label-Free failed: {e}")
    
    async def run_labo_method(self, dataset_name, class_names):
        """Helper method for running LaBo in batch mode"""
        print(" Running LaBo CBM...")
        querier = LaBoQuerier(self.config_path)
        
        try:
            if asyncio.iscoroutinefunction(querier.generate_concepts):
                class2concepts = await querier.generate_concepts(class_names, dataset_name)
            else:
                class2concepts = querier.generate_concepts(class_names, dataset_name)
                
            default_k = self.config.get('query_settings', {}).get('labo', {}).get('concepts_per_class', 25)
            
            if asyncio.iscoroutinefunction(querier.submodular_selection):
                selected_concepts = await querier.submodular_selection(class2concepts, dataset_name, default_k)
            else:
                selected_concepts = querier.submodular_selection(class2concepts, dataset_name, default_k)
                
            if selected_concepts:
                total = sum(len(concepts) for concepts in selected_concepts.values())
                print(f"LaBo: {total} concepts")
            else:
                print("LaBo: completed")
        except Exception as e:
            print(f" LaBo failed: {e}")
    
    async def run_lm4cv_method(self, dataset_name, class_names):
        """Helper method for running LM4CV in batch mode"""
        print(" Running LM4CV...")
        querier = LM4CVQuerier(self.config_path)
        
        try:
            if asyncio.iscoroutinefunction(querier.generate_attributes):
                attributes, cls2attributes = await querier.generate_attributes(class_names, dataset_name)
            else:
                attributes, cls2attributes = querier.generate_attributes(class_names, dataset_name)
                
            print(f"LM4CV: {len(attributes)} attributes")
        except Exception as e:
            print(f" LM4CV failed: {e}")
    
    async def run_cb_llm_method(self):
        """Helper method for running CB-LLM in batch mode"""
        print(" Running CB-LLM...")
        querier = CBLLMQuerier(self.config_path)
        
        try:
            concepts = await querier.generate_concepts("SST2")
            filtered_concepts = await querier.apply_filtering(concepts, "SST2")
            print(f"CB-LLM: {len(filtered_concepts)} concepts for SST2")
        except Exception as e:
            print(f" CB-LLM failed: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Async LLM Concept Query System")
    
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--show-config", action="store_true", help="Show configuration")
    parser.add_argument("--method", "-m", choices=["label-free", "labo", "lm4cv", "cb-llm", "all"], help="Method to run")
    parser.add_argument("--dataset", "-d", help="Dataset to use")
    parser.add_argument("--classes", nargs="+", help="Custom class names")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--config", "-c", default="config/query_config.yaml", help="Config file path")
    
    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_arguments()
    
    if not Path(args.config).exists():
        print(f" Config file not found: {args.config}")
        return
    
    try:
        interface = AsyncLLMQueryInterface(args.config)
        print(" Interface initialized successfully!")
    except Exception as e:
        print(f" Failed to initialize: {e}")
        return
    
    if args.list_datasets:
        print("\n Available datasets:")
        for dataset in interface.get_available_datasets():
            description = interface.config['datasets'][dataset].get('description', 'No description')
            print(f"  • {dataset} - {description}")
        return
    
    if args.show_config:
        interface.show_configuration()
        return
    
    if args.interactive or not args.method:
        await interface.main_menu()
    else:
        print("Command line mode not implemented yet - use --interactive")


if __name__ == "__main__":
    asyncio.run(main())
