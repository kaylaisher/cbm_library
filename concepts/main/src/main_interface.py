import yaml
import os
from pathlib import Path
from label_free_querier import LabelFreeQuerier
from labo_querier import LaBoQuerier
from lm4cv_querier import LM4CVQuerier
from utils.detailed_logger import DetailedLogger


class LLMQueryInterface:
    def __init__(self, config_path: str = "config/query_config.yaml"):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize session logger
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

    def get_dataset_info(self):
        """Get dataset information from user"""
        print("\n📂 Available datasets:")
        
        # Get datasets from config
        available_datasets = list(self.config['datasets'].keys())
        
        # Display options
        for i, dataset in enumerate(available_datasets, 1):
            description = self.config['datasets'][dataset].get('description', 'No description')
            print(f"[{i}] {dataset} - {description}")
        print(f"[{len(available_datasets)+1}] Custom dataset")
        
        choice = input(f"\nChoose dataset (1-{len(available_datasets)+1}): ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_datasets):
                dataset_name = available_datasets[choice_idx]
                
                # Load classes from config-specified file
                try:
                    class_names = self._load_classes_from_config(dataset_name)
                    print(f"✅ Loaded {len(class_names)} classes for {dataset_name}")
                    return dataset_name, class_names
                    
                except (FileNotFoundError, ValueError) as e:
                    print(f"❌ Error loading {dataset_name}: {e}")
                    return self._fallback_to_custom()
                
            elif choice_idx == len(available_datasets):
                return self._fallback_to_custom()
                
        except ValueError:
            pass
        
        print("❌ Invalid choice. Please try again.")
        return self.get_dataset_info()

    def _fallback_to_custom(self):
        """Fallback to custom dataset input"""
        dataset_name = input("Enter dataset name: ").strip()
        class_names_input = input("Enter class names (comma-separated): ").strip()
        class_names = [name.strip() for name in class_names_input.split(',')]
        return dataset_name, class_names

    def main_menu(self):
        """Interactive main menu"""
        while True:
            print("\n" + "="*50)
            print("🤖 LLM Concept Query System")
            print("="*50)
            print("Choose a method:")
            print("[1] Label-Free CBM (3 JSON files + filtered TXT)")
            print("[2] LaBo CBM (2 JSON files: concepts + selected)")
            print("[3] LM4CV (TXT attributes + JSON class mapping)")
            print("[4] Run All Methods")
            print("[5] Show Configuration")
            print("[0] Exit")
            
            choice = input("\nEnter choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                self.run_label_free()
            elif choice == "2":
                self.run_labo()
            elif choice == "3":
                self.run_lm4cv()
            elif choice == "4":
                self.run_all_methods()
            elif choice == "5":
                self.show_configuration()
            else:
                print("❌ Invalid choice. Please try again.")

    def show_configuration(self):
        """Display current configuration"""
        print("\n📋 Current Configuration:")
        print("-" * 30)
        
        print("📂 Available Datasets:")
        for dataset, config in self.config['datasets'].items():
            classes_file = config.get('classes_file', 'Not specified')
            description = config.get('description', 'No description')
            file_exists = "✅" if Path(classes_file).exists() else "❌"
            print(f"  {file_exists} {dataset}: {classes_file}")
            print(f"     {description}")
        
        print(f"\n🤖 LLM Configuration:")
        llm_config = self.config.get('llm', {})
        print(f"  Provider: {llm_config.get('provider', 'Not specified')}")
        print(f"  Model: {llm_config.get('model', 'Not specified')}")
        print(f"  Temperature: {llm_config.get('temperature', 'Not specified')}")
        
        print(f"\n📁 Output Directory: {self.config.get('output', {}).get('base_dir', 'outputs')}")
    
    def run_label_free(self):
        """Run Label-Free CBM pipeline"""
        print("\n🔬 Label-Free CBM Pipeline")
        
        dataset_name, class_names = self.get_dataset_info()
        
        querier = LabelFreeQuerier(self.config_path, enable_detailed_logging=True)
        
        # Generate concepts
        concepts = querier.generate_concepts(class_names, dataset_name)
        
        # Apply filtering
        filtered_concepts = querier.apply_filtering(concepts, dataset_name)
        
        print(f"\n✅ Label-Free CBM Complete!")
        print(f"📊 Generated {len(filtered_concepts)} filtered concepts")
        print(f"📁 Files saved in outputs/label_free/")
        print(f"📝 Detailed logs saved in query_logs/")

        # Create summary
        if hasattr(querier, 'logger') and querier.logger:
            querier.logger.create_summary_report()

    def run_labo(self):
        """Run LaBo pipeline"""
        print("\n🍾 LaBo CBM Pipeline")
        
        dataset_name, class_names = self.get_dataset_info()
        
        querier = LaBoQuerier(self.config_path)
        
        # Generate concepts
        class2concepts = querier.generate_concepts(class_names, dataset_name)
        
        # Apply submodular selection
        default_k = self.config.get('query_settings', {}).get('labo', {}).get('concepts_per_class', 25)
        k_per_class = int(input(f"Enter concepts per class (default {default_k}): ") or str(default_k))
        selected_concepts = querier.submodular_selection(class2concepts, dataset_name, k_per_class)
        
        total_selected = sum(len(concepts) for concepts in selected_concepts.values())
        print(f"\n✅ LaBo CBM Complete!")
        print(f"📊 Selected {total_selected} concepts total")
        print(f"📁 Files saved in outputs/labo/")
    
    def run_lm4cv(self):
        """Run LM4CV pipeline"""
        print("\n🔍 LM4CV Pipeline")
        
        dataset_name, class_names = self.get_dataset_info()
        
        querier = LM4CVQuerier(self.config_path)
        
        # Generate attributes
        attributes, cls2attributes = querier.generate_attributes(class_names, dataset_name)
        
        print(f"\n✅ LM4CV Complete!")
        print(f"📊 Generated {len(attributes)} unique attributes")
        print(f"📊 Created mappings for {len(cls2attributes)} classes")
        print(f"📁 Files saved in outputs/lm4cv/")
    
    def run_all_methods(self):
        """Run all three methods"""
        print("\n🚀 Running All Methods")
        
        dataset_name, class_names = self.get_dataset_info()
        
        print("\n" + "="*30)
        self.run_label_free_method(dataset_name, class_names)
        
        print("\n" + "="*30)
        self.run_labo_method(dataset_name, class_names)
        
        print("\n" + "="*30)
        self.run_lm4cv_method(dataset_name, class_names)
        
        print(f"\n🎉 All methods completed for {dataset_name}!")
    
    def run_label_free_method(self, dataset_name, class_names):
        querier = LabelFreeQuerier(self.config_path)
        concepts = querier.generate_concepts(class_names, dataset_name)
        filtered_concepts = querier.apply_filtering(concepts, dataset_name)
        print(f"Label-Free: {len(filtered_concepts)} concepts")
    
    def run_labo_method(self, dataset_name, class_names):
        querier = LaBoQuerier(self.config_path)
        class2concepts = querier.generate_concepts(class_names, dataset_name)
        default_k = self.config.get('query_settings', {}).get('labo', {}).get('concepts_per_class', 25)
        selected_concepts = querier.submodular_selection(class2concepts, dataset_name, default_k)
        total = sum(len(concepts) for concepts in selected_concepts.values())
        print(f"LaBo: {total} concepts")
    
    def run_lm4cv_method(self, dataset_name, class_names):
        querier = LM4CVQuerier(self.config_path)
        attributes, cls2attributes = querier.generate_attributes(class_names, dataset_name)
        print(f"LM4CV: {len(attributes)} attributes")


def main():
    interface = LLMQueryInterface()
    interface.main_menu()


if __name__ == "__main__":
    main()