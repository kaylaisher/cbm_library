import json
import os
import datetime
from pathlib import Path
from typing import Dict, List, Any
import yaml

class DetailedLogger:
    def __init__(self, output_dir: str = "query_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this session
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize log structure
        self.session_log = {
            "session_info": {
                "timestamp": self.timestamp,
                "start_time": datetime.datetime.now().isoformat(),
                "methods_run": []
            },
            "label_free_cbm": {},
            "labo_cbm": {},
            "lm4cv": {}
        }
        
        print(f"📝 Detailed logging enabled - Session: {self.timestamp}")
        print(f"📁 Logs will be saved to: {self.output_dir.absolute()}")
    
    def log_method_start(self, method_name: str, dataset_name: str, class_names: List[str]):
        """Log the start of a method execution"""
        method_info = {
            "method": method_name,
            "dataset": dataset_name,
            "classes": class_names,
            "num_classes": len(class_names),
            "start_time": datetime.datetime.now().isoformat(),
            "steps": []
        }
        
        self.session_log["session_info"]["methods_run"].append(method_info)
        
        # Create method-specific log file
        log_file = self.output_dir / f"{method_name}_{dataset_name}_{self.timestamp}.md"
        with open(log_file, 'w') as f:
            f.write(f"# {method_name.upper()} Detailed Execution Log\n\n")
            f.write(f"**Dataset:** {dataset_name}\n")
            f.write(f"**Classes:** {', '.join(class_names)}\n")
            f.write(f"**Start Time:** {method_info['start_time']}\n\n")
            f.write("---\n\n")
        
        return log_file
    
    def log_query_step(self, log_file: Path, step_name: str, prompt: str, 
                      response: str, parsed_result: List[str], class_name: str = None):
        """Log each individual query step with full details"""
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        with open(log_file, 'a') as f:
            f.write(f"## {step_name}")
            if class_name:
                f.write(f" - {class_name}")
            f.write(f" ({timestamp})\n\n")
            
            f.write("### 📝 Prompt Sent to LLM:\n```\n")
            f.write(prompt)
            f.write("\n```\n\n")
            
            f.write("### 🤖 Raw LLM Response:\n```\n")
            f.write(response)
            f.write("\n```\n\n")
            
            f.write("### ✅ Parsed Results:\n")
            for i, item in enumerate(parsed_result, 1):
                f.write(f"{i}. {item}\n")
            f.write(f"\n**Total items extracted:** {len(parsed_result)}\n\n")
            f.write("---\n\n")
    
    def log_filtering_step(self, log_file: Path, step_name: str, 
                          input_concepts: List[str], output_concepts: List[str],
                          removed_concepts: List[str] = None, criteria: str = None):
        """Log filtering steps with before/after comparison"""
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        with open(log_file, 'a') as f:
            f.write(f"## 🔍 Filtering Step: {step_name} ({timestamp})\n\n")
            
            if criteria:
                f.write(f"**Criteria:** {criteria}\n\n")
            
            f.write(f"**Input:** {len(input_concepts)} concepts\n")
            f.write(f"**Output:** {len(output_concepts)} concepts\n")
            f.write(f"**Removed:** {len(input_concepts) - len(output_concepts)} concepts\n\n")
            
            if removed_concepts:
                f.write("### ❌ Removed Concepts:\n")
                for concept in removed_concepts:
                    f.write(f"- {concept}\n")
                f.write("\n")
            
            f.write("### ✅ Remaining Concepts:\n")
            for i, concept in enumerate(output_concepts[:50], 1):  # Show first 50
                f.write(f"{i}. {concept}\n")
            
            if len(output_concepts) > 50:
                f.write(f"\n... and {len(output_concepts) - 50} more\n")
            
            f.write("\n---\n\n")
    
    def log_submodular_selection(self, log_file: Path, class_name: str,
                               candidate_concepts: List[str], selected_concepts: List[str],
                               selection_scores: Dict = None):
        """Log submodular selection process"""
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        with open(log_file, 'a') as f:
            f.write(f"## 🎯 Submodular Selection - {class_name} ({timestamp})\n\n")
            f.write(f"**Candidates:** {len(candidate_concepts)} concepts\n")
            f.write(f"**Selected:** {len(selected_concepts)} concepts\n\n")
            
            if selection_scores:
                f.write("### 📊 Selection Scores:\n")
                for concept, score in selection_scores.items():
                    f.write(f"- {concept}: {score:.3f}\n")
                f.write("\n")
            
            f.write("### ✅ Selected Concepts:\n")
            for i, concept in enumerate(selected_concepts, 1):
                f.write(f"{i}. {concept}\n")
            
            f.write("\n### ❌ Not Selected:\n")
            not_selected = [c for c in candidate_concepts if c not in selected_concepts]
            for concept in not_selected[:20]:  # Show first 20
                f.write(f"- {concept}\n")
            
            if len(not_selected) > 20:
                f.write(f"\n... and {len(not_selected) - 20} more\n")
            
            f.write("\n---\n\n")
    
    def log(self, message: str):
        log_file = self.output_dir / f"SESSION_LOG_{self.timestamp}.log"
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
        with open(log_file, 'a') as f:
            f.write(timestamp + message + "\n")

    def log_and_print(self, message: str):
        """Print to terminal and log to the general session file"""
        print(message)
        self.log(message)  # this assumes you've already added the `log()` method



    def create_summary_report(self):
        """Create a comprehensive summary report"""
        summary_file = self.output_dir / f"SUMMARY_REPORT_{self.timestamp}.md"
        
        with open(summary_file, 'w') as f:
            f.write("# LLM Query Module - Execution Summary Report\n\n")
            f.write(f"**Session ID:** {self.timestamp}\n")
            f.write(f"**Generated:** {datetime.datetime.now().isoformat()}\n\n")
            
            f.write("## 📊 Session Overview\n\n")
            f.write(f"**Methods Run:** {len(self.session_log['session_info']['methods_run'])}\n\n")
            
            for method_info in self.session_log['session_info']['methods_run']:
                f.write(f"### {method_info['method'].upper()}\n")
                f.write(f"- **Dataset:** {method_info['dataset']}\n")
                f.write(f"- **Classes:** {method_info['num_classes']}\n")
                f.write(f"- **Start Time:** {method_info['start_time']}\n\n")
            
            f.write("## 📁 Generated Files\n\n")
            f.write("### Detailed Logs:\n")
            for log_file in self.output_dir.glob("*_202*.md"):
                if "SUMMARY" not in log_file.name:
                    f.write(f"- [{log_file.name}]({log_file.name})\n")
            
            f.write("\n### Data Files:\n")
            f.write("Check the `outputs/` directory for generated concept files.\n\n")
        
        print(f"📋 Summary report created: {summary_file}")
        return summary_file