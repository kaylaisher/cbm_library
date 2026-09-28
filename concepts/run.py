import sys
import os
import asyncio
from pathlib import Path

# Get the root directory of the project (where this file is located)
ROOT_DIR = Path(__file__).parent.absolute()

# Add shared code and each concept-generation method folder to Python path
for sub in ('common', 'label_free', 'labo', 'lm4cv', 'cb_llm'):
    sys.path.insert(0, str(ROOT_DIR / sub))

# Change working directory to project root (important for relative paths)
os.chdir(ROOT_DIR)

# Now import and run
from async_main_interface_test import main

if __name__ == "__main__":
    print(f"🚀 Async LLM Query Module")
    print(f"📁 Running from: {ROOT_DIR}")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Run the async main function
    asyncio.run(main())

log_path = "cbllm_llm_outputs.log"

print("\n📄 ==== LLM Query Log ====\n")
try:
    with open(log_path, 'r') as f:
        print(f.read())
except FileNotFoundError:
    print(f"(No log file found at {log_path})")