import sys
import os
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()

sys.path.insert(0, str(ROOT_DIR / 'src'))

os.chdir(ROOT_DIR)

from async_main_interface_test import main

if __name__ == "__main__":
    print(f" Async LLM Query Module")
    print(f" Running from: {ROOT_DIR}")
    print(f" Working directory: {os.getcwd()}")
    
    asyncio.run(main())

log_path = "cbllm_llm_outputs.log"

print("\n ==== LLM Query Log ====\n")
try:
    with open(log_path, 'r') as f:
        print(f.read())
except FileNotFoundError:
    print(f"(No log file found at {log_path})")
