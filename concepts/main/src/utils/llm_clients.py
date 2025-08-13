import httpx
import time
import yaml
import json
import datetime

class LLMClient:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config.get("model_name", "/openchat/openchat-3.5")
        self.base_url = self.config.get("openai_base_url", "http://localhost:8000/v1").rstrip('/')
        self.log_path = "cbllm_llm_outputs.log"

    async def query(self, prompt: str) -> str:
        MAX_CONTEXT_TOKENS = 8192
        prompt_token_estimate = len(prompt) // 4
        max_tokens = MAX_CONTEXT_TOKENS - prompt_token_estimate
        max_tokens = max(64, min(max_tokens, 4096))  # ← updated upper limit

        async with httpx.AsyncClient(timeout=600.0) as client:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens
            }

            try:
                response = await client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()

                content = response.json()["choices"][0]["message"]["content"]

                # Logging
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                output = (
                    f"\n🕒 {timestamp}\n📤 Prompt:\n{prompt}\n\n"
                    f"📥 Raw LLM Response:\n{content}\n" + ("=" * 60)
                )
                print(output)

                with open(self.log_path, 'a') as log_file:
                    log_file.write(output + "\n")

                return content

            except httpx.HTTPStatusError as e:
                print(f"❌ HTTP {e.response.status_code}: {e.response.text}")
                raise
