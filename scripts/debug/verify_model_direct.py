"""
Direct connectivity test for the configured LLM endpoint.

Provider-agnostic: reads LLM_BASE_URL / LLM_MODEL from .env, so it works against
local Ollama (http://localhost:11434/v1) as well as any OpenAI-compatible API.
"""
import os
import sys

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
model_name = os.getenv("LLM_MODEL", "llama3.1")
api_key = os.getenv("OPENROUTER_API_KEY") or "ollama"  # local servers ignore the value

print(f"Endpoint: {base_url}")
print(f"Model:    {model_name}")

payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": "Hello, are you working? Answer in one sentence."}],
    "max_tokens": 50,
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

try:
    response = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=120)

    if response.status_code == 200:
        print("\nSUCCESS! Model is responding.")
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print(f"\nFAILED. Status: {response.status_code}")
        print(response.text[:500])

except requests.exceptions.ConnectionError:
    print(f"\nCould not reach {base_url}. Is the server running? For Ollama: 'ollama serve'")
except Exception as e:
    print(f"\nError: {e}")
