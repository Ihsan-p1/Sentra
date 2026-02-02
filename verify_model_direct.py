import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("LLM_MODEL")

print(f"Testing Model: {model_name}")
print(f"API Key: {api_key[:5]}...")

payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": "Hello, are you working?"}],
    "max_tokens": 50
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "Sentra Test"
}

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=30
    )
    
    if response.status_code == 200:
        print("\nSUCCESS! Model is working.")
        print(response.json()['choices'][0]['message']['content'])
    else:
        print(f"\nFAILED. Status: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"\nError: {e}")
