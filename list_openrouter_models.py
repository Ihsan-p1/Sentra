import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY not found in .env")
    exit(1)

print(f"Querying OpenRouter API with key: {api_key[:5]}...")

try:
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\nAvailable FREE models:")
        count = 0
        for model in data['data']:
            if ":free" in model['id']:
                print(f"- {model['id']} ({model.get('name', 'Unknown')})")
                count += 1
        
        if count == 0:
            print("No models with ':free' suffix found. Listing all models containing 'free' in ID...")
            for model in data['data']:
                if "free" in model['id'].lower():
                    print(f"- {model['id']}")
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Error: {e}")
