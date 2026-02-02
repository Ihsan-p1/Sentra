import requests
import json

try:
    print("Checking Ollama API at http://localhost:11434/api/tags ...")
    response = requests.get("http://localhost:11434/api/tags")
    
    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])
        print(f"Connection Successful! Found {len(models)} models.")
        for m in models:
            print(f"- {m['name']}")
            
        if not any('llama3.1' in m['name'] for m in models):
            print("\nWARNING: 'llama3.1' is NOT found.")
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Connection Failed: {e}")
    print("Is Ollama running? Try running 'ollama serve' in a separate terminal.")
