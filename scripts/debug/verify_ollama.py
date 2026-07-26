"""
Check that the Ollama server is reachable and serving the configured model.
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

# Derive the host from LLM_BASE_URL so this checks the endpoint the app actually uses
host = settings.LLM_BASE_URL.rsplit("/v1", 1)[0].rstrip("/")
expected_model = settings.LLM_MODEL

try:
    print(f"Checking Ollama API at {host}/api/tags ...")
    response = requests.get(f"{host}/api/tags", timeout=10)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text[:200]}")
        sys.exit(1)

    models = [m["name"] for m in response.json().get("models", [])]
    print(f"Connection successful. Found {len(models)} model(s):")
    for name in models:
        marker = "  <- configured in .env" if name == expected_model else ""
        print(f"- {name}{marker}")

    if expected_model not in models:
        print(f"\nWARNING: '{expected_model}' (LLM_MODEL) is NOT available.")
        print("Build it with: ollama create sentra-qwen:3b -f ollama/Modelfile")
        sys.exit(1)

    print(f"\nOK: '{expected_model}' is ready.")

except requests.exceptions.RequestException as e:
    print(f"Connection failed: {e}")
    print("Is Ollama running? Start the Ollama app, or run 'ollama serve'.")
    sys.exit(1)
