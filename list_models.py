import google.generativeai as genai
import os
from config.settings import settings

api_key = settings.GOOGLE_API_KEY
if not api_key:
    # Try reading directly from .env if settings fail
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

print(f"Using API Key: {api_key[:5]}...")

try:
    genai.configure(api_key=api_key)
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")
