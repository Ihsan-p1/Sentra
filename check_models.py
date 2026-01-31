
import google.generativeai as genai

API_KEY = "AIzaSyAa_m-XbKJ6YLDWV0TiKi3aEiHNyzM1UXU"

print(f"Configuring with Key: {API_KEY[:5]}...{API_KEY[-5:]}")
genai.configure(api_key=API_KEY)

print("Listing models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
