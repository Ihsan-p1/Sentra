import requests
import json

url = "http://127.0.0.1:8000/api/chat"
payload = {
    "message": "prabowo cabinet",
    "mode": "default"
}

try:
    print(f"Sending request...")
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"FAILED: {response.text}")
        exit()
    
    data = response.json()
    sources = data.get("sources", {})
    
    if not sources:
        print("SOURCES IS EMPTY")
        exit()

    print(f"Sources found: {list(sources.keys())}")
    
    for media, chunks in sources.items():
        print(f"Media: {media} | Count: {len(chunks)}")
        if chunks:
            c = chunks[0]
            print(f"  Title: {c.get('title')}")
            print(f"  URL: {c.get('url')}")
            print(f"  Text (first 50 chars): {c.get('text', '')[:50]}...")
            print(f"  Sim: {c.get('similarity')}")

except Exception as e:
    print(f"Error: {e}")
