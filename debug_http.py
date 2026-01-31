
import asyncio
import httpx
import os

API_KEY = "AIzaSyAa_m-XbKJ6YLDWV0TiKi3aEiHNyzM1UXU"

async def test_model(model_name, version="v1beta"):
    url = f"https://generativelanguage.googleapis.com/{version}/models/{model_name}:generateContent?key={API_KEY}"
    print(f"\nTesting {model_name} ({version})...")
    print(f"URL: {url}")
    
    payload = {
        "contents": [{"parts": [{"text": "Hello, are you working?"}]}]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                print("SUCCESS!")
            else:
                print("FAILED.")
    except Exception as e:
        print(f"EXCEPTION: {e}")

async def main():
    models = ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-pro"]
    for m in models:
        await test_model(m)

if __name__ == "__main__":
    asyncio.run(main())
