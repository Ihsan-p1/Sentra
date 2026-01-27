"""
Test the chatbot API with the new dataset.
"""
import requests
import json

API_BASE = "http://localhost:8000"

def test_chat(question: str):
    """Send a question to the chatbot and display the response."""
    print(f"\n🗨️ Question: {question}")
    print("-" * 60)
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"message": question},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📝 Answer:")
            print(result.get("answer", "No answer"))
            
            print(f"\n📊 Confidence: {result.get('confidence', 'N/A')}")
            
            comparison = result.get("comparison", {})
            
            # Hallucination
            hall = comparison.get("hallucination", {})
            print(f"\n🔍 Hallucination Detection:")
            print(f"  Model A ({hall.get('model_a', {}).get('name', 'N/A')}): {hall.get('model_a', {}).get('result', 'N/A')}")
            print(f"  Model B ({hall.get('model_b', {}).get('name', 'N/A')}): {hall.get('model_b', {}).get('result', 'N/A')}")
            
            # Framing
            framing = comparison.get("framing", {})
            print(f"\n📰 Framing Keywords:")
            for model_key in ["model_a", "model_b"]:
                model = framing.get(model_key, {})
                print(f"  {model.get('name', model_key)}:")
                keywords = model.get("keywords", {})
                for media, kws in keywords.items():
                    print(f"    {media}: {', '.join(kws) if kws else 'None'}")
            
            print("\n✅ Chatbot is working!")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the server running?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_chat("What are the key differences in how media sources reported Prabowo's election victory?")
