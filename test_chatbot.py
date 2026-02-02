"""
Test the improved chatbot output format.
"""
import requests
import json

API_BASE = "http://localhost:8000"

def test_chat(question: str):
    """Send a question to the chatbot and display the full response."""
    print(f"\nQuestion: {question}")
    print("=" * 70)
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"message": question},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\nFULL ANSWER:")
            print("-" * 70)
            print(result.get("answer", "No answer"))
            print("-" * 70)
            
            # Verification summary
            verification = result.get("verification_summary", {})
            if verification:
                level = verification.get("level", "unknown")
                support_rate = verification.get("support_rate", 0)
                print(f"\nVerification Level: {level.upper()} ({support_rate:.1%})")
                print(f"   Supported: {verification.get('supported_count', 0)}/{verification.get('total_checked', 0)}")
            
            verification = result.get("verification_summary", {})
            if verification:
                level = verification.get("level", "unknown")
                support_rate = verification.get("support_rate", 0)
                print(f"\nVerification Level: {level.upper()} ({support_rate:.1%})")
                print(f"   Supported: {verification.get('supported_count', 0)}/{verification.get('total_checked', 0)}")
                
                print("\n   Sentence Breakdown:")
                for item in verification.get("items", []):
                    status = "[PASS]" if item.get("is_supported") else "[FAIL]"
                    print(f"   {status} {item.get('sentence')[:50]}...")
                    if not item.get("is_supported"):
                        print(f"          Reason: {item.get('reason')}")
            
            comparison = result.get("model_comparison", {})
            if comparison.get("confidence"):
                print(f"\nConfidence: Model A: {comparison['confidence']['model_a']['score_percent']}, "
                      f"Model B: {comparison['confidence']['model_b']['score_percent']}")
                
            if comparison.get("hallucination"):
                 print(f"\nHallucination Detection:")
                 print(f"  Model A: {comparison['hallucination']['model_a'].get('supported_ratio')} ({comparison['hallucination']['model_a'].get('accuracy_estimate')})")
                 print(f"  Model B: {comparison['hallucination']['model_b'].get('supported_ratio')} ({comparison['hallucination']['model_b'].get('accuracy_estimate')})")
            
            print("\nTest complete.")
            
        else:
            print(f"[ERROR] Status code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to API. Is the server running?")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    # Test with reduce_hallucination mode
    question = "who is the new president of indonesia?"
    print(f"\nTesting Question: {question}")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={
                "message": question,
                "mode": "reduce_hallucination"
            },
            timeout=120
        )
        if response.status_code == 200:
            result = response.json()
            
            # Save to file for debug
            with open("debug_result.json", "w") as f:
                json.dump(result, f, indent=2)
                
            print("\nFULL ANSWER:")
            print("-" * 70)
            print(result.get("answer", "No answer"))
            print("-" * 70)
            
            verification = result.get("verification_summary", {})
            if verification:
                level = verification.get("level", "unknown")
                support_rate = verification.get("support_rate", 0)
                print(f"\nVerification Level: {level.upper()} ({support_rate:.1%})")
                print(f"   Supported: {verification.get('supported_count', 0)}/{verification.get('total_checked', 0)}")
            
            comparison = result.get("model_comparison", {})
            if comparison.get("hallucination"):
                 print(f"\nHallucination Detection:")
                 print(f"  Model A: {comparison['hallucination']['model_a'].get('supported_ratio')} ({comparison['hallucination']['model_a'].get('accuracy_estimate')})")
                 print(f"  Model B: {comparison['hallucination']['model_b'].get('supported_ratio')} ({comparison['hallucination']['model_b'].get('accuracy_estimate')})")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
