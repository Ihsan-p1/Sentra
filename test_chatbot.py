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
            
            # Check for old-style per-sentence warnings
            answer = result.get("answer", "")
            warning_count = answer.count("[Unverified]")
            if warning_count > 0:
                print(f"\n[WARN] Found {warning_count} old-style per-sentence warnings.")
            else:
                print("\n[OK] No per-sentence warning spam detected.")
            
            # Model comparison
            comparison = result.get("model_comparison", {})
            if comparison.get("confidence"):
                print(f"\nConfidence: Model A: {comparison['confidence']['model_a']['score_percent']}, "
                      f"Model B: {comparison['confidence']['model_b']['score_percent']}")
            
            print("\nTest complete.")
            
        else:
            print(f"[ERROR] Status code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to API. Is the server running?")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test_chat("How did different media cover Prabowo's cabinet formation?")
