"""
LLM Client Module (Sync Requests Version)
Uses standard 'requests' library to reliably connect to Gemini 1.5 Flash.
"""
import requests
import json
import os
from config.settings import settings
from typing import Dict, Any, List

class LLMClient:
    """
    Client for interacting with Google Gemini Models via Standard Requests.
    """
    def __init__(self):
        # Hardcoded Key as fallback
        self.api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY") or "AIzaSyAa_m-XbKJ6YLDWV0TiKi3aEiHNyzM1UXU"
        
        if self.api_key and "," in self.api_key:
            self.api_key = self.api_key.split(",")[0].strip()
            
        # Confirmed Model from Discovery (2026 Era)
        self.model_name = "gemini-2.5-flash"
        self.version = "v1beta"
        self.base_url = f"https://generativelanguage.googleapis.com/{self.version}/models/{self.model_name}:generateContent"
        
        print(f"[INFO] LLM Config: Sync HTTP to {self.model_name} ({self.version})")

    async def generate_comparative_answer(
        self, 
        user_query: str, 
        retrieved_chunks: Dict[str, List[Dict]], 
        framing_analysis: Dict[str, Any],
        mode: str = "default"
    ) -> str:
        
        if not self.api_key: return "Error: API Key missing."

        # Format retrieved chunks
        chunks_str = ""
        for media, chunks in retrieved_chunks.items():
            chunks_str += f"\nSOURCE ({media.upper()}):\n"
            for c in chunks:
                chunks_str += f"- {c.get('text', '')[:300]}...\n"

        framing_str = json.dumps(framing_analysis, indent=2)

        prompt_text = f"""
        You are Sentra.
        USER QUERY: {user_query}
        RETRIEVED: {chunks_str}
        FRAMING: {framing_str}
        
        Answer based on sources and compare framing.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": 0.3}
        }

        try:
            # Sync Request (OK in async wrapper for reliability)
            print(f"[DEBUG] POST to {self.base_url}")
            response = requests.post(
                f"{self.base_url}?key={self.api_key}", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[ERROR] API {response.status_code}: {response.text}")
                return f"API Error {response.status_code}: {response.text}"
            
            data = response.json()
            try:
                text = data['candidates'][0]['content']['parts'][0]['text']
                return text
            except Exception as e:
                return f"Error parsing response: {str(data)[:200]}"

        except Exception as e:
            return f"Connection Failed: {str(e)}"

    async def safe_generate(self, prompt: str) -> str:
        if not self.api_key: return ""
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(
                f"{self.base_url}?key={self.api_key}", 
                json=payload, 
                timeout=10
            )
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
        except: pass
        return ""
