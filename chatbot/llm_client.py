"""
LLM Client for Sentra Chatbot
Robust implementation for Google Gemini 2.0 Flash.
"""
import os
import json
import google.generativeai as genai
from config.settings import settings
from typing import Dict, Any, List

class LLMClient:
    """
    Client for interacting with Google Gemini Models.
    """
    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            print("[WARN] GOOGLE_API_KEY not found. LLM features will fail.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                # Use model from settings, or fallback to 'gemini-2.0-flash'
                model_name = settings.LLM_MODEL or "gemini-2.0-flash"
                self.model = genai.GenerativeModel(model_name)
                print(f"[INFO] LLM Config: Model={model_name} (Gemini)")
            except Exception as e:
                print(f"[ERROR] Configuring Gemini: {e}")
                self.model = None

    async def generate_comparative_answer(
        self, 
        user_query: str, 
        retrieved_chunks: Dict[str, List[Dict]], 
        framing_analysis: Dict[str, Any],
        mode: str = "default"
    ) -> str:
        """
        Generate answer using the specific framing analysis prompt.
        """
        if not self.model:
            return "Error: LLM not configured (Key missing or invalid)."

        # Format retrieved chunks for prompt
        chunks_str = ""
        for media, chunks in retrieved_chunks.items():
            chunks_str += f"\nSOURCE: {media.upper()}\n"
            for c in chunks:
                chunks_str += f"- {c['text']}\n"
        
        # Format framing analysis for prompt
        framing_str = json.dumps(framing_analysis, indent=2)
        
        from chatbot.prompts import FRAMING_ANALYSIS_PROMPT, REDUCE_HALLUCINATION_PROMPT
        
        # Select prompt based on mode
        base_prompt = REDUCE_HALLUCINATION_PROMPT if mode == "reduce_hallucination" else FRAMING_ANALYSIS_PROMPT
        
        system_prompt = base_prompt.format(
            user_question=user_query,
            retrieved_chunks_by_media=chunks_str,
            computed_framing=framing_str
        )
        
        try:
            # Gemini doesn't have system prompt in generate_content (mostly), 
            # so we just combine it or use system_instruction if supported by the specific model version.
            # For robustness, we'll append.
            # Newer Gemini models support system_instruction in constructor, but we initialized already.
            # We'll just pass it as text.
            
            full_prompt = f"System: You are a helpful AI assistant specialized in media framing analysis.\n{system_prompt}"
            
            response = await self.model.generate_content_async(full_prompt)
            return response.text
        except Exception as e:
            print(f"[ERROR] LLM Generation: {e}")
            return f"Error interacting with AI model: {str(e)}"

    async def safe_generate(self, prompt: str) -> str:
        """Generic generation for other tasks"""
        if not self.model: return "Error: No LLM"
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
