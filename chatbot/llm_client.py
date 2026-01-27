"""
LLM Client 
Generates responses using OpenAI API (or compatible local LLM).
"""
from openai import AsyncOpenAI
import json
from typing import Dict, List, Any

import sys
sys.path.append('..')
from config.settings import settings
from .prompts import FRAMING_ANALYSIS_PROMPT

class LLMClient:
    """Client for generating text responses"""
    
    def __init__(self):
        # Can be configured to point to local Ollama (base_url="http://localhost:11434/v1")
        # if using local LLM, set OPENAI_API_KEY to 'ollama'
        self.client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY, 
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL
        print(f"🔧 LLM Config: BaseURL={settings.LLM_BASE_URL}, Model={self.model}")
        print(f"🔑 API Key: {settings.GROQ_API_KEY[:5]}...{settings.GROQ_API_KEY[-4:] if settings.GROQ_API_KEY else 'NONE'}")
        
    async def generate_comparative_answer(
        self, 
        query: str, 
        retrieved_chunks: Dict[str, List[Dict]],
        framing_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate answer comparing media perspectives.
        """
        # Format context for prompt
        context_str = self._format_context(retrieved_chunks)
        framing_str = json.dumps(framing_analysis, indent=2)
        
        prompt = FRAMING_ANALYSIS_PROMPT.format(
            user_question=query,
            retrieved_chunks_by_media=context_str,
            # We can inject the computed framing analysis to guide the LLM
            computed_framing=framing_str 
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a neutral media analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLM Generation Error: {error_msg}")
            return f"Maaf, terjadi kesalahan saat menyusun jawaban. (LLM Error: {error_msg})"
            
    def _format_context(self, chunks: Dict[str, List[Dict]]) -> str:
        formatted = []
        for media, items in chunks.items():
            formatted.append(f"\n=== {media.upper()} ===")
            for item in items:
                formatted.append(f"- {item['text']}")
        return "\n".join(formatted)
