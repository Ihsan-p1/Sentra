"""
LLM Client Module (OpenRouter / OpenAI Compatible)
Optimized for multi-turn conversational RAG chatbot.
"""
import requests
import json
import os
from config.settings import settings
from typing import Dict, Any, List, Optional
from chatbot.prompts import FRAMING_ANALYSIS_PROMPT, REDUCE_HALLUCINATION_PROMPT

class LLMClient:
    """
    OpenRouter API Client (OpenAI-compatible).
    Supports multiple providers (Google, Meta, etc.) via a single interface.
    """
    def __init__(self):
        # Prioritize OpenRouter, then fallback to others
        self.api_key = settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            # Fallback legacy checks
            self.api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
            
        if not self.api_key:
            print("[ERROR] API_KEY not found! (Checked OpenRouter & Groq)")
        else:
            print("[INFO] API Key loaded successfully")
        
        self.model_name = settings.LLM_MODEL or "google/gemini-2.0-flash-lite-preview-02-05:free"
        self.base_url = settings.LLM_BASE_URL or "https://openrouter.ai/api/v1"
        print(f"[INFO] LLM: {self.model_name} via {self.base_url}")

    async def generate_conversational_response(
        self, 
        user_query: str, 
        retrieved_chunks: Dict[str, List[Dict]], 
        conversation_history: str = "",
        mode: str = "default",
        focus_topic: str = None  # Specific topic to focus on
    ) -> str:
        """
        Generate response using proper multi-turn conversation format.
        """
        if not self.api_key: 
            return "Error: API_KEY not configured."

        # Format news sources context
        news_context = self._format_news_context(retrieved_chunks)
        
        # Build the messages array (OpenAI format - Groq compatible)
        messages = []
        
        # 1. System prompt - Select between Structured Analysis or Strict Fact-Checking
        if mode == "reduce_hallucination":
            # Strict mode: Minimal hallucination, bullet points, citations
            # We must format the prompt with data since it's a template
            system_prompt = REDUCE_HALLUCINATION_PROMPT.format(
                user_question=user_query,
                retrieved_chunks_by_media=news_context,
                computed_framing="(Not used in strict mode)"
            )
        else:
            # Default mode: Structured Framing Analysis (Direct Answer -> Summary -> Media Breakdown)
            system_prompt = FRAMING_ANALYSIS_PROMPT.format(
                user_question=user_query,
                retrieved_chunks_by_media=news_context,
                computed_framing="(Framing analysis implied in source comparison)"
            )
            
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. Parse and add conversation history as separate messages
        if conversation_history and conversation_history != "(This is the start of the conversation)":
            history_messages = self._parse_history_to_messages(conversation_history)
            messages.extend(history_messages)
        
        # 3. Current user message
        # Since the system prompt already contains the context and instructions, 
        # we can keep the user message simple or reinforce the focus.
        
        if focus_topic:
            # If focusing on a specific topic, reinforce it
            current_prompt = f"""Focus ONLY on: {focus_topic}
            
            Refer to the news sources provided in the system prompt.
            """
        else:
            current_prompt = user_query

        messages.append({"role": "user", "content": current_prompt})

        # OpenRouter / OpenAI API request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7 if mode == "default" else 0.2,
            "max_tokens": 2048,
            "top_p": 0.9
        }

        # Headers required for OpenRouter
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000", # Required by OpenRouter
            "X-Title": "Sentra News Analysis"        # Required by OpenRouter
        }

        url = f"{self.base_url}/chat/completions"

        try:
            print(f"[DEBUG] Sending {len(messages)} messages to OpenRouter")
            response = requests.post(
                url, 
                json=payload,
                headers=headers,
                timeout=90  # Longer timeout for detailed responses
            )
            
            if response.status_code != 200:
                print(f"[ERROR] API {response.status_code}: {response.text[:200]}")
                return f"API Error: {response.text[:200]}"
            
            data = response.json()
            # Extract text from OpenAI-compatible response format
            if 'choices' in data and len(data['choices']) > 0:
                text = data['choices'][0]['message']['content']
                return text
            else:
                return "Error: Empty response from LLM provider."

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return f"Connection Error: {str(e)}"

    def _format_news_context(self, retrieved_chunks: Dict[str, List[Dict]]) -> str:
        """Format retrieved chunks as readable news context."""
        if not retrieved_chunks:
            return "(No news sources available for this query)"
        
        context = ""
        for media, chunks in retrieved_chunks.items():
            if chunks:
                context += f"\n📰 {media.upper()}:\n"
                for i, c in enumerate(chunks[:4], 1):  # Max 4 per source
                    text = c.get('text', '')[:500]  # More text per chunk
                    title = c.get('title', 'Article')
                    context += f"  [{i}] {title}\n      {text}\n"
        
        return context if context else "(No relevant news found)"

    def _parse_history_to_messages(self, history_str: str) -> List[Dict]:
        """Parse conversation history string into messages array."""
        messages = []
        
        # Handle formatted history from memory
        lines = history_str.strip().split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('[USER]'):
                # Save previous message if exists
                if current_role and current_content:
                    content = ' '.join(current_content).strip()
                    if len(content) > 10:  # Skip very short messages
                        messages.append({"role": current_role, "content": content[:800]})
                current_role = "user"
                current_content = [line.replace('[USER]', '').strip()]
            elif line.startswith('[ASSISTANT]'):
                if current_role and current_content:
                    content = ' '.join(current_content).strip()
                    if len(content) > 10:
                        messages.append({"role": current_role, "content": content[:800]})
                current_role = "assistant"
                current_content = [line.replace('[ASSISTANT]', '').strip()]
            elif line and current_role:
                current_content.append(line)
        
        # Don't forget the last message
        if current_role and current_content:
            content = ' '.join(current_content).strip()
            if len(content) > 10:
                messages.append({"role": current_role, "content": content[:800]})
        
        # Limit to last 6 messages to avoid token overflow
        return messages[-6:]

    async def safe_generate(self, prompt: str) -> str:
        if not self.api_key: return ""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256
            }
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Sentra"
            }
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except: pass
        return ""
