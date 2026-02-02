"""
LLM Client Module (Groq API Version)
Optimized for multi-turn conversational RAG chatbot.
Perfect for Indonesian Election 2024 news analysis.
"""
import requests
import json
import os
from config.settings import settings
from typing import Dict, Any, List, Optional

# Base system prompt - focused on election analysis
ELECTION_SYSTEM_PROMPT = """You are Sentra, an expert AI assistant for Indonesian Election 2024 and political news analysis.

YOUR EXPERTISE:
- Indonesian presidential election 2024 (Prabowo Subianto, Anies Baswedan, Ganjar Pranowo)
- Government policies, cabinet members, and political developments
- Media framing analysis - comparing how different outlets cover stories
- Current events in Indonesian politics

CONVERSATION RULES:
1. ALWAYS maintain conversation context - if user says "yes", "tell me more", or asks follow-up questions, continue the SAME topic
2. Reference previous messages naturally (e.g., "As I mentioned earlier about Prabowo...")
3. Provide DETAILED, COMPREHENSIVE answers (3-4 paragraphs minimum)
4. Cite your sources naturally (e.g., "According to TEMPO...", "ANTARA reports...")
5. Compare different media perspectives when available

CRITICAL: FOLLOW-UP SUGGESTIONS
- At the end of your response, suggest ONLY ONE specific follow-up topic
- Format: "Would you like to know more about [SINGLE SPECIFIC TOPIC]?"
- Do NOT list multiple options or alternatives
- The topic must be directly related to what you just discussed
- Example: "Would you like to know more about Prabowo's cabinet appointments?" (NOT "...cabinet, economy, or foreign policy?")

RESPONSE STYLE:
- Be conversational and friendly, not robotic
- Structure answers clearly with paragraphs
- Include specific details, dates, and quotes when available
- Always stay grounded in the provided news sources"""

STRICT_SYSTEM_PROMPT = """You are Sentra, a fact-checking AI for Indonesian political news.

STRICT RULES:
- ONLY state facts explicitly in the provided sources
- Always cite sources: [TEMPO], [ANTARA], [Jakarta Post]
- If not in sources, say "I don't have that information"
- NO speculation or inference
- Keep responses concise and factual"""


class LLMClient:
    """
    Groq API Client optimized for multi-turn conversations.
    """
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("[ERROR] GROQ_API_KEY not found!")
        else:
            print("[INFO] Groq API key loaded")
        
        self.model_name = settings.LLM_MODEL or "llama-3.3-70b-versatile"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        print(f"[INFO] LLM: {self.model_name}")

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
            return "Error: GROQ_API_KEY not configured."

        # Format news sources context
        news_context = self._format_news_context(retrieved_chunks)
        
        # Build the messages array (proper multi-turn format)
        messages = []
        
        # 1. System prompt
        system_prompt = STRICT_SYSTEM_PROMPT if mode == "reduce_hallucination" else ELECTION_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})
        
        # 2. Parse and add conversation history as separate messages
        if conversation_history and conversation_history != "(This is the start of the conversation)":
            history_messages = self._parse_history_to_messages(conversation_history)
            messages.extend(history_messages)
        
        # 3. Current user message with news context
        # If we have a specific focus_topic (from "yes" response), use strict instructions
        if focus_topic:
            current_prompt = f"""⚠️ MANDATORY TOPIC: "{focus_topic}"

YOU MUST ANSWER ONLY ABOUT THIS TOPIC. Start your response by directly addressing "{focus_topic}".

Here are some news sources (some may be unrelated - IGNORE unrelated ones):
{news_context}

STRICT RULES:
1. START your response with information about "{focus_topic}" - do not start with other topics
2. ONLY discuss "{focus_topic}" - nothing else
3. If NO sources mention "{focus_topic}", say: "I don't have specific information about {focus_topic} in my current sources."
4. Do NOT mention MRT, protests, wages, or other unrelated topics
5. Write 3-4 focused paragraphs about "{focus_topic}" ONLY

BEGIN your response about: {focus_topic}"""
        else:
            current_prompt = f"""NEWS SOURCES FOR THIS QUERY:
{news_context}

USER QUESTION: {user_query}

Provide a comprehensive, detailed response based on the news sources above. Focus on the specific topic asked."""

        messages.append({"role": "user", "content": current_prompt})

        # API request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7 if mode == "default" else 0.2,
            "max_tokens": 2048,  # Increased for longer responses
            "top_p": 0.9
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            print(f"[DEBUG] Sending {len(messages)} messages to Groq")
            response = requests.post(
                self.base_url, 
                json=payload,
                headers=headers,
                timeout=90  # Longer timeout for detailed responses
            )
            
            if response.status_code != 200:
                print(f"[ERROR] Groq {response.status_code}: {response.text[:200]}")
                return f"API Error: {response.text[:200]}"
            
            data = response.json()
            text = data['choices'][0]['message']['content']
            return text

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

    # Legacy method
    async def generate_comparative_answer(
        self, 
        user_query: str, 
        retrieved_chunks: Dict[str, List[Dict]], 
        framing_analysis: Dict[str, Any],
        mode: str = "default",
        conversation_history: str = ""
    ) -> str:
        return await self.generate_conversational_response(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            mode=mode
        )

    async def safe_generate(self, prompt: str) -> str:
        if not self.api_key: return ""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except: pass
        return ""
