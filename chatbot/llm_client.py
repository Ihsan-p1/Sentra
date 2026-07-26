"""
LLM Client Module (OpenAI-compatible).

Provider-agnostic: the endpoint is driven entirely by LLM_BASE_URL / LLM_MODEL in
.env, so the same client talks to a local Ollama server, LM Studio, OpenRouter, or
any other OpenAI-compatible API. Local endpoints need no API key.
"""
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from config.settings import settings
from chatbot.conversation_memory import NO_HISTORY_SENTINEL
from chatbot.language import ENGLISH, LANGUAGE_NAMES, headings
from chatbot.prompts import FRAMING_ANALYSIS_PROMPT, REDUCE_HALLUCINATION_PROMPT

# Hostnames treated as a local inference server (no credentials required)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}


class LLMUnavailable(RuntimeError):
    """
    The model server could not produce a completion.

    Raised instead of returning the error text as the answer. Returning it meant
    "Connection Error: could not reach http://localhost:11434/v1" was treated as
    a model response: split into sentences, scored by the hallucination detector,
    given a confidence percentage, rendered with a full A/B panel, and written
    into conversation memory where it then polluted follow-up resolution. An
    infrastructure failure is not a low-quality answer, and the UI should not
    grade it like one.
    """

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def user_message(self) -> str:
        return f"{self.message} {self.hint}".strip()


def _is_local_endpoint(base_url: str) -> bool:
    """True if base_url points at a model server running on this machine."""
    try:
        return (urlparse(base_url).hostname or "").lower() in LOCAL_HOSTS
    except ValueError:
        return False


class LLMClient:
    """
    OpenAI-compatible chat client.

    Default target is local Ollama (http://localhost:11434/v1). Remote providers
    still work: set LLM_BASE_URL and supply the matching key in .env.
    """

    def __init__(self):
        self.base_url = (settings.LLM_BASE_URL or "http://localhost:11434/v1").rstrip("/")
        self.model_name = settings.LLM_MODEL or "llama3.1"
        self.timeout = settings.LLM_TIMEOUT
        self.is_local = _is_local_endpoint(self.base_url)

        # Local runtimes ignore the credential, but some still expect the header
        # to be present, so send a placeholder rather than omitting it.
        self.api_key = (
            settings.OPENROUTER_API_KEY
            or os.getenv("OPENROUTER_API_KEY")
            or settings.GROQ_API_KEY
            or os.getenv("GROQ_API_KEY")
        )

        if self.is_local:
            self.api_key = self.api_key or "local"
            print(f"[INFO] LLM: {self.model_name} via local endpoint {self.base_url} (no API key required)")
        elif not self.api_key:
            print(f"[ERROR] {self.base_url} is a remote endpoint but no API key was found in .env")
        else:
            print(f"[INFO] LLM: {self.model_name} via {self.base_url}")

    def _build_headers(self) -> Dict[str, str]:
        """Auth header, plus the attribution headers OpenRouter requires."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "Sentra News Analysis"
        return headers

    def _post_chat(self, payload: Dict[str, Any]) -> str:
        """
        Send one chat completion request and return the message text.

        Raises LLMUnavailable on any failure to obtain a completion, so the
        caller can distinguish "the model said something poor" from "there was no
        model response at all".
        """
        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(url, json=payload, headers=self._build_headers(), timeout=self.timeout)

            if response.status_code != 200:
                print(f"[ERROR] API {response.status_code}: {response.text[:200]}")
                raise LLMUnavailable(
                    f"API Error {response.status_code} from the model server.",
                    response.text[:200],
                )

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise LLMUnavailable("The model server returned an empty response.")

            content = choices[0]["message"]["content"]
            if not (content or "").strip():
                raise LLMUnavailable("The model returned an empty message.")
            return content

        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Could not connect to {self.base_url}")
            raise LLMUnavailable(
                f"Connection Error: could not reach {self.base_url}.",
                "Is 'ollama serve' running?" if self.is_local else "",
            )
        except requests.exceptions.Timeout:
            print(f"[ERROR] LLM request timed out after {self.timeout}s")
            raise LLMUnavailable(
                f"Timeout Error: the model did not respond within {self.timeout}s.",
                "Local models on CPU can be slow - raise LLM_TIMEOUT in .env or use a smaller model.",
            )
        except (KeyError, ValueError) as e:
            print(f"[ERROR] Malformed LLM response: {e}")
            raise LLMUnavailable(f"Malformed response from the model server ({e}).")

    async def generate_conversational_response(
        self,
        user_query: str,
        retrieved_chunks: Dict[str, List[Dict]],
        conversation_history: str = "",
        mode: str = "default",
        focus_topic: Optional[str] = None,  # Specific topic to focus on
        language: str = ENGLISH,  # answer in the language the user wrote in
    ) -> str:
        """
        Generate response using proper multi-turn conversation format.
        """
        if not self.api_key:
            raise LLMUnavailable(
                "No API key configured for the remote LLM endpoint.",
                "Set OPENROUTER_API_KEY or GROQ_API_KEY in .env, or point "
                "LLM_BASE_URL at a local server.",
            )

        # Format news sources context
        news_context = self._format_news_context(retrieved_chunks)

        messages = []

        language_name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[ENGLISH])
        # Section headings are injected already translated - a 3B model copies the
        # headings it can see and ignores an instruction to translate them.
        section_headings = headings(language)

        # 1. System prompt - Select between Structured Analysis or Strict Fact-Checking
        if mode == "reduce_hallucination":
            # Strict mode: Minimal hallucination, bullet points, citations
            # We must format the prompt with data since it's a template
            system_prompt = REDUCE_HALLUCINATION_PROMPT.format(
                user_question=user_query,
                retrieved_chunks_by_media=news_context,
                computed_framing="(Not used in strict mode)",
                response_language=language_name,
                **section_headings,
            )
        else:
            # Default mode: Structured Framing Analysis (Direct Answer -> Summary -> Media Breakdown)
            system_prompt = FRAMING_ANALYSIS_PROMPT.format(
                user_question=user_query,
                retrieved_chunks_by_media=news_context,
                computed_framing="(Framing analysis implied in source comparison)",
                response_language=language_name,
                **section_headings,
            )

        messages.append({"role": "system", "content": system_prompt})

        # 2. Parse and add conversation history as separate messages
        if conversation_history and conversation_history != NO_HISTORY_SENTINEL:
            messages.extend(self._parse_history_to_messages(conversation_history))

        # 3. Current user message
        # Since the system prompt already contains the context and instructions,
        # we can keep the user message simple or reinforce the focus.
        if focus_topic:
            current_prompt = (
                f"Focus ONLY on: {focus_topic}\n\n"
                "Refer to the news sources provided in the system prompt."
            )
        else:
            current_prompt = user_query

        messages.append({"role": "user", "content": current_prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            # Strict mode is fully deterministic; default mode uses the configured
            # temperature. Both send a fixed seed so runs are reproducible.
            "temperature": 0.0 if mode == "reduce_hallucination" else settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "top_p": 0.9,
            "seed": settings.LLM_SEED,
        }

        # Message count and model land in the request's [TRACE] line; no separate
        # per-request print.
        return self._post_chat(payload)

    def _format_news_context(self, retrieved_chunks: Dict[str, List[Dict]]) -> str:
        """
        Format retrieved chunks as readable news context.

        Kept deliberately compact: local models have a much smaller effective
        context window than hosted ones (Ollama defaults to num_ctx=2048).
        """
        if not retrieved_chunks:
            return "(No news sources available for this query)"

        max_chunks = settings.MAX_CONTEXT_CHUNKS_PER_SOURCE
        max_chars = settings.MAX_CONTEXT_CHARS_PER_CHUNK

        available = [media.upper() for media, chunks in retrieved_chunks.items() if chunks]
        if not available:
            return "(No relevant news found)"

        # Small local models abbreviate or invent outlet names (observed:
        # ANTARANEWS -> "ANTV"). Naming the allowed set up front, as a closed
        # list, is far more reliable than telling the model to copy the headings.
        context = (
            f"AVAILABLE SOURCES - use ONLY these exact names, never abbreviate or "
            f"invent others: {', '.join(available)}\n"
        )

        for media, chunks in retrieved_chunks.items():
            if chunks:
                context += f"\n{media.upper()}:\n"
                for i, c in enumerate(chunks[:max_chunks], 1):
                    text = c.get("text", "")[:max_chars]
                    title = c.get("title", "Article")
                    context += f"  [{i}] {title}\n      {text}\n"

        return context

    def _parse_history_to_messages(self, history_str: str) -> List[Dict]:
        """Parse conversation history string into messages array."""
        messages = []

        # Handle formatted history from memory
        lines = history_str.strip().split("\n")
        current_role = None
        current_content = []

        def flush():
            if current_role and current_content:
                content = " ".join(current_content).strip()
                if len(content) > 10:  # Skip very short messages
                    messages.append({"role": current_role, "content": content[:800]})

        for line in lines:
            line = line.strip()
            if line.startswith("[USER]"):
                flush()
                current_role = "user"
                current_content = [line.replace("[USER]", "").strip()]
            elif line.startswith("[ASSISTANT]"):
                flush()
                current_role = "assistant"
                current_content = [line.replace("[ASSISTANT]", "").strip()]
            elif line and current_role:
                current_content.append(line)

        # Don't forget the last message
        flush()

        # Limit to last 6 messages to avoid token overflow
        return messages[-6:]
