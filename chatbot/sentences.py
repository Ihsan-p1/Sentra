"""
How an answer is cut into sentences for the hallucination check.

The engine and the evaluation scripts both use these two functions, so the
evaluation scores exactly the sentences the app checks, cleaned the same way.
"""
import re
from typing import List, Optional

# Phrases from refusals and section labels. They are not claims about the news.
_SKIP_PHRASES = (
    "retrieved sources do not contain",
    "sufficient information",
    "framing analysis",
    "explicit framing",
)


def split_sentences(text: str) -> List[str]:
    """Split on sentence punctuation or on newlines (markdown headings stand alone)."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def prepare_for_check(sentence: str) -> Optional[str]:
    """Return the text to verify, or None when the engine does not check this sentence."""
    if len(sentence.split()) < 4:
        return None
    lower = sentence.lower()
    if any(phrase in lower for phrase in _SKIP_PHRASES):
        return None
    if sentence.strip().startswith("##"):
        return None

    # Remove [Source] citations, then a leading bullet.
    clean = re.sub(r'\[.*?\]', '', sentence).strip()
    clean = re.sub(r'^[\-\*•]\s+', '', clean).strip()
    if not clean or len(clean.split()) < 3:
        return None
    return clean
