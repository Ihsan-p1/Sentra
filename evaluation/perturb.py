"""
Synthetic negatives for the hallucination evaluation (eval phase 1, step 3).

Real answers rarely contain unsupported claims, so supported sentences are
copied with one fact swapped: a name, then a year, then any other number. The
copy is labelled "unsupported" by construction and kept apart from the real
data (synthetic: True). A swap is rejected when the new value appears anywhere
in the chunks, because then the copy might still be supported.
"""
import re
from typing import Dict, Iterator, List, Optional, Tuple

NAME_SWAPS = {
    "Prabowo": "Anies",
    "Anies": "Ganjar",
    "Ganjar": "Prabowo",
    "Gibran": "Mahfud",
    "Mahfud": "Gibran",
    "Jokowi": "Megawati",
}


def _in_chunks(value: str, chunk_texts: List[str]) -> bool:
    pattern = re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)
    return any(pattern.search(text) for text in chunk_texts)


def _candidates(sentence: str) -> Iterator[Tuple[re.Match, str, str]]:
    """(match, replacement, kind) in the order they are tried."""
    for name, other in NAME_SWAPS.items():
        for m in re.finditer(rf"\b{name}\b", sentence):
            yield m, other, "name"
    numbers = list(re.finditer(r"\b\d+\b", sentence))
    for m in numbers:
        n = int(m.group())
        if len(m.group()) == 4 and 1900 <= n <= 2100:
            yield m, str(n - 5), "year"
    for m in numbers:
        n = int(m.group())
        if not (len(m.group()) == 4 and 1900 <= n <= 2100):
            yield m, str(n * 2 + 1), "number"


def perturb_sentence(sentence: str, chunk_texts: List[str]) -> Optional[Tuple[str, str]]:
    for m, replacement, kind in _candidates(sentence):
        if not _in_chunks(replacement, chunk_texts):
            return sentence[:m.start()] + replacement + sentence[m.end():], kind
    return None


def synthetic_rows(rows: List[Dict], answers_by_qid: Dict[str, Dict]) -> List[Dict]:
    out = []
    for row in rows:
        if row["label"] != "supported" or row["check_text"] is None:
            continue
        chunk_texts = [c["text"] for c in answers_by_qid[row["query_id"]]["chunks"]]
        swapped = perturb_sentence(row["check_text"], chunk_texts)
        if swapped is None:
            continue
        text, kind = swapped
        out.append({
            "query_id": row["query_id"],
            "sentence_idx": row["sentence_idx"],
            "check_text": text,
            "kind": kind,
            "label": "unsupported",
            "synthetic": True,
        })
    return out
