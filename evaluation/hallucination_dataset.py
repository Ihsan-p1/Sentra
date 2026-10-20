"""
Files and rows of the real hallucination dataset (eval phase 1).

Two JSONL files, so the retrieved chunks are stored once per answer:

- hallucination_answers.jsonl: one record per question - the answer and the chunks
  the hallucination check compared it against.
- hallucination_real.jsonl: one row per answer sentence. `label` is filled in by
  the owner with scripts/label_hallucination.py and by nothing else.
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chatbot.sentences import prepare_for_check, split_sentences

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
QUESTIONS_PATH = DATASETS_DIR / "questions.json"
ANSWERS_PATH = DATASETS_DIR / "hallucination_answers.jsonl"
SENTENCES_PATH = DATASETS_DIR / "hallucination_real.jsonl"
META_PATH = DATASETS_DIR / "hallucination_meta.json"

CHUNK_FIELDS = ("media", "title", "url", "text", "similarity")


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for q in questions:
        if q["id"] in seen:
            raise ValueError(f"duplicate question id: {q['id']}")
        seen.add(q["id"])
    return questions


def load_jsonl(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    """Write via a temp file and rename, so a crash never leaves half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def answer_record(query_id: str, question: str, answer: str, chunks: List[Dict], language: str) -> Dict:
    return {
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "language": language,
        "chunks": [{k: c[k] for k in CHUNK_FIELDS} for c in chunks],
    }


def answer_to_rows(query_id: str, answer: Optional[str]) -> List[Dict]:
    """One row per sentence; check_text is None for sentences the engine skips."""
    if not answer:
        return []
    return [
        {
            "query_id": query_id,
            "sentence_idx": i,
            "sentence": sentence,
            "check_text": prepare_for_check(sentence),
            "label": None,
        }
        for i, sentence in enumerate(split_sentences(answer))
    ]


def classify_result(result: Dict) -> Tuple[str, Optional[str]]:
    """("answered", None), ("refused", reason) - final - or ("retry", reason) for an LLM outage."""
    if result.get("raw_answer"):
        return "answered", None
    if result.get("error"):
        return "retry", result["error"]
    return "refused", (result.get("source_quality") or {}).get("refusal_reason") or "no raw_answer"


def pending_questions(questions: List[Dict], done_ids: set) -> List[Dict]:
    return [q for q in questions if q["id"] not in done_ids]
