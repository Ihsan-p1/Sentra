"""
Generate the real hallucination dataset (eval phase 1, step 1).

Runs every question in evaluation/datasets/questions.json through the real
pipeline (ChatbotEngine.process_query, default mode) and stores each answer, the
chunks the hallucination check compared it against, and one unlabelled row per
sentence. Labels are added afterwards by the owner with
scripts/label_hallucination.py.

Resumable: questions already in hallucination_answers.jsonl are skipped, and the
files are saved after every question.

Usage:
    python scripts/generate_hallucination_set.py [--limit N]

Requires Ollama (settings.LLM_MODEL) and a populated database.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chatbot.engine import ChatbotEngine
from config.settings import settings
from data.corpus import corpus_version
from database.connection import db_manager
from evaluation.hallucination_dataset import (
    ANSWERS_PATH,
    META_PATH,
    SENTENCES_PATH,
    answer_record,
    answer_to_rows,
    classify_result,
    load_jsonl,
    load_questions,
    pending_questions,
    save_jsonl,
)


def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "mode": "default",
        "llm_model": settings.LLM_MODEL,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "llm_seed": settings.LLM_SEED,
        "embedding_model": settings.EMBEDDING_MODEL,
        "corpus_version": corpus_version(),
        "refused": {},
        "seconds": {},
    }


def save_meta(meta: dict) -> None:
    tmp = META_PATH.with_name(META_PATH.name + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, META_PATH)


async def main(limit: int) -> None:
    answers = load_jsonl(ANSWERS_PATH)
    rows = load_jsonl(SENTENCES_PATH)
    meta = load_meta()
    done = {a["query_id"] for a in answers} | set(meta["refused"])
    todo = pending_questions(load_questions(), done)
    if limit:
        todo = todo[:limit]
    print(f"[INFO] {len(done)} questions done, {len(todo)} to run")

    await db_manager.connect()
    engine = ChatbotEngine()
    try:
        for q in todo:
            qid = q["id"]
            start = time.perf_counter()
            # One session per question, so no answer is resolved against another.
            result = await engine.process_query(q["text"], session_id=f"eval-{qid}", mode="default")
            meta["seconds"][qid] = round(time.perf_counter() - start, 1)

            raw = result.get("raw_answer")
            status, reason = classify_result(result)
            if status == "retry":
                # Not recorded, so the next run asks this question again.
                print(f"[WARN] {qid} not answered ({reason}); rerun the script to retry it")
            elif status == "refused":
                meta["refused"][qid] = reason
                print(f"[WARN] {qid} refused: {reason}")
            else:
                chunks = engine.retriever.flatten_results(result["sources"])
                answers.append(answer_record(qid, q["text"], raw, chunks, result.get("language", "en")))
                new_rows = answer_to_rows(qid, raw)
                rows.extend(new_rows)
                checked = sum(r["check_text"] is not None for r in new_rows)
                print(f"[INFO] {qid} {meta['seconds'][qid]}s: {len(new_rows)} sentences, {checked} checked")

            save_jsonl(ANSWERS_PATH, answers)
            save_jsonl(SENTENCES_PATH, rows)
            save_meta(meta)
    finally:
        await db_manager.disconnect()

    checked = sum(r["check_text"] is not None for r in rows)
    print(f"[INFO] total: {len(answers)} answers, {len(meta['refused'])} refused, "
          f"{len(rows)} sentences, {checked} checked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--limit", type=int, default=0, help="run at most N pending questions")
    asyncio.run(main(parser.parse_args().limit))
