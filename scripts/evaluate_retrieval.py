"""
Score retrieval on the owner's gold articles (eval phase 2).

Runs the 50 in-corpus questions and the 20 out-of-corpus questions through the
dense retriever (rag/retrieval.py) and the TF-IDF baseline, three times each
after one discarded warm-up call. Rankings and best-chunk similarity come from
the first run; latency from all three. Needs the database, not the LLM.

Usage:
    python scripts/evaluate_retrieval.py
"""
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

from config.settings import settings
from data.corpus import corpus_version, load_articles
from database.connection import db_manager
from evaluation.hallucination_dataset import DATASETS_DIR, load_questions
from evaluation.lexical_retriever import TfidfRetriever
from evaluation.retrieval_gold import GOLD_PATH
from evaluation.retrieval_metrics import percentile, rank_articles
from evaluation.retrieval_report import build_report, refusal_table, score_rankings
from rag.retrieval import RAGRetriever

RESULTS_DIR = DATASETS_DIR.parent / "results"
REPEATS = 3


def ooc_group(q: dict):
    """Refusal-table groups for one out-of-corpus question."""
    if q["origin"] == "calibration":
        return "ooc_calibration"
    return ["ooc_held_out", "ooc_near"] if q["kind"] == "near" else "ooc_held_out"


async def run(questions: list, tfidf: TfidfRetriever):
    retriever = RAGRetriever()
    await retriever.retrieve("warm-up")
    rankings = {"dense": {}, "tfidf": {}}
    times = {"dense": [], "tfidf": []}
    max_sims = {}
    for q in questions:
        for rep in range(REPEATS):
            start = time.perf_counter()
            flat = retriever.flatten_results(await retriever.retrieve(q["text"]))
            times["dense"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            lexical = tfidf.rank(q["text"], 10)
            times["tfidf"].append((time.perf_counter() - start) * 1000)
            if rep == 0:
                rankings["dense"][q["id"]] = rank_articles(flat)
                rankings["tfidf"][q["id"]] = [url for url, _ in lexical]
                max_sims[q["id"]] = max((c["similarity"] for c in flat), default=0.0)
    return rankings, times, max_sims


def main() -> int:
    in_corpus = load_questions()
    ooc = load_questions(DATASETS_DIR / "ooc_questions.json")
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    articles = load_articles()

    async def with_db():
        await db_manager.connect()
        try:
            return await run(in_corpus + ooc, TfidfRetriever(articles))
        finally:
            await db_manager.disconnect()

    rankings, times, max_sims = asyncio.run(with_db())

    thresholds = {"default": settings.RETRIEVAL_THRESHOLD, "strict": settings.RETRIEVAL_THRESHOLD_STRICT}
    scores = {name: score_rankings(r, gold) for name, r in rankings.items()}
    groups = {q["id"]: "in_corpus_calibration" if q["origin"] == "calibration" else "in_corpus_held_out"
              for q in in_corpus}
    groups.update({q["id"]: ooc_group(q) for q in ooc})
    refusals = refusal_table(max_sims, groups, thresholds)
    latency = {name: {"p50": percentile(t, 50), "p95": percentile(t, 95), "n": len(t)}
               for name, t in times.items()}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "commit": commit,
        "corpus_version": corpus_version(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "top_k": settings.TOP_K_RETRIEVAL,
        "thresholds": thresholds,
        "n_articles": len(articles),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gold_provenance": "the owner judged every candidate, with an AI tool helping to read "
                           "or summarise articles",
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    report = build_report(scores, refusals, latency, meta)
    (RESULTS_DIR / "retrieval.md").write_text(report, encoding="utf-8", newline="\n")
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps({
        "meta": meta, "scores": scores, "refusals": refusals, "latency": latency,
        "max_similarity": max_sims, "rankings": rankings,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(report)
    print(f"[INFO] wrote {RESULTS_DIR / 'retrieval.md'} and retrieval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
