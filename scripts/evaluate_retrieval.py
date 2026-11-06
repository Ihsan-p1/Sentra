"""
Score retrieval on the owner's gold articles (eval phase 2).

Runs the 50 in-corpus questions and the 20 out-of-corpus questions through the
dense retriever (rag/retrieval.py), the TF-IDF baseline and, since phase 3,
RRF fusion of the two and a cross-encoder reranking dense or dense + TF-IDF
candidates, three times each after one discarded warm-up call. Rankings and
best scores come from the first run; latency from all three. Needs the
database and the reranker model, not the LLM.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --questions evaluation/datasets/fulltext_questions.json \
        --gold evaluation/datasets/retrieval_gold_fulltext.json --out-name retrieval_fulltext \
        --gold-check evaluation/datasets/retrieval_gold_fulltext_check.json
"""
import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import settings
from data.corpus import corpus_version, load_articles
from database.connection import db_manager
from evaluation.hallucination_dataset import DATASETS_DIR, QUESTIONS_PATH, load_questions
from evaluation.lexical_retriever import TfidfRetriever
from evaluation.retrieval_gold import EXTRA_POOL_PATH, GOLD_PATH
from evaluation.reranking import RERANKER_MODEL, CrossEncoderReranker, reciprocal_rank_fusion, rerank
from evaluation.retrieval_metrics import percentile, rank_articles
from evaluation.retrieval_report import (build_report, pick_threshold, refusal_table, score_rankings,
                                         unjudged_at_k)
from rag.retrieval import RAGRetriever

RESULTS_DIR = DATASETS_DIR.parent / "results"
REPEATS = 3
SYSTEM_KEYS = ("dense", "tfidf", "rrf", "rerank", "rerank_hybrid")


def parse_args(argv=None) -> argparse.Namespace:
    """No arguments reproduce the old corpus's run and its output names."""
    parser = argparse.ArgumentParser(description="Score retrieval against gold articles.")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH), help="in-corpus questions")
    parser.add_argument("--gold", default=str(GOLD_PATH), help="gold articles per question")
    parser.add_argument("--out-name", default="retrieval", help="base name of the .md/.json results")
    parser.add_argument("--gold-check", default=None, help="owner's check of a constructed gold, if any")
    return parser.parse_args(argv)


def is_event_gold(gold: dict) -> bool:
    return bool(gold) and all(e.get("source") == "event" for e in gold.values())


def median_words(articles: list) -> int:
    return int(statistics.median(len(a["content"].split()) for a in articles))


def output_paths(out_name: str):
    return RESULTS_DIR / f"{out_name}.md", RESULTS_DIR / f"{out_name}.json"


def gold_provenance(gold: dict) -> str:
    if is_event_gold(gold):
        return ("none; relevance is constructed from event membership (a question about an event "
                "is answered by that event's news), checked by the owner on a sample")
    return "the owner judged every candidate, with an AI tool helping to read or summarise articles"


def extra_judged(gold: dict) -> int:
    """Articles from the phase 3 extra sheet that are already merged into the gold."""
    if not EXTRA_POOL_PATH.exists():
        return 0
    pool = json.loads(EXTRA_POOL_PATH.read_text(encoding="utf-8"))["pool"]
    return sum(u in gold.get(qid, {}).get("judged", []) for qid, urls in pool.items() for u in urls)


def ooc_group(q: dict):
    """Refusal-table groups for one out-of-corpus question."""
    if q["origin"] == "calibration":
        return "ooc_calibration"
    return ["ooc_held_out", "ooc_near"] if q["kind"] == "near" else "ooc_held_out"


def since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def run(questions: list, tfidf: TfidfRetriever, reranker: CrossEncoderReranker, articles: dict):
    """Each stage is timed once per repeat; a combined system's latency is the sum of its stages."""
    retriever = RAGRetriever()
    await retriever.retrieve("warm-up")
    reranker.score("warm-up", ["warm-up"])
    rankings = {s: {} for s in SYSTEM_KEYS}
    times = {s: [] for s in SYSTEM_KEYS}
    max_sims = {}
    rerank_max = {"rerank": {}, "rerank_hybrid": {}}
    for q in questions:
        for rep in range(REPEATS):
            start = time.perf_counter()
            flat = retriever.flatten_results(await retriever.retrieve(q["text"]))
            dense_ms = since(start)
            start = time.perf_counter()
            lexical = [url for url, _ in tfidf.rank(q["text"], 10)]
            tfidf_ms = since(start)
            dense = rank_articles(flat)
            start = time.perf_counter()
            fused = reciprocal_rank_fusion([dense, lexical])
            fuse_ms = since(start)
            start = time.perf_counter()
            reranked = rerank(q["text"], dense, articles, reranker.score)
            rerank_ms = since(start)
            start = time.perf_counter()
            union = dense + [u for u in lexical if u not in set(dense)]
            reranked_hybrid = rerank(q["text"], union, articles, reranker.score)
            hybrid_ms = since(start)

            times["dense"].append(dense_ms)
            times["tfidf"].append(tfidf_ms)
            times["rrf"].append(dense_ms + tfidf_ms + fuse_ms)
            times["rerank"].append(dense_ms + rerank_ms)
            times["rerank_hybrid"].append(dense_ms + tfidf_ms + hybrid_ms)
            if rep == 0:
                qid = q["id"]
                rankings["dense"][qid] = dense
                rankings["tfidf"][qid] = lexical
                rankings["rrf"][qid] = fused
                rankings["rerank"][qid] = [u for u, _ in reranked]
                rankings["rerank_hybrid"][qid] = [u for u, _ in reranked_hybrid]
                max_sims[qid] = max((c["similarity"] for c in flat), default=0.0)
                # No candidate at all must be refused, and 0.0 is a mid-range logit.
                rerank_max["rerank"][qid] = reranked[0][1] if reranked else float("-inf")
                rerank_max["rerank_hybrid"][qid] = reranked_hybrid[0][1] if reranked_hybrid else float("-inf")
    return rankings, times, max_sims, rerank_max


def main(argv=None) -> int:
    args = parse_args(argv)
    in_corpus = load_questions(args.questions)
    ooc = load_questions(DATASETS_DIR / "ooc_questions.json")
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    articles = load_articles()

    reranker = CrossEncoderReranker()
    by_url = {a["url"]: a for a in articles}

    async def with_db():
        await db_manager.connect()
        try:
            return await run(in_corpus + ooc, TfidfRetriever(articles), reranker, by_url)
        finally:
            await db_manager.disconnect()

    rankings, times, max_sims, rerank_max = asyncio.run(with_db())

    thresholds = {"default": settings.RETRIEVAL_THRESHOLD, "strict": settings.RETRIEVAL_THRESHOLD_STRICT}
    scores = {name: score_rankings(r, gold) for name, r in rankings.items()}
    groups = {q["id"]: "in_corpus_calibration" if q["origin"] == "calibration" else "in_corpus_held_out"
              for q in in_corpus}
    groups.update({q["id"]: ooc_group(q) for q in ooc})
    refusals = refusal_table(max_sims, groups, thresholds)
    gates = {}
    # A reranker gate is fitted on in-corpus calibration questions; a question set without any
    # (the full-text corpus) would fit it on out-of-corpus scores alone, so it gets no gate.
    has_calibration = "in_corpus_calibration" in groups.values()
    for system, scores_by_q in (rerank_max.items() if has_calibration else ()):
        t = pick_threshold([scores_by_q[q] for q, g in groups.items() if g == "in_corpus_calibration"],
                           [scores_by_q[q] for q, g in groups.items() if g == "ooc_calibration"])
        gates[system] = {"threshold": t, "refusals": refusal_table(scores_by_q, groups, {"picked": t})}
    unjudged = {name: unjudged_at_k(r, gold, 10) for name, r in rankings.items()}
    latency = {name: {"p50": percentile(t, 50), "p95": percentile(t, 95), "n": len(t)}
               for name, t in times.items()}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "commit": commit,
        "corpus_version": corpus_version(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "top_k": settings.TOP_K_RETRIEVAL,
        "thresholds": thresholds,
        "n_articles": len(articles),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gold_provenance": gold_provenance(gold),
        "gold_extra": extra_judged(gold),
        "event_gold": is_event_gold(gold),
        "median_words": median_words(articles),
        "gold_check": (json.loads(Path(args.gold_check).read_text(encoding="utf-8"))
                       if args.gold_check else None),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    md_path, json_path = output_paths(args.out_name)
    report = build_report(scores, refusals, latency, meta, gates, unjudged)
    md_path.write_text(report, encoding="utf-8", newline="\n")
    json_path.write_text(json.dumps({
        "meta": meta, "scores": scores, "refusals": refusals, "latency": latency,
        "max_similarity": max_sims, "rerank_max": rerank_max, "gates": gates,
        "unjudged": unjudged, "rankings": rankings,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(report)
    print(f"[INFO] wrote {md_path} and {json_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
