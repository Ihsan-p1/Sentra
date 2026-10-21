"""
Build the candidate pool and the gold sheet for the retrieval evaluation
(eval phase 2). The owner judges relevance; this script only prepares and reads.

Usage:
    python scripts/make_retrieval_pool.py pool              # needs the database
    python scripts/make_retrieval_pool.py sheet OUT.md [QID ...]
    python scripts/make_retrieval_pool.py parse OUT.md      # merges into retrieval_gold.json
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.corpus import corpus_version, load_articles
from evaluation.hallucination_dataset import load_questions
from evaluation.lexical_retriever import TfidfRetriever
from evaluation.retrieval_gold import GOLD_PATH, POOL_PATH, build_pool, make_sheet, parse_sheet
from evaluation.retrieval_metrics import rank_articles

DEPTH = 10


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


async def make_pool() -> int:
    from config.settings import settings
    from database.connection import db_manager
    from rag.retrieval import RAGRetriever

    tfidf = TfidfRetriever(load_articles())
    await db_manager.connect()
    try:
        retriever = RAGRetriever()
        pool, overlap = {}, []
        for q in load_questions():
            grouped = await retriever.retrieve(q["text"])
            dense = rank_articles(retriever.flatten_results(grouped))[:DEPTH]
            lexical = [url for url, _ in tfidf.rank(q["text"], DEPTH)]
            pool[q["id"]] = build_pool(dense, lexical, DEPTH)
            overlap.append(len(set(dense) & set(lexical)))
    finally:
        await db_manager.disconnect()

    write_json(POOL_PATH, {
        "corpus_version": corpus_version(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "depth": DEPTH,
        "pool": pool,
    })
    sizes = [len(v) for v in pool.values()]
    print(f"[INFO] {len(pool)} questions, {sum(sizes)} candidates "
          f"(min {min(sizes)}, mean {sum(sizes) / len(sizes):.1f}, max {max(sizes)}); "
          f"dense and TF-IDF share {sum(overlap) / len(overlap):.1f} of top-{DEPTH} on average")
    print(f"[INFO] wrote {POOL_PATH}")
    return 0


def write_sheet(out: str, qids: list) -> int:
    saved = read_json(POOL_PATH)
    if saved["corpus_version"] != corpus_version():
        print(f"[ERROR] the pool was built from {saved['corpus_version']}, the corpus is now "
              f"{corpus_version()}; run `pool` again")
        return 1
    pool = saved["pool"]
    unknown = [q for q in qids if q not in pool]
    if unknown:
        print(f"[ERROR] not in the pool: {', '.join(unknown)}")
        return 1
    if qids:
        pool = {q: pool[q] for q in qids}
    questions = {q["id"]: q["text"] for q in load_questions()}
    articles = {a["url"]: a for a in load_articles()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(make_sheet(pool, questions, articles), encoding="utf-8", newline="\n")
    print(f"[INFO] wrote {out}: {len(pool)} questions, {sum(len(v) for v in pool.values())} candidates")
    return 0


def read_sheet(path: str) -> int:
    pool = read_json(POOL_PATH)["pool"]
    gold, errors = parse_sheet(Path(path).read_text(encoding="utf-8"), pool)
    for e in errors:
        print(f"[WARN] {e}")
    merged = read_json(GOLD_PATH)
    for qid, urls in gold.items():
        merged[qid] = {"relevant": urls, "judged": pool[qid]}
    write_json(GOLD_PATH, dict(sorted(merged.items())))
    print(f"[INFO] {len(gold)} questions read from the sheet; {len(merged)}/{len(pool)} in {GOLD_PATH.name}")
    return 1 if errors else 0


def main(argv: list) -> int:
    if argv[:1] == ["pool"]:
        return asyncio.run(make_pool())
    if argv[:1] == ["sheet"] and len(argv) >= 2:
        return write_sheet(argv[1], argv[2:])
    if argv[:1] == ["parse"] and len(argv) == 2:
        return read_sheet(argv[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
