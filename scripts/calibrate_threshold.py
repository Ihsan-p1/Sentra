"""
Calibrate the retrieval refusal threshold against the actual corpus.

Sentra refuses to answer when the best-matching chunk scores below
settings.RETRIEVAL_THRESHOLD. Picking that number by feel is how you end up
refusing questions the corpus clearly covers: the original 0.35 rejected 7 of 15
valid queries while blocking nothing that 0.28 did not already block.

This script measures the two distributions that actually matter - queries the
corpus can answer, and queries it cannot - and reports the error rates at each
candidate threshold so the choice can be defended with data.

Usage:
    python scripts/calibrate_threshold.py

Requires a populated database (scripts/tools/refresh_database.py).
"""
import asyncio
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import settings
from database.connection import db_manager
from rag.retrieval import RAGRetriever

# Questions the corpus (2024 Indonesian election aftermath) genuinely covers.
# Deliberately includes short, bare-name and Indonesian-language phrasings,
# because those are what real users type and what the old threshold rejected.
IN_CORPUS = [
    "who won the 2024 indonesian election",
    "prabowo cabinet appointments",
    "constitutional court election ruling",
    "voter turnout in indonesia",
    "gibran vice president",
    "jokowi endorsement",
    "coalition parties indonesia",
    "free meal program",
    "siapa itu prabowo subianto",
    "who is prabowo subianto",
    "siapa itu gibran rakabuming",
    "indonesian election results",
    "prabowo foreign policy",
    "KPU announcement",
    "siapa presiden indonesia baru",
    # Added after a UAT scenario was refused at a threshold this set had called
    # error-free: 23 queries is small enough that any realistic query missing
    # from it can invalidate the calibration.
    "compare how the outlets framed the cabinet formation",
    "how did each outlet cover the election result",
    "what were the main developments after the election",
    "what did prabowo say about food self sufficiency",
    "apa yang dikatakan prabowo tentang swasembada pangan",
    "bagaimana media meliput pembentukan kabinet",
    # The Indonesian phrasing of the most basic question in the corpus. It used
    # to be refused because the winner was named exactly once, in passing, in a
    # single article - articles stating the official result were added rather
    # than lowering the threshold to reach the one weak mention.
    "siapa pemenang pemilu 2024",
    "siapa yang menang pemilu presiden indonesia",
    "kapan prabowo dilantik jadi presiden",
    "how many votes did prabowo get",
]

# Questions the corpus cannot answer. The threshold must reject all of these.
OUT_OF_CORPUS = [
    "what is the price of bitcoin",
    "how to cook fried rice",
    "premier league top scorer",
    "python list comprehension tutorial",
    "weather forecast in tokyo",
    "best smartphone 2025",
    "how to lose weight fast",
    "quantum computing basics",
]

CANDIDATES = [0.25, 0.30, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.55]


async def max_similarities(retriever, queries):
    """Best-matching chunk score for each query."""
    results = []
    for query in queries:
        grouped = await retriever.retrieve(query)
        flat = retriever.flatten_results(grouped)
        results.append((query, max((c["similarity"] for c in flat), default=0.0)))
    return results


def describe(label, scored):
    values = [s for _, s in scored]
    print(
        f"{label:<16} n={len(values):<3} "
        f"min={min(values):.3f}  median={statistics.median(values):.3f}  max={max(values):.3f}"
    )
    return values


async def main():
    await db_manager.connect()
    retriever = RAGRetriever()

    print("=" * 78)
    print("RETRIEVAL THRESHOLD CALIBRATION")
    print("=" * 78)
    print(f"Embedding model: {settings.EMBEDDING_MODEL}")
    print(f"Current setting: RETRIEVAL_THRESHOLD = {settings.RETRIEVAL_THRESHOLD}\n")

    in_scored = await max_similarities(retriever, IN_CORPUS)
    out_scored = await max_similarities(retriever, OUT_OF_CORPUS)

    in_values = describe("In-corpus", in_scored)
    out_values = describe("Out-of-corpus", out_scored)

    separation = min(in_values) - max(out_values)
    print(
        f"\nSeparation between the two distributions: {separation:+.3f}"
        f"  ({'clean' if separation > 0 else 'overlapping'})"
    )

    print("\nError rates by threshold:")
    print(f"{'threshold':>10} | {'false refusals':>16} | {'false accepts':>14} | verdict")
    print("-" * 78)

    best = None
    for t in CANDIDATES:
        false_refusals = sum(1 for s in in_values if s < t)
        false_accepts = sum(1 for s in out_values if s >= t)
        # Prefer zero false accepts (never answer what we cannot support),
        # then the fewest false refusals.
        score = (false_accepts, false_refusals)
        if best is None or score < best[0]:
            best = (score, t)
        print(
            f"{t:>10.2f} | {false_refusals:>7}/{len(in_values):<8} | "
            f"{false_accepts:>6}/{len(out_values):<7} |"
            f"{'  <- current' if abs(t - settings.RETRIEVAL_THRESHOLD) < 1e-9 else ''}"
        )

    (fa, fr), recommended = best
    print(f"\nRecommended: {recommended:.2f}  ({fa} false accepts, {fr} false refusals)")
    if abs(recommended - settings.RETRIEVAL_THRESHOLD) > 1e-9:
        print(f"  -> differs from the configured {settings.RETRIEVAL_THRESHOLD}; "
              f"update RETRIEVAL_THRESHOLD in config/settings.py")
    else:
        print("  -> matches the configured value.")

    refused = [(q, s) for q, s in in_scored if s < settings.RETRIEVAL_THRESHOLD]
    if refused:
        print(f"\nValid queries still refused at {settings.RETRIEVAL_THRESHOLD}:")
        for query, score in sorted(refused, key=lambda x: x[1]):
            print(f"   {score:.3f}  {query}")

    leaked = [(q, s) for q, s in out_scored if s >= settings.RETRIEVAL_THRESHOLD]
    if leaked:
        print(f"\nWARNING - unanswerable queries accepted at {settings.RETRIEVAL_THRESHOLD}:")
        for query, score in sorted(leaked, key=lambda x: -x[1]):
            print(f"   {score:.3f}  {query}")

    print("\nNote: this is a calibration on a small hand-built query set, not a")
    print("rigorous ROC study. Re-run it whenever the corpus or embedding model changes.")

    await db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
