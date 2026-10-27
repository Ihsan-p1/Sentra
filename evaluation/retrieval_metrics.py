"""
Scores for the retrieval evaluation (eval phase 2).

Relevance is judged per article, keyed by URL, because an article's chunks are
an ingestion detail and its URL survives a database refresh. A metric that
cannot be computed is None, never 0, matching hallucination_metrics.
"""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def rank_articles(chunks: List[Dict]) -> List[str]:
    """Article URLs by best chunk similarity, each URL once."""
    best: Dict[str, float] = {}
    for c in chunks:
        if c["similarity"] > best.get(c["url"], float("-inf")):
            best[c["url"]] = c["similarity"]
    return sorted(best, key=best.get, reverse=True)


def recall_at_k(ranked: List[str], gold: Set[str], k: int) -> Optional[float]:
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: List[str], gold: Set[str], k: int = 10) -> Optional[float]:
    if not gold:
        return None
    for rank, url in enumerate(ranked[:k], 1):
        if url in gold:
            return 1 / rank
    return 0.0


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(values, q))


def mean_ci(values: List[float], n_boot: int = 2000, seed: int = 0) -> Optional[Tuple[float, float]]:
    """95% bootstrap interval of the mean, resampling whole questions."""
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    means = data[rng.integers(0, len(data), size=(n_boot, len(data)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
