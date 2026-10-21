"""
Metrics for the retrieval evaluation (eval phase 2).

Rankings are lists of article URLs; gold is the set of URLs the owner judged
relevant. An empty gold set has no defined recall or MRR, so those return None.
"""
from evaluation.retrieval_metrics import (
    mean_ci,
    percentile,
    rank_articles,
    recall_at_k,
    reciprocal_rank,
)


def test_rank_articles_dedupes_by_best_chunk():
    chunks = [{"url": "a", "similarity": 0.5}, {"url": "b", "similarity": 0.7},
              {"url": "a", "similarity": 0.9}]
    assert rank_articles(chunks) == ["a", "b"]


def test_rank_articles_empty():
    assert rank_articles([]) == []


def test_recall_at_k():
    assert recall_at_k(["x", "a", "y", "b"], {"a", "b", "c"}, 2) == 1 / 3
    assert recall_at_k(["a"], set(), 5) is None


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert reciprocal_rank(["x"] * 10 + ["a"], {"a"}) == 0.0  # rank 11 is past k=10
    assert reciprocal_rank(["a"], set()) is None


def test_percentile_and_ci():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) is None
    assert mean_ci([0.5]) is None
    lo, hi = mean_ci([0.0, 1.0, 0.5, 0.5])
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
