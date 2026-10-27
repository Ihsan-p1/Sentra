"""Tests for evaluation/reranking.py (eval phase 3). Offline: the cross-encoder is replaced by a fake."""
from evaluation.reranking import article_text, reciprocal_rank_fusion, rerank


def test_article_text_joins_title_and_content():
    assert article_text({"title": "T", "content": "C"}) == "T. C"


def test_rerank_orders_by_score_and_passes_texts_in_url_order():
    seen = {}

    def fn(q, texts):
        seen["args"] = (q, texts)
        return [0.1, 0.9, 0.5]

    arts = {u: {"title": u.upper(), "content": "x"} for u in "abc"}
    assert rerank("q", ["a", "b", "c"], arts, fn) == [("b", 0.9), ("c", 0.5), ("a", 0.1)]
    assert seen["args"] == ("q", ["A. x", "B. x", "C. x"])


def test_rerank_empty_never_calls_the_model():
    def boom(q, texts):
        raise AssertionError("called")

    assert rerank("q", [], {}, boom) == []


def test_rrf_fuses_and_keeps_urls_found_by_one_system():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]]) == ["b", "a", "c"]


def test_rrf_ties_keep_first_seen_order():
    assert reciprocal_rank_fusion([["a"], ["b"]]) == ["a", "b"]
