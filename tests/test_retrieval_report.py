"""Scoring and report for the retrieval evaluation (eval phase 2)."""
from evaluation.retrieval_report import build_report, paired_difference, refusal_table, score_rankings

GOLD = {"q1": {"relevant": ["a"], "judged": ["a", "b"]},
        "q2": {"relevant": [], "judged": ["b"]}}


def test_score_rankings_skips_no_gold_queries():
    s = score_rankings({"q1": ["b", "a"], "q2": ["b"]}, GOLD)
    assert s["recall_at_5"]["mean"] == 1.0 and s["mrr_at_10"]["mean"] == 0.5
    assert s["n_scored"] == 1 and s["no_gold"] == ["q2"]


def test_score_rankings_ignores_queries_without_gold_entry():
    s = score_rankings({"q9": ["a"]}, {})
    assert s["n_scored"] == 0
    assert s["recall_at_5"]["mean"] is None


def test_empty_ranking_scores_zero_not_none():
    # Every chunk under MIN_CHUNK_SIMILARITY leaves nothing ranked: a miss, not a skip.
    s = score_rankings({"q1": []}, GOLD)
    assert s["recall_at_5"]["mean"] == 0.0 and s["mrr_at_10"]["mean"] == 0.0


def test_refusal_table_counts_by_group_and_threshold():
    t = refusal_table({"c01": 0.5, "c11": 0.33, "o01": 0.2, "o09": 0.4},
                      {"c01": "in_corpus", "c11": "in_corpus",
                       "o01": "ooc_calibration", "o09": ["ooc_held_out", "ooc_near"]},
                      {"default": 0.35, "strict": 0.42})
    assert t["default"]["in_corpus"] == {"refused": 1, "total": 2}
    assert t["default"]["ooc_near"] == {"refused": 0, "total": 1}
    assert t["default"]["ooc_held_out"] == {"refused": 0, "total": 1}
    assert t["strict"]["ooc_near"] == {"refused": 1, "total": 1}


SCORES = {name: score_rankings({"q1": ["a"], "q2": ["b"]}, GOLD) for name in ("dense", "tfidf")}
REFUSALS = refusal_table({"q1": 0.5}, {"q1": "in_corpus"}, {"default": 0.35, "strict": 0.42})
LATENCY = {"dense": {"p50": 12.0, "p95": 30.0, "n": 3}, "tfidf": {"p50": 1.0, "p95": 2.0, "n": 3}}
META = {"commit": "abc1234", "corpus_version": "x@1", "embedding_model": "m", "top_k": 8,
        "n_articles": 154, "generated_at": "2026-10-21T00:00:00",
        "gold_provenance": "judged by the owner"}


def test_report_states_corpus_limit_and_single_judge():
    md = build_report(SCORES, REFUSALS, LATENCY, META)
    assert "154" in md and "summaries" in md
    assert "single judge" in md and "pooled" in md


def test_report_names_questions_without_gold():
    md = build_report(SCORES, REFUSALS, LATENCY, META)
    assert "q2" in md


def test_report_splits_in_corpus_calibration_from_held_out():
    # c01-c25 are the in-corpus half of the threshold calibration set; c11 and c16
    # are refused there, so pooling them with n01-n25 hid where the refusals came from.
    refusals = refusal_table({"c11": 0.33, "n01": 0.5},
                             {"c11": "in_corpus_calibration", "n01": "in_corpus_held_out"},
                             {"default": 0.35})
    md = build_report(SCORES, refusals, LATENCY, META)
    assert "In-corpus, calibration set" in md and "In-corpus, held out" in md
    assert "only the held-out groups test it" in md


def test_paired_difference_uses_shared_questions():
    a = {"per_query": {"q1": {"recall_at_5": 1.0}, "q2": {"recall_at_5": 0.5}, "q3": {"recall_at_5": 0.0}}}
    b = {"per_query": {"q1": {"recall_at_5": 0.5}, "q2": {"recall_at_5": 0.5}}}
    d = paired_difference(a, b, "recall_at_5")
    assert d["mean"] == 0.25 and d["n"] == 2
    assert d["ci"][0] <= 0.25 <= d["ci"][1]


def test_report_shows_paired_difference_row():
    md = build_report(SCORES, REFUSALS, LATENCY, META)
    assert "Dense minus TF-IDF (paired)" in md


def test_recall_ceiling_when_gold_exceeds_k():
    gold = {"q1": {"relevant": ["a"], "judged": []},
            "q2": {"relevant": list("abcdefghij"), "judged": []}}
    s = score_rankings({"q1": ["a"], "q2": list("abcdefghij")}, gold)
    assert s["recall_at_5_ceiling"] == 0.75   # mean of 1.0 and 5/10
    assert "Best possible Recall@5" in build_report({"dense": s}, REFUSALS, LATENCY, META)
