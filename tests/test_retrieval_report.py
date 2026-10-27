"""Scoring and report for the retrieval evaluation (eval phase 2)."""
import pytest
from evaluation.retrieval_report import (build_report, paired_difference, pick_threshold, refusal_table,
                                         score_rankings, unjudged_at_k)

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


# --- eval phase 3 ---

def test_pick_threshold_separable_takes_midpoint():
    assert pick_threshold([0.8, 0.9], [0.1, 0.3]) == pytest.approx(0.55)


def test_pick_threshold_overlap_prefers_fewer_leaks():
    # 0.2 and 0.46 both make one error; 0.46 leaks nothing
    assert pick_threshold([0.30, 0.6, 0.7], [0.32, 0.1]) == pytest.approx(0.46)


def test_unjudged_counts_top_k_outside_judged():
    gold = {"q1": {"relevant": ["a"], "judged": ["a", "b"]}}
    assert unjudged_at_k({"q1": ["a", "x", "b", "y"]}, gold, k=3) == 1


def test_report_adds_paired_row_per_new_system():
    scores = {**SCORES, "rerank": score_rankings({"q1": ["a"]}, GOLD)}
    latency = {**LATENCY, "rerank": {"p50": 40.0, "p95": 90.0, "n": 3}}
    md = build_report(scores, REFUSALS, latency, META)
    assert "| Cross-encoder rerank of dense candidates minus dense (paired) |" in md
    assert "| Dense minus TF-IDF (paired) |" in md


def test_report_gate_section_names_calibration_only():
    gate = refusal_table({"q1": 1.5}, {"q1": "in_corpus_held_out"}, {"picked": 0.42})
    md = build_report(SCORES, REFUSALS, LATENCY, META,
                      gates={"rerank": {"threshold": 0.42, "refusals": gate}})
    assert "## Reranker score as refusal signal" in md
    assert "threshold 0.420, picked on the calibration groups only" in md


def test_report_lists_unjudged_counts():
    md = build_report(SCORES, REFUSALS, LATENCY, META, unjudged={"dense": 0, "rerank": 7})
    assert "Articles in a top 10 that were never judged: dense 0, rerank 7" in md


def test_report_states_extended_gold():
    md = build_report(SCORES, REFUSALS, LATENCY, {**META, "gold_extra": 12})
    assert ("12 articles first ranked by the phase 3 systems were judged later by the same owner; "
            "every system is scored on the same extended gold.") in md


# --- final review fixes ---

def test_pick_threshold_separates_minus_inf_score():
    # a question with no candidates scores -inf; the midpoint with it is -inf, so it was never offered
    t = pick_threshold([-0.5, 1.0], [float("-inf")])
    assert float("-inf") < t <= -0.5


def test_pick_threshold_needs_two_distinct_scores():
    with pytest.raises(ValueError):
        pick_threshold([0.5], [0.5])


KEEP_GOLD = {q: {"relevant": ["good"], "judged": ["good", "bad"]} for q in ("q1", "q2", "q3", "q4")}
DENSE_MISS = score_rankings({q: ["bad"] for q in KEEP_GOLD}, KEEP_GOLD)
NEW_HIT = score_rankings({q: ["good"] for q in KEEP_GOLD}, KEEP_GOLD)


def _keep_line(md, name):
    return next(line for line in md.splitlines() if line.startswith(f"- {name}:"))


def test_keep_rule_keeps_system_that_passes_all_three():
    latency = {"dense": {"p50": 1, "p95": 70.0, "n": 3}, "rrf": {"p50": 1, "p95": 172.0, "n": 3}}
    md = build_report({"dense": DENSE_MISS, "rrf": NEW_HIT}, REFUSALS, latency, META)
    assert "## Keep rule (pre-registered)" in md
    line = _keep_line(md, "Hybrid RRF (dense + TF-IDF, no model)")
    assert "keep" in line and "better than dense" in line and "172 ms" in line


def test_keep_rule_drops_system_over_latency_budget():
    latency = {"dense": {"p50": 1, "p95": 70.0, "n": 3}, "rerank": {"p50": 1, "p95": 1080.0, "n": 3}}
    md = build_report({"dense": DENSE_MISS, "rerank": NEW_HIT}, REFUSALS, latency, META)
    line = _keep_line(md, "Cross-encoder rerank of dense candidates")
    assert "drop" in line and "p95 over 1000 ms" in line


def test_keep_rule_gate_needs_four_near_refusals():
    groups = {"n1": ["ooc_held_out", "ooc_near"], "n2": ["ooc_held_out", "ooc_near"], "c1": "in_corpus_held_out"}
    gate = refusal_table({"n1": -5.0, "n2": 2.0, "c1": 3.0}, groups, {"picked": 0.0})
    md = build_report(SCORES, REFUSALS, LATENCY, META, gates={"rerank": {"threshold": 0.0, "refusals": gate}})
    line = _keep_line(md, "Gate, Cross-encoder rerank of dense candidates")
    assert "not useful" in line and "1/2" in line and "0/1" in line
