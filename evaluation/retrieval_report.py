"""
Scoring and the markdown report for the retrieval evaluation (eval phase 2).

Pure functions: rankings, similarity scores and timings come in, numbers and
text go out. scripts/evaluate_retrieval.py does the database work.
"""
from typing import Dict, List, Optional, Union

from evaluation.retrieval_metrics import mean_ci, recall_at_k, reciprocal_rank

SYSTEMS = {
    "dense": "Dense (current, cosine over chunk embeddings)",
    "tfidf": "TF-IDF baseline",
    "rrf": "Hybrid RRF (dense + TF-IDF, no model)",
    "rerank": "Cross-encoder rerank of dense candidates",
    "rerank_hybrid": "Cross-encoder rerank of dense + TF-IDF candidates",
}

GROUPS = {
    "in_corpus_calibration": "In-corpus, calibration set (the threshold was tuned on these; "
                             "refusal is a false refusal)",
    "in_corpus_held_out": "In-corpus, held out (refusal is a false refusal)",
    "ooc_calibration": "Out-of-corpus, calibration set (the threshold was tuned on these)",
    "ooc_held_out": "Out-of-corpus, held out",
    "ooc_near": "of which near-domain (elections outside Indonesia)",
}

# Keep rule, fixed in the phase 3 plan before any result was seen.
KEEP_RECALL_GAIN = 0.05   # paired Recall@5 gain over dense, at least
KEEP_MRR_DROP = 0.02      # paired MRR@10 may fall by at most this
KEEP_P95_MS = 1000.0
GATE_NEAR_MIN = 4         # near-domain held-out questions refused, at least
GATE_FALSE_REFUSALS_MAX = 1  # in-corpus held-out questions refused, at most


def _summary(values: List[float]) -> Dict:
    return {"mean": sum(values) / len(values) if values else None, "ci": mean_ci(values)}


def score_rankings(rankings: Dict[str, List[str]], gold: Dict[str, Dict]) -> Dict:
    """Recall@5 and MRR@10 over questions with at least one relevant article."""
    per_query, no_gold = {}, []
    for qid, ranked in rankings.items():
        if qid not in gold:
            continue
        relevant = set(gold[qid]["relevant"])
        if not relevant:
            no_gold.append(qid)
            continue
        per_query[qid] = {
            "recall_at_5": recall_at_k(ranked, relevant, 5),
            "mrr_at_10": reciprocal_rank(ranked, relevant, 10),
        }
    ceilings = [min(5, len(gold[q]["relevant"])) / len(gold[q]["relevant"]) for q in per_query]
    return {
        "recall_at_5": _summary([v["recall_at_5"] for v in per_query.values()]),
        "recall_at_5_ceiling": sum(ceilings) / len(ceilings) if ceilings else None,
        "mrr_at_10": _summary([v["mrr_at_10"] for v in per_query.values()]),
        "n_scored": len(per_query),
        "no_gold": no_gold,
        "per_query": per_query,
    }


def paired_difference(a: Dict, b: Dict, metric: str) -> Dict:
    """Mean per-question difference a - b over the questions both scored, with a bootstrap interval."""
    shared = [q for q in a["per_query"] if q in b["per_query"]]
    diffs = [a["per_query"][q][metric] - b["per_query"][q][metric] for q in shared]
    return {**_summary(diffs), "n": len(diffs)}


def refusal_table(
    max_sims: Dict[str, float],
    kinds: Dict[str, Union[str, List[str]]],
    thresholds: Dict[str, float],
) -> Dict:
    """Questions refused (best chunk below the threshold) per threshold and per group."""
    table = {}
    for name, threshold in thresholds.items():
        counts: Dict[str, Dict[str, int]] = {}
        for qid, groups in kinds.items():
            for group in [groups] if isinstance(groups, str) else groups:
                c = counts.setdefault(group, {"refused": 0, "total": 0})
                c["total"] += 1
                c["refused"] += max_sims[qid] < threshold
        table[name] = counts
    return table


def pick_threshold(in_corpus: List[float], out_of_corpus: List[float]) -> float:
    """Threshold with the fewest errors on calibration scores; ties go to fewer leaks, then the lower value.

    Candidates are midpoints between neighbouring scores. A question is refused
    when its score is below the threshold, so an out-of-corpus score at or above
    it is a leak and an in-corpus score below it is a false refusal.
    """
    scores = sorted(set(in_corpus) | set(out_of_corpus))
    if len(scores) < 2:
        raise ValueError("pick_threshold needs at least two distinct scores")
    best = None
    for lo, hi in zip(scores, scores[1:]):
        # The midpoint with -inf (a question with no candidates) is -inf and would separate nothing.
        t = hi - 1.0 if lo == float("-inf") else (lo + hi) / 2
        leaks = sum(s >= t for s in out_of_corpus)
        key = (sum(s < t for s in in_corpus) + leaks, leaks, t)
        best = key if best is None or key < best else best
    return best[2]


def unjudged_at_k(rankings: Dict[str, List[str]], gold: Dict[str, Dict], k: int = 10) -> int:
    """Articles in a top k that the owner never judged, summed over questions with gold."""
    return sum(
        url not in set(gold[qid]["judged"])
        for qid, ranked in rankings.items() if qid in gold
        for url in ranked[:k]
    )


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _points(summary: Dict) -> str:
    if summary["mean"] is None:
        return "n/a"
    ci = summary["ci"]
    text = f"{summary['mean'] * 100:+.1f} pts"
    return text + (f" [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]" if ci else "")


def _with_ci(summary: Dict) -> str:
    if summary["mean"] is None:
        return "n/a"
    ci = summary["ci"]
    return _pct(summary["mean"]) + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "")


def keep_rule_lines(scores: Dict, latency: Dict, gates: Optional[Dict]) -> List[str]:
    """The pre-registered keep rule applied to every system other than dense and TF-IDF, and to each gate."""
    lines = ["", "## Keep rule (pre-registered)", "",
             f"A system is kept when its paired Recall@5 gain over dense is at least "
             f"{KEEP_RECALL_GAIN * 100:.0f} pts, its MRR@10 falls by no more than {KEEP_MRR_DROP * 100:.0f} pts "
             f"and its p95 is at most {KEEP_P95_MS:.0f} ms. It is called better than dense only when the "
             f"Recall@5 interval excludes 0. A gate is useful when it refuses at least {GATE_NEAR_MIN} "
             f"near-domain held-out questions with at most {GATE_FALSE_REFUSALS_MAX} in-corpus held-out "
             "false refusal.", ""]
    for key in scores:
        if key in ("dense", "tfidf") or "dense" not in scores or key not in latency:
            continue
        recall = paired_difference(scores[key], scores["dense"], "recall_at_5")
        mrr = paired_difference(scores[key], scores["dense"], "mrr_at_10")
        p95 = latency[key]["p95"]
        fails = []
        if recall["mean"] is None or recall["mean"] < KEEP_RECALL_GAIN:
            fails.append(f"Recall@5 gain under {KEEP_RECALL_GAIN * 100:.0f} pts")
        if mrr["mean"] is None or mrr["mean"] < -KEEP_MRR_DROP:
            fails.append(f"MRR@10 down more than {KEEP_MRR_DROP * 100:.0f} pts")
        if p95 > KEEP_P95_MS:
            fails.append(f"p95 over {KEEP_P95_MS:.0f} ms")
        verdict = "drop (" + "; ".join(fails) + ")" if fails else "keep"
        better = recall["ci"] is not None and recall["ci"][0] > 0
        lines.append(f"- {SYSTEMS.get(key, key)}: {verdict}. Recall@5 {_points(recall)}, MRR@10 {_points(mrr)}, "
                     f"p95 {p95:.0f} ms; " + ("better than dense." if better else "not shown better than dense."))
    for key, gate in (gates or {}).items():
        table = gate["refusals"]["picked"]
        near = table.get("ooc_near", {"refused": 0, "total": 0})
        false = table.get("in_corpus_held_out", {"refused": 0, "total": 0})
        useful = near["refused"] >= GATE_NEAR_MIN and false["refused"] <= GATE_FALSE_REFUSALS_MAX
        lines.append(f"- Gate, {SYSTEMS.get(key, key)}: {'useful' if useful else 'not useful'}. Near-domain "
                     f"refused {near['refused']}/{near['total']}, in-corpus held-out false refusals "
                     f"{false['refused']}/{false['total']}.")
    return lines


def build_report(
    scores: Dict,
    refusals: Dict,
    latency: Dict,
    meta: Dict,
    gates: Optional[Dict] = None,
    unjudged: Optional[Dict[str, int]] = None,
) -> str:
    first = next(iter(scores.values()))
    names = list(refusals)
    lines = [
        "# Retrieval evaluation",
        "",
        f"{first['n_scored']} questions scored against owner-judged gold articles. "
        "Recall@5 is the share of a question's relevant articles in the top 5; MRR@10 is the "
        "reciprocal rank of the first relevant article in the top 10. Intervals are 95% "
        "bootstrap intervals over questions.",
        "",
        "| System | Recall@5 | MRR@10 | Questions |",
        "|---|---|---|---|",
    ]
    for key, s in scores.items():
        lines.append(f"| {SYSTEMS.get(key, key)} | {_with_ci(s['recall_at_5'])} | "
                     f"{_with_ci(s['mrr_at_10'])} | {s['n_scored']} |")
    if "dense" in scores and "tfidf" in scores:
        cells = []
        for metric in ("recall_at_5", "mrr_at_10"):
            d = paired_difference(scores["dense"], scores["tfidf"], metric)
            cells.append(_points(d))
        lines.append(f"| Dense minus TF-IDF (paired) | {cells[0]} | {cells[1]} | {d['n']} |")
    for key in scores:
        if key in ("dense", "tfidf") or "dense" not in scores:
            continue
        cells = []
        for metric in ("recall_at_5", "mrr_at_10"):
            d = paired_difference(scores[key], scores["dense"], metric)
            cells.append(_points(d))
        lines.append(f"| {SYSTEMS.get(key, key)} minus dense (paired) | {cells[0]} | {cells[1]} | {d['n']} |")
    if first.get("recall_at_5_ceiling") is not None:
        lines += ["", f"Best possible Recall@5 with this gold: {_pct(first['recall_at_5_ceiling'])}, "
                      "because some questions have more than 5 relevant articles."]
    if first["no_gold"]:
        lines += ["", f"No candidate was judged relevant for {', '.join(first['no_gold'])}; "
                      "those questions are left out of both averages."]

    thresholds = meta.get("thresholds", {})
    header = " | ".join(f"Refused at {thresholds.get(n, '?')} ({n})" for n in names)
    lines += [
        "",
        "## Refusal gate (dense only; TF-IDF has no gate)",
        "",
        "A question is refused when its best chunk scores below the threshold.",
        "",
        f"| Group | {header} |",
        "|---|" + "---|" * len(names),
    ]
    for group, label in GROUPS.items():
        if group not in refusals[names[0]]:
            continue
        cells = []
        for n in names:
            c = refusals[n][group]
            cells.append(f"{c['refused']}/{c['total']}")
        lines.append(f"| {label} | {' | '.join(cells)} |")

    if gates:
        default = names[0]
        lines += [
            "",
            "## Reranker score as refusal signal",
            "",
            "The best reranker score per question takes the place of the best cosine. Each "
            "threshold was picked on the calibration groups only (fewest errors, ties to fewer "
            "leaks); the held-out groups test it. Reranker scores are raw logits.",
            "",
        ]
        for key, gate in gates.items():
            lines.append(f"- {SYSTEMS.get(key, key)}: threshold {gate['threshold']:.3f}, "
                         "picked on the calibration groups only.")
        header = " | ".join(SYSTEMS.get(k, k) for k in gates)
        lines += ["", f"| Group | Dense, cosine at {thresholds.get(default, '?')} | {header} |",
                  "|---|---|" + "---|" * len(gates)]
        tables = [refusals[default]] + [g["refusals"]["picked"] for g in gates.values()]
        for group, label in GROUPS.items():
            if not any(group in t for t in tables):
                continue
            cells = [f"{t[group]['refused']}/{t[group]['total']}" if group in t else "n/a" for t in tables]
            lines.append(f"| {label} | {' | '.join(cells)} |")

    if any(k not in ("dense", "tfidf") for k in scores) or gates:
        lines += keep_rule_lines(scores, latency, gates)

    lines += ["", "## Latency per retrieval call", "", "| System | p50 | p95 | Calls |", "|---|---|---|---|"]
    for key, lat in latency.items():
        lines.append(f"| {SYSTEMS.get(key, key)} | {lat['p50']:.0f} ms | {lat['p95']:.0f} ms | {lat['n']} |")

    lines += [
        "",
        "## Limits",
        "",
        f"- The corpus holds {meta['n_articles']} article summaries (median 97 words), not full text.",
        f"- Gold comes from a single judge: {meta['gold_provenance']}.",
        "- Gold is pooled from the top 10 of both systems; a relevant article neither system "
        "ranked in its top 10 was never judged and counts as not relevant.",
        *([f"- {meta['gold_extra']} articles first ranked by the phase 3 systems were judged later by "
           "the same owner; every system is scored on the same extended gold."]
          if meta.get("gold_extra") else []),
        "- The calibration questions, in-corpus and out-of-corpus, are the data the refusal "
        "threshold was tuned on; only the held-out groups test it.",
        *([f"- Articles in a top 10 that were never judged: "
           + ", ".join(f"{k} {v}" for k, v in unjudged.items())
           + ". They count as not relevant."] if unjudged else []),
        "- Latency was measured on one laptop against a local PostgreSQL; the dense figure "
        "includes embedding the query and fetching every chunk from the database.",
        "",
        "## Setup",
        "",
        f"Commit `{meta['commit']}`, corpus `{meta['corpus_version']}`, embedder "
        f"`{meta['embedding_model']}`, top_k {meta['top_k']} per outlet, generated "
        f"{meta['generated_at']}.",
        "",
    ]
    return "\n".join(lines)
