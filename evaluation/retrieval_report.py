"""
Scoring and the markdown report for the retrieval evaluation (eval phase 2).

Pure functions: rankings, similarity scores and timings come in, numbers and
text go out. scripts/evaluate_retrieval.py does the database work.
"""
from typing import Dict, List, Optional, Union

from evaluation.retrieval_metrics import mean_ci, recall_at_k, reciprocal_rank

SYSTEMS = {"dense": "Dense (current, cosine over chunk embeddings)", "tfidf": "TF-IDF baseline"}

GROUPS = {
    "in_corpus": "In-corpus questions (refusal here is a false refusal)",
    "ooc_calibration": "Out-of-corpus, calibration set (the threshold was tuned on these)",
    "ooc_held_out": "Out-of-corpus, held out",
    "ooc_near": "of which near-domain (elections outside Indonesia)",
}


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
    return {
        "recall_at_5": _summary([v["recall_at_5"] for v in per_query.values()]),
        "mrr_at_10": _summary([v["mrr_at_10"] for v in per_query.values()]),
        "n_scored": len(per_query),
        "no_gold": no_gold,
        "per_query": per_query,
    }


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


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _with_ci(summary: Dict) -> str:
    if summary["mean"] is None:
        return "n/a"
    ci = summary["ci"]
    return _pct(summary["mean"]) + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "")


def build_report(scores: Dict, refusals: Dict, latency: Dict, meta: Dict) -> str:
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
        "- The calibration out-of-corpus questions are the data the refusal threshold was tuned "
        "on; only the held-out group tests it.",
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
