"""
Predictions and the report for the real hallucination evaluation.

The checkers are passed in, so the tests can use small fakes and the script
passes the real Model A, Model B and embedder.
"""
from typing import Dict, List, Optional, Tuple

from evaluation.hallucination_metrics import binary_scores, cluster_bootstrap_ci, label_agreement

MODELS = ("model_a", "model_b", "always_supported")
MODEL_NAMES = {
    "model_a": "Model A (Logistic Regression)",
    "model_b": "Model B (Keyword overlap)",
    "always_supported": "Always supported",
}
METRICS = ("recall", "precision", "f1")


def claim_rows(rows: List[Dict]) -> List[Dict]:
    """Checked sentences the owner labelled supported or unsupported."""
    return [r for r in rows if r["check_text"] is not None and r["label"] in ("supported", "unsupported")]


def predict_all(rows: List[Dict], answers_by_qid: Dict[str, Dict], detector, baseline, embedder) -> Dict[str, List[bool]]:
    """True means the checker calls the sentence unsupported."""
    source_cache = {}
    preds = {m: [] for m in MODELS}
    for row in rows:
        qid = row["query_id"]
        if qid not in source_cache:
            texts = [c["text"] for c in answers_by_qid[qid]["chunks"]]
            source_cache[qid] = (texts, embedder.generate(texts, show_progress=False))
        texts, embs = source_cache[qid]
        text = row["check_text"]
        result_a = detector.predict(embedder.generate_single(text), embs, text, texts)
        preds["model_a"].append(not result_a["is_supported"])
        preds["model_b"].append(not baseline.check_sentence(text, texts)["is_supported"])
        preds["always_supported"].append(False)
    return preds


def claude_agreement(rows: List[Dict], readings: List[Dict], corrections: List[Dict]) -> Optional[Dict]:
    """Owner vs Claude's pre-recorded reading, using the owner's answer from before any correction."""
    claude = {(r["query_id"], r["sentence_idx"]): r["label"] for r in readings}
    before = {}
    for c in corrections:
        if c.get("after_claude_reading"):
            before.setdefault((c["query_id"], c["sentence_idx"]), c["from"])
    keys = [(r["query_id"], r["sentence_idx"]) for r in rows if r["label"] is not None]
    keys = [k for k in keys if k in claude]
    if not keys:
        return None
    final = {(r["query_id"], r["sentence_idx"]): r["label"] for r in rows}
    result = label_agreement([before.get(k, final[k]) for k in keys], [claude[k] for k in keys])
    result["corrected"] = sum(k in before for k in keys)
    return result


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _with_ci(value: Optional[float], ci: Optional[Tuple[float, float]]) -> str:
    if value is None:
        return "n/a"
    if ci is None:
        return _pct(value)
    return f"{_pct(value)} [{_pct(ci[0])}, {_pct(ci[1])}]"


def build_report(
    real_rows: List[Dict],
    real_preds: Dict[str, List[bool]],
    syn_rows: List[Dict],
    syn_preds: Dict[str, List[bool]],
    agreement: Optional[Dict],
    counts: Dict[str, int],
    claude: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    y_true = [r["label"] == "unsupported" for r in real_rows]
    groups = [r["query_id"] for r in real_rows]
    real = {}
    for m in MODELS:
        scores = binary_scores(y_true, real_preds[m])
        for metric in METRICS:
            scores[f"{metric}_ci"] = cluster_bootstrap_ci(groups, y_true, real_preds[m], metric)
        real[m] = scores

    synthetic = {}
    kinds = sorted({r["kind"] for r in syn_rows})
    for m in MODELS[:2]:
        if not syn_rows:
            break
        synthetic[m] = {"all": sum(syn_preds[m]) / len(syn_rows)}
        for kind in kinds:
            picked = [p for r, p in zip(syn_rows, syn_preds[m]) if r["kind"] == kind]
            synthetic[m][kind] = sum(picked) / len(picked)

    n_questions = len(set(groups))
    positives = sum(y_true)
    lines = [
        "# Hallucination check on real Sentra answers",
        "",
        f"{len(real_rows)} labelled claim sentences from {n_questions} questions, "
        f"{positives} of them unsupported. Positive class: `unsupported`, so recall is the catch rate. "
        "Intervals are 95% bootstrap intervals over questions.",
        "",
        "| Metric | " + " | ".join(MODEL_NAMES[m] for m in MODELS) + " |",
        "|---|" + "---|" * len(MODELS),
    ]
    for metric in METRICS:
        cells = [_with_ci(real[m][metric], real[m][f"{metric}_ci"]) for m in MODELS]
        lines.append(f"| {metric.capitalize()} | " + " | ".join(cells) + " |")
    cells = [f"{real[m]['tp']} / {real[m]['fp']} / {real[m]['fn']}" for m in MODELS]
    lines.append("| TP / FP / FN | " + " | ".join(cells) + " |")

    lines += ["", "## Synthetic negatives (reported separately)", ""]
    if syn_rows:
        lines += [
            f"{len(syn_rows)} supported sentences with one fact swapped, unsupported by construction. "
            "Catch rate = share flagged unsupported.",
            "",
            "| Swap | n | " + " | ".join(MODEL_NAMES[m] for m in MODELS[:2]) + " |",
            "|---|---|---|---|",
            f"| all | {len(syn_rows)} | " + " | ".join(_pct(synthetic[m]["all"]) for m in MODELS[:2]) + " |",
        ]
        for kind in kinds:
            n = sum(r["kind"] == kind for r in syn_rows)
            lines.append(f"| {kind} | {n} | " + " | ".join(_pct(synthetic[m][kind]) for m in MODELS[:2]) + " |")
    else:
        lines.append("No synthetic rows: no supported sentence had a name, year or number to swap.")

    lines += ["", "## Label consistency", ""]
    if agreement:
        kappa = "n/a" if agreement["kappa"] is None else f"{agreement['kappa']:.2f}"
        lines.append(f"{agreement['n']} sentences relabelled blind: agreement {_pct(agreement['agreement'])}, "
                     f"Cohen's kappa {kappa}.")
    else:
        lines.append("Not measured yet (run `scripts/label_hallucination.py --relabel` about a week after labelling).")

    lines += [
        "",
        "## Limits",
        "",
        "- The corpus holds article summaries (median 97 words), not full text.",
        "- One person chose every label; for most questions an AI tool tidied the written reasons.",
        "- Model A was trained on synthetic data.",
        "- Model B is scored per sentence with `check_sentence` on the same text as Model A; "
        "the app runs `check_response`, which splits sentences differently.",
        f"- {counts['skipped']} sentences were skipped by the engine (headings, under four words) and "
        f"{counts['unlabelled']} checked sentences are not labelled yet; neither is scored.",
    ]
    if claude:
        kappa = "n/a" if claude["kappa"] is None else f"{claude['kappa']:.2f}"
        plural = "" if claude["corrected"] == 1 else "s"
        lines.append(
            f"- Claude's reading of {claude['n']} sentences was recorded before each question and shown "
            f"after the owner answered. On the owner's first answers: agreement {_pct(claude['agreement'])}, "
            f"Cohen's kappa {kappa}. The owner then changed {claude['corrected']} label{plural} "
            "(`hallucination_corrections.jsonl`), and later answers may lean towards Claude's readings."
        )

    data = {
        "n_sentences": len(real_rows),
        "n_unsupported": positives,
        "n_questions": n_questions,
        "real": real,
        "synthetic": {"n": len(syn_rows), "catch_rate": synthetic},
        "agreement": agreement,
        "claude_agreement": claude,
        "counts": counts,
    }
    return "\n".join(lines) + "\n", data
