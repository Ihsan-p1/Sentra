"""
Score the hallucination checkers on the owner's labels (eval phase 1, step 5).

Reads the labelled sentences, runs Model A (trained detector), Model B (keyword
overlap) and an "always supported" baseline on every claim, and reports
recall / precision / F1 with `unsupported` as the positive class. Synthetic
negatives are built from the supported sentences and reported separately; the
relabel pass, when it exists, gives Cohen's kappa.

Offline: needs the embedder and models/hallucination_detector.pkl, not the
database or the LLM.

Usage:
    python scripts/evaluate_hallucination.py [--datasets DIR] [--out DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluation.hallucination_dataset import DATASETS_DIR, load_jsonl, save_jsonl
from evaluation.hallucination_metrics import label_agreement
from evaluation.hallucination_report import build_report, claim_rows, claude_agreement, predict_all
from evaluation.perturb import synthetic_rows

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def main(datasets: Path, out: Path) -> int:
    rows = load_jsonl(datasets / "hallucination_real.jsonl")
    answers = {a["query_id"]: a for a in load_jsonl(datasets / "hallucination_answers.jsonl")}
    claims = claim_rows(rows)
    if not claims:
        print("[ERROR] No labelled claims yet. Run scripts/label_hallucination.py first.")
        return 1

    # Heavy imports only once there is something to score.
    from config.settings import settings
    from models.baseline.models import get_baseline_models
    from models.hallucination.detector import HallucinationDetector
    from pipeline.embeddings import get_embedder

    detector = HallucinationDetector(os.path.join(settings.MODELS_DIR, "hallucination_detector.pkl"))
    baseline = get_baseline_models()["hallucination"]
    embedder = get_embedder()

    real_preds = predict_all(claims, answers, detector, baseline, embedder)
    synthetic = synthetic_rows(claims, answers)
    save_jsonl(datasets / "hallucination_synthetic.jsonl", synthetic)
    syn_preds = predict_all(synthetic, answers, detector, baseline, embedder) if synthetic else {}

    agreement = None
    relabels = load_jsonl(datasets / "hallucination_relabel.jsonl")
    if relabels:
        first = {(r["query_id"], r["sentence_idx"]): r["label"] for r in rows if r["label"] is not None}
        pairs = [(first[k], r["label"]) for r in relabels if (k := (r["query_id"], r["sentence_idx"])) in first]
        agreement = label_agreement([a for a, _ in pairs], [b for _, b in pairs])

    counts = {
        "skipped": sum(r["check_text"] is None for r in rows),
        "unlabelled": sum(r["check_text"] is not None and r["label"] is None for r in rows),
        "not_claim": sum(r["label"] == "not_claim" for r in rows),
    }
    claude = claude_agreement(
        rows,
        load_jsonl(datasets / "hallucination_claude_reading.jsonl"),
        load_jsonl(datasets / "hallucination_corrections.jsonl"),
    )
    markdown, data = build_report(claims, real_preds, synthetic, syn_preds, agreement, counts, claude)

    out.mkdir(parents=True, exist_ok=True)
    (out / "hallucination_real.md").write_text(markdown, encoding="utf-8", newline="\n")
    (out / "hallucination_real.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(markdown)
    print(f"Report written to {out / 'hallucination_real.md'} and {out / 'hallucination_real.json'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the hallucination checkers on real labels.")
    parser.add_argument("--datasets", type=Path, default=DATASETS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    sys.exit(main(args.datasets, args.out))
