"""
Score neutral's DeBERTa against the owner's labels on the Sentra sample.

Reads data/raw/deberta_sample.jsonl, deberta_labels.jsonl and
deberta_sample_meta.json, and writes evaluation/results/deberta_sentra.md:
counts and scores only, next to the BABE and BASIL figures the model served
with its predictions. When data/raw/deberta_claude_reading.jsonl exists, the
report also states how often that second reading matches the owner.

Usage: python scripts/fulltext/measure_deberta.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.deberta_domain import build_report
from scripts.fulltext.deberta_sample import META_OUT, SAMPLE_OUT
from scripts.fulltext.deberta_sheet import LABELS_OUT, read_jsonl

READING = Path("data/raw/deberta_claude_reading.jsonl")
REPORT = Path("evaluation/results/deberta_sentra.md")


def main() -> int:
    sample, labels = read_jsonl(SAMPLE_OUT), read_jsonl(LABELS_OUT)
    meta = json.loads(META_OUT.read_text(encoding="utf-8"))
    reading = read_jsonl(READING) if READING.exists() else None
    REPORT.write_text(build_report(sample, labels, meta, reading), encoding="utf-8", newline="\n")
    print(f"[INFO] {len(labels)} labels on {len(sample)} sentences -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
