"""
The owner's labelling sheet for the DeBERTa sample.

`make` writes data/raw/deberta_sheet.md from data/raw/deberta_sample.jsonl:
one sentence and one empty `label:` line each, with no outlet and no model
output. `parse` reads the filled sheet into data/raw/deberta_labels.jsonl
(biased, neutral or skip) and can be rerun while labelling is in progress.
All three files hold outlet text and stay out of git.

Usage: python scripts/fulltext/deberta_sheet.py make|parse
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.deberta_domain import make_sheet, parse_sheet
from scripts.fulltext.deberta_sample import SAMPLE_OUT, write_jsonl

SHEET = Path("data/raw/deberta_sheet.md")
LABELS_OUT = Path("data/raw/deberta_labels.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv) -> int:
    if argv not in (["make"], ["parse"]):
        print(__doc__.strip().splitlines()[-1])
        return 2
    sample = read_jsonl(SAMPLE_OUT)
    if argv == ["make"]:
        if SHEET.exists():
            print(f"[ERROR] {SHEET} exists and may hold labels; move it away first")
            return 1
        SHEET.write_text(make_sheet(sample), encoding="utf-8", newline="\n")
        print(f"[INFO] {len(sample)} sentences -> {SHEET}")
        return 0

    labels, errors = parse_sheet(SHEET.read_text(encoding="utf-8"), sample)
    for error in errors:
        print(f"[ERROR] {error}")
    if errors:
        print(f"[ERROR] {len(errors)} unreadable entries; nothing written")
        return 1
    write_jsonl(labels, LABELS_OUT)
    counts = Counter(entry["label"] for entry in labels)
    print(f"[INFO] {len(labels)} of {len(sample)} labelled -> {LABELS_OUT} "
          f"(biased {counts['biased']}, neutral {counts['neutral']}, skip {counts['skip']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
