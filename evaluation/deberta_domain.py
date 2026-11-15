"""
Measuring neutral's DeBERTa on Sentra sentences labelled by the owner.

Third domain-shift point after BABE and BASIL: sentences come from the full-text
corpus as `POST /analyze` segments them, the owner labels a stratified sample
blind to the model, and the model's labels are scored against the owner's.
Everything here is pure, so it is tested offline; the scripts under
scripts/fulltext/ do the I/O.
"""
import hashlib
import random
import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score

LABELS = ("biased", "neutral")
MODEL_VERSION = "deberta-babe-f0s3"
SENTRA_DOMAIN = "Sentra full-text corpus"
SKIP = "skip"
ANSWERS = {"b": "biased", "biased": "biased", "n": "neutral", "neutral": "neutral",
           "s": SKIP, "skip": SKIP}

SHEET_HEAD = re.compile(r"^# .*\(sampel ([0-9a-f]{12})\)\s*$")
ITEM = re.compile(r"^## (\d+)\s*$")
ANSWER = re.compile(r"^label:(.*)$")
LOOSE_ANSWER = re.compile(r"(?i)^\s*label\s*:")


def macro_f1(gold: List[str], pred: List[str]) -> float:
    """Macro F1 over both classes; a class absent from gold and pred scores 0, as in neutral."""
    return float(f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0))


def best_constant(gold: List[str]) -> Tuple[str, float]:
    """The single label that scores highest when predicted for every sentence, and its macro F1."""
    return max(((label, macro_f1(gold, [label] * len(gold))) for label in LABELS),
               key=lambda pair: pair[1])


def bootstrap_ci(rows: List[Dict], n: int = 2000, seed: int = 7) -> Tuple[float, float]:
    """95% interval for macro F1 from resampling whole articles: sentences of one article are not independent."""
    by_url: Dict[str, List[Dict]] = {}
    for row in rows:
        by_url.setdefault(row["url"], []).append(row)
    articles = list(by_url.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        drawn = [row for k in rng.integers(0, len(articles), size=len(articles)) for row in articles[k]]
        values.append(macro_f1([r["gold"] for r in drawn], [r["pred"] for r in drawn]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _quotas(sizes: Dict[str, int], total: int, what: str) -> Dict[str, int]:
    """Split `total` evenly over the keys; what a small key cannot give comes from the largest."""
    order = sorted(sizes, key=lambda key: (-sizes[key], key))
    take = {key: min(total // len(order), sizes[key]) for key in order}
    for key in order:
        take[key] += min(total - sum(take.values()), sizes[key] - take[key])
    if sum(take.values()) < total:
        raise ValueError(f"{what} has {sum(sizes.values())} sentences, fewer than its share of {total}")
    return take


def stratified_sample(sentences: List[Dict], n: int = 150, seed: int = 7) -> List[Dict]:
    """
    `n` sentences, the same number per outlet, split evenly over that outlet's events.

    Rows come back shuffled with an `id` from 1, so the sheet does not group
    sentences by outlet or event.
    """
    strata: Dict[str, Dict[str, List[Dict]]] = {}
    for row in sentences:
        strata.setdefault(row["outlet"], {}).setdefault(row["event"], []).append(row)
    outlets = sorted(strata)
    rng = random.Random(seed)
    picked: List[Dict] = []
    for position, outlet in enumerate(outlets):
        share = n // len(outlets) + (position < n % len(outlets))
        events = strata[outlet]
        take = _quotas({event: len(rows) for event, rows in events.items()}, share, f"outlet {outlet}")
        for event in sorted(events):
            picked += rng.sample(events[event], take[event])
    rng.shuffle(picked)
    return [{"id": i, **row} for i, row in enumerate(picked, 1)]


def join_labels(sample: List[Dict], labels: List[Dict]) -> List[Dict]:
    """Sample rows that carry an owner label, as `gold`. Skipped and unlabelled sentences are left out."""
    by_id = {row["id"]: row for row in sample}
    rows = []
    for entry in labels:
        if entry["id"] not in by_id:
            raise ValueError(f"label for id {entry['id']}, which is not in the sample")
        if entry["label"] in LABELS:
            rows.append({**by_id[entry["id"]], "gold": entry["label"]})
    return rows


def sample_fingerprint(sample: List[Dict]) -> str:
    return hashlib.sha256("\n".join(row["text"] for row in sample).encode("utf-8")).hexdigest()[:12]


def make_sheet(sample: List[Dict]) -> str:
    """The owner's labelling sheet: sentence text only, no outlet, no model output."""
    lines = [
        f"# Lembar label DeBERTa (sampel {sample_fingerprint(sample)})",
        "",
        "Isi tiap baris `label:` dengan `b` (biased), `n` (neutral) atau `s` (skip, bukan kalimat",
        "yang bisa dinilai). Biarkan kosong kalau belum dinilai. Nilai kalimatnya saja, menurut",
        "`evaluation/datasets/DEBERTA_LABELING_GUIDE.md`.",
        "",
    ]
    for row in sample:
        lines += [f"## {row['id']:03d}", " ".join(row["text"].split()), "label: ", ""]
    return "\n".join(lines)


def parse_sheet(text: str, sample: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Labels read from a filled sheet, plus one message per line that could not be read."""
    lines = text.lstrip("﻿").splitlines()  # Notepad saves UTF-8 with a BOM
    head = SHEET_HEAD.match(lines[0]) if lines else None
    if not head or head.group(1) != sample_fingerprint(sample):
        return [], ["the sheet was made from a different sample; make a new sheet"]
    ids = {row["id"] for row in sample}
    labels: List[Dict] = []
    errors: List[str] = []
    current = None
    for no, line in enumerate(lines, 1):
        item = ITEM.match(line)
        if item:
            current = int(item.group(1))
            if current not in ids:
                errors.append(f"line {no}: {current} is not in the sample")
                current = None
            continue
        answer = ANSWER.match(line)
        if not answer and LOOSE_ANSWER.match(line):
            errors.append(f"line {no}: write the label line exactly as `label: b`, lower case, no indent")
            continue
        if not (answer and current):
            continue
        value, sentence_id, current = answer.group(1).strip().lower(), current, None
        if not value:
            continue
        if any(entry["id"] == sentence_id for entry in labels):
            errors.append(f"line {no}: {sentence_id} is labelled more than once")
        elif value in ANSWERS:
            labels.append({"id": sentence_id, "label": ANSWERS[value]})
        else:
            errors.append(f"line {no}: {sentence_id} -> {value!r} is not b, n or s")
    return labels, errors


def response_error(body: Dict) -> str | None:
    """Why an `/analyze` response cannot be used for this measurement, or None when it can."""
    if body.get("model_version") != MODEL_VERSION:
        return f"model_version is {body.get('model_version')!r}, expected {MODEL_VERSION!r}"
    if not body.get("evaluation"):
        return "evaluation is null: the server is not the onnx backend at threshold 0.5"
    if "sentences" not in body:
        return "no sentences in the response; send return_sentences: true"
    return None


def _score_cells(rows: List[Dict]) -> Tuple[str, str]:
    gold = [row["gold"] for row in rows]
    label, floor = best_constant(gold)
    return f"{macro_f1(gold, [row['pred'] for row in rows]):.4f}", f"{floor:.4f} (all {label})"


def build_report(sample: List[Dict], labels: List[Dict], meta: Dict, reading: List[Dict] | None = None) -> str:
    """The results file: counts and scores only, never a sentence."""
    rows = join_labels(sample, labels)
    skipped = sum(entry["label"] == SKIP for entry in labels)
    lo, hi = bootstrap_ci(rows)
    score, floor = _score_cells(rows)
    lines = [
        "# neutral's DeBERTa on Sentra sentences",
        "",
        f"Model `{meta['model_version']}` served by `neutral`'s `POST /analyze` (onnx, threshold 0.5), "
        f"scored on {meta['generated_at']} over corpus `{meta['corpus_version']}`. "
        "Labels are the repository owner's, given without seeing the model's output. "
        "Built by `scripts/fulltext/measure_deberta.py`.",
        "",
        "| Domain | n | Macro F1 | 95% CI | Best constant |",
        "|---|---|---|---|---|",
    ]
    for domain in meta["evaluation"]["domains"]:
        if domain["macro_f1"] is not None:
            lines.append(f"| {domain['description']} | {domain['n_sentences']} | {domain['macro_f1']:.4f} | "
                         f"not reported here | {domain['constant_baseline_macro_f1']:.4f} |")
    lines += [
        f"| {SENTRA_DOMAIN} | {len(rows)} | {score} | [{lo:.4f}, {hi:.4f}] | {floor} |",
        "",
        f"Sample: {len(sample)} sentences drawn, {len(rows)} scored, {skipped} skipped, "
        f"{len(sample) - len(labels)} not labelled.",
        "",
        "| Outlet | n | Owner: biased | Model: biased | Macro F1 | Best constant |",
        "|---|---|---|---|---|---|",
    ]
    for outlet in sorted({row["outlet"] for row in rows}):
        part = [row for row in rows if row["outlet"] == outlet]
        part_score, part_floor = _score_cells(part)
        lines.append(f"| {outlet} | {len(part)} | {sum(r['gold'] == 'biased' for r in part)} | "
                     f"{sum(r['pred'] == 'biased' for r in part)} | {part_score} | {part_floor} |")
    lines += ["", "| | Model: biased | Model: neutral |", "|---|---|---|"]
    for gold in LABELS:
        cells = [sum(r["gold"] == gold and r["pred"] == pred for r in rows) for pred in LABELS]
        lines.append(f"| Owner: {gold} | {cells[0]} | {cells[1]} |")
    if reading:
        owner = {entry["id"]: entry["label"] for entry in labels}
        both = [(owner[entry["id"]], entry["label"]) for entry in reading if entry["id"] in owner]
        same = sum(a == b for a, b in both)
        lines += ["", f"A second reading by Claude, written after the owner finished, matches the owner on "
                      f"{same} of {len(both)} sentences (biased, neutral or skip). It is not used as gold."]
    lines += [
        "",
        "## Limits",
        "",
        "- One annotator and one pass. No second annotator, so no inter-annotator agreement.",
        f"- {len(rows)} sentences. The interval resamples whole articles (2,000 draws), because "
        "sentences of one article are not independent.",
        "- The sample is stratified: the same number of sentences per outlet, split evenly over each "
        "outlet's events, with any remainder taken from the largest event. It does not follow the "
        "corpus's own proportions.",
        "- Macro F1 counts both classes equally. With few sentences labelled biased, a handful of them "
        "moves the score; read it next to the best constant.",
        "- Sentences are the units `/analyze` produced (pysbd). BASIL's figure uses BASIL's own segmentation.",
        "- The owner labelled from a summary of BABE's annotation manual "
        "(`evaluation/datasets/DEBERTA_LABELING_GUIDE.md`), not as a trained BABE annotator.",
        "- Loaded wording inside quoted or reported speech counts as biased, as in BABE's manual. "
        "A biased label means the sentence prints loaded language, not that an outlet is biased.",
        "",
    ]
    return "\n".join(lines)
