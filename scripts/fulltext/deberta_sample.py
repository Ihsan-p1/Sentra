"""
Score every news article of the full-text corpus with neutral's DeBERTa and draw the sample to label.

Sends each article to neutral's POST /analyze on this machine, keeps the
sentences as that endpoint segments them, and writes a stratified sample of
150 to data/raw/deberta_sample.jsonl (text, outlet, event, url, pred, prob)
plus data/raw/deberta_sample_meta.json. Both hold outlet text or describe it
and stay out of git. Start neutral first, from its repo:

    NEUTRAL_MODEL_BACKEND=onnx .venv/Scripts/python.exe -m uvicorn neutral.app:app --port 8001

Usage: python scripts/fulltext/deberta_sample.py [--url http://127.0.0.1:8001]
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.corpus import corpus_version, load_articles
from evaluation.deberta_domain import response_error, stratified_sample
from pipeline.fulltext.manifest import load_manifest

SAMPLE_OUT = Path("data/raw/deberta_sample.jsonl")
META_OUT = Path("data/raw/deberta_sample_meta.json")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
N, SEED = 150, 7


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="base URL of neutral's API")
    args = parser.parse_args(argv)

    if urlparse(args.url).hostname not in LOCAL_HOSTS:
        print(f"[ERROR] {args.url} is not this machine; outlet text stays on this laptop")
        return 1
    if SAMPLE_OUT.exists():
        print(f"[ERROR] {SAMPLE_OUT} exists; a new sample would orphan labels made from it. Delete it to resample")
        return 1

    events = {entry["url"]: entry["event"] for entry in load_manifest() if entry["genre"] == "news"}
    articles = load_articles()
    unknown = [a["url"] for a in articles if a["url"] not in events]
    if unknown:
        print(f"[ERROR] {len(unknown)} of {len(articles)} articles are not news in the manifest "
              f"(first: {unknown[0]}); set CORPUS_PATH to the full-text corpus")
        return 1

    sentences, evaluation, model_version = [], None, None
    for article in articles:
        response = requests.post(f"{args.url}/analyze", timeout=300,
                                 json={"text": article["content"], "return_sentences": True})
        response.raise_for_status()
        body = response.json()
        error = response_error(body)
        if error:
            print(f"[ERROR] {article['url']}: {error}")
            return 1
        evaluation, model_version = body["evaluation"], body["model_version"]
        sentences += [
            {"text": s["text"], "outlet": article["media_source"], "event": events[article["url"]],
             "url": article["url"], "pred": s["label"], "prob": s["prob"]}
            for s in body["sentences"]
        ]

    sample = stratified_sample(sentences, n=N, seed=SEED)
    write_jsonl(sample, SAMPLE_OUT)
    with open(META_OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"model_version": model_version, "evaluation": evaluation,
                   "generated_at": date.today().isoformat(), "corpus_version": corpus_version(),
                   "n_articles": len(articles), "n_sentences": len(sentences), "seed": SEED}, f, indent=1)
        f.write("\n")
    print(f"[INFO] {len(articles)} articles, {len(sentences)} sentences scored by {model_version}")
    print(f"[INFO] {len(sample)} sampled -> {SAMPLE_OUT}; next: python scripts/fulltext/deberta_sheet.py make")
    return 0


if __name__ == "__main__":
    sys.exit(main())
