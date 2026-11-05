"""
Build the full-text corpus from the pages saved under data/raw/.

Reads every manifest entry's saved page, parses and cleans it, and writes two
corpus files in the old corpus format plus corpus-level keys: news articles to
data/raw/fulltext_corpus.json (ingested) and opinion pieces to
data/raw/fulltext_opinion.json (kept, not ingested). Both stay out of git.
Exits 1 when any page is missing or has no article body, naming the file.

Usage: python scripts/fulltext/build_corpus.py [MANIFEST]
"""
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.fulltext.manifest import MANIFEST_PATH, load_manifest
from pipeline.fulltext.parse import clean_paragraphs, parse_article, to_ascii
from scripts.fulltext.fetch_antara import RAW_DIR, raw_path

NAME = "indonesia-election-2024-fulltext"
DESCRIPTION = ("Full text of English-language coverage of the 2024 Indonesian election and its "
               "aftermath from ANTARA News and Tempo English, picked per event (stage 1). Local only.")
CORPUS_KEYS = {
    "sources": ["antaranews", "tempo"],
    # Tempo text may feed indexes and calibration, never learned weights (owner, 2026-10-29).
    "no_train_sources": ["tempo"],
    "unevaluated_models": ["framing_bert"],
}
NEWS_OUT = RAW_DIR / "fulltext_corpus.json"
OPINION_OUT = RAW_DIR / "fulltext_opinion.json"


def _payload(name: str, description: str) -> dict:
    return {"name": name, "description": description, **CORPUS_KEYS, "articles": []}


def _article(entry: dict, page: dict, paragraphs: list[str]) -> dict:
    return {
        "title": to_ascii(entry["title"]),
        "content": to_ascii("\n\n".join(paragraphs)),
        "media_source": entry["outlet"],
        "url": entry["url"],
        "author": to_ascii(page["author"]),
        # The page date is the real one; Tempo manifest dates are estimates from the article id.
        "published_date": page["published_date"] or entry["published_date"],
        "category": "election",
        "event": entry["event"],
        "genre": entry["genre"],
    }


def build_corpus(entries: list[dict], raw_dir: Path) -> tuple[dict, dict, list[str]]:
    """Return (news payload, opinion payload, errors); an entry with an error is left out."""
    news = _payload(NAME, DESCRIPTION)
    opinion = _payload(f"{NAME}-opinion", "Opinion pieces found with the news corpus; not ingested.")
    errors = []
    for entry in entries:
        path = raw_path(entry["url"], entry["outlet"], raw_dir)
        if not path.is_file():
            errors.append(f"{entry['url']}: missing raw file {path}")
            continue
        page = parse_article(path.read_text(encoding="utf-8"), entry["outlet"])
        paragraphs = clean_paragraphs(page["paragraphs"], entry["outlet"])
        if not paragraphs:
            errors.append(f"{entry['url']}: empty body in {path} (saved in another format?)")
            continue
        target = opinion if entry["genre"] == "opinion" else news
        target["articles"].append(_article(entry, page, paragraphs))
    return news, opinion, errors


def _write(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")


def main(argv) -> int:
    entries = load_manifest(argv[0] if argv else MANIFEST_PATH)
    news, opinion, errors = build_corpus(entries, RAW_DIR)
    for error in errors:
        print(f"[ERROR] {error}")
    if errors:
        print(f"[ERROR] {len(errors)} of {len(entries)} pages failed; nothing written")
        return 1
    _write(news, NEWS_OUT)
    _write(opinion, OPINION_OUT)
    words = [len(a["content"].split()) for a in news["articles"]]
    print(f"[INFO] {len(news['articles'])} news -> {NEWS_OUT}, "
          f"{len(opinion['articles'])} opinion -> {OPINION_OUT}")
    print(f"[INFO] words per news article: median {statistics.median(words)}, "
          f"min {min(words)}, max {max(words)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
