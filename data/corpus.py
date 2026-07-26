"""
Corpus loader.

The article set used to live in `data/election_articles.py` as a Python literal.
Storing data as code meant it could only be read by importing it, it could not be
diffed or validated without executing it, and nothing tied a populated database
back to the corpus it came from.

It is now `data/election_articles.json`, and this module is the single way to read
it. `corpus_version()` produces a stable fingerprint that is stored alongside every
ingested article, so a database can always be traced to the exact corpus that
filled it.
"""
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

CORPUS_PATH = os.path.join("data", "election_articles.json")

# Required on every article; ingestion rejects anything missing these.
REQUIRED_FIELDS = ("title", "content", "media_source")

_cache: Dict[str, Any] = {}


def _read(path: str) -> Dict[str, Any]:
    """Read and cache the raw JSON payload."""
    resolved = os.path.abspath(path)
    if resolved not in _cache:
        with open(resolved, "r", encoding="utf-8") as f:
            _cache[resolved] = json.load(f)
    return _cache[resolved]


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    """Accept a bare date or a full ISO timestamp; None stays None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_articles(path: str = CORPUS_PATH) -> List[Dict[str, Any]]:
    """
    Return the corpus as a list of article dicts, dates parsed to datetime.

    The shape matches what ArticleIngestor.ingest_article expects, so callers do
    not need to know that the corpus is stored as JSON.
    """
    payload = _read(path)
    articles = []

    for index, raw in enumerate(payload.get("articles", [])):
        missing = [f for f in REQUIRED_FIELDS if not raw.get(f)]
        if missing:
            print(f"[WARN] Skipping corpus entry {index}: missing {', '.join(missing)}")
            continue

        articles.append({
            "title": raw["title"],
            "content": raw["content"],
            "media_source": raw["media_source"],
            "url": raw.get("url", ""),
            "author": raw.get("author", ""),
            "published_date": _parse_date(raw.get("published_date")),
            "category": raw.get("category", ""),
        })

    return articles


def corpus_version(path: str = CORPUS_PATH) -> str:
    """
    Short content fingerprint of the corpus, e.g. "indonesia-election-2024@a1b2c3d4e5f6".

    Hashed from the article list only, so reformatting the file or editing its
    description does not invalidate an otherwise-identical corpus.
    """
    payload = _read(path)
    canonical = json.dumps(payload.get("articles", []), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{payload.get('name', 'corpus')}@{digest}"


def corpus_metadata(path: str = CORPUS_PATH) -> Dict[str, Any]:
    """Name, description, version and per-source counts. Used by scripts and docs."""
    payload = _read(path)
    articles = load_articles(path)

    counts: Dict[str, int] = {}
    for article in articles:
        counts[article["media_source"]] = counts.get(article["media_source"], 0) + 1

    return {
        "name": payload.get("name", "corpus"),
        "description": payload.get("description", ""),
        "version": corpus_version(path),
        "article_count": len(articles),
        "counts_by_source": counts,
    }
