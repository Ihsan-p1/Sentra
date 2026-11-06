"""
Event list and URL manifest for the full-text corpus.

The manifest holds URLs and metadata only, never article text: the text stays on
the owner's laptop under data/raw/ (see the full-text corpus spec, section 4).
validate_manifest() enforces the selection rules so a hand-edited manifest cannot
quietly drift from them.
"""
import json
from collections import Counter
from typing import Dict, List

EVENTS_PATH = "data/fulltext/events.json"
MANIFEST_PATH = "data/fulltext/manifest.json"

OUTLETS = ("antaranews", "tempo")
GENRES = ("news", "opinion")

# Per-event news quota per outlet, inclusive. Tempo is saved by hand, so it is fixed.
QUOTA = {"tempo": (4, 4), "antaranews": (4, 8)}


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_events(path: str = EVENTS_PATH) -> List[Dict]:
    return _read_json(path)["events"]


def load_manifest(path: str = MANIFEST_PATH) -> List[Dict]:
    return _read_json(path)["articles"]


def validate_manifest(entries: List[Dict], events: List[Dict]) -> List[str]:
    """Return every rule violation as a message; an empty list means the manifest is valid."""
    errors: List[str] = []
    by_id = {e["id"]: e for e in events}

    url_counts = Counter(e["url"] for e in entries)
    for url, count in sorted(url_counts.items()):
        if count > 1:
            errors.append(f"duplicate url ({count}x): {url}")

    news = Counter()
    for e in entries:
        url = e["url"]
        event = by_id.get(e["event"])
        if event is None:
            errors.append(f"unknown event {e['event']!r}: {url}")
        if e["outlet"] not in OUTLETS:
            errors.append(f"unknown outlet {e['outlet']!r}: {url}")
        if e["genre"] not in GENRES:
            errors.append(f"unknown genre {e['genre']!r}: {url}")
        # ISO dates compare correctly as strings.
        if event is not None and not event["start"] <= e["published_date"] <= event["end"]:
            errors.append(
                f"{e['event']}: date {e['published_date']} outside {event['start']}..{event['end']}: {url}"
            )
        if e["genre"] == "news":
            news[(e["event"], e["outlet"])] += 1

    for event in events:
        if not event["required"] and not any(e["event"] == event["id"] for e in entries):
            continue
        for outlet, (low, high) in QUOTA.items():
            count = news[(event["id"], outlet)]
            if count < low:
                errors.append(f"{event['id']}: {outlet} has {count} news, needs >= {low}")
            elif count > high:
                errors.append(f"{event['id']}: {outlet} has {count} news, allows <= {high}")

    return errors
