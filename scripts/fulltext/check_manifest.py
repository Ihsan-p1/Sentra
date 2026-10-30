"""
Check the full-text corpus manifest against its selection rules.

Prints every violation from validate_manifest(), then the article count per
event x outlet x genre. Exits 1 when there is any violation, so a hand-edited
manifest that drifts from the rules fails loudly.

Usage: python scripts/fulltext/check_manifest.py [MANIFEST] [EVENTS]
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.fulltext.manifest import (EVENTS_PATH, GENRES, MANIFEST_PATH, OUTLETS, load_events,
                                        load_manifest, validate_manifest)


def main(argv) -> int:
    manifest_path = argv[0] if len(argv) > 0 else MANIFEST_PATH
    events_path = argv[1] if len(argv) > 1 else EVENTS_PATH
    events = load_events(events_path)
    entries = load_manifest(manifest_path)

    errors = validate_manifest(entries, events)
    for error in errors:
        print(f"[ERROR] {error}")

    counts = Counter((e["event"], e["outlet"], e["genre"]) for e in entries)
    columns = [(o, g) for o in OUTLETS for g in GENRES]
    print("event  " + "  ".join(f"{o}/{g}" for o, g in columns))
    for event in events:
        row = [counts[(event["id"], o, g)] for o, g in columns]
        if not any(row):
            continue
        cells = "  ".join(f"{n:>{len(o) + len(g) + 1}}" for n, (o, g) in zip(row, columns))
        print(f"{event['id']:<6} {cells}")
    print(f"[INFO] {len(entries)} articles, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
