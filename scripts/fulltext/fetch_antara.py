"""
Fetch the Antara pages listed in the full-text corpus manifest.

Saves each page's raw HTML to data/raw/antaranews/<id>.html, skipping files
that already exist, with a pause between requests. Tempo entries are never
requested: the owner saves those pages through a browser (spec section 6).

Usage: python scripts/fulltext/fetch_antara.py [MANIFEST]
"""
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.fulltext.manifest import MANIFEST_PATH, load_manifest

USER_AGENT = "SentraResearch/1.0 (+https://github.com/Ihsan-p1/Sentra)"
TIMEOUT_S = 15
RAW_DIR = Path("data/raw")


def raw_path(url: str, outlet: str, raw_dir: Path = RAW_DIR) -> Path:
    """Local file for a page: <raw_dir>/<outlet>/<first numeric path segment>.html."""
    for segment in urlparse(url).path.split("/"):
        if segment.isdigit():
            return Path(raw_dir) / outlet / f"{segment}.html"
    raise ValueError(f"no numeric id in URL path: {url}")


def fetch_all(entries: list[dict], session, raw_dir: Path, delay_s: float = 2.0) -> dict[str, str]:
    """Fetch every antaranews entry; returns url -> fetched | skipped | failed: <status>."""
    results = {}
    sent = 0
    for entry in entries:
        if entry["outlet"] != "antaranews":
            continue
        url = entry["url"]
        target = raw_path(url, "antaranews", raw_dir)
        if target.exists():
            results[url] = "skipped"
            continue
        if sent and delay_s:
            time.sleep(delay_s)
        sent += 1
        try:
            response = session.get(url, timeout=TIMEOUT_S)
        except requests.RequestException as exc:
            results[url] = f"failed: {type(exc).__name__}"
            continue
        if response.status_code != 200:
            results[url] = f"failed: {response.status_code}"
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        results[url] = "fetched"
    return results


def main(argv) -> int:
    entries = load_manifest(argv[0] if argv else MANIFEST_PATH)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    results = fetch_all(entries, session, RAW_DIR)
    for url, status in results.items():
        print(f"[INFO] {status:<12} {url}")
    failed = [s for s in results.values() if s.startswith("failed")]
    print(f"[INFO] {len(results)} Antara entries, {len(failed)} failed")
    if "failed: 403" in failed:
        print("[ERROR] Antara refused the research User-Agent (403); ask the owner before changing it.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
