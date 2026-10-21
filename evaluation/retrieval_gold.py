"""
Candidate pool and gold sheet for the retrieval evaluation (eval phase 2).

The owner judges which articles answer each question. Judging all 154 articles
for 50 questions is not feasible, so each question gets a pool: the union of
the top articles from the dense retriever and the TF-IDF baseline. The pool is
sorted by URL, not by score, so the sheet does not show which system ranked
what. Relevant articles outside the pool are never judged; the report says so.

A sheet's candidate numbers point into the pool. Each question header carries a
fingerprint of its pool, and a sheet whose fingerprint no longer matches is
refused for that question rather than read against the wrong articles.
"""
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
POOL_PATH = DATASETS_DIR / "retrieval_pool.json"
GOLD_PATH = DATASETS_DIR / "retrieval_gold.json"

HEAD = re.compile(r"^## (\w+): .*\((\d+) kandidat, pool (\w+)\)\s*$")
ANSWER = re.compile(r"^relevan:(.*)$")


def build_pool(dense: List[str], lexical: List[str], depth: int = 10) -> List[str]:
    return sorted(set(dense[:depth]) | set(lexical[:depth]))


def pool_fingerprint(urls: List[str]) -> str:
    return hashlib.sha1("\n".join(urls).encode("utf-8")).hexdigest()[:8]


def make_sheet(pool: Dict[str, List[str]], questions: Dict[str, str], articles: Dict[str, Dict]) -> str:
    lines = [
        "# Lembar gold retrieval",
        "",
        "Untuk setiap pertanyaan, tulis nomor artikel yang relevan di baris relevan, dipisah spasi",
        "(contoh `2 5 11`). Relevan berarti artikel memuat informasi yang menjawab sebagian atau",
        "seluruh pertanyaan. Tulis `-` kalau tidak ada yang relevan; biarkan kosong kalau belum",
        "dinilai. Urutan kandidat tidak mengikuti peringkat sistem mana pun.",
        "Aturan lengkap: `evaluation/datasets/RETRIEVAL_GUIDE.md`.",
        "",
    ]
    for qid, urls in pool.items():
        lines += [f"## {qid}: {questions[qid]} ({len(urls)} kandidat, pool {pool_fingerprint(urls)})", ""]
        for n, url in enumerate(urls, 1):
            a = articles[url]
            date = a["published_date"].date().isoformat() if a.get("published_date") else "tanpa tanggal"
            lines += [f"[{n}] *{a['title']}* ({a['media_source']}, {date})", a["content"], ""]
        lines += ["relevan: ", ""]
    return "\n".join(lines)


def parse_sheet(text: str, pool: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Gold URLs per answered question, plus one message per entry that could not be read."""
    gold: Dict[str, List[str]] = {}
    errors: List[str] = []
    qid = None
    for no, line in enumerate(text.splitlines(), 1):
        m = HEAD.match(line)
        if m:
            qid = m.group(1)
            if qid not in pool:
                errors.append(f"line {no}: {qid} is not in the pool")
                qid = None
            elif m.group(3) != pool_fingerprint(pool[qid]):
                errors.append(f"line {no}: {qid} was made from a different pool; make a new sheet")
                qid = None
            continue
        m = ANSWER.match(line)
        if not (m and qid):
            continue
        urls = pool[qid]
        value, picked = m.group(1).strip(), []
        current, qid = qid, None
        if not value:
            continue
        if value == "-":
            gold[current] = []
            continue
        for token in value.split():
            if token.isdigit() and 1 <= int(token) <= len(urls):
                if urls[int(token) - 1] not in picked:
                    picked.append(urls[int(token) - 1])
            else:
                errors.append(f"line {no}: {current} -> {token!r} is not a candidate number")
        if picked:
            gold[current] = picked
    return gold, errors
