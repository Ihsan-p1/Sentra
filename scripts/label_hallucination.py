"""
Label answer sentences for the real hallucination dataset (eval phase 1, step 2).

For each question it prints the retrieved chunks once, then each checked sentence
of the answer. Judge the sentence against the chunks only, not against what you
know about the news:

    s  supported    the chunks state it
    u  unsupported  the chunks do not state it, or contradict it
    n  not_claim    nothing to verify (a heading, an offer, a sign-off)
    c  show the chunks again     k  skip for now     q  quit

Model predictions are never shown, so they cannot steer the label. Every label is
saved immediately; run the script again to continue.

Usage:
    python scripts/label_hallucination.py
    python scripts/label_hallucination.py --relabel   # 30 blind repeats, about a week later
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluation.hallucination_dataset import ANSWERS_PATH, DATASETS_DIR, SENTENCES_PATH, load_jsonl, save_jsonl
from evaluation.labeling import LABELS, next_unlabeled, progress, query_order, sample_for_relabel

TARGET_CLAIMS = 150
RELABEL_PATH = DATASETS_DIR / "hallucination_relabel.jsonl"


def show_chunks(answer: dict) -> None:
    print("\n" + "=" * 78)
    print(f"PERTANYAAN [{answer['query_id']}]: {answer['question']}")
    print("=" * 78)
    by_media = {}
    for chunk in answer["chunks"]:
        by_media.setdefault(chunk["media"], []).append(chunk)
    n = 0
    for media, chunks in by_media.items():
        print(f"\n--- {media.upper()} ---")
        for chunk in sorted(chunks, key=lambda c: -c["similarity"]):
            n += 1
            print(f"[{n}] {chunk['title']}\n    {chunk['text']}")
    print()


def ask(prompt: str) -> str:
    """One key from the user; 'q' on end of input, so a closed pipe quits cleanly."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def show_sentence(rows: list, i: int) -> None:
    row = rows[i]
    before = [r for r in rows if r["query_id"] == row["query_id"] and r["sentence_idx"] == row["sentence_idx"] - 1]
    if before:
        print(f"  (kalimat sebelumnya: {before[0]['sentence']})")
    print(f"  >> {row['sentence']}")


def label_loop(rows_path: Path, answers_path: Path) -> None:
    rows = load_jsonl(rows_path)
    answers = {a["query_id"]: a for a in load_jsonl(answers_path)}
    order = query_order(list(answers))
    skipped, current_qid = set(), None

    while True:
        pending = [r for r in rows if id(r) not in skipped]
        idx_in_pending = next_unlabeled(pending, order)
        if idx_in_pending is None:
            print("Semua kalimat sudah dilabel." if not skipped else "Tinggal kalimat yang dilewati.")
            break
        row = pending[idx_in_pending]
        i = rows.index(row)
        if row["query_id"] != current_qid:
            current_qid = row["query_id"]
            show_chunks(answers[current_qid])

        p = progress(rows)
        print(f"[{p['claims']}/{TARGET_CLAIMS} klaim, {p['unsupported']} tidak didukung, sisa {p['remaining']}]")
        show_sentence(rows, i)
        key = ask("  label [s=didukung u=tidak didukung n=bukan klaim | c=chunk k=lewati q=keluar]: ")
        if key in LABELS:
            rows[i]["label"] = LABELS[key]
            save_jsonl(rows_path, rows)
        elif key == "c":
            show_chunks(answers[current_qid])
        elif key == "k":
            skipped.add(id(row))
        elif key == "q":
            break
        else:
            print("  tombol tidak dikenal")

    p = progress(rows)
    print(f"Tersimpan. {p['claims']} klaim dilabel ({p['unsupported']} tidak didukung), sisa {p['remaining']}.")


def relabel_loop(rows_path: Path, answers_path: Path, relabel_path: Path, n: int) -> None:
    rows = load_jsonl(rows_path)
    answers = {a["query_id"]: a for a in load_jsonl(answers_path)}
    by_key = {(r["query_id"], r["sentence_idx"]): r for r in rows}
    done = load_jsonl(relabel_path)
    done_keys = {(d["query_id"], d["sentence_idx"]) for d in done}
    rank = {qid: i for i, qid in enumerate(query_order(list(answers)))}
    sample = sorted(sample_for_relabel(rows, n), key=lambda k: (rank.get(k[0], 0), k[1]))
    todo = [k for k in sample if k not in done_keys]
    print(f"Label ulang: {len(done)} selesai, {len(todo)} lagi. Label sebelumnya disembunyikan.")

    current_qid = None
    for key in todo:
        if key[0] != current_qid:
            current_qid = key[0]
            show_chunks(answers[current_qid])
        while True:
            show_sentence(rows, rows.index(by_key[key]))
            choice = ask("  label [s=didukung u=tidak didukung n=bukan klaim | c=chunk q=keluar]: ")
            if choice == "c":
                show_chunks(answers[current_qid])
                continue
            break
        if choice == "q":
            break
        if choice not in LABELS:
            print("  tombol tidak dikenal, kalimat ini ditunda ke sesi berikutnya")
            continue
        done.append({"query_id": key[0], "sentence_idx": key[1], "label": LABELS[choice],
                     "relabelled_at": datetime.now().isoformat(timespec="seconds")})
        save_jsonl(relabel_path, done)
    print(f"Tersimpan. {len(done)} dilabel ulang.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label hallucination sentences.")
    parser.add_argument("--path", type=Path, default=SENTENCES_PATH, help="sentence rows (JSONL)")
    parser.add_argument("--answers", type=Path, default=ANSWERS_PATH, help="answers with chunks (JSONL)")
    parser.add_argument("--relabel", action="store_true", help="blind repeat of a fixed sample")
    parser.add_argument("--relabel-path", type=Path, default=RELABEL_PATH)
    parser.add_argument("-n", type=int, default=30, help="relabel sample size")
    args = parser.parse_args()
    try:
        if args.relabel:
            relabel_loop(args.path, args.answers, args.relabel_path, args.n)
        else:
            label_loop(args.path, args.answers)
    except KeyboardInterrupt:
        print("\nBerhenti. Semua label yang sudah diberikan tersimpan.")
