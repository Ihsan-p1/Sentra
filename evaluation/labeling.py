"""
Ordering, resume and sampling rules for the hallucination labelling tool.

Questions are labelled in a fixed random order, so stopping early still leaves
a random sample of questions rather than the first few in the file.
"""
import random
from typing import Dict, List, Optional, Tuple

LABELS = {"s": "supported", "u": "unsupported", "n": "not_claim"}
CLAIM_LABELS = ("supported", "unsupported")


def query_order(query_ids: List[str], seed: int = 20261020) -> List[str]:
    order = sorted(set(query_ids))
    random.Random(seed).shuffle(order)
    return order


def next_unlabeled(rows: List[Dict], order: List[str]) -> Optional[int]:
    """Index of the first checked, unlabelled row in question order, or None."""
    rank = {qid: i for i, qid in enumerate(order)}
    pending = [
        (rank.get(r["query_id"], len(rank)), r["sentence_idx"], i)
        for i, r in enumerate(rows)
        if r["check_text"] is not None and r["label"] is None
    ]
    return min(pending)[2] if pending else None


def progress(rows: List[Dict]) -> Dict[str, int]:
    checked = [r for r in rows if r["check_text"] is not None]
    labels = [r["label"] for r in checked]
    return {
        "labelled": sum(label is not None for label in labels),
        "claims": sum(label in CLAIM_LABELS for label in labels),
        "unsupported": labels.count("unsupported"),
        "remaining": labels.count(None),
    }


def sample_for_relabel(rows: List[Dict], n: int = 30, seed: int = 7) -> List[Tuple[str, int]]:
    labelled = sorted((r["query_id"], r["sentence_idx"]) for r in rows if r["label"] is not None)
    return random.Random(seed).sample(labelled, min(n, len(labelled)))
