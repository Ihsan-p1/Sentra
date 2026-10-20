"""
Pure helpers behind scripts/label_hallucination.py.

The CLI only reads keys and prints; the ordering, resume and sampling rules
live here so they can be tested without a terminal.
"""
from evaluation.labeling import LABELS, next_unlabeled, progress, query_order, sample_for_relabel


def _row(qid, idx, label=None, checked=True):
    return {"query_id": qid, "sentence_idx": idx, "sentence": f"{qid}-{idx}",
            "check_text": f"{qid}-{idx}" if checked else None, "label": label}


def test_labels_cover_the_three_classes():
    assert LABELS == {"s": "supported", "u": "unsupported", "n": "not_claim"}


def test_query_order_is_deterministic_and_complete():
    ids = [f"c{i:02d}" for i in range(1, 26)] + [f"n{i:02d}" for i in range(1, 26)]
    first, second = query_order(ids), query_order(list(reversed(ids)))
    assert first == second
    assert sorted(first) == sorted(ids)
    assert first != sorted(ids)


def test_next_unlabeled_follows_order_and_skips_unchecked():
    rows = [_row("a", 0), _row("b", 1), _row("b", 0, checked=False), _row("a", 1)]
    assert next_unlabeled(rows, ["b", "a"]) == 1
    rows[1]["label"] = "supported"
    assert next_unlabeled(rows, ["b", "a"]) == 0


def test_next_unlabeled_none_when_done():
    rows = [_row("a", 0, label="not_claim"), _row("a", 1, checked=False)]
    assert next_unlabeled(rows, ["a"]) is None


def test_progress_counts_claims():
    rows = [_row("a", 0, "supported"), _row("a", 1, "unsupported"), _row("a", 2, "not_claim"),
            _row("a", 3), _row("a", 4, checked=False)]
    assert progress(rows) == {"labelled": 3, "claims": 2, "unsupported": 1, "remaining": 1}


def test_relabel_sample_only_from_labelled_rows_and_capped():
    rows = [_row("a", i, "supported") for i in range(3)] + [_row("b", 0)]
    sample = sample_for_relabel(rows, n=30)
    assert sorted(sample) == [("a", 0), ("a", 1), ("a", 2)]
    assert sample_for_relabel(rows, n=2) == sample_for_relabel(rows, n=2)
    assert len(sample_for_relabel(rows, n=2)) == 2
