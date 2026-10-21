"""
Metrics for the real hallucination evaluation.

True means "unsupported" everywhere: that is the class the checker exists to
catch, so recall is the catch rate.
"""
import numpy as np

from evaluation.hallucination_metrics import binary_scores, cluster_bootstrap_ci, label_agreement
from evaluation.hallucination_report import build_report, claim_rows, claude_agreement, predict_all

T, F = True, False


def test_scores_with_unsupported_positive():
    s = binary_scores([T, T, F, F], [T, F, T, F])
    assert (s["n"], s["positives"], s["tp"], s["fp"], s["fn"]) == (4, 2, 1, 1, 1)
    assert (s["recall"], s["precision"], s["f1"]) == (0.5, 0.5, 0.5)


def test_always_supported_has_no_precision():
    s = binary_scores([T, T, F], [F, F, F])
    assert s["precision"] is None
    assert s["recall"] == 0.0
    assert s["f1"] == 0.0


def test_no_positives_gives_none_recall():
    s = binary_scores([F, F, F], [T, F, F])
    assert s["positives"] == 0
    assert s["recall"] is None
    assert s["f1"] is None
    assert s["precision"] == 0.0


def test_bootstrap_is_deterministic_and_brackets_point():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    y_true = [T, F, T, T, F, F, T, F]
    y_pred = [T, F, F, T, F, T, T, F]
    point = binary_scores(y_true, y_pred)["recall"]

    ci = cluster_bootstrap_ci(groups, y_true, y_pred, "recall", n_boot=500)
    assert ci == cluster_bootstrap_ci(groups, y_true, y_pred, "recall", n_boot=500)
    lo, hi = ci
    assert lo <= point <= hi


def test_bootstrap_none_when_metric_undefined():
    assert cluster_bootstrap_ci(["a", "b"], [F, F], [F, F], "recall", n_boot=100) is None


def test_kappa_perfect_and_worse_than_chance():
    assert label_agreement(["s", "u", "n"], ["s", "u", "n"]) == {"n": 3, "agreement": 1.0, "kappa": 1.0}
    assert label_agreement(["s", "u"], ["u", "s"])["kappa"] < 0


def test_kappa_undefined_for_one_class_or_one_item():
    assert label_agreement(["s", "s"], ["s", "s"])["kappa"] is None
    assert label_agreement(["s"], ["u"])["kappa"] is None


# --- predict_all and the report (scripts/evaluate_hallucination.py) ---


class FakeEmbedder:
    def __init__(self):
        self.batches = 0

    def generate(self, texts, show_progress=False):
        self.batches += 1
        return np.ones((len(texts), 3))

    def generate_single(self, text):
        return np.ones(3)


class FakeDetector:
    def predict(self, sentence_emb, source_embs, sentence_text, source_texts):
        return {"is_supported": "invented" not in sentence_text}


class FakeBaseline:
    def check_sentence(self, sentence, source_texts):
        return {"is_supported": True}


def _claims():
    return [
        {"query_id": "a", "sentence_idx": 0, "check_text": "fine claim here", "label": "supported"},
        {"query_id": "a", "sentence_idx": 1, "check_text": "an invented claim", "label": "unsupported"},
        {"query_id": "b", "sentence_idx": 0, "check_text": "another fine claim", "label": "supported"},
    ]


def test_claim_rows_keep_only_checked_supported_or_unsupported():
    rows = _claims() + [
        {"query_id": "b", "sentence_idx": 1, "check_text": "x y z", "label": "not_claim"},
        {"query_id": "b", "sentence_idx": 2, "check_text": None, "label": None},
        {"query_id": "b", "sentence_idx": 3, "check_text": "x y z", "label": None},
    ]
    assert claim_rows(rows) == _claims()


def test_predict_all_shapes():
    answers = {q: {"chunks": [{"text": "source one"}, {"text": "source two"}]} for q in ("a", "b")}
    embedder = FakeEmbedder()
    preds = predict_all(_claims(), answers, FakeDetector(), FakeBaseline(), embedder)

    assert preds == {
        "model_a": [False, True, False],
        "model_b": [False, False, False],
        "always_supported": [False, False, False],
    }
    assert embedder.batches == 2  # chunks embedded once per answer


def test_report_writes_na_for_undefined_metrics():
    rows = [dict(r, label="supported") for r in _claims()]
    preds = {"model_a": [False] * 3, "model_b": [False] * 3, "always_supported": [False] * 3}
    markdown, data = build_report(rows, preds, [], {}, None, {"skipped": 0, "unlabelled": 0})
    assert "n/a" in markdown
    assert data["real"]["model_a"]["recall"] is None


def test_claude_agreement_uses_owner_label_before_correction():
    rows = _claims()  # final labels after any correction
    readings = [{"query_id": r["query_id"], "sentence_idx": r["sentence_idx"], "label": r["label"]} for r in rows]
    first = rows[0]
    flipped = "supported" if first["label"] == "unsupported" else "unsupported"
    corrections = [{"query_id": first["query_id"], "sentence_idx": first["sentence_idx"],
                    "from": flipped, "to": first["label"], "after_claude_reading": True}]
    result = claude_agreement(rows, readings, corrections)
    assert result["n"] == 3
    assert result["agreement"] == 2 / 3  # the owner's first answer disagreed once
    assert result["corrected"] == 1


def test_claude_agreement_is_none_without_readings():
    assert claude_agreement(_claims(), [], []) is None


def test_report_states_claude_readings_when_present():
    rows = _claims()
    preds = {"model_a": [False] * 3, "model_b": [False] * 3, "always_supported": [False] * 3}
    counts = {"skipped": 0, "unlabelled": 0}
    claude = {"n": 3, "agreement": 2 / 3, "kappa": 0.5, "corrected": 1}
    with_claude, data = build_report(rows, preds, [], {}, None, counts, claude)
    without, _ = build_report(rows, preds, [], {}, None, counts)
    assert "Claude's reading" in with_claude and "1 label" in with_claude
    assert "Claude's reading" not in without
    assert data["claude_agreement"] == claude
