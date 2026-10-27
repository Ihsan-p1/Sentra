"""Tests for evaluation/hallucination_retrain.py (eval phase 3). Offline: the embedder is a fake."""
import numpy as np

from evaluation.hallucination_retrain import claim_features, out_of_fold, predict_with_fold
from models.hallucination.detector import HallucinationDetector


def _data():
    # four questions, four rows each; feature 0 equals the label except in q4, where it is flipped
    X, y, g = [], [], []
    for q in ("q1", "q2", "q3", "q4"):
        for label in (True, False, True, False):
            x = float(label) if q != "q4" else float(not label)
            X.append([x, 0.5, 0.5, 0.1, 0.5]); y.append(label); g.append(q)
    return np.array(X), y, g


def test_out_of_fold_returns_one_model_per_question():
    X, y, g = _data()
    _, models = out_of_fold(X, y, g)
    assert set(models) == {"q1", "q2", "q3", "q4"}


def test_out_of_fold_predicts_each_question_from_the_others():
    X, y, g = _data()
    preds, _ = out_of_fold(X, y, g)
    q4 = [i for i, q in enumerate(g) if q == "q4"]
    assert [preds[i] for i in q4] == [not y[i] for i in q4]


def test_predict_with_fold_uses_that_questions_model():
    class Fixed:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.array([self.v] * len(X))
    X = np.zeros((2, 5))
    assert predict_with_fold({"a": Fixed(1), "b": Fixed(0)}, X, ["a", "b"]) == [True, False]


class CountingEmbedder:
    def __init__(self):
        self.batches = 0

    def generate(self, texts, show_progress=False):
        self.batches += 1
        return np.ones((len(texts), 3))

    def generate_single(self, text):
        return np.ones(3)


def test_claim_features_embeds_chunks_once_per_question():
    rows = [
        {"query_id": "a", "check_text": "source one words"},
        {"query_id": "a", "check_text": "nothing shared"},
    ]
    answers = {"a": {"chunks": [{"text": "source one"}, {"text": "source two"}]}}
    embedder = CountingEmbedder()
    X = claim_features(rows, answers, HallucinationDetector(), embedder)
    assert embedder.batches == 1
    assert X.shape == (2, 5)
    assert X[0, 4] == 2 / 3 and X[1, 4] == 0.0
