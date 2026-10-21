"""
Model A retrained on the owner's real labels, scored out of fold (eval phase 3).

The deployed detector was trained on synthetic data. Here the same five
features feed a fresh logistic regression, trained with one question held out
at a time (GroupKFold by query_id), so no sentence is scored by a model that saw
its own question. The detector's heuristic overrides and hard gate are left out
on purpose: they exist because the synthetic weights are not trusted, and the
point here is to measure weights learned from real labels. The deployed .pkl is
not touched.
"""
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def claim_features(rows: List[Dict], answers_by_qid: Dict[str, Dict], detector, embedder) -> np.ndarray:
    """The detector's five features per row; each question's chunks are embedded once."""
    source_cache = {}
    features = []
    for row in rows:
        qid = row["query_id"]
        if qid not in source_cache:
            texts = [c["text"] for c in answers_by_qid[qid]["chunks"]]
            source_cache[qid] = (texts, embedder.generate(texts, show_progress=False))
        texts, embs = source_cache[qid]
        text = row["check_text"]
        features.append(detector.extract_features(embedder.generate_single(text), embs, text, texts))
    return np.array(features)


def out_of_fold(X: np.ndarray, unsupported: List[bool], groups: List[str]) -> Tuple[List[bool], Dict[str, Pipeline]]:
    """Predictions (True = unsupported) from models that never saw the row's question, plus each model."""
    y = np.array(unsupported, dtype=int)
    preds = np.zeros(len(y), dtype=bool)
    models = {}
    for train, test in GroupKFold(n_splits=len(set(groups))).split(X, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
        model.fit(X[train], y[train])
        preds[test] = model.predict(X[test]) == 1
        models[groups[test[0]]] = model
    return preds.tolist(), models


def predict_with_fold(models: Dict[str, Pipeline], X: np.ndarray, groups: List[str]) -> List[bool]:
    """Each row scored by the model that held its question out."""
    return [bool(models[g].predict(X[i:i + 1])[0] == 1) for i, g in enumerate(groups)]
