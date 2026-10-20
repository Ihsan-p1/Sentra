"""
Scores for the hallucination checkers, with "unsupported" as the positive class.

Recall is the catch rate: the share of unsupported sentences the checker flags.
A metric that cannot be computed is None, never 0, so a report can say "n/a"
instead of a number that looks like a measurement.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score


def binary_scores(y_true: List[bool], y_pred: List[bool]) -> Dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum(p and not t for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    positives = tp + fn
    return {
        "n": len(y_true),
        "positives": positives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / positives if positives else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if positives else None,
    }


def cluster_bootstrap_ci(
    groups: List[str],
    y_true: List[bool],
    y_pred: List[bool],
    metric: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> Optional[Tuple[float, float]]:
    """95% interval from resampling whole questions, since sentences of one answer are not independent."""
    by_group: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    members = list(by_group.values())
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), size=len(members))
        idx = [i for k in picked for i in members[k]]
        value = binary_scores([y_true[i] for i in idx], [y_pred[i] for i in idx])[metric]
        if value is not None:
            values.append(value)
    if len(values) < n_boot / 2:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def label_agreement(a: List[str], b: List[str]) -> Dict:
    """Raw agreement and Cohen's kappa between two labelling passes over the same sentences."""
    n = len(a)
    agreement = sum(x == y for x, y in zip(a, b)) / n if n else None
    kappa = None
    if n >= 2 and len(set(a) | set(b)) >= 2:
        value = float(cohen_kappa_score(a, b))
        kappa = None if math.isnan(value) else value
    return {"n": n, "agreement": agreement, "kappa": kappa}
