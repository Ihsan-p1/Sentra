"""
Metrics for the real hallucination evaluation.

True means "unsupported" everywhere: that is the class the checker exists to
catch, so recall is the catch rate.
"""
from evaluation.hallucination_metrics import binary_scores, cluster_bootstrap_ci, label_agreement

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
