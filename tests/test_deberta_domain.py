"""Tests for measuring neutral's DeBERTa on owner-labelled Sentra sentences (invented text only)."""
import pytest

from evaluation.deberta_domain import (
    best_constant,
    build_report,
    bootstrap_ci,
    join_labels,
    macro_f1,
    make_sheet,
    parse_sheet,
    response_error,
    stratified_sample,
)

B, N = "biased", "neutral"


def sentences(sizes):
    """One row per sentence for {(outlet, event): count}, with a unique invented text each."""
    return [
        {"text": f"{outlet} {event} sentence {i}", "outlet": outlet, "event": event,
         "url": f"https://example.test/{outlet}/{event}/{i % 4}", "pred": N, "prob": 0.1}
        for (outlet, event), count in sizes.items()
        for i in range(count)
    ]


def counts(sample):
    out = {}
    for row in sample:
        out[(row["outlet"], row["event"])] = out.get((row["outlet"], row["event"]), 0) + 1
    return out


def test_macro_f1_known_values():
    # biased: tp 1, fn 1 -> 2/3; neutral: tp 2, fp 1 -> 4/5
    assert macro_f1([B, B, N, N], [B, N, N, N]) == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert macro_f1([B, N], [B, N]) == 1.0
    # Both classes always count, so a perfect score on one-class gold is still 0.5.
    assert macro_f1([N, N], [N, N]) == 0.5


def test_best_constant():
    label, score = best_constant([N, N, N, B])
    assert label == N
    assert score == pytest.approx((6 / 7) / 2)
    assert best_constant([B, B, N])[0] == B


def test_bootstrap_groups_by_url():
    right = [{"url": "a", "gold": B, "pred": B}, {"url": "a", "gold": N, "pred": N}]
    wrong = [{"url": "b", "gold": B, "pred": N}, {"url": "b", "gold": N, "pred": B}]
    lo, hi = bootstrap_ci(right + wrong)
    # Whole articles are drawn: a draw is all right, all wrong, or the two together.
    # Drawing sentences one by one would give an interval strictly inside (0, 1).
    assert (lo, hi) == (0.0, 1.0)
    assert bootstrap_ci(right + wrong) == (lo, hi)


def test_stratified_sample_balance():
    pool = sentences({
        ("antaranews", "e1"): 100, ("antaranews", "e2"): 60,
        ("tempo", "e1"): 90, ("tempo", "e2"): 50,
    })
    sample = stratified_sample(pool, n=150, seed=7)
    assert len(sample) == 150
    assert len({row["text"] for row in sample}) == 150
    # 75 per outlet, split evenly over events, the odd one from the largest event.
    assert counts(sample) == {
        ("antaranews", "e1"): 38, ("antaranews", "e2"): 37,
        ("tempo", "e1"): 38, ("tempo", "e2"): 37,
    }
    assert [row["id"] for row in sample] == list(range(1, 151))
    assert stratified_sample(pool, n=150, seed=7) == sample
    assert stratified_sample(pool, n=150, seed=8) != sample


def test_stratified_sample_fills_a_short_event_from_the_largest():
    pool = sentences({
        ("antaranews", "e1"): 100, ("antaranews", "e2"): 5,
        ("tempo", "e1"): 40, ("tempo", "e2"): 40,
    })
    got = counts(stratified_sample(pool, n=40, seed=7))
    assert got == {
        ("antaranews", "e1"): 15, ("antaranews", "e2"): 5,
        ("tempo", "e1"): 10, ("tempo", "e2"): 10,
    }


def test_stratified_sample_refuses_an_outlet_that_is_too_small():
    pool = sentences({("antaranews", "e1"): 100, ("tempo", "e1"): 3})
    with pytest.raises(ValueError, match="tempo"):
        stratified_sample(pool, n=20, seed=7)


def test_skip_labels_excluded():
    sample = stratified_sample(sentences({("antaranews", "e1"): 2, ("tempo", "e1"): 2}), n=4)
    labels = [{"id": 1, "label": B}, {"id": 2, "label": "skip"}, {"id": 3, "label": N}]
    rows = join_labels(sample, labels)
    # id 2 is skipped and id 4 was never labelled: neither is scored.
    assert [(row["id"], row["gold"]) for row in rows] == [(1, B), (3, N)]
    assert rows[0]["text"] == sample[0]["text"]


def test_join_labels_rejects_an_unknown_id():
    sample = stratified_sample(sentences({("antaranews", "e1"): 1, ("tempo", "e1"): 1}), n=2)
    with pytest.raises(ValueError, match="9"):
        join_labels(sample, [{"id": 9, "label": B}])


def test_sheet_round_trip_hides_the_model():
    sample = stratified_sample(sentences({("antaranews", "e1"): 3, ("tempo", "e1"): 3}), n=4)
    for row in sample:
        row["pred"], row["prob"] = B, 0.987654
    sheet = make_sheet(sample)
    assert "0.987654" not in sheet
    assert sheet.count("label: ") == 4
    assert all(row["text"] in sheet for row in sample)

    filled = sheet
    for value in ("b", "neutral", "skip"):
        filled = filled.replace("label: \n", f"label: {value}\n", 1)
    labels, errors = parse_sheet(filled, sample)
    assert errors == []
    # The fourth sentence is still empty, so it is simply not in the labels yet.
    assert labels == [{"id": 1, "label": B}, {"id": 2, "label": N}, {"id": 3, "label": "skip"}]


def test_parse_sheet_reports_what_it_cannot_read():
    sample = stratified_sample(sentences({("antaranews", "e1"): 2, ("tempo", "e1"): 2}), n=2)
    sheet = make_sheet(sample).replace("label: \n", "label: maybe\n", 1)
    labels, errors = parse_sheet(sheet, sample)
    assert labels == []
    assert len(errors) == 1 and "maybe" in errors[0]

    other = stratified_sample(sentences({("antaranews", "e9"): 2, ("tempo", "e9"): 2}), n=2)
    labels, errors = parse_sheet(make_sheet(other), sample)
    assert labels == []
    assert len(errors) == 1 and "different sample" in errors[0]


EVALUATION = {
    "model_version": "deberta-babe-f0s3", "metric": "macro_f1", "threshold": 0.5,
    "domains": [
        {"name": "babe", "description": "BABE fold 0 test", "macro_f1": 0.8765,
         "n_sentences": 751, "constant_baseline_macro_f1": 0.3598},
        {"name": "basil", "description": "BASIL test", "macro_f1": 0.5656,
         "n_sentences": 5780, "constant_baseline_macro_f1": 0.4868},
        {"name": "sentra", "description": "Sentra full-text corpus", "macro_f1": None,
         "n_sentences": None, "constant_baseline_macro_f1": None},
    ],
}


def test_response_error_rejects_unmeasured_or_other_models():
    good = {"model_version": "deberta-babe-f0s3", "evaluation": EVALUATION, "sentences": []}
    assert response_error(good) is None
    assert "evaluation" in response_error({**good, "evaluation": None})
    assert "stub-v0" in response_error({**good, "model_version": "stub-v0"})
    assert "sentences" in response_error({k: v for k, v in good.items() if k != "sentences"})


def test_report_carries_other_domains_and_no_sentence_text():
    sample = stratified_sample(sentences({("antaranews", "e1"): 6, ("tempo", "e1"): 6}), n=8)
    labels = [{"id": i, "label": B if i % 3 == 0 else N} for i in range(1, 7)] + [{"id": 7, "label": "skip"}]
    report = build_report(sample, labels, {"model_version": "deberta-babe-f0s3", "evaluation": EVALUATION,
                                           "generated_at": "2026-11-13", "corpus_version": "c@1"})
    assert "0.8765" in report and "0.5656" in report and "0.3598" in report and "0.4868" in report
    assert "| Sentra full-text corpus | 6 |" in report
    assert "1 skipped" in report and "1 not labelled" in report
    assert "antaranews" in report and "tempo" in report
    assert "quoted or reported speech" in report and "not that an outlet is biased" in report
    assert not any(row["text"] in report for row in sample)
