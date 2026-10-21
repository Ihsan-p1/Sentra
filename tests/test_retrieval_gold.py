"""Candidate pool and gold sheet for the retrieval evaluation (eval phase 2)."""
from datetime import datetime

from evaluation.retrieval_gold import build_pool, make_sheet, parse_sheet

POOL = {"q1": ["u1", "u2", "u3"], "q2": ["u2"], "q3": ["u1", "u3"]}
QUESTIONS = {"q1": "free meal program", "q2": "court ruling", "q3": "who won"}
ARTICLES = {
    "u1": {"title": "Prabowo wins", "content": "Prabowo won the vote", "media_source": "tempo",
           "published_date": datetime(2024, 3, 20)},
    "u2": {"title": "Court ruling", "content": "The court rejected it", "media_source": "abc_news",
           "published_date": None},
    "u3": {"title": "Free meals", "content": "School lunch program", "media_source": "antaranews",
           "published_date": datetime(2024, 8, 12)},
}


def sheet(answers):
    """A sheet with q1..q3 answered as given; None leaves the line blank."""
    text = make_sheet(POOL, QUESTIONS, ARTICLES)
    for qid in ("q1", "q2", "q3"):
        head = text.index(f"## {qid}:")
        at = text.index("relevan: ", head) + len("relevan: ")
        text = text[:at] + (answers.get(qid) or "") + text[at:]
    return text


def test_build_pool_unions_and_hides_rank():
    assert build_pool(["c", "a", "b"], ["d", "a"], depth=2) == ["a", "c", "d"]


def test_parse_sheet_reads_numbers_dash_and_blank():
    gold, errors = parse_sheet(sheet({"q1": "1 3", "q2": "-"}), POOL)
    assert gold == {"q1": ["u1", "u3"], "q2": []}
    assert errors == []


def test_parse_sheet_reports_bad_entries():
    gold, errors = parse_sheet(sheet({"q1": "1 99 x"}), POOL)
    assert gold == {"q1": ["u1"]}
    assert len(errors) == 2


def test_parse_sheet_rejects_sheet_built_from_another_pool():
    # Numbers point into the pool; if the pool changed since the sheet was made,
    # "1" may name a different article, so the whole question is refused.
    gold, errors = parse_sheet(sheet({"q1": "1"}), {**POOL, "q1": ["u1", "u2"]})
    assert "q1" not in gold
    assert len(errors) == 1


def test_make_sheet_shows_every_candidate_once_in_pool_order():
    text = make_sheet(POOL, QUESTIONS, ARTICLES)
    q1 = text[text.index("## q1:"):text.index("## q2:")]
    assert "(3 kandidat, pool " in q1
    assert q1.index("[1] *Prabowo wins*") < q1.index("[2] *Court ruling*") < q1.index("[3] *Free meals*")
    assert "School lunch program" in q1 and "2024-08-12" in q1


def test_make_sheet_round_trips_with_parse():
    text = make_sheet(POOL, QUESTIONS, ARTICLES).replace("relevan: ", "relevan: 1", 1)
    gold, _ = parse_sheet(text, POOL)
    assert gold == {"q1": ["u1"]}
