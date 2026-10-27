"""TF-IDF retrieval baseline for the retrieval evaluation (eval phase 2)."""
from evaluation.lexical_retriever import TfidfRetriever

ARTS = [
    {"url": "u1", "title": "Prabowo wins", "content": "Prabowo won the vote"},
    {"url": "u2", "title": "Free meals", "content": "school lunch program"},
    {"url": "u3", "title": "Court ruling", "content": "constitutional court rejects"},
]


def test_rank_puts_matching_article_first():
    assert TfidfRetriever(ARTS).rank("free meal program")[0][0] == "u2"


def test_rank_drops_zero_scores_and_respects_k():
    assert TfidfRetriever(ARTS).rank("bitcoin price") == []
    assert len(TfidfRetriever(ARTS).rank("prabowo court program", k=2)) == 2


def test_function_words_alone_do_not_match():
    # Without stop words, "what is the price of bitcoin" scored 0.18 against an
    # unrelated corpus article through "the" and "of" alone.
    assert TfidfRetriever(ARTS).rank("what is the price of bitcoin") == []
