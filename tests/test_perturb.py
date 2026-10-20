"""
Synthetic negatives: supported sentences with one fact swapped.

A swap only counts if the new name or number appears nowhere in the chunks,
otherwise the "unsupported" label given by construction could be wrong.
"""
from evaluation.perturb import perturb_sentence, synthetic_rows

CHUNKS = ["Prabowo won the presidential vote in 2024 with 96 million votes."]


def test_name_swap_first():
    assert perturb_sentence("Prabowo won the vote.", CHUNKS) == ("Anies won the vote.", "name")


def test_name_swap_rejected_when_replacement_in_chunks_falls_back_to_year():
    chunks = CHUNKS + ["Anies came second."]
    assert perturb_sentence("Prabowo won in 2024.", chunks) == ("Prabowo won in 2019.", "year")


def test_number_swap():
    assert perturb_sentence("He got 96 million votes.", CHUNKS) == ("He got 193 million votes.", "number")


def test_decimal_changes_its_whole_part():
    assert perturb_sentence("Turnout was 58.59 percent.", CHUNKS) == ("Turnout was 117.59 percent.", "number")


def test_replacement_present_in_chunks_is_rejected():
    chunks = ["The count reached 193 thousand."]
    assert perturb_sentence("He got 96 million votes.", chunks) is None


def test_name_inside_a_longer_word_is_not_swapped():
    assert perturb_sentence("The Jokowian era ended quietly.", CHUNKS) is None


def test_nothing_to_perturb_returns_none():
    assert perturb_sentence("The outlets framed it differently.", CHUNKS) is None


def test_synthetic_rows_only_from_supported():
    rows = [
        {"query_id": "c01", "sentence_idx": 0, "check_text": "Prabowo won the vote.", "label": "supported"},
        {"query_id": "c01", "sentence_idx": 1, "check_text": "Gibran is vice president.", "label": "unsupported"},
        {"query_id": "c01", "sentence_idx": 2, "check_text": None, "label": None},
        {"query_id": "c01", "sentence_idx": 3, "check_text": "Media framed it calmly.", "label": "supported"},
    ]
    answers = {"c01": {"chunks": [{"text": t} for t in CHUNKS]}}
    assert synthetic_rows(rows, answers) == [{
        "query_id": "c01", "sentence_idx": 0, "check_text": "Anies won the vote.",
        "kind": "name", "label": "unsupported", "synthetic": True,
    }]
