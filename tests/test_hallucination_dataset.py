"""
Dataset schema for the real hallucination evaluation (eval phase 1).

Offline: no database, no LLM. Pins the question set, the per-sentence rows
and the JSONL round trip the labelling tool depends on.
"""
import ast
import json
from pathlib import Path

import pytest

from evaluation.hallucination_dataset import (
    answer_record,
    answer_to_rows,
    classify_result,
    load_jsonl,
    load_questions,
    pending_questions,
    save_jsonl,
)


def _calibration_in_corpus():
    # Read the list with ast: importing the script pulls in torch through
    # rag.retrieval and costs ~20 s per test run.
    source = (Path(__file__).resolve().parents[1] / "scripts" / "calibrate_threshold.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "IN_CORPUS":
            return ast.literal_eval(node.value)
    raise AssertionError("IN_CORPUS not found")


def test_questions_start_with_calibration_set():
    questions = load_questions()
    assert [q["text"] for q in questions[:25]] == _calibration_in_corpus()
    assert {q["origin"] for q in questions[:25]} == {"calibration"}


def test_questions_have_unique_ids_and_fifty_items():
    questions = load_questions()
    assert len(questions) == 50
    assert len({q["id"] for q in questions}) == 50
    assert {q["origin"] for q in questions[25:]} == {"new"}


def test_duplicate_ids_raise(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"id": "x1", "text": "a", "origin": "new"},
        {"id": "x1", "text": "b", "origin": "new"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="x1"):
        load_questions(path)


def test_answer_to_rows_marks_skipped_sentences():
    answer = "## Direct Answer\nPrabowo won the 2024 presidential election. Yes.\n- Gibran became vice president in October."
    rows = answer_to_rows("c01", answer)

    assert [r["sentence_idx"] for r in rows] == [0, 1, 2, 3]
    assert [r["check_text"] for r in rows] == [
        None,
        "Prabowo won the 2024 presidential election.",
        None,
        "Gibran became vice president in October.",
    ]
    assert rows[3]["sentence"] == "- Gibran became vice president in October."
    assert all(r["query_id"] == "c01" and r["label"] is None for r in rows)


@pytest.mark.parametrize("answer", ["", None])
def test_empty_answer_gives_no_rows(answer):
    assert answer_to_rows("c01", answer) == []


def test_answer_record_keeps_only_chunk_fields_needed():
    chunk = {"media": "tempo", "title": "T", "url": "u", "text": "body",
             "similarity": 0.51, "date": "2024-03-20", "metadata": {"i": 1}}
    record = answer_record("c01", "q?", "answer", [chunk], "en")
    assert record == {
        "query_id": "c01", "question": "q?", "answer": "answer", "language": "en",
        "chunks": [{"media": "tempo", "title": "T", "url": "u", "text": "body", "similarity": 0.51}],
    }


def test_jsonl_round_trip_keeps_unicode(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"sentence": "Pemilu — é", "label": None}, {"sentence": "two", "label": "supported"}]
    save_jsonl(path, rows)

    assert load_jsonl(path) == rows
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert "—".encode("utf-8") in raw


def test_missing_jsonl_reads_as_empty(tmp_path):
    assert load_jsonl(tmp_path / "absent.jsonl") == []


def test_save_is_atomic(tmp_path):
    path = tmp_path / "rows.jsonl"
    save_jsonl(path, [{"a": 1}])
    save_jsonl(path, [{"a": 2}])
    assert load_jsonl(path) == [{"a": 2}]
    assert [p.name for p in tmp_path.iterdir()] == ["rows.jsonl"]


def test_pending_questions_skips_done():
    questions = [{"id": "c01"}, {"id": "c02"}, {"id": "n01"}]
    assert pending_questions(questions, {"c02"}) == [{"id": "c01"}, {"id": "n01"}]


def test_answered_result_is_kept():
    assert classify_result({"raw_answer": "Prabowo won."}) == ("answered", None)


def test_retrieval_refusal_is_recorded_with_its_reason():
    result = {"answer": "No.", "source_quality": {"refusal_reason": "max_sim=0.33 < 0.35"}}
    assert classify_result(result) == ("refused", "max_sim=0.33 < 0.35")


def test_llm_outage_is_retried_not_recorded_as_refusal():
    # A refusal is skipped on resume; an outage must not be, or the question is lost.
    result = {"answer": "down", "raw_answer": "", "error": "llm_unavailable"}
    assert classify_result(result) == ("retry", "llm_unavailable")
