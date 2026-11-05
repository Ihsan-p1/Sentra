"""Command-line paths of the retrieval eval scripts (full-text corpus, Task 8)."""
import json
from pathlib import Path

from evaluation.hallucination_dataset import DATASETS_DIR, QUESTIONS_PATH
from evaluation.retrieval_gold import GOLD_PATH
from scripts import calibrate_threshold, evaluate_retrieval


def test_default_output_names_unchanged():
    # Running the script with no arguments must still write the old corpus's results.
    args = evaluate_retrieval.parse_args([])
    assert Path(args.questions) == QUESTIONS_PATH
    assert Path(args.gold) == GOLD_PATH
    md, js = evaluate_retrieval.output_paths(args.out_name)
    assert (md.name, js.name) == ("retrieval.md", "retrieval.json")

    args = evaluate_retrieval.parse_args(["--questions", "q.json", "--gold", "g.json",
                                          "--out-name", "retrieval_fulltext"])
    assert (args.questions, args.gold) == ("q.json", "g.json")
    assert [p.name for p in evaluate_retrieval.output_paths(args.out_name)] == [
        "retrieval_fulltext.md", "retrieval_fulltext.json"]


def test_calibration_questions_file(tmp_path):
    args = calibrate_threshold.parse_args([])
    assert calibrate_threshold.load_question_sets(args.questions) == (
        calibrate_threshold.IN_CORPUS, calibrate_threshold.OUT_OF_CORPUS)

    path = tmp_path / "q.json"
    path.write_text(json.dumps({"in_corpus": ["a"], "out_of_corpus": ["b"]}), encoding="utf-8")
    args = calibrate_threshold.parse_args(["--questions", str(path)])
    assert calibrate_threshold.load_question_sets(args.questions) == (["a"], ["b"])

    in_corpus, out_of_corpus = calibrate_threshold.load_question_sets(
        DATASETS_DIR / "fulltext_calibration.json")
    assert len(in_corpus) >= 12
    # None of the old out-of-corpus questions touches the 2024 election, so all of them still apply.
    assert out_of_corpus == calibrate_threshold.OUT_OF_CORPUS
