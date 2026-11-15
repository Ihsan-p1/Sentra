"""
Unit tests for the DistilBERT framing gate.

DistilBERT is trained on the old corpus only, so on a corpus that lists it under
"unevaluated_models" its scores must not be computed or shown as a match.
"""
from chatbot.framing_gate import neural_framing, style_match_label


class RecordingBert:
    def __init__(self, scores):
        self.scores = scores
        self.texts = []

    def predict_source_style(self, text):
        self.texts.append(text)
        return self.scores


class ExplodingBert:
    def predict_source_style(self, text):
        raise AssertionError("DistilBERT must not run on an unevaluated corpus")


def test_neural_framing_unevaluated_returns_none_without_calling_bert():
    texts = {"antaranews": ["one"], "tempo": ["two"]}

    result = neural_framing(texts, ExplodingBert(), ["framing_bert"])

    assert result == {"antaranews": None, "tempo": None}


def test_neural_framing_evaluated_matches_old_behaviour():
    bert = RecordingBert({"antaranews": 0.7})
    texts = {"antaranews": ["x" * 400, "y" * 400], "abc_news": ["z"], "empty": []}

    result = neural_framing(texts, bert, [])

    assert result == {"antaranews": 0.7, "abc_news": 0.0}
    assert bert.texts[0] == (" ".join(["x" * 400, "y" * 400]))[:512]


def test_style_match_label_renders_none_as_not_evaluated():
    assert style_match_label(None) == "Not evaluated on this corpus"
    assert style_match_label(0.734) == "Style Match: 73.4%"
