"""
Unit tests for the hallucination detector's feature extraction.

Feature extraction is pure numeric code, so it is tested with hand-built
embedding vectors rather than a real sentence-transformer model. That keeps the
suite fast and offline while still pinning down the contract the trained
scaler/classifier depends on: exactly N_FEATURES values, in a fixed order.
"""
import numpy as np
import pytest

from models.hallucination.detector import FEATURE_NAMES, N_FEATURES, HallucinationDetector

MAX_SIM, AVG_SIM, MIN_SIM, STD_SIM, OVERLAP = range(N_FEATURES)


@pytest.fixture
def detector():
    return HallucinationDetector()


def test_feature_names_match_vector_width():
    assert len(FEATURE_NAMES) == N_FEATURES


def test_extract_features_returns_fixed_width(detector):
    sentence = np.array([1.0, 0.0, 0.0])
    sources = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    features = detector.extract_features(sentence, sources, "some text", ["some text"])

    assert features.shape == (N_FEATURES,)


def test_empty_sources_return_zero_vector_of_correct_width(detector):
    # Regression: this branch used to return a 6-wide vector, which would blow up
    # the scaler (fitted on 5 features) with a shape mismatch at predict time.
    features = detector.extract_features(np.array([1.0, 0.0]), np.array([]), "text", [])

    assert features.shape == (N_FEATURES,)
    assert not features.any()


def test_identical_vectors_give_max_similarity_one(detector):
    sentence = np.array([1.0, 0.0, 0.0])
    sources = np.array([[1.0, 0.0, 0.0]])

    features = detector.extract_features(sentence, sources, "prabowo won", ["prabowo won"])

    assert features[MAX_SIM] == pytest.approx(1.0)
    assert features[OVERLAP] == pytest.approx(1.0)


def test_orthogonal_vectors_give_zero_similarity(detector):
    sentence = np.array([1.0, 0.0])
    sources = np.array([[0.0, 1.0]])

    features = detector.extract_features(sentence, sources, "totally unrelated", ["different words here"])

    assert features[MAX_SIM] == pytest.approx(0.0)
    assert features[OVERLAP] == pytest.approx(0.0)


def test_max_is_never_below_min_and_avg_sits_between(detector):
    sentence = np.array([1.0, 0.0])
    sources = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])

    f = detector.extract_features(sentence, sources, "text", ["text"])

    assert f[MIN_SIM] <= f[AVG_SIM] <= f[MAX_SIM]
    assert f[STD_SIM] >= 0.0


def test_overlap_is_ratio_of_sentence_words_found_in_sources(detector):
    sentence = np.array([1.0, 0.0])
    sources = np.array([[1.0, 0.0]])

    # 2 of 4 sentence words appear in the source text
    f = detector.extract_features(sentence, sources, "prabowo won unknown election", ["prabowo the won"])

    assert f[OVERLAP] == pytest.approx(0.5)


def test_predict_before_training_raises(detector):
    with pytest.raises(ValueError, match="not trained"):
        detector.predict(np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), "text", ["text"])


def test_hard_gate_rejects_low_similarity_after_training(detector):
    # Train on a trivially separable synthetic set so predict() is usable.
    X = np.array(
        [[0.9, 0.8, 0.7, 0.05, 0.9]] * 20 + [[0.1, 0.05, 0.0, 0.05, 0.05]] * 20
    )
    y = np.array([1] * 20 + [0] * 20)
    detector.train(X, y)

    # max_similarity 0.0 is below the 0.30 hard gate -> unsupported regardless
    # of what the classifier says.
    result = detector.predict(
        np.array([1.0, 0.0]), np.array([[0.0, 1.0]]), "fabricated claim", ["unrelated source text"]
    )

    assert result["is_supported"] is False
    assert result["reason"]


def test_strong_match_is_supported_after_training(detector):
    X = np.array(
        [[0.9, 0.8, 0.7, 0.05, 0.9]] * 20 + [[0.1, 0.05, 0.0, 0.05, 0.05]] * 20
    )
    y = np.array([1] * 20 + [0] * 20)
    detector.train(X, y)

    result = detector.predict(
        np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), "prabowo won", ["prabowo won"]
    )

    assert result["is_supported"] is True
    assert result["reason"] == ""
    assert 0.0 <= result["score"] <= 1.0
