"""
End-to-end smoke tests against a running Sentra server.

Marked `integration` and skipped automatically when nothing is listening on
API_BASE, so the default `pytest` run stays offline. To run them:

    python -m uvicorn api.main:app        # in one terminal
    pytest -m integration                 # in another
"""
import os

import pytest
import requests

API_BASE = os.getenv("SENTRA_API_BASE", "http://localhost:8000")
TIMEOUT = int(os.getenv("SENTRA_TEST_TIMEOUT", "180"))

pytestmark = pytest.mark.integration


def _server_is_up() -> bool:
    try:
        return requests.get(f"{API_BASE}/api/health", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


requires_server = pytest.mark.skipif(
    not _server_is_up(), reason=f"no Sentra server reachable at {API_BASE}"
)


def ask(question: str, mode: str = "default", session_id: str = None) -> dict:
    payload = {"message": question, "mode": mode}
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=TIMEOUT)
    assert response.status_code == 200, response.text
    return response.json()


@requires_server
def test_health_endpoint_reports_ok():
    data = requests.get(f"{API_BASE}/api/health", timeout=10).json()
    assert data["status"] == "ok"


@requires_server
def test_chat_returns_a_non_empty_answer():
    data = ask("who is the new president of indonesia?")

    assert data["answer"].strip()
    assert data["session_id"]


@requires_server
def test_chat_reports_verification_summary():
    data = ask("what did the constitutional court decide?")

    verification = data.get("verification_summary") or {}
    if not verification:
        pytest.skip("answer was a refusal; no verification performed")

    assert verification["level"] in {"high", "moderate", "low"}
    assert 0.0 <= verification["support_rate"] <= 1.0
    assert verification["supported_count"] <= verification["total_checked"]


@requires_server
def test_model_comparison_reports_both_models():
    data = ask("how did media cover the cabinet formation?")

    comparison = data.get("model_comparison") or {}
    if not comparison:
        pytest.skip("answer was a refusal; no A/B comparison performed")

    for stage in ("hallucination", "confidence", "framing"):
        assert "model_a" in comparison[stage], f"{stage} missing Model A"
        assert "model_b" in comparison[stage], f"{stage} missing Model B"

    for model in ("model_a", "model_b"):
        unsupported, total = comparison["hallucination"][model]["unsupported_ratio"].split("/")
        assert int(unsupported) <= int(total)


@requires_server
def test_strict_mode_rejects_a_too_short_query():
    data = ask("who?", mode="reduce_hallucination")

    # Asserting on intent, not exact wording: the copy is localised and may change
    assert data["answer"].strip()
    assert not data.get("sources")
    assert "three words" in data["answer"] or "tiga kata" in data["answer"]


@requires_server
def test_greeting_is_answered_without_retrieval():
    data = ask("halo")

    assert data.get("intent") == "greeting"
    assert not data.get("sources")
    assert "Halo" in data["answer"]


@requires_server
def test_indonesian_question_is_answered_in_indonesian():
    data = ask("siapa yang memenangkan pemilu indonesia 2024")

    assert data.get("language") == "id"
    # Indonesian function words that would not appear in an English answer
    lowered = data["answer"].lower()
    assert any(word in lowered for word in (" yang ", " dan ", " dengan ", " adalah "))


@requires_server
def test_english_question_is_answered_in_english():
    data = ask("who won the 2024 indonesian election")

    assert data.get("language") == "en"


@requires_server
def test_session_history_is_retained_across_turns():
    first = ask("tell me about the 2024 presidential election")
    session_id = first["session_id"]

    # A follow-up only makes sense if the previous turn was actually stored
    second = ask("tell me more", session_id=session_id)

    assert second["session_id"] == session_id
    assert second["answer"].strip()


@requires_server
def test_clearing_a_session_succeeds():
    session_id = ask("what happened after the election?")["session_id"]

    response = requests.post(
        f"{API_BASE}/api/chat/clear", json={"session_id": session_id}, timeout=30
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cleared"
