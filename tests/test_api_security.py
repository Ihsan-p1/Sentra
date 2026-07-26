"""
Unit tests for the ingest endpoint's auth gate.

The gate is tested directly rather than through the app, so no database,
model loading, or running server is required.
"""
import pytest
from fastapi import HTTPException

from api.routes import _require_ingest_key
from config.settings import settings


def test_open_when_no_key_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "INGEST_API_KEY", None)

    _require_ingest_key(None)  # must not raise - local development default


def test_correct_key_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "INGEST_API_KEY", "s3cret")

    _require_ingest_key("s3cret")


def test_missing_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "INGEST_API_KEY", "s3cret")

    with pytest.raises(HTTPException) as exc:
        _require_ingest_key(None)

    assert exc.value.status_code == 401


def test_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "INGEST_API_KEY", "s3cret")

    with pytest.raises(HTTPException) as exc:
        _require_ingest_key("guess")

    assert exc.value.status_code == 401


def test_key_of_different_length_is_rejected(monkeypatch):
    # compare_digest raises on mismatched types, not lengths; make sure a short
    # candidate is handled as a plain rejection.
    monkeypatch.setattr(settings, "INGEST_API_KEY", "s3cret")

    with pytest.raises(HTTPException):
        _require_ingest_key("s")


def test_cors_is_not_a_wildcard():
    assert "*" not in settings.CORS_ALLOW_ORIGINS
