"""Tests for the Antara fetcher of the full-text corpus (no network)."""
from pathlib import Path

import pytest

from scripts.fulltext.fetch_antara import fetch_all, raw_path

ANTARA = "https://en.antaranews.com/news/298146/some-slug"
TEMPO = "https://en.tempo.co/read/1784688/other-slug"


class FakeResponse:
    def __init__(self, status_code, content=b"<html>body</html>"):
        self.status_code = status_code
        self.content = content


class FakeSession:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return FakeResponse(self.status_code)


def entry(url, outlet="antaranews"):
    return {"url": url, "outlet": outlet}


def test_raw_path_uses_numeric_id(tmp_path):
    assert raw_path(ANTARA, "antaranews") == Path("data/raw/antaranews/298146.html")
    assert raw_path(TEMPO, "tempo", tmp_path) == tmp_path / "tempo" / "1784688.html"


def test_raw_path_rejects_url_without_id():
    with pytest.raises(ValueError, match="no numeric id"):
        raw_path("https://en.antaranews.com/news/some-slug", "antaranews")


def test_skips_existing_file(tmp_path):
    target = raw_path(ANTARA, "antaranews", tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already here")
    session = FakeSession()

    result = fetch_all([entry(ANTARA)], session, tmp_path, delay_s=0)

    assert result == {ANTARA: "skipped"}
    assert session.calls == []
    assert target.read_bytes() == b"already here"


def test_never_requests_tempo(tmp_path):
    session = FakeSession()

    result = fetch_all([entry(TEMPO, "tempo")], session, tmp_path, delay_s=0)

    assert result == {}
    assert session.calls == []
    assert not (tmp_path / "tempo").exists()


def test_fetched_file_written(tmp_path):
    session = FakeSession()

    result = fetch_all([entry(ANTARA)], session, tmp_path, delay_s=0)

    assert result == {ANTARA: "fetched"}
    assert raw_path(ANTARA, "antaranews", tmp_path).read_bytes() == b"<html>body</html>"


def test_failed_status_recorded(tmp_path):
    session = FakeSession(status_code=403)

    result = fetch_all([entry(ANTARA)], session, tmp_path, delay_s=0)

    assert result == {ANTARA: "failed: 403"}
    assert not raw_path(ANTARA, "antaranews", tmp_path).exists()
