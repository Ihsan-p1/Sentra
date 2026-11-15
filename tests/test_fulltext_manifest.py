"""Tests for the full-text corpus event list and manifest validator."""
from pipeline.fulltext.manifest import load_events, validate_manifest

EVENTS = [
    {"id": "e1", "name": "MK age ruling", "start": "2023-10-16", "end": "2023-11-13",
     "keywords": ["age limit"], "required": True},
    {"id": "e9", "name": "Spare event", "start": "2024-10-19", "end": "2024-10-21",
     "keywords": ["inauguration"], "required": False},
]


def entry(n, outlet="tempo", event="e1", genre="news", date="2023-11-01"):
    host = "en.tempo.co/read" if outlet == "tempo" else "en.antaranews.com/news"
    return {"url": f"https://{host}/{1000 + n}/slug-{n}", "outlet": outlet, "event": event,
            "genre": genre, "title": f"Title {n}", "published_date": date, "found_by": "age limit"}


def valid_entries():
    tempo = [entry(i, "tempo") for i in range(4)]
    antara = [entry(10 + i, "antaranews") for i in range(5)]
    return tempo + antara


def test_valid_manifest_passes():
    assert validate_manifest(valid_entries(), EVENTS) == []


def test_rejects_date_outside_window():
    entries = valid_entries() + [entry(50, "antaranews", date="2023-12-01")]
    errors = validate_manifest(entries, EVENTS)
    assert len(errors) == 1
    assert entries[-1]["url"] in errors[0]


def test_rejects_duplicate_url():
    entries = valid_entries()
    entries.append(dict(entries[0], genre="opinion"))
    errors = validate_manifest(entries, EVENTS)
    assert any("duplicate" in e and entries[0]["url"] in e for e in errors)


def test_rejects_unknown_event_outlet_genre():
    entries = valid_entries() + [
        entry(60, event="e7"),
        dict(entry(61), outlet="abc_news"),
        entry(62, genre="feature"),
    ]
    errors = validate_manifest(entries, EVENTS)
    assert any("e7" in e for e in errors)
    assert any("abc_news" in e for e in errors)
    assert any("feature" in e for e in errors)


def test_quota_news_per_outlet():
    entries = [e for e in valid_entries() if e["url"] != entry(3)["url"]]
    entries.append(entry(70, genre="opinion"))  # opinion does not count towards the quota
    errors = validate_manifest(entries, EVENTS)
    assert "e1: tempo has 3 news, needs >= 4" in errors


def test_tempo_quota_is_exactly_four():
    errors = validate_manifest(valid_entries() + [entry(80)], EVENTS)
    assert any("e1: tempo has 5 news" in e for e in errors)


def test_antara_quota_four_to_eight():
    entries = valid_entries() + [entry(90 + i, "antaranews") for i in range(4)]
    errors = validate_manifest(entries, EVENTS)
    assert any("e1: antaranews has 9 news" in e for e in errors)


def test_optional_event_without_entries_is_fine():
    assert validate_manifest(valid_entries(), EVENTS) == []


def test_committed_event_list_has_the_required_events():
    ids = {e["id"] for e in load_events() if e["required"]}
    assert {"e1", "e2", "e3", "e3b", "e4", "e5"} <= ids
