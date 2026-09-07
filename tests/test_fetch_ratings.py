"""A rater who redoes the study must count once, not twice."""

import json

from src import fetch_ratings


def test_manifest_helper_exists():
    assert hasattr(fetch_ratings, "fetch")


def test_newest_submission_wins(monkeypatch, tmp_path):
    blobs = [
        {"pathname": "ratings/ann-1.json", "url": "u1", "uploadedAt": "2026-09-07T10:00:00.000Z"},
        {"pathname": "ratings/ann-2.json", "url": "u2", "uploadedAt": "2026-09-07T12:00:00.000Z"},
        {"pathname": "ratings/bob-1.json", "url": "u3", "uploadedAt": "2026-09-07T11:00:00.000Z"},
    ]
    bodies = {
        "u1": {"rater": "Ann", "ratings": [{"rating": 1}]},
        "u2": {"rater": "Ann", "ratings": [{"rating": 5}]},
        "u3": {"rater": "Bob", "ratings": [{"rating": 3}]},
    }

    class _Resp:
        def __init__(self, payload): self._p = json.dumps(payload).encode()
        def read(self): return self._p
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        url = req if isinstance(req, str) else getattr(req, "full_url", "")
        if "prefix" in str(url):
            return _Resp({"blobs": blobs})
        return _Resp(bodies[url])

    monkeypatch.setattr(fetch_ratings.urllib.request, "urlopen", fake_urlopen)
    files = fetch_ratings.fetch("tok", "ratings/", tmp_path)
    names = sorted(f.name for f in files)
    assert names == ["ann-2.json", "bob-1.json"], names
    ann = json.loads((tmp_path / "ann-2.json").read_text())
    assert ann["ratings"][0]["rating"] == 5, "the later submission must win"
