"""The study page is a real deliverable sent to real people, so the parts that
would silently corrupt the result are pinned here."""

import json

from src import human_eval


def _examples(**over):
    clip = {"clip_id": "c1", "ytid": "abc123", "start_s": 30, "end_s": 40,
            "caption": "a clip", "score": 0.5, "is_correct": False}
    clip.update(over)
    return [{"query_caption": "a sad piano ballad", "top3": [dict(clip, clip_id="c%d" % i)
                                                             for i in range(1, 4)]}]


def _build(tmp_path, examples, endpoint=""):
    src = tmp_path / "ex.json"
    src.write_text(json.dumps(examples))
    human_eval.STUDY = tmp_path / "study"
    human_eval.RATINGS = human_eval.STUDY / "ratings"
    human_eval.build(str(src), endpoint=endpoint)
    return (human_eval.STUDY / "rating_sheet.html").read_text()


def test_clip_is_cued_to_its_own_window_not_the_start(tmp_path):
    html = _build(tmp_path, _examples())
    data = json.loads(html.split("const STUDY = ", 1)[1].split(";\nconst ENDPOINT", 1)[0])
    assert data["queries"][0]["clips"][0]["start_s"] == 30
    assert data["queries"][0]["clips"][0]["end_s"] == 40


def test_a_clip_without_a_known_window_gets_no_player(tmp_path):
    html = _build(tmp_path, _examples(start_s=None, end_s=None))
    data = json.loads(html.split("const STUDY = ", 1)[1].split(";\nconst ENDPOINT", 1)[0])
    assert data["queries"][0]["clips"][0]["start_s"] is None


def test_endpoint_is_embedded_when_given(tmp_path):
    assert '"/api/submit"' in _build(tmp_path, _examples(), endpoint="/api/submit")


def test_without_an_endpoint_the_page_falls_back_to_download(tmp_path):
    html = _build(tmp_path, _examples())
    assert 'const ENDPOINT = "";' in html


def test_every_clip_reaches_the_page(tmp_path):
    html = _build(tmp_path, _examples())
    data = json.loads(html.split("const STUDY = ", 1)[1].split(";\nconst ENDPOINT", 1)[0])
    assert data["total"] == 3
    assert len(data["queries"][0]["clips"]) == 3
