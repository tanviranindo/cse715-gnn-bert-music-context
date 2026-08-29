"""Task 1 MusicCaps variant: vocabulary, leakage stripping, lexical baseline."""

import csv

import pytest

from src import musiccaps_data as mc


@pytest.fixture
def caps(tmp_path):
    path = tmp_path / "musiccaps-public.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ytid", "caption", "aspect_list", "is_audioset_eval"])
        w.writerow(["a1", "A sad ballad with soft piano.", "['sad', 'ballad']", "False"])
        w.writerow(["a2", "An energetic techno track.", "['energetic', 'rare_tag']", "True"])
        w.writerow(["a3", "A sad song, slow and quiet.", "['sad']", "False"])
    return path


def test_rare_aspects_are_filtered_out(caps):
    vocab, _ = mc.build_dataset(caps, n_aspects=50, min_count=2)
    assert "sad" in vocab           # appears twice
    assert "rare_tag" not in vocab  # appears once
    assert "ballad" not in vocab


def test_clip_with_no_surviving_label_is_dropped(caps):
    _, recs = mc.build_dataset(caps, n_aspects=50, min_count=2)
    assert {r["clip_id"] for r in recs} == {"a1", "a3"}


def test_strip_leakage_removes_the_aspect_word(caps):
    _, recs = mc.build_dataset(caps, n_aspects=50, min_count=2, strip_leakage=True)
    for r in recs:
        assert "sad" not in r["text"].lower(), "label word must not survive in input"
    # surrounding context is preserved
    assert "piano" in next(r["text"] for r in recs if r["clip_id"] == "a1")


def test_strip_leakage_off_by_default(caps):
    _, recs = mc.build_dataset(caps, n_aspects=50, min_count=2)
    assert "sad" in next(r["text"] for r in recs if r["clip_id"] == "a1").lower()


def test_lexical_baseline_detects_only_literal_matches():
    vocab = ["sad", "techno", "piano"]
    assert mc.lexical_match_predict("A sad piano piece", vocab) == {"sad", "piano"}
    assert mc.lexical_match_predict("Nothing here", vocab) == set()


def test_strip_aspects_is_case_insensitive_and_repeats():
    out = mc.strip_aspects_from_caption("Sad and sad and SAD.", {"sad"})
    assert "sad" not in out.lower()
