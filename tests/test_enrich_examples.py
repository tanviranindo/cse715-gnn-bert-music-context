"""MusicCaps captions describe one ten-second window, so the offsets are what
make the Task 4 listening study valid rather than decorative."""

import csv

from src import enrich_examples, musiccaps_data


def _examples():
    return [{"top3": [{"ytid": "aaa"}, {"ytid": "bbb"}, {"ytid": "unknown"}]}]


def test_enrich_attaches_the_window_for_known_clips():
    out, filled, missing = enrich_examples.enrich(
        _examples(), {"aaa": (30, 40), "bbb": (0, 10)})
    clips = out[0]["top3"]
    assert (clips[0]["start_s"], clips[0]["end_s"]) == (30, 40)
    assert (clips[1]["start_s"], clips[1]["end_s"]) == (0, 10)
    assert filled == 2


def test_unknown_clips_are_left_without_offsets_not_defaulted_to_zero():
    out, _, missing = enrich_examples.enrich(_examples(), {"aaa": (30, 40)})
    assert "start_s" not in out[0]["top3"][2]
    assert missing == 2


def test_clip_windows_parses_the_published_csv(tmp_path):
    p = tmp_path / "musiccaps.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ytid", "start_s", "end_s"])
        w.writerow(["-0Gj8-vB1q4", "30", "40"])
        w.writerow(["broken", "", ""])
    windows = musiccaps_data.clip_windows(p)
    assert windows == {"-0Gj8-vB1q4": (30, 40)}
