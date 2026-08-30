"""DEAM loading: three incompatible yearly schemas, captions, target scaling."""

import csv

import pytest

from src import deam_data as dd


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def meta_dir(tmp_path):
    # 2013 schema, with the stray tabs the real files contain
    _write(tmp_path / "metadata_2013.csv",
           ["song_id", "file_name", "Artist", "Song title", "Genre"],
           [["2", "2.mp3", "\tThe Troubadours\t", "Tonight", "\tBlues\t"]])
    # 2014 schema: different key names, plus a trailing empty column
    _write(tmp_path / "metadata_2014.csv",
           ["Id", "Artist", "Album", "Track", "Genre", "last.fm labels", ""],
           [["1001", "Rope River", "Muddy", "Slow Burn", "Soul-Rock", "moody", ""]])
    # 2015 schema: lowercase everything
    _write(tmp_path / "metadata_2015.csv",
           ["id", "Filename", "title", "artist", "album", "genre"],
           [["2001", "2001.mp3", "Latest", "Someone", "An Album", "Jazz"]])
    return tmp_path


@pytest.fixture
def ann_dir(tmp_path):
    d = tmp_path / "ann"
    d.mkdir()
    _write(d / "static_annotations_averaged_songs_1_2000.csv",
           ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"],
           [["2", "3.1", "0.94", "3.0", "0.63"], ["1001", "6.0", "1.0", "7.0", "1.0"]])
    _write(d / "static_annotations_averaged_songs_2000_2058.csv",
           ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"],
           [["2001", "4.0", "1.0", "5.0", "1.0"]])
    return d


def test_all_three_schemas_are_unified(meta_dir):
    meta = dd.load_metadata(meta_dir)
    assert set(meta) == {"2", "1001", "2001"}
    assert meta["2"]["artist"] == "The Troubadours"      # tabs stripped
    assert meta["2"]["genre"] == "Blues"
    assert meta["1001"]["title"] == "Slow Burn"          # 'Track' -> title
    assert meta["2001"]["album"] == "An Album"           # lowercase schema


def test_trailing_empty_column_does_not_break_parsing(meta_dir):
    meta = dd.load_metadata(meta_dir)
    assert meta["1001"]["tags"] == "moody"


def test_annotations_read_from_both_files(ann_dir):
    ann = dd.load_annotations(ann_dir)
    assert ann["2"] == (3.1, 3.0)
    assert ann["2001"] == (4.0, 5.0)


def test_pseudo_caption_splits_hyphenated_genres():
    text = dd.build_pseudo_caption(
        {"title": "Slow Burn", "album": "Muddy", "genre": "Soul-Rock", "tags": "moody"}
    )
    assert "Soul, Rock" in text
    assert "Slow Burn" in text and "moody" in text


def test_pseudo_caption_skips_missing_fields():
    assert dd.build_pseudo_caption({"title": "Only"}) == "Only"
    assert dd.build_pseudo_caption({}) == ""


def test_build_dataset_joins_annotations_and_metadata(meta_dir, ann_dir):
    recs = dd.build_dataset(ann_dir, meta_dir)
    assert {r["clip_id"] for r in recs} == {"2", "1001", "2001"}
    r = next(x for x in recs if x["clip_id"] == "2")
    assert r["valence"] == 3.1 and r["arousal"] == 3.0
    assert r["labels"] == set(), "DEAM carries no tag supervision"


def test_record_without_metadata_is_dropped(ann_dir, tmp_path):
    empty = tmp_path / "nometa"
    empty.mkdir()
    assert dd.build_dataset(ann_dir, empty) == []


def test_standardise_targets_zero_means_and_is_reusable(meta_dir, ann_dir):
    recs = dd.build_dataset(ann_dir, meta_dir)
    recs, stats = dd.standardise_targets(recs)
    zs = [r["valence_z"] for r in recs]
    assert abs(sum(zs) / len(zs)) < 1e-9
    # applying train stats to a held-out record must not recompute them
    other = dd.build_dataset(ann_dir, meta_dir)
    _, stats2 = dd.standardise_targets(other, stats=stats)
    assert stats2 == stats
