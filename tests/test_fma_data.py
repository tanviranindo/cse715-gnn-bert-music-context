"""Task 2 FMA loading: paths, subset filtering, split integrity."""

import pytest

from src import fma_data


def test_audio_path_uses_three_digit_bucket():
    assert fma_data.track_audio_path("/data/fma_small", 123).as_posix() == \
        "/data/fma_small/000/000123.mp3"
    assert fma_data.track_audio_path("/x", 154321).as_posix() == \
        "/x/154/154321.mp3"


def test_genre_vocabulary_is_sorted_and_unique():
    recs = [{"genre": "Rock"}, {"genre": "Folk"}, {"genre": "Rock"}]
    assert fma_data.genre_vocabulary(recs) == ["Folk", "Rock"]


def test_official_splits_renames_and_partitions():
    recs = [
        {"track_id": 1, "genre": "Rock", "artist": "A", "split": "training"},
        {"track_id": 2, "genre": "Folk", "artist": "B", "split": "validation"},
        {"track_id": 3, "genre": "Rock", "artist": "C", "split": "test"},
        {"track_id": 4, "genre": "Folk", "artist": "D", "split": "nonsense"},
    ]
    out = fma_data.official_splits(recs)
    assert [r["track_id"] for r in out["train"]] == [1]
    assert [r["track_id"] for r in out["val"]] == [2]
    assert [r["track_id"] for r in out["test"]] == [3]


def test_artist_leakage_detects_a_shared_artist():
    splits = {
        "train": [{"artist": "A"}, {"artist": "B"}],
        "val": [],
        "test": [{"artist": "B"}],
    }
    assert fma_data.artist_leakage(splits)["train_test"] == 1


def test_artist_leakage_reports_zero_when_disjoint():
    splits = {"train": [{"artist": "A"}], "val": [{"artist": "B"}],
              "test": [{"artist": "C"}]}
    assert fma_data.artist_leakage(splits) == \
        {"train_test": 0, "train_val": 0, "val_test": 0}


def test_class_balance_counts_per_genre():
    recs = [{"genre": "Rock"}, {"genre": "Rock"}, {"genre": "Folk"}]
    assert fma_data.class_balance(recs) == {"Folk": 1, "Rock": 2}
