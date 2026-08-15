import pytest
from src.splits import artist_grouped_split


def make_records():
    records = []
    for artist_id in range(20):
        for track_id in range(5):  # 5 tracks per artist, 100 tracks total
            records.append({"track_id": f"{artist_id}-{track_id}", "artist": f"artist_{artist_id}"})
    return records


def test_split_has_no_artist_overlap():
    records = make_records()
    train, val, test = artist_grouped_split(records, train_frac=0.7, val_frac=0.15)
    train_artists = {r["artist"] for r in train}
    val_artists = {r["artist"] for r in val}
    test_artists = {r["artist"] for r in test}
    assert train_artists.isdisjoint(val_artists)
    assert train_artists.isdisjoint(test_artists)
    assert val_artists.isdisjoint(test_artists)


def test_split_covers_all_records():
    records = make_records()
    train, val, test = artist_grouped_split(records, train_frac=0.7, val_frac=0.15)
    assert len(train) + len(val) + len(test) == len(records)


def test_split_is_deterministic_given_seed():
    records = make_records()
    train1, val1, test1 = artist_grouped_split(records, seed=42)
    train2, val2, test2 = artist_grouped_split(records, seed=42)
    assert [r["track_id"] for r in train1] == [r["track_id"] for r in train2]


def test_split_respects_approximate_fractions():
    records = make_records()  # 20 artists
    train, val, test = artist_grouped_split(records, train_frac=0.7, val_frac=0.15)
    train_artists = {r["artist"] for r in train}
    val_artists = {r["artist"] for r in val}
    test_artists = {r["artist"] for r in test}
    assert len(train_artists) == 14  # round(20 * 0.7)
    assert len(val_artists) == 3     # round(20 * 0.15)
    assert len(test_artists) == 3    # remainder
