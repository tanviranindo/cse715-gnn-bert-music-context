"""The split manifests are a submission artifact (spec S10), so the shape they
are written in is worth pinning."""

from src import dump_splits


def _records():
    return {
        "train": [{"clip_id": "b"}, {"clip_id": "a"}],
        "val": [{"clip_id": "c"}],
        "test": [{"clip_id": "e"}, {"clip_id": "d"}],
    }


def test_manifest_reports_sizes_per_split():
    doc = dump_splits.manifest(_records(), "clip_id")
    assert doc["sizes"] == {"train": 2, "val": 1, "test": 2}


def test_manifest_sorts_ids_so_the_file_is_diffable():
    doc = dump_splits.manifest(_records(), "clip_id")
    assert doc["train"] == ["a", "b"]
    assert doc["test"] == ["d", "e"]


def test_manifest_stringifies_numeric_track_ids():
    doc = dump_splits.manifest({"train": [{"track_id": 12}]}, "track_id")
    assert doc["train"] == ["12"]


def test_splits_are_disjoint_in_the_manifest():
    doc = dump_splits.manifest(_records(), "clip_id")
    ids = doc["train"] + doc["val"] + doc["test"]
    assert len(ids) == len(set(ids))
