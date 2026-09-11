"""The split manifests are a submission artifact (spec S10), so the shape they
are written in is worth pinning."""

from src import dump_splits
from src.splits import split_digest, split_records


class _Item:
    def __init__(self, clip_id, is_eval):
        self.clip_id = clip_id
        self.is_eval = is_eval


def _records():
    return {
        "train": [{"clip_id": "b"}, {"clip_id": "a"}],
        "val": [{"clip_id": "c"}],
        "test": [{"clip_id": "e"}, {"clip_id": "d"}],
    }


def test_manifest_reports_sizes_per_split():
    doc = dump_splits.manifest(_records(), "artist_grouped_seed42")
    assert doc["sizes"] == {"train": 2, "val": 1, "test": 2}


def test_manifest_sorts_ids_so_the_file_is_diffable():
    doc = dump_splits.manifest(_records(), "artist_grouped_seed42")
    assert doc["train"] == ["a", "b"]
    assert doc["test"] == ["d", "e"]


def test_manifest_stringifies_numeric_track_ids():
    doc = dump_splits.manifest({"train": [{"track_id": 12}]}, "official_fma_split")
    assert doc["train"] == ["12"]


def test_clip_id_falls_back_across_the_corpora_id_fields():
    assert dump_splits._clip_id({"clip_id": "a"}) == "a"
    assert dump_splits._clip_id({"track_id": 7}) == "7"
    assert dump_splits._clip_id({"ytid": "-0Gj8-vB1q4"}) == "-0Gj8-vB1q4"


def test_manifest_records_which_split_policy_produced_it():
    doc = dump_splits.manifest({"train": [{"clip_id": "a"}]}, "official_audioset_eval")
    assert doc["split_kind"] == "official_audioset_eval"


def test_splits_are_disjoint_in_the_manifest():
    doc = dump_splits.manifest(_records(), "artist_grouped_seed42")
    ids = doc["train"] + doc["val"] + doc["test"]
    assert len(ids) == len(set(ids))


def test_musiccaps_manifest_uses_the_training_split_implementation():
    records = [
        _Item("z", False),
        _Item("a", False),
        _Item("e", True),
        _Item("m", False),
    ]
    by_split, kind = split_records("musiccaps", records, seed=42)

    doc = dump_splits.manifest_for_dataset("musiccaps", records, seed=42)

    assert doc["split_kind"] == kind
    assert doc["train"] == sorted(x.clip_id for x in by_split["train"])
    assert doc["val"] == sorted(x.clip_id for x in by_split["val"])
    assert doc["test"] == ["e"]
    assert doc["split_digest"] == split_digest(by_split)
