from src.run_provenance import build_provenance


def _splits():
    return {
        "train": [{"clip_id": "b"}, {"clip_id": "a"}],
        "val": [{"clip_id": "c"}],
        "test": [{"clip_id": "d"}],
    }


def test_provenance_digests_change_when_split_membership_changes():
    base = build_provenance({}, _splits(), ["x", "y"], "val", "val_macro_f1")
    changed = _splits()
    changed["test"] = [{"clip_id": "e"}]
    other = build_provenance({}, changed, ["x", "y"], "val", "val_macro_f1")
    assert base["split_digest"] != other["split_digest"]


def test_provenance_digests_change_when_vocabulary_order_changes():
    base = build_provenance({}, _splits(), ["x", "y"], "val", "val_macro_f1")
    other = build_provenance({}, _splits(), ["y", "x"], "val", "val_macro_f1")
    assert base["vocabulary_digest"] != other["vocabulary_digest"]
    assert base["vocabulary_from"] == "train"
    assert base["threshold_from"] == "val"


def test_provenance_accepts_explicit_revision_for_rsync_remote(monkeypatch):
    revision = "1" * 40
    monkeypatch.setenv("EXPERIMENT_SOURCE_REVISION", revision)
    provenance = build_provenance({}, _splits(), ["x"], "val", "val_macro_f1")
    assert provenance["source_revision"] == revision
