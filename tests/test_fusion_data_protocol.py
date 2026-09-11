"""Task 3 must not trust a vocabulary selected before graph-aware splitting."""

from types import SimpleNamespace

from src.train_fusion import prepare_tag_protocol


def _items():
    items = []
    clip_id = 0
    for artist_id in range(10):
        repeats = 20 if artist_id == 1 else 1
        for _ in range(repeats):
            clip_id += 1
            items.append(
                SimpleNamespace(
                    artist=f"artist_{artist_id}",
                    clip_id=str(clip_id),
                    labels_set={"held_out_only"} if artist_id == 1 else {"train_only"},
                    has_tags=1.0,
                    has_emotion=0.0,
                    valence=0.0,
                    arousal=0.0,
                    is_eval=False,
                )
            )
    return items


def test_task3_ignores_cache_vocabulary_and_uses_training_labels():
    vocab_a, splits_a, meta_a = prepare_tag_protocol(
        _items(), "artist", seed=42, n_tags=1,
        cached_vocab=["held_out_only"],
    )
    vocab_b, splits_b, meta_b = prepare_tag_protocol(
        _items(), "artist", seed=42, n_tags=1, cached_vocab=[]
    )

    assert vocab_a == vocab_b == ["train_only"]
    assert meta_a["vocabulary_from"] == "train"
    assert meta_a["cache_vocabulary_ignored"] is True
    assert meta_b["cache_vocabulary_ignored"] is False
    assert meta_a["split_digest"] == meta_b["split_digest"]
    assert [d.clip_id for d in splits_a["test"]] == [
        d.clip_id for d in splits_b["test"]
    ]
    assert all(d.y.shape == (1, 1) for part in splits_a.values() for d in part)
