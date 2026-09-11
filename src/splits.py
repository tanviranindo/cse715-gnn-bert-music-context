"""Artist-grouped train/val/test splitting (no artist leakage)."""

import hashlib
import json
import random


def record_id(record) -> str:
    """Return the stable corpus ID from either a record dict or data object."""
    for key in ("clip_id", "track_id", "ytid"):
        value = record.get(key) if isinstance(record, dict) else getattr(record, key, None)
        if value is not None and value != "":
            return str(value)
    raise KeyError("record carries no clip_id / track_id / ytid")


def _field(record, key, default=None):
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def artist_grouped_split(
    records: list[dict],
    artist_key: str = "artist",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    artists = sorted({r[artist_key] for r in records})
    rng = random.Random(seed)
    rng.shuffle(artists)

    n_train = round(len(artists) * train_frac)
    n_val = round(len(artists) * val_frac)

    train_artists = set(artists[:n_train])
    val_artists = set(artists[n_train:n_train + n_val])
    test_artists = set(artists[n_train + n_val:])

    train = [r for r in records if r[artist_key] in train_artists]
    val = [r for r in records if r[artist_key] in val_artists]
    test = [r for r in records if r[artist_key] in test_artists]
    return train, val, test


def official_eval_split(items, seed: int = 42, val_frac: float = 0.15,
                        id_of=record_id):
    """Hold out the published eval partition, split the rest reproducibly.

    Two scripts must be able to produce the *same* split independently, or a
    supervised and a zero-shot model compared "on the same data" are not. An
    earlier version had train_contrastive shuffle the pool while train_fusion
    took the first 15% unshuffled: the test partition matched and the sizes
    matched, so the mismatch was invisible in every printed number.

    The order is therefore fixed by sorting on clip id before shuffling with a
    dedicated RNG, so the result depends on the seed and the data, never on how
    many random numbers the caller happened to draw first.
    """
    import random as _random

    eval_items = [d for d in items if _field(d, "is_eval", False)]
    if not eval_items:
        raise ValueError("no clip carries is_eval; cannot use the official split")
    pool = sorted((d for d in items if not _field(d, "is_eval", False)),
                  key=id_of)
    _random.Random(seed).shuffle(pool)
    n_val = max(1, int(val_frac * len(pool)))
    return pool[n_val:], pool[:n_val], eval_items


def split_records(dataset: str, records: list, seed: int = 42) -> tuple[dict[str, list], str]:
    """Apply the one authoritative partition policy for a supported corpus."""
    if dataset == "fma_small":
        from src import fma_data

        by_split = fma_data.official_splits(records)
        return by_split, "official_fma_split"
    if dataset == "musiccaps":
        train, val, test = official_eval_split(records, seed=seed)
        return {"train": train, "val": val, "test": test}, "official_audioset_eval"
    if dataset == "mtat":
        train, val, test = artist_grouped_split(records, seed=seed)
        return {"train": train, "val": val, "test": test}, f"artist_grouped_seed{seed}"
    raise ValueError(f"unsupported dataset {dataset!r}")


def split_digest(by_split: dict[str, list], id_of=record_id) -> str:
    """SHA-256 of canonical, sorted split membership."""
    payload = {
        name: sorted(id_of(record) for record in by_split.get(name, []))
        for name in ("train", "val", "test")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
