"""Artist-grouped train/val/test splitting (no artist leakage)."""

import random


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
