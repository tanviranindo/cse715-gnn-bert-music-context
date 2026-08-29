"""FMA-small loading for Task 2 genre classification (PDF S4.2).

`fma_metadata/tracks.csv` uses a two-level column header, so it must be read
with `header=[0, 1]`. The columns that matter:

    ('set', 'subset')      small / medium / large
    ('set', 'split')       training / validation / test  (official)
    ('track', 'genre_top') the 8 top-level genres in FMA-small
    ('artist', 'name')     needed for artist-grouped splitting

Audio lives at `fma_small/<first 3 digits of zero-padded id>/<id>.mp3`.
"""

from pathlib import Path


def track_audio_path(root: str | Path, track_id: int) -> Path:
    """FMA stores 000123.mp3 under a directory named by its first 3 digits."""
    tid = f"{int(track_id):06d}"
    return Path(root) / tid[:3] / f"{tid}.mp3"


def load_tracks(tracks_csv: str | Path, subset: str = "small") -> list[dict]:
    """Read tracks.csv and return records for the requested FMA subset.

    Each record: track_id, genre, artist, split (the official one).
    Rows without a top-level genre are dropped.
    """
    import pandas as pd

    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1], low_memory=False)
    if subset:
        # the subset column is Categorical and ordered small < medium < large
        mask = df[("set", "subset")].astype(str) == subset
        df = df[mask]

    records = []
    for track_id, row in df.iterrows():
        genre = row[("track", "genre_top")]
        if not isinstance(genre, str) or not genre.strip():
            continue
        artist = row[("artist", "name")]
        records.append(
            {
                "track_id": int(track_id),
                "genre": genre.strip(),
                "artist": artist.strip() if isinstance(artist, str) else "unknown",
                "split": str(row[("set", "split")]).strip(),
            }
        )
    return records


def genre_vocabulary(records: list[dict]) -> list[str]:
    """Sorted genre list — index position is the class label."""
    return sorted({r["genre"] for r in records})


def official_splits(records: list[dict]) -> dict[str, list[dict]]:
    """Partition by FMA's own split column.

    FMA's split IS artist-aware (unlike MagnaTagATune's), so this is usable
    directly. `artist_leakage` below verifies that claim rather than trusting
    it, since the same assumption cost us on MagnaTagATune.
    """
    name = {"training": "train", "validation": "val", "test": "test"}
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        key = name.get(r["split"])
        if key:
            out[key].append(r)
    return out


def artist_leakage(splits: dict[str, list[dict]]) -> dict[str, int]:
    """Artists shared between splits. Verify, never assume."""
    a = {k: {r["artist"] for r in v} for k, v in splits.items()}
    return {
        "train_test": len(a.get("train", set()) & a.get("test", set())),
        "train_val": len(a.get("train", set()) & a.get("val", set())),
        "val_test": len(a.get("val", set()) & a.get("test", set())),
    }


def class_balance(records: list[dict]) -> dict[str, int]:
    """Clips per genre — FMA-small is balanced at 1000 each by construction."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r["genre"]] = counts.get(r["genre"], 0) + 1
    return dict(sorted(counts.items()))
