"""DEAM loading for Task 3's auxiliary valence/arousal term (PDF S4.3).

    L = L_tags + alpha*||v - v_hat||^2 + beta*||a - a_hat||^2

Two things this module exists to handle:

1. **Metadata lives in a separate download.** `DEAM_Annotations.zip` carries
   only valence/arousal — no genre, artist or title. The project's
   `data/README.md` does not mention it, but `metadata.zip` from the same
   host does have them, for 1744 of 1802 songs (96.8%). Without it, spec
   S2.1's documented fallback would apply (attach the emotion head to the
   graph vector alone); with it, the fused representation can be used as
   originally designed.

2. **Three incompatible schemas.** metadata_2013 uses
   `song_id / Artist / Song title / Genre`, metadata_2014 uses
   `Id / Artist / Album / Track / Genre / last.fm labels`, and metadata_2015
   uses `id / title / artist / album / genre`. Values also carry stray tabs.
   `load_metadata` normalises all three.

Valence and arousal are on a 1-9 scale (observed 1.6-8.4 and 1.6-8.1).
`standardise_targets` z-scores them, because raw 1-9 targets make the MSE
term dwarf the tag BCE and the tag head stops learning.
"""

import csv
from pathlib import Path

# canonical key -> the names each yearly file uses
_FIELD_ALIASES = {
    "song_id": ("song_id", "Id", "id"),
    "artist": ("Artist", "artist"),
    "title": ("Song title", "Track", "title"),
    "album": ("Album", "album"),
    "genre": ("Genre", "genre"),
    "tags": ("last.fm labels",),
}


def _clean_row(row: dict) -> dict:
    """Strip whitespace/tabs from keys and values, dropping the None key
    produced by files with a trailing delimiter."""
    out = {}
    for k, v in row.items():
        if k is None:
            continue
        key = k.strip()
        out[key] = v.strip() if isinstance(v, str) else ""
    return out


def _pick(row: dict, canonical: str) -> str:
    for name in _FIELD_ALIASES[canonical]:
        if name in row and row[name]:
            return row[name]
    return ""


def load_metadata(metadata_dir: str | Path) -> dict[str, dict]:
    """Read every metadata_*.csv -> {song_id: {artist, title, album, genre, tags}}."""
    out: dict[str, dict] = {}
    for path in sorted(Path(metadata_dir).glob("metadata_*.csv")):
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for raw in csv.DictReader(f):
                row = _clean_row(raw)
                song_id = _pick(row, "song_id")
                if not song_id:
                    continue
                out[song_id] = {
                    "artist": _pick(row, "artist"),
                    "title": _pick(row, "title"),
                    "album": _pick(row, "album"),
                    "genre": _pick(row, "genre"),
                    "tags": _pick(row, "tags"),
                }
    return out


def load_annotations(annotations_dir: str | Path) -> dict[str, tuple[float, float]]:
    """Read the averaged song-level statics -> {song_id: (valence, arousal)}."""
    out: dict[str, tuple[float, float]] = {}
    root = Path(annotations_dir)
    for path in root.rglob("static_annotations_averaged_songs_*.csv"):
        with open(path, newline="") as f:
            for raw in csv.DictReader(f):
                row = _clean_row(raw)
                sid = row.get("song_id", "")
                try:
                    out[sid] = (float(row["valence_mean"]), float(row["arousal_mean"]))
                except (KeyError, ValueError):
                    continue
    return out


def build_pseudo_caption(meta: dict) -> str:
    """Metadata -> the text BERT sees for a DEAM sample (spec S2.1).

    Formatted like the MagnaTagATune pseudo-caption ('title | album | ...')
    so the text tower sees a consistent input shape across corpora. Genre
    strings such as 'SoulRB-Country-Folk-Pop' are split into a comma list.
    """
    genre = meta.get("genre", "").replace("-", ", ")
    parts = [meta.get("title", ""), meta.get("album", ""), genre,
             meta.get("tags", "")]
    return " | ".join(p.strip() for p in parts if p and p.strip())


def build_dataset(
    annotations_dir: str | Path,
    metadata_dir: str | Path,
    audio_dir: str | Path | None = None,
) -> list[dict]:
    """Assemble DEAM records with pseudo-captions and emotion targets.

    Songs lacking metadata are kept only if some text survives; a record with
    an empty caption would train the text tower on nothing.
    """
    ann = load_annotations(annotations_dir)
    meta = load_metadata(metadata_dir)
    out = []
    for song_id, (valence, arousal) in sorted(ann.items(), key=lambda kv: int(kv[0])):
        m = meta.get(song_id, {})
        text = build_pseudo_caption(m)
        if not text:
            continue
        record = {
            "clip_id": song_id,
            "artist": m.get("artist") or f"unknown_{song_id}",
            "text": text,
            "valence": valence,
            "arousal": arousal,
            "labels": set(),          # DEAM carries no tag supervision
        }
        if audio_dir is not None:
            record["audio_path"] = str(Path(audio_dir) / f"{song_id}.mp3")
        out.append(record)
    return out


def standardise_targets(
    records: list[dict], stats: tuple[float, float, float, float] | None = None
) -> tuple[list[dict], tuple[float, float, float, float]]:
    """Z-score valence/arousal. Pass `stats` from train to apply to val/test.

    Returns (records, (v_mean, v_std, a_mean, a_std)) so the scaling can be
    inverted when reporting MAE in the original 1-9 units.
    """
    if stats is None:
        vs = [r["valence"] for r in records]
        as_ = [r["arousal"] for r in records]
        vm = sum(vs) / len(vs)
        am = sum(as_) / len(as_)
        vsd = (sum((x - vm) ** 2 for x in vs) / len(vs)) ** 0.5 or 1.0
        asd = (sum((x - am) ** 2 for x in as_) / len(as_)) ** 0.5 or 1.0
        stats = (vm, vsd, am, asd)
    vm, vsd, am, asd = stats
    for r in records:
        r["valence_z"] = (r["valence"] - vm) / vsd
        r["arousal_z"] = (r["arousal"] - am) / asd
    return records, stats
