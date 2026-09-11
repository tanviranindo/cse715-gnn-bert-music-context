"""MagnaTagATune loading for Task 1 (BERT multi-label tag classifier).

Three things this module exists to get right, all of which materially change
the reported numbers:

1. **Synonym merging.** MTAT's 188 tags contain heavy near-duplicates
   ('classical'/'clasical'/'classic', six variants of 'female vocal').
   Merging before ranking is standard practice; skipping it splits a single
   concept across several weak labels.
2. **Top-N tags.** 67 of the 188 tags have <100 positive clips and cannot be
   learned. The literature convention is top-50 after merging.
3. **Artist-grouped splits.** The official MTT split leaks 45 artists across
   train/test, covering 61.6% of test clips. See `load_splits`.
"""

import csv
import collections
from pathlib import Path

# Groups of tags that denote the same concept. First element is the canonical
# name. Derived by inspecting the 188-tag vocabulary; counts confirmed present.
SYNONYMS: list[tuple[str, ...]] = [
    ("classical", "clasical", "classic"),
    ("choir", "chorus", "choral"),
    ("female vocal", "female", "female voice", "female singing", "woman", "woman singing"),
    ("male vocal", "male", "male voice", "male singing", "man", "man singing"),
    ("electronic", "electro", "electronica"),
    ("beat", "beats"),
    ("vocal", "vocals", "voice", "voices", "singing", "singer"),
    ("no vocal", "no vocals", "no voice", "no singing", "instrumental"),
    ("harpsichord", "harpsicord"),
    ("synth", "synthesizer"),
    ("drum", "drums"),
    ("guitar", "guitars"),
    ("violin", "violins"),
    ("string", "strings"),
    ("horn", "horns"),
    ("flute", "flutes"),
    ("india", "indian"),
    ("weird", "strange"),
    ("quiet", "silence"),
]


def canonical_tag_map(tags: list[str]) -> dict[str, str]:
    """Map each raw tag to its canonical form (identity if not a synonym)."""
    mapping = {t: t for t in tags}
    present = set(tags)
    for group in SYNONYMS:
        canon = group[0]
        if canon not in present:
            continue
        for alias in group[1:]:
            if alias in present:
                mapping[alias] = canon
    return mapping


def load_annotations(path: str | Path) -> tuple[list[str], list[dict]]:
    """Read annotations_final.csv (tab separated).

    Returns (raw_tag_names, records) where each record is
    {'clip_id': str, 'mp3_path': str, 'tags': set[str]} using raw tag names.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        tag_names = header[1:-1]
        records = []
        for row in reader:
            if len(row) != len(header):
                continue
            positives = {tag_names[i] for i, v in enumerate(row[1:-1]) if v == "1"}
            records.append(
                {"clip_id": row[0], "mp3_path": row[-1], "tags": positives}
            )
    return tag_names, records


def load_clip_info(path: str | Path) -> dict[str, dict]:
    """Read clip_info_final.csv -> {clip_id: {title, artist, album}}."""
    info = {}
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx = {k: header.index(k) for k in ("clip_id", "title", "artist", "album")}
        for row in reader:
            if len(row) <= max(idx.values()):
                continue
            info[row[idx["clip_id"]]] = {
                "title": row[idx["title"]],
                "artist": row[idx["artist"]],
                "album": row[idx["album"]],
            }
    return info


def top_tags(records: list[dict], tag_map: dict[str, str], n: int = 50) -> list[str]:
    """Most frequent `n` canonical tags, ordered by clip count (desc)."""
    counts: collections.Counter = collections.Counter()
    for r in records:
        for t in {tag_map[t] for t in r["tags"] if t in tag_map}:
            counts[t] += 1
    return [t for t, _ in counts.most_common(n)]


def build_text(info: dict, use_artist: bool = False) -> str:
    """Pseudo-caption fed to BERT.

    `use_artist` defaults to False on purpose. Artist name is the single
    strongest memorisation shortcut in this corpus — with it the model can
    map 'American Bach Soloists' straight to 'classical' without reading the
    music. Title and album still carry genuine lexical signal.
    """
    parts = [info.get("title", ""), info.get("album", "")]
    if use_artist:
        parts.insert(0, info.get("artist", ""))
    return " | ".join(p.strip() for p in parts if p and p.strip())


def build_dataset(
    annotations_path: str | Path,
    clip_info_path: str | Path,
    n_tags: int = 50,
    use_artist: bool = False,
) -> tuple[list[str], list[dict]]:
    """Assemble Task 1 records.

    Returns (tag_vocabulary, records) where each record carries
    clip_id, artist (for grouping), text, mp3_path and a `labels` set
    restricted to the tag vocabulary. Clips with no positive tag in the
    vocabulary are dropped, matching standard MTAT practice.
    """
    records = load_labeled_records(
        annotations_path, clip_info_path, use_artist=use_artist
    )
    vocab = top_labels(records, n_tags)
    return vocab, project_vocabulary(records, vocab)


def load_labeled_records(
    annotations_path: str | Path,
    clip_info_path: str | Path,
    use_artist: bool = False,
) -> list[dict]:
    """Load all canonical labeled clips without selecting a top-N vocabulary."""
    raw_tags, records = load_annotations(annotations_path)
    info = load_clip_info(clip_info_path)
    tag_map = canonical_tag_map(raw_tags)

    out = []
    for r in records:
        meta = info.get(r["clip_id"])
        if meta is None:
            continue
        labels = {tag_map[t] for t in r["tags"] if t in tag_map}
        if not labels:
            continue
        out.append(
            {
                "clip_id": r["clip_id"],
                "artist": meta["artist"],
                "text": build_text(meta, use_artist=use_artist),
                "mp3_path": r["mp3_path"],
                "labels": labels,
            }
        )
    return out


def top_labels(records: list[dict], n: int = 50) -> list[str]:
    """Most frequent canonical labels from the records supplied by the caller."""
    counts: collections.Counter = collections.Counter()
    for record in records:
        counts.update(set(record["labels"]))
    return [tag for tag, _ in counts.most_common(n)]


def project_vocabulary(records: list[dict], vocab: list[str]) -> list[dict]:
    """Copy records whose labels intersect a preselected training vocabulary."""
    vocab_set = set(vocab)
    out = []
    for record in records:
        labels = set(record["labels"]) & vocab_set
        if not labels:
            continue
        projected = dict(record)
        projected["labels"] = labels
        out.append(projected)
    return out


def labels_to_vector(labels: set[str], vocab: list[str]) -> list[int]:
    """Multi-hot encode a label set against the tag vocabulary."""
    index = {t: i for i, t in enumerate(vocab)}
    vec = [0] * len(vocab)
    for t in labels:
        if t in index:
            vec[index[t]] = 1
    return vec


def load_official_splits(splits_dir: str | Path) -> dict[str, set[str]]:
    """Read the official MTT train/val/test clip ids.

    Provided for comparability with published numbers ONLY. These splits leak
    45 artists between train and test, covering 61.6% of test clips, so they
    overstate generalisation. Prefer `splits.artist_grouped_split`.
    """
    out = {}
    for name in ("train", "val", "test"):
        path = Path(splits_dir) / f"{name}_gt_mtt.tsv"
        ids = set()
        with open(path) as f:
            for line in f:
                clip_id = line.split("\t")[0].strip()
                if clip_id:
                    ids.add(clip_id)
        out[name] = ids
    return out


def artist_overlap(splits: dict[str, list[dict]]) -> dict[str, int]:
    """Count artists shared between splits — 0 everywhere means no leakage."""
    artists = {k: {r["artist"] for r in v} for k, v in splits.items()}
    return {
        "train_test": len(artists.get("train", set()) & artists.get("test", set())),
        "train_val": len(artists.get("train", set()) & artists.get("val", set())),
        "val_test": len(artists.get("val", set()) & artists.get("test", set())),
    }
