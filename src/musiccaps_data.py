"""MusicCaps loading for Task 1's 'caption -> tag proxy' variant (PDF S4.1).

The PDF offers two alternatives for Task 1: MagnaTagATune top-50 tags, or a
MusicCaps caption -> tag proxy task. We build both, because each has a defect
the other does not:

- MagnaTagATune (see `mtat_data`): only 24.6% of clips have a distinct text
  input, capping any text-only model at Micro-F1 0.673.
- MusicCaps: every caption is unique, but 52.1% of aspect strings appear
  verbatim inside their own caption. A lexical-match baseline is therefore
  mandatory to show what BERT contributes beyond substring detection.

Aspects come from the annotator's own `aspect_list`, so this is a proxy task,
not ground-truth audio tagging. The report must say so.
"""

import ast
import collections
import csv
from pathlib import Path


def load_musiccaps(path: str | Path) -> list[dict]:
    """Read musiccaps-public.csv into records with parsed aspect lists."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                aspects = [a.strip() for a in ast.literal_eval(row["aspect_list"])]
            except (ValueError, SyntaxError):
                continue
            out.append(
                {
                    "ytid": row["ytid"],
                    "caption": row["caption"],
                    "aspects": {a for a in aspects if a},
                    "is_eval": row.get("is_audioset_eval", "").strip().lower()
                    in ("true", "1"),
                }
            )
    return out


def top_aspects(records: list[dict], n: int = 50, min_count: int = 100) -> list[str]:
    """Most frequent aspects, subject to a minimum clip count.

    13,219 distinct aspects exist but only 80 reach 100 clips, so an
    unfiltered vocabulary would be mostly unlearnable singletons.
    """
    counts: collections.Counter = collections.Counter()
    for r in records:
        for a in r["aspects"]:
            counts[a] += 1
    return [a for a, c in counts.most_common(n) if c >= min_count]


def strip_aspects_from_caption(caption: str, aspects: set[str]) -> str:
    """Remove aspect strings that occur verbatim in the caption.

    Used for the 'hard' variant of the proxy task. Without this, a model can
    reach high F1 by substring detection alone, which measures nothing about
    music understanding.
    """
    out = caption
    for a in sorted(aspects, key=len, reverse=True):
        idx = out.lower().find(a.lower())
        while idx != -1:
            out = out[:idx] + out[idx + len(a):]
            idx = out.lower().find(a.lower())
    return " ".join(out.split())


def build_dataset(
    path: str | Path,
    n_aspects: int = 50,
    min_count: int = 100,
    strip_leakage: bool = False,
) -> tuple[list[str], list[dict]]:
    """Assemble Task 1 MusicCaps records.

    With `strip_leakage=True` the aspect words are deleted from the caption,
    forcing the model to infer a label from surrounding context rather than
    detect its literal presence.
    """
    records = load_musiccaps(path)
    vocab = top_aspects(records, n_aspects, min_count)
    vocab_set = set(vocab)

    out = []
    for r in records:
        labels = r["aspects"] & vocab_set
        if not labels:
            continue
        text = (
            strip_aspects_from_caption(r["caption"], labels)
            if strip_leakage
            else r["caption"]
        )
        out.append(
            {
                "clip_id": r["ytid"],
                "artist": r["ytid"],  # no artist metadata; group by clip
                "text": text,
                "labels": labels,
                "is_eval": r["is_eval"],
            }
        )
    return vocab, out


def lexical_match_predict(text: str, vocab: list[str]) -> set[str]:
    """Baseline: predict a tag iff its literal string is in the text.

    This is the control that says how much of the score is substring
    detection rather than language understanding.
    """
    low = text.lower()
    return {t for t in vocab if t.lower() in low}


def clip_windows(path: str | Path) -> dict[str, tuple[int, int]]:
    """{ytid: (start_s, end_s)} from musiccaps-public.csv.

    Every caption describes one specific ten-second window, and that window is
    usually not at 0:00 — the first row of the corpus starts at 30 s. A rating
    study that plays each video from its beginning therefore asks listeners
    about audio the caption never described, and produces numbers that look
    fine and mean nothing.
    """
    out: dict[str, tuple[int, int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["ytid"]] = (int(float(row["start_s"])), int(float(row["end_s"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out
