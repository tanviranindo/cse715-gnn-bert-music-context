"""Add MusicCaps start/end offsets to a saved retrieval-examples file.

The Task 4 examples are written from the graph cache, which carries the YouTube
id but not the clip window. Without the window the rating sheet plays each video
from 0:00, so raters judge audio the caption never described. This joins the
offsets back on by ytid.

    python -m src.enrich_examples --examples results/retrieval_examples/task4_examples_gnnlr.json \
        --csv data/raw/musiccaps/musiccaps-public.csv
"""

import argparse
import json
import pathlib

from src import musiccaps_data


def enrich(examples: list[dict], windows: dict[str, tuple[int, int]]) -> tuple[list[dict], int, int]:
    """Returns (examples, filled, missing). Clips with no known window are left
    without offsets so the sheet can refuse them rather than imply 0:00."""
    filled = missing = 0
    for ex in examples:
        for clip in ex["top3"]:
            win = windows.get(clip.get("ytid", ""))
            if win:
                clip["start_s"], clip["end_s"] = win
                filled += 1
            else:
                missing += 1
    return examples, filled, missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--examples", required=True)
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    path = pathlib.Path(args.examples)
    examples = json.loads(path.read_text())
    windows = musiccaps_data.clip_windows(args.csv)
    examples, filled, missing = enrich(examples, windows)
    path.write_text(json.dumps(examples, indent=2))
    print("%s: %d clips given their window, %d without one" % (path, filled, missing))


if __name__ == "__main__":
    main()
