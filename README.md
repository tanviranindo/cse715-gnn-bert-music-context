# GNN-BERT Music Context Understanding

CSE715 Neural Networks course project. Hybrid BERT + GNN system for
understanding musical context (genre, mood, emotion) from audio structure
and text (lyrics/tags/captions).

See `PROGRESS.md` for week-by-week status. The full design spec lives in the
course notes repo at `docs/superpowers/specs/2026-08-15-gnn-bert-music-context-design.md`.

## Setup

Python 3.12 (the ML stack does not yet have reliable wheels for 3.13+):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest -v
```

The root `conftest.py` is what puts the repo root on `sys.path` so that
`from src.… import …` resolves — don't delete it.

## Datasets

See `data/README.md` for download instructions (FMA-medium, MagnaTagATune,
DEAM, MusicCaps — all downloaded manually, not committed to git).
