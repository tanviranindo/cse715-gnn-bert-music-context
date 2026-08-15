# Progress Log

Tracks weekly progress against `docs/superpowers/specs/2026-08-15-gnn-bert-music-context-design.md`
(spec lives in the CSE715 course notes repo). Each entry: what was done,
links to proof (commits/plots/metrics), and open blockers.

## Week 1 (Aug 15-21): Repo scaffold + preprocessing foundations

- [x] Private repo created and scaffolded — commit `8407ff6`, verified `isPrivate: true`
- [x] Python 3.12 venv created (`.venv/`). **Only Week 1 deps installed**
      (librosa 1.0.0, numpy 2.5.2, pytest 9.1.1) — torch, torch-geometric and
      transformers are listed in `requirements.txt` but not yet installed;
      install them at the start of Week 2. Python 3.14 was avoided
      deliberately: the ML stack lacks reliable wheels for it.
- [x] `src/audio_features.py` — resample, log-mel, chroma, segmentation.
      Commit `1fa39c7`, 5 tests passing (`tests/test_audio_features.py`).
- [x] `src/splits.py` — artist-grouped splitting, no artist leakage.
      Commit (see `git log`), 4 tests passing (`tests/test_splits.py`).
- [x] Dataset download instructions — `data/README.md`, commit `a231b83`
- [x] GitHub Milestones (7, one per week) and Issues (#1-#6, one per task) created
- [ ] **GitHub Project board — BLOCKED.** `gh` token lacks the `project`
      scope. Fix with `gh auth refresh -s project,read:project` (interactive
      browser flow), then `gh project create --owner @me --title "GNN-BERT Music Context"`
      and add Backlog / This Week / In Progress / Done columns in the web UI.
- [ ] **Datasets not yet downloaded.** Start the MusicCaps `yt-dlp` scrape
      and the ~22GB FMA-medium download first — both are long-running and
      Task 4 (Week 6) depends on the scrape. Record the actual MusicCaps clip
      count here when it finishes; do not assume all 5,521.

Full suite at end of Week 1: **9 passed**.

Blockers: project board auth scope (above); no dataset bytes on disk yet.

## Week 2 (Aug 21-28): Task 1 — BERT tag classifier

(to be filled in when Week 2 starts — see issue #1)

## Week 3 (Aug 28-Sep 4): Task 2 — GNN on music structure graphs

(to be filled in when Week 3 starts — see issue #2)

## Week 4 (Sep 4-11): Task 3 — GNN-BERT fusion (part 1)

(to be filled in when Week 4 starts — see issue #3)

## Week 5 (Sep 11-17): Task 3 — ablations, t-SNE, case studies

(to be filled in when Week 5 starts — see issue #4)

## Week 6 (Sep 17-24): Task 4 — contrastive retrieval

(to be filled in when Week 6 starts — see issue #5)

## Week 7 (Sep 24-29): Report, demo notebook, cleanup

(to be filled in when Week 7 starts — see issue #6)

## Buffer (Sep 29-Oct 2): Submission only
