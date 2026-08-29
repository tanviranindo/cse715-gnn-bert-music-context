# Progress Log

**Deadline: 11 September 2026** (corrected 2026-08-16, was 2 October).
Same week structure and task order; the weeks are just shorter.

Tracks weekly progress against `docs/superpowers/specs/2026-08-15-gnn-bert-music-context-design.md`
(spec lives in the parent CSE715 notes repo at `../../docs/superpowers/specs/`). Each entry: what was done,
links to proof (commits/plots/metrics), and open blockers.

## Week 1 (Aug 15-18): Repo scaffold + preprocessing foundations

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
- [x] **GPU training environment stood up (2026-08-24).** Rented a Vast.ai
      on-demand instance (RTX 4090, 24GB VRAM, 205GB disk, ~$0.15-0.36/hr
      depending on host) since local machine (M2 Pro, 16GB RAM, 78GB free
      disk) and Deepnote's free tier (CPU-only, ephemeral browser terminal)
      were both insufficient. Repo cloned, `requirements.txt` installed
      (torch 2.13.0, torch-geometric 2.8.0, transformers 5.15.1), all 9
      Week 1 tests re-verified passing on the instance, CUDA confirmed
      available (`torch.cuda.is_available() == True`). Work runs inside
      `tmux` sessions on the instance so long jobs survive SSH disconnects.
- [x] **Datasets downloaded and validated (2026-08-24), including FMA-medium integrity check (unzip -tq, no errors):**
  - FMA-medium: `fma_medium.zip` (~22GB) + `fma_metadata.zip` (342MB) —
    zip integrity verified (`unzip -tq`, no errors).
  - MagnaTagATune: all 3 split-zip parts + `annotations_final.csv` (21MB)
    downloaded. Parts 2-3 correctly show as raw split-archive data (not
    corruption) — merge with `zip -F` before extracting.
  - DEAM: `deam_audio.zip` (1.3GB) — zip integrity verified, no errors.
  - MusicCaps: **captions CSV validated at exactly 5,521 rows** (full
    expected count), correct columns
    (`ytid,start_s,end_s,audioset_positive_labels,aspect_list,caption,author_id,is_balanced_subset,is_audioset_eval`).
    **Audio: 5,355 of 5,521 clips (97%) — NOT scraped via yt-dlp.**
    Direct YouTube scraping from the Vast.ai datacenter IP was 100%
    blocked by YouTube's bot detection ("Sign in to confirm you're not a
    bot") even with `--extractor-args player_client=android/tv`
    workarounds. Used a pre-scraped mirror instead:
    `mahendra0203/musiccaps_processed_full` on Hugging Face (7 parquet
    shards, 3.4GB, columns `audio,caption,youtube_id,start_time,end_time,aspect_list`),
    row count validated via `pyarrow` (765 × 7 = 5,355 exactly). This is
    the actual retrieved clip count — do not assume all 5,521 exist per
    the spec's own caution about this.

Full suite at end of Week 1: **9 passed** (re-verified on GPU instance).

Blockers: project board auth scope (above, still open).


### Compute environment rebuilt on a persistent volume (2026-08-29)

The Aug 24 setup was lost when that instance was destroyed. Rebuilt on
**Vast.ai machine 143795** with a **persistent local volume** so this cannot
recur:

- **Volume `49151852`** (`22241134_gnnbert_data`), 60 GB, mounted at `/data`.
  Survives instance destroy — **proven by experiment**, see `infra/vast-connect.md`.
  Pinned to machine 143795. Costs $0.133/day, billed 24/7.

  **The volume must be created standalone and attached with `--link-volume`.**
  Both `--env '-v V.<id>:/data'` (what the Vast docs show) and `--create-volume`
  silently destroy the volume with the instance; each cost us 17 GB on 2026-08-29.
- **Instance**: offer `45573262`, 1x RTX 4080 SUPER 16 GB, 64 threads, 62 GB RAM,
  **$0.143/hr all-in**. Chosen via `vastai search offers --storage 90`, which
  exposes per-host storage price — invisible in the web UI and varying 3-4x.
  The RTX 5090 first rented cost $0.499/hr and would have given ~12 h of runway
  on the remaining credit; this gives ~47 h.
- **Stack**: image ships Python 3.10.12 + torch 2.5.1+cu121, *not* the
  `requirements.txt` pins (3.12 / torch 2.13.0). librosa 0.11.0,
  transformers 5.16.1, torch-geometric 2.8.0. **All 9 Week-1 tests pass**
  unchanged, so `requirements.txt` should be relaxed to match rather than
  rebuilding torch on every rental.
- **Datasets on `/data/raw` (17 GB of 40 GB), all verified**:

  | dataset | count | expected |
  |---|---|---|
  | FMA-small | 8,000 mp3 | ~8,000 |
  | MagnaTagATune | 25,863 mp3 + 3 MTT split TSVs | ~25,863 |
  | DEAM | 1,802 mp3 | ~1,802 |
  | MusicCaps | 5,355 rows | 5,355 (matches Aug 24) |

  **FMA-small, not FMA-medium.** `data/README.md` confirms Task 2 uses only the
  FMA-small subset; medium's extra 15 GB does not fit the volume alongside
  extracted features. Revisit only if a task genuinely needs medium.

Scripts: `infra/fetch_datasets.sh`, `infra/extract_datasets.sh` (both idempotent),
connection details and re-rent command in `infra/vast-connect.md`.

Remaining credit after setup: **$6.86**.

## Week 2 (Aug 19-22): Task 1 — BERT tag classifier

(to be filled in when Week 2 starts — see issue #9)

## Week 3 (Aug 23-27): Task 2 — GNN on music structure graphs

(to be filled in when Week 3 starts — see issue #10)

## Week 4 (Aug 28-31): Task 3 — GNN-BERT fusion (part 1)

(to be filled in when Week 4 starts — see issue #11)

## Week 5 (Sep 1-4): Task 3 — ablations, t-SNE, case studies

(to be filled in when Week 5 starts — see issue #12)

## Week 6 (Sep 5-8): Task 4 — contrastive retrieval

(to be filled in when Week 6 starts — see issue #13)

## Week 7 (Sep 9-10): Report, demo notebook, cleanup

(to be filled in when Week 7 starts — see issue #14)

## Buffer (Sep 11): Submission only
