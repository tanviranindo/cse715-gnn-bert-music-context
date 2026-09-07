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

**Status: DONE (run 2026-08-30).** Both PDF-sanctioned dataset variants built
and evaluated. Artefacts: `results/metrics_task1_*.json`,
`results/examples_task1_*.json`, logs in `artifacts/logs/`.

| variant | Micro-F1 | Macro-F1 | best baseline |
|---|---|---|---|
| MagnaTagATune, metadata text, artist-grouped split | 0.258 | 0.182 | random-prevalence 0.111 / 0.061 |
| MusicCaps caption -> aspect | **0.687** | **0.646** | lexical match 0.582 / 0.560 |
| MusicCaps, aspect words stripped from caption | 0.608 | 0.551 | lexical match 0.000 / 0.000 |

Reading these honestly:

- **MusicCaps stripped is the headline number.** 0.608 Micro-F1 against a
  lexical baseline of exactly 0.000 — the aspect words are deleted from the
  input, so every point is inference from surrounding context.
- On raw captions BERT scores 0.687 but a pure substring matcher already gets
  0.582, so **only ~0.105 of that is language understanding**. Reporting the
  raw number alone would have overstated the result by ~5x.
- MagnaTagATune is much harder (0.258) and that is expected, not a failure:
  the text is track-level metadata while clips are 29 s segments, so 96.8% of
  clips share their input with another clip. Oracle ceiling for ANY text-only
  model on this corpus is Micro-F1 0.673 / Macro-F1 0.575.
- Splits are artist-grouped with **0 leakage**. The official MTT split shares
  45 artists between train and test (61.6% of test clips) and was rejected;
  the PDF asks for both official splits and no artist leakage, which conflict.

**This is the quantitative case for Task 3.** The text branch is capped by
inputs it cannot distinguish; the GNN branch sees each clip's own audio. The
fusion is not decoration, it is the fix for a measured ceiling.

Baseline B4 (PCA+MLP on audio features) is still outstanding — it needs the
audio seed, so it is folded into Week 3 alongside Task 2's CNN baseline.

## Week 3 (Aug 23-27): Task 2 — GNN on music structure graphs

**Status: DONE (run 2026-08-30).** Artefacts: `results/metrics_task2.json`,
25 example graphs in `data/processed/graph_samples/`, logs in `artifacts/logs/`.

Dataset: FMA-small, 7,994 of 8,000 tracks usable (6 corrupt mp3s dropped),
8 genres balanced at 1,000 each. FMA's official splits were used and their
artist-disjointness was **verified, not assumed**: 0 leakage in all three
pairings, unlike MagnaTagATune's.

| model | params | test acc | macro-F1 |
|---|---|---|---|
| GraphSAGE | 49,416 | 0.414 | 0.407 |
| **GAT** | **26,120** | 0.438 | **0.432** |
| CNN baseline (B2) | 241,992 | **0.443** | 0.408 |

Chance is 0.125 for 8 genres, so all three are ~3.5x chance. The honest
reading: **the CNN edges out the GNNs on accuracy, but GAT matches it on
macro-F1 with 9x fewer parameters.** Structure is competitive with, not
superior to, treating the spectrogram as an image — which is exactly the
comparison the PDF asks for, and Task 3 exists because neither branch alone
is enough.

Per-class F1 (GAT) ranges from Hip-Hop 0.664 down to Experimental 0.250;
"Experimental" is a catch-all genre and behaves like one.

**A bug worth recording.** The first preprocessing pass used `tau=0.9` on raw
MFCC features and produced graphs with 19.4 nodes and average degree 18.15 —
essentially complete graphs, on which message passing degenerates into global
mean pooling. Cause: MFCC[0] encodes loudness and dominates the vector, so
raw cosine similarity between segments has median 0.9971 and 5th percentile
0.9509. Fix: z-score features within each track before computing similarity
(median falls to -0.078, p90 to 0.331) and use tau=0.35, giving average
degree 3.2 with no isolated nodes. Because the cache stores node features
separately from edges, rebuilding took 10 s rather than a 60 min
re-extraction. A regression test now covers it.

Baseline B4 (PCA+MLP on hand-crafted audio features) remains outstanding.

## Week 4 (Aug 28-31): Task 3 — GNN-BERT fusion

**Status: ablation DONE, emotion term OUTSTANDING (run 2026-08-30).**
Artefacts: `results/metrics_task3.json`, `results/case_studies_task3.json`,
`results/tsne_task3.json`, `results/plots/tsne_task3.png`.

Dataset: MagnaTagATune top-50, 21,315 clips, artist-grouped splits
(train 14,972 / val 3,129 / test 3,214), 0 leakage.

| mode | params | macro-F1 | micro-F1 | AUC-PR |
|---|---|---|---|---|
| BERT-only | 66,402,868 | **0.1864** | 0.3110 | 0.1697 |
| GNN-only | 31,796 | 0.1141 | 0.2358 | 0.1026 |
| early concat | 66,434,612 | 0.1748 | 0.3176 | **0.1836** |
| cross-attention | 66,834,740 | 0.1817 | **0.3230** | 0.1779 |

**The honest result: fusion does not clearly beat BERT alone.** Against
BERT-only, cross-attention is +0.012 micro-F1 and +0.008 AUC-PR but
**-0.005 macro-F1**. Early concat is +0.014 AUC-PR but -0.012 macro-F1.
These are small margins and the project's central claim is not demonstrated
by them.

**Why — measured, not guessed.** The cross-attention has collapsed to
uniform averaging. Across the case studies the max/min attention weight
ratio over the top-8 tokens is 1.03-1.11, and entropy sits at 65-88% of the
uniform maximum. The graph vector is not discriminating between caption
tokens, so `A H_text` degenerates to a mean over the text and the model
reduces to something close to early concat.

Two plausible causes, both testable:
1. **The query is weak.** GNN-only reaches only 0.114 macro-F1 on 19-node
   segment graphs, so `g` carries little signal to query with.
2. **The text tower dominates.** 66.4M BERT parameters against 31.8k GNN
   parameters, trained jointly at a single lr=2e-5 tuned for BERT. The graph
   branch is plausibly undertrained rather than uninformative.

Next steps before writing this up as final: train with `--freeze-bert` so the
graph branch must contribute, try a higher separate lr for the GNN, and
enrich node features. Do not report the current numbers as a refutation of
fusion — they are a result about *this* configuration.

**Outstanding:** the DEAM valence/arousal auxiliary term. The loader,
masked multi-task loss and graph builder are all implemented and tested
(`src/deam_data.py`, `fusion_model.multitask_loss`); only the run is missing.
DEAM downloaded at 196 KB/s on that host, and waiting ~70 min on a $0.46/hr
box was poor economics, so it is deferred to a cheap box. The PDF marks
valence/arousal "(optional)" for Task 3.

## Week 5 (Sep 1-4): Task 3 — ablations, t-SNE, case studies

(to be filled in when Week 5 starts — see issue #12)

## Week 6 (Sep 5-8): Task 4 — contrastive retrieval

(to be filled in when Week 6 starts — see issue #13)

## Week 7 (Sep 9-10): Report, demo notebook, cleanup

**Report: DONE (2026-08-31).** `report/final_report.pdf`, 6 pages, two-column,
built from `report/final_report.tex` with `pdflatex`. Every number is generated
programmatically from `results/*.json` via `report/_numbers.json`, so the paper
cannot drift from the artefacts.

Structure: Introduction, Related Work, Method (all four task formulations with
the specification's equations), Experimental Setup, Results, Engineering
Findings, Discussion and Limitations, Conclusion, 10 references.
7 tables, 2 figures.

The paper is framed around the negative result rather than hiding it: fusion
does not reliably beat its stronger unimodal branch, and the contribution is
the diagnosis. It leads with the four measured findings --- the 0.673 text
ceiling from duplicated inputs, the 61.6% artist leak in the official split,
the attention collapse at 65-88% of uniform entropy, and the multi-task
trade-off.

Also closes a gap: the specification requires "Macro-F1 / Micro-F1 curves vs.
training epochs" for Task 1, which had data but no plot. Now
`results/plots/training_curves.png`, which additionally shows Task 4's InfoNCE
loss falling monotonically while validation R@10 peaks at epoch 6 and degrades.

**Notebooks: DONE (2026-08-31).** Both required by the spec's repo structure,
both executed end-to-end with `nbconvert` rather than assumed to work.

- `notebooks/eda.ipynb` (17 cells) reproduces the three dataset findings from
  scratch: the 61.6% artist leak, the 0.673 oracle ceiling, and the 52.1%
  verbatim-aspect overlap, plus a faithful demonstration of the graph
  saturation problem. Downloads ~35 MB of metadata on demand; needs no audio.
- `notebooks/demo_context.ipynb` (24 cells) walks all four tasks from the
  committed artefacts, including the attention-collapse measurement and a
  cross-task summary against baselines. Runs in seconds, no GPU.

Executing them caught two real defects: cell sources lacked trailing newlines
so every cell collapsed to one line, and a loop variable `c` shadowed the Task 3
results dict several cells later.

**Still outstanding for submission:**
- The two Task 3 follow-up runs (--freeze-bert, separate --gnn-lr) that the
  report names as the first experiments to try next. Two attempts were lost to
  an oversubscribed host (load average 19.9 with our workers starved to 28s of
  CPU); the code is committed and tested.

## Buffer (Sep 11): Submission only
### Task 3 multi-task with DEAM (run 2026-08-31)

The auxiliary valence/arousal term now runs, using masked alternating batches
per spec S2.1: 21,315 MagnaTagATune clips plus 1,802 DEAM songs, 1,304 of
them emotion-supervised in train. Artefacts:
`results/metrics_task3_multitask.json`.

| cross-attention | tags only | + DEAM | delta |
|---|---|---|---|
| macro-F1 | 0.1817 | 0.1160 | **-0.0657** |
| micro-F1 | 0.3230 | 0.2787 | -0.0443 |
| AUC-PR | 0.1779 | 0.1324 | -0.0455 |

| emotion head | MAE (z) | MAE (1-9 scale) | R2 |
|---|---|---|---|
| valence | 0.710 | 0.83 | **+0.181** |
| arousal | 0.721 | 0.92 | **+0.085** |

**The emotion heads work but the multi-task loss costs tag performance.**
Both R2 values are positive, so the model beats predicting the mean on
held-out songs, and MAE of 0.83-0.92 on a 1-9 scale is usable. But tag
macro-F1 drops by 0.066 — a 36% relative fall. The MagnaTagATune training
data is identical between the two runs, so this is attributable to the
auxiliary term competing for shared capacity rather than to a data change.

Report both. The PDF marks valence/arousal "(optional)" for Task 3, and the
honest framing is that the auxiliary term buys emotion regression at a
measurable cost to tagging, not that it is free.


## Correction pass (2026-09-07): official splits, checkpoints, and a segfault

Instance `50108358` (1x RTX 4060 Ti, $0.0979/hr, `ssh6.vast.ai:28358`). Every
number below was read back off that box after the run, not predicted.

### What was actually wrong

1. **The "FMA multiprocessing hang" was a segmentation fault.** `librosa`'s
   `chroma_stft` -> `estimate_tuning` -> `piptrack` -> `_parabolic_interpolation`
   is a numba `guvectorize` kernel, and the image's numba 0.67.0 / llvmlite
   0.49.0 against numpy 2.1.2 segfaulted inside it on the *first* track.
   `mp.Pool` silently respawns a worker that dies mid-task and never delivers
   that task's result, so the parent blocked forever in `imap_unordered` — which
   is why every worker showed ~0s CPU in `futex_wait` and it looked like a
   deadlock. Two earlier "hangs" (MTAT 10500/21318, FMA 4000/8000) were the same
   bug. `maxtasksperchild` made it worse, not better.
   - Fixed by pinning `numba>=0.61,<0.62` / `llvmlite>=0.44,<0.45` / `numpy<2.2`
     and replacing `mp.Pool` with `ProcessPoolExecutor`, which raises
     `BrokenProcessPool` immediately instead of stalling.
   - Throughput 8.8 -> **16.2 tracks/s**; FMA-small finished **7994/8000 in
     14.4 min** having previously never passed 4000.
2. **MusicCaps ignored its own official split.** The metadata ships
   `is_audioset_eval`; Tasks 1 and 4 were using random splits. The flag is now
   honoured and carried into the graph cache (verified: 2773 eval / 2582 train
   in `musiccaps_graphs.pt`). Neither task will silently fall back to a random
   split any more — Task 1 raises, Task 4 records `split_kind`.
3. **The B1 random baseline leaked test statistics** by estimating per-tag
   prevalence from test labels. Now training labels only.
4. **Task 1 and Task 3 scored the final epoch, not the best validation
   checkpoint.** Material, not cosmetic: Task 3's GNN-only peaks at epoch 1 and
   has decayed to ~0 validation Macro-F1 by epoch 8. Correcting it changed the
   Task 3 ranking.
5. **`--n-examples 0` clobbered `graph_samples/index.json`**, emptying the
   manifest for 50 committed samples. Guarded.
6. **`infra/fetch_datasets.sh`'s DEAM Zenodo fallback points at record
   1188976, which is RAVDESS**, not DEAM — a latent bug that would have failed
   silently.

### Verified results (all official-split unless noted)

| Task 1 (best-val checkpoint) | Micro-F1 | Macro-F1 |
|---|---|---|
| MagnaTagATune (artist-grouped, epoch 5) | 0.2751 | 0.1923 |
| MusicCaps raw (official eval, epoch 6) | 0.5647 | 0.5315 |
| — B5 lexical match | 0.5402 | **0.5499** |
| MusicCaps stripped (official eval) | 0.4877 | 0.4385 |

MusicCaps splits moved from a random 3496/749/750 to the official
**2054/362/2579**. BERT now *loses to substring matching on Macro-F1*; this is
reported as-is.

| Task 2 (FMA-small, 6394/800/800, 0 artist leakage) | Acc | Macro-F1 |
|---|---|---|
| B4 PCA+MLP (new) | 0.3787 | 0.3708 |
| GraphSAGE | 0.3962 | 0.4006 |
| GAT | 0.4325 | 0.4259 |
| CNN (B2) | **0.4825** | **0.4794** |

| Task 3 (MTAT, best-val) | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| BERT-only | 0.1709 | 0.3057 | 0.1679 |
| GNN-only | 0.1228 | 0.1243 | 0.0840 |
| early concat | 0.1681 | 0.3139 | 0.1820 |
| cross-attention | 0.1925 | 0.3245 | 0.1765 |
| + freeze BERT | 0.1176 | 0.3152 | 0.1967 |
| **+ separate GNN lr 1e-3** | **0.2457** | **0.4152** | **0.2484** |

**The separate GNN learning rate is the largest single effect in the project.**
Attention entropy vs uniform: shared lr **1.000** (total collapse, peak 1.03x),
freeze-BERT 0.079 (peak 10.02x), GNN-lr **0.695** (peak 4.19x). The change that
fixes the metric is the one that un-collapses the attention — the mechanism the
earlier draft predicted. Single seed; variance not measured.

| Task 4 (official eval, gallery 2773) | R@1 | R@5 | R@10 |
|---|---|---|---|
| caption -> audio | 0.0029 | 0.0108 | 0.0220 |
| audio -> caption | 0.0022 | 0.0126 | 0.0224 |
| random | 0.0004 | — | 0.0036 |

Median rank 527/2773. Absolute R@K fell versus the old random-split run because
the gallery is 5.2x larger (2773 vs 536) and training pairs halved (2195 vs
4284); the two are not comparable. Zero-shot tagging micro-F1 0.1070.

### Not done, and why

- **Task 3 DEAM multi-task was NOT re-run.** `cvml.unige.ch` now times out
  entirely; a HuggingFace mirror (`herrjyj/herrjyj-deam-assets`, 1343.2 MB,
  matching our recorded 1.3 GB) supplied the audio, but no surviving source has
  `metadata.zip`, whose genre/artist/title fields build DEAM's pseudo-captions.
  Re-running with empty captions would change the setup rather than reproduce
  it. The existing numbers stay, explicitly labelled in the report as
  final-epoch methodology.
- **Task 4 human evaluation (spec S6, >=5 listeners rating 1-5)** has never been
  done and cannot be fabricated. Still an open gap against the PDF.
- Single-seed throughout; no seed sweep on the GNN-lr result.

### Artifact state

- Report regenerated: **7 pages** (spec allows 6-10), tables and conclusions
  rewritten for the corrected numbers.
- Both notebooks re-executed end-to-end with `nbconvert` against the new
  metrics; `results/plots/training_curves.png` now has a committed generator
  (`src/make_plots.py`) instead of an uncommitted ad-hoc script.
- t-SNE and case studies regenerated from the new cross-attention checkpoint.
- 50 graph samples (25 FMA across 8 genres + 25 MTAT), index rebuilt.
- **Not submitted, not merged, not published.**

## Seed sweep (2026-09-07, instance 50118002)

Tesla V100 16GB, 72 cores, $0.0877/hr. The Task 3 learning-rate result was the
project's central positive claim and rested on one seed, so both arms were
re-run over five seeds (42, 1, 2, 3, 4) on a single machine. The arms share
seeds, so the per-seed difference is paired and tested with a paired t-test.

| cross-attention | shared lr 2e-5 | separate GNN lr 1e-3 | paired delta | p |
|---|---|---|---|---|
| macro-F1 | 0.1724 ± 0.0228 | 0.2398 ± 0.0197 | **+0.0674 ± 0.0226** | 0.0026 |
| micro-F1 | 0.3347 ± 0.0243 | 0.4155 ± 0.0275 | **+0.0808 ± 0.0184** | 0.0006 |
| AUC-PR | 0.1813 ± 0.0148 | 0.2598 ± 0.0180 | **+0.0786 ± 0.0104** | 0.0001 |

Per-seed macro-F1 deltas: +0.0716, +0.0401, +0.0766, +0.0977, +0.0510 —
**positive on every seed**, ~3x the seed-to-seed spread.

Two things this also established, both worth keeping:

- **Seed variance (±0.023 macro-F1) exceeds several between-ablation margins in
  the main Task 3 table.** The BERT-only vs early-concat vs cross-attention
  differences at a shared learning rate should not be read as firm rankings.
  Only the learning-rate effect is comfortably outside the noise.
- **Seed 42 reproduced as 0.2537 here vs 0.2457 on the RTX 4060 Ti box** — same
  code, same seed, different GPU and library build. The third decimal is not
  portable across machines; the committed single-seed report numbers are from
  the original box and are, if anything, slightly conservative versus the
  sweep means.

Artifacts: `results/_sweep/{shared,gnnlr}_s{42,1,2,3,4}/metrics_task3.json`,
aggregated by `src/aggregate_sweep.py` (paired t-test implemented in-file and
validated against known t critical values: t=2.776/df=4 -> p=0.0500,
t=4.604/df=4 -> p=0.0100).

Instance destroyed after pulling artifacts. Still not submitted.
