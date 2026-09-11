# GNN-BERT Music Context Project — Design & Tracking Plan

> **Written before implementation, on 2026-08-15, and deliberately not
> rewritten.** It records what was planned and why, which is the point of
> keeping it; several decisions changed once the data was in hand — MusicCaps
> audio came from a pre-scraped mirror rather than a YouTube scrape, and the
> official MagnaTagATune split was rejected after a leakage audit. What the
> project actually did is in `README.md`, `results/metrics.json` and the
> report; this is the plan it started from.


**Course:** CSE715 (Neural Networks) — 15% of course grade
**Deadline:** 11 September 2026 (corrected 2026-08-16 — the 2 October date in the PDF was a misreading; treat 11 Sep as authoritative)
**Author:** Tanvir Rahman (solo submission)
**Spec source:** `Project/CSE425_Project_GNN_BERT_Music_Context.pdf` (instructor: Moin Mostakim)

## 1. Scope

Full four-task roadmap, committed (not staged as stretch goals):

1. **Task 1 (Easy, 18 pts):** BERT multi-label tag classifier.
2. **Task 2 (Medium, 22 pts):** GraphSAGE/GAT on chord/segment graphs vs. CNN baseline.
3. **Task 3 (Hard, 22 pts):** GNN–BERT cross-attention fusion, multi-task loss including DEAM valence/arousal regression, ablations, t-SNE, case studies.
4. **Task 4 (Advanced, 18 pts):** Contrastive dual-encoder (InfoNCE) retrieval on MusicCaps.
5. **Report + GitHub (20 pts):** reproducible code, 6–10 page NeurIPS/IEEE/ICML-style report, demo notebook.

Total addressable: 100/100 rubric marks.

## 2. Datasets

| Dataset | Role | Used in |
|---|---|---|
| FMA-medium | Audio + genre/tags, chord/segment graph source | Tasks 2, 3 |
| MagnaTagATune | 188 multi-label tags | Tasks 1, 3 |
| DEAM | Valence/arousal (continuous, 0.5s resolution) | Task 3 (multi-task loss) |
| MusicCaps | 5,521 natural-language captions | Task 4 |

No artist leakage across train/val/test splits. Use official dataset splits where available (FMA, MagnaTagATune); standard train/val partition for DEAM.

**FMA subsets.** Only `fma_medium` is downloaded. The PDF's Task 2 deliverable names "GTZAN or FMA-small"; we satisfy it with the **FMA-small subset of the FMA-medium download**, selected via the `set/subset` column in `fma_metadata/tracks.csv` (values: `small` / `medium` / `large`). GTZAN is not used, and no extra download is needed.

### 2.1 Handling DEAM's disjointness from the tag corpora (Task 3)

FMA/MagnaTagATune and DEAM are **disjoint corpora** — no track carries both tag labels and valence/arousal targets. The PDF's loss `L = L_tags + α‖v−v̂‖² + β‖a−â‖²` therefore cannot have both terms non-zero on the same batch (the PDF hedges: "when available"). Design decision:

- **Shared encoder, two heads.** The fused representation `z` feeds a tag head (sigmoid multi-label → BCE) and an emotion head (2-dim linear → MSE on valence/arousal).
- **Masked, alternating batches.** Each batch is drawn from a single corpus and carries a per-sample mask; the inactive loss term is zeroed for that batch. Batches alternate between the tag corpus and DEAM, sampled proportionally to corpus size.
- **Text input for DEAM samples.** DEAM ships per-song metadata (including genre); a metadata-derived pseudo-caption is fed through BERT so the fusion path is exercised on DEAM batches rather than receiving an empty string.
- **Documented fallback.** If DEAM metadata proves unusable as text, attach the emotion head to the graph representation `g` alone instead of the fused `z`, and state this deviation explicitly in the report. Do not silently drop the emotion term.

## 3. Repository

**Course submission repository** — `https://github.com/tanviranindo/cse715-gnn-bert-music-context` (standalone GitHub repo for submission, matching the spec's prescribed structure verbatim):

```
cse715-gnn-bert-music-context/
  README.md
  requirements.txt
  config.yaml
  data/{raw,processed,splits}/
  notebooks/{eda.ipynb,demo_context.ipynb}
  src/{audio_features,graph_builder,bert_encoder,gnn_model,fusion_model,contrastive,train,evaluate}.py
  results/{metrics.json,plots/,retrieval_examples/}
  report/final_report.pdf
```

Maintained locally at `Project/cse715-gnn-bert-music-context-public/`.

## 4. Timeline (Aug 15 – Sep 11, 2026)

Dates revised 2026-08-16: the deadline is 11 September, not 2 October. Same
seven-week structure and same task order as originally planned — each week is
simply shorter, since the window is ~26 days rather than ~47.

| Week | Dates | Focus |
|---|---|---|
| 1 | Aug 15–18 | Repo scaffold, data download (all 4 datasets), preprocessing pipeline (resample, mel/chroma extraction, segmentation) |
| 2 | Aug 19–22 | Task 1: BERT tag classifier + B1 (random)/B4 (PCA+MLP) baselines |
| 3 | Aug 23–27 | Task 2: chord/segment graph construction + GraphSAGE/GAT + CNN baseline (B2) |
| 4 | Aug 28–31 | Task 3: cross-attention fusion + DEAM multi-task loss |
| 5 | Sep 1–4 | Task 3 ablations (BERT-only/GNN-only/concat/cross-attention), t-SNE, case studies |
| 6 | Sep 5–8 | Task 4: contrastive dual-encoder + retrieval eval (R@1/5/10) |
| 7 | Sep 9–10 | Final report (NeurIPS/IEEE template), demo notebook, repo cleanup |
| — | Sep 11 | Buffer / submission only — no new work planned |

## 5. Tracking mechanism

Two layers, kept in sync:

1. **`PROGRESS.md`** in the new repo's root. One section per week (matching the table above). Each entry updated at the end of a work session: what was completed, links to artifacts (plots, metrics, commits) proving the claim, and open blockers. Committed to git — the commit history itself becomes a progress log.
2. **GitHub Issues + Project board** on the new repo. One issue per task (Task 1–4, Report), broken into sub-issues per concrete deliverable (e.g., "Task 2: graph construction script", "Task 2: CNN baseline"). Kanban board (Backlog / This Week / In Progress / Done). GitHub Milestones map 1:1 to the 7 weeks above, due-dated to week boundaries.

No "done" without evidence: a PROGRESS.md entry or issue close must link to the artifact (metrics.json entry, plot file, notebook cell) that substantiates it — not just "implemented."

## 6. Verification per task

- **Task 1:** Macro-F1/Micro-F1 curves vs. epoch logged; baselines B1 (random) + B4 (PCA+MLP) compared; 5 example predictions saved with optional attention viz.
- **Task 2:** Genre classification F1 on the FMA-small subset (see §2) vs. CNN baseline (B2); graph builder produces ≥20 example `.pt`/`.json` graphs (hard submission requirement).
- **Task 3:** Full ablation table (BERT-only, GNN-only, early concat, cross-attention); t-SNE of fused representation colored by genre/mood; 3 case studies (graph path + caption/lyric alignment); DEAM MAE/R² reported, with the masked alternating-batch scheme from §2.1 documented in the report.
- **Task 4:** Retrieval table (Caption→Audio and Audio→Caption R@1/5/10) on MusicCaps test split; 10 qualitative retrieval examples; zero-shot tag prediction from captions compared against Task 3's supervised model.
- All quantitative results land in `results/metrics.json`, plots in `results/plots/`, retrieval examples in `results/retrieval_examples/`.

## 7. Baselines (rubric requires ≥2)

- B1: Majority-class / random tag predictor
- B2: CNN on mel-spectrogram (no graph, no text)
- B3: BERT-only (= Task 1 output, reused)
- B4: PCA + MLP on hand-crafted audio features (optional but planned, used alongside B1 for Task 1)

## 8. Out of scope / explicitly deferred

- Nothing deferred — all four tasks are committed per user decision (2026-08-15). If Week 5 buffer is consumed by Task 3 overruns, Task 4 scope is the first thing to renegotiate (not silently dropped — flag to user).

## 9. Known schedule risks

- **MusicCaps audio is not distributed.** The dataset ships captions plus YouTube video IDs; audio must be scraped with `yt-dlp`, a meaningful fraction of links are dead, and rate-limiting is likely. Task 4 (Week 6) depends on it, so the scrape starts on day 1 of Week 1 and runs in the background. Record the actual retrieved clip count in `PROGRESS.md` — do not assume all 5,521.
- **FMA-medium is ~22GB.** Start this download early in Week 1 alongside the MusicCaps scrape.
- **Disk.** ~84 GB free as of 2026-08-16 against a need of ~58 GB across all four datasets. It fits, but delete each archive right after extracting.
