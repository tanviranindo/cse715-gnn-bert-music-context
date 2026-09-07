# GNN-BERT Music Context Understanding

CSE715 Neural Networks course project. Hybrid BERT + GNN system for
understanding musical context (genre, mood, emotion) from audio structure
and text (lyrics/tags/captions). All four tasks in the specification are
implemented, run and reported.

**Report:** [`report/final_report.pdf`](report/final_report.pdf) (8 pages)
· **All headline numbers:** [`results/metrics.json`](results/metrics.json)
· **Week-by-week log:** [`PROGRESS.md`](PROGRESS.md)
· **Demo:** [`notebooks/demo_context.ipynb`](notebooks/demo_context.ipynb)

The full design spec lives in the parent CSE715 notes repo at
`../../docs/superpowers/specs/2026-08-15-gnn-bert-music-context-design.md`.
This is an independent git repo that happens to sit inside that notes repo;
the path is gitignored there, so the two never interfere.

## Results at a glance

Every number below is read from `results/metrics.json`, which is regenerated
from the per-run artifacts by `src/aggregate_metrics.py`.

**Task 1 — BERT multi-label tagging** (best-validation checkpoint)

| Variant | Micro-F1 | Macro-F1 | Best baseline (Micro/Macro) |
|---|---|---|---|
| MagnaTagATune, metadata text, artist-grouped | 0.275 | 0.192 | B1 random-prevalence 0.103 / 0.058 |
| MusicCaps caption → aspect (official eval split) | 0.565 | 0.531 | B5 lexical match 0.540 / **0.550** |
| MusicCaps, aspect words stripped from caption | 0.488 | 0.438 | B5 lexical match 0.000 / 0.000 |

On raw captions BERT loses to substring matching on Macro-F1; reported as-is.
The stripped variant is the honest measure of language understanding.

**Task 2 — GNN on segment graphs** (FMA-small, 8 genres, chance = 0.125, zero artist leakage)

| Model | Params | Accuracy | Macro-F1 |
|---|---|---|---|
| B4 PCA+MLP | — | 0.379 | 0.371 |
| GraphSAGE | 49,416 | 0.396 | 0.401 |
| GAT | 26,120 | 0.433 | 0.426 |
| **B2 CNN on mel-spectrogram** | 241,992 | **0.483** | **0.479** |

The CNN wins. GAT reaches 89% of its Macro-F1 with 9× fewer parameters.

The spec (§3.3) names two graph structures, so both were trained on identical
splits and labels:

| Graph input | GraphSAGE Macro-F1 | GAT Macro-F1 |
|---|---|---|
| Segment similarity | 0.401 | **0.426** |
| Chord transition | 0.316 | 0.271 |

Chord graphs carry real signal (2.5× chance) but lose 0.110 Macro-F1 to segment
similarity: a 30 s clip collapses to ~21 nodes over a 24-triad vocabulary and
the node feature is the chord's own pitch-class template, so all timbre is
discarded before the GNN sees anything.

**Task 3 — GNN-BERT fusion** (MagnaTagATune top-50, artist-grouped, best-val)

| Mode | Macro-F1 | Micro-F1 | AUC-PR | Attn. entropy vs uniform |
|---|---|---|---|---|
| BERT-only (B3) | 0.171 | 0.306 | 0.168 | — |
| GNN-only | 0.123 | 0.124 | 0.084 | — |
| Early concat | 0.168 | 0.314 | 0.182 | — |
| Cross-attention | 0.193 | 0.325 | 0.177 | 1.000 (collapsed) |
| + freeze BERT | 0.118 | 0.315 | 0.197 | 0.079 |
| **+ separate GNN lr 1e-3** | **0.246** | **0.415** | **0.248** | 0.695 |

The learning-rate effect is the project's central positive result and is the
only one confirmed over five paired seeds: +0.067 ± 0.023 Macro-F1, p = 0.0026,
positive on every seed. Seed variance (±0.023) exceeds several of the other
between-mode margins, so those should not be read as firm rankings.

**Task 4 — Contrastive retrieval** (MusicCaps official AudioSet eval, gallery 2,773)

| Run | Direction | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| Shared lr | Caption → audio | 0.0029 | 0.0108 | 0.0220 |
| Shared lr | Audio → caption | 0.0022 | 0.0126 | 0.0224 |
| **Separate graph lr** | Caption → audio | **0.0058** | **0.0173** | **0.0339** |
| **Separate graph lr** | Audio → caption | 0.0054 | 0.0184 | 0.0317 |
| Random | — | 0.0004 | — | 0.0036 |

The Task 3 learning-rate finding transfers: it doubles R@1 and lifts R@10 to
9.4× random, with median rank improving from 527 to 400 of 2,773. The absolute
numbers stay low because the task is data-limited, and that is measured rather
than asserted — holding the gallery fixed and subsampling only the training
pairs, R@10 grows log-linearly in paired examples (R² = 0.914, +0.0099 per
doubling), putting a usable R@10 of 0.10 about 100× beyond MusicCaps' official
split. Zero-shot tagging from captions reaches Micro-F1 0.107 against Task 1's
supervised 0.488 on the same corpus.

## Setup

Python 3.12 (the ML stack does not yet have reliable wheels for 3.13+):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `numba` / `llvmlite` / `numpy` deliberately — a newer
pairing segfaults inside librosa's `chroma_stft`. See the comment in that file.

A GPU is needed to reproduce training (we used single rented RTX 4060 Ti /
4080 SUPER / V100 boxes; see `infra/vast-connect.md`). The notebooks and all
evaluation run on CPU in seconds from the committed artifacts.

## Tests

```bash
pytest -q     # 109 passed
```

The root `conftest.py` is what puts the repo root on `sys.path` so that
`from src.… import …` resolves — don't delete it.

## Datasets

See [`data/README.md`](data/README.md) for download instructions (FMA,
MagnaTagATune, DEAM, MusicCaps — all downloaded manually, not committed to
git). `infra/fetch_datasets.sh` and `infra/extract_datasets.sh` automate this
on a fresh GPU box and are idempotent.

## Reproducing the results

Preprocessing writes a graph cache per corpus; every training script then
reads that cache, so audio is touched exactly once.

```bash
# 0. Validate every corpus against its published shape BEFORE preprocessing.
#    Checks file counts, decodability, CSV columns, tag vocabulary, row counts
#    and split flags; exits non-zero and names replacement mirrors on failure.
python infra/validate_datasets.py --raw /data/raw

# 1. Build segment + chord graphs (once per dataset; --n-examples commits samples)
# --out is the output DIRECTORY; each run writes <dataset>_graphs.pt inside it.
python src/build_graphs.py --dataset fma  --out /data/processed --n-examples 25 \
       --audio-root /data/raw/fma_small --tracks-csv /data/raw/fma_metadata/tracks.csv
python src/build_graphs.py --dataset mtat --out /data/processed --n-examples 25 \
       --audio-root /data/raw/magnatagatune/audio --mtat-dir /data/raw/magnatagatune --no-mel
python src/build_graphs.py --dataset musiccaps --out /data/processed \
       --musiccaps-dir /data/raw/musiccaps/shards \
       --musiccaps-csv /data/raw/musiccaps/musiccaps-public.csv
python src/build_graphs.py --dataset deam --out /data/processed \
       --audio-root /data/raw/deam/audio/MEMD_audio

# 2. Task 1 — BERT tag classifier (three variants + B1 baselines)
python src/train.py --dataset mtat      --save-checkpoint
python src/train.py --dataset musiccaps
python src/train.py --dataset musiccaps --strip-leakage

# 3. Task 2 — GraphSAGE / GAT / CNN baseline, then B4
python src/train_gnn.py --cache /data/processed/fma_small_graphs.pt --model all
python src/train_b4.py  --cache /data/processed/fma_small_graphs.pt
# the chord-transition graph as the GNN input (PDF S3.3), same splits and labels
python src/train_gnn.py --cache /data/processed/fma_small_graphs.pt --graph chord --model all

# 4. Task 3 — full ablation, then the two follow-up arms
python src/train_fusion.py --mtat-cache /data/processed/mtat_graphs.pt --mode all
python src/train_fusion.py --mtat-cache /data/processed/mtat_graphs.pt --mode crossattn \
       --freeze-bert --out-dir results/_freeze
python src/train_fusion.py --mtat-cache /data/processed/mtat_graphs.pt --mode crossattn \
       --gnn-lr 1e-3 --out-dir results/_gnnlr
# multi-task valence/arousal (optional term in the spec)
python src/train_fusion.py --mtat-cache /data/processed/mtat_graphs.pt \
       --deam-cache /data/processed/deam_graphs.pt --mode crossattn

# 5. Task 3 seed sweep (5 paired seeds per arm)
for s in 42 1 2 3 4; do
  python src/train_fusion.py --mode crossattn --seed $s --out-dir results/_sweep/shared_s$s
  python src/train_fusion.py --mode crossattn --seed $s --gnn-lr 1e-3 --out-dir results/_sweep/gnnlr_s$s
done
python src/aggregate_sweep.py results/_sweep

# 6. Task 4 — contrastive dual encoder + retrieval + zero-shot tagging
python src/train_contrastive.py --cache /data/processed/musiccaps_graphs.pt
# does the Task 3 learning-rate finding transfer to the dual encoder?
python src/train_contrastive.py --cache /data/processed/musiccaps_graphs.pt \
       --gnn-lr 1e-3 --batch-size 128 --metrics-suffix _gnnlr
# is retrieval limited by the model or by 2.2k training pairs?
for f in 0.25 0.50 0.75; do
  python src/train_contrastive.py --cache /data/processed/musiccaps_graphs.pt \
    --gnn-lr 1e-3 --batch-size 128 --train-frac $f --metrics-suffix _frac${f/./}
done

# 7. Analysis and figures
python src/analyze_fusion.py --checkpoint results/task3_crossattn.pt   # t-SNE + case studies
python src/attention_viz.py --dataset mtat                            # Task 1 attention heatmap
python src/make_plots.py                                              # training curves
python src/analyze_scale.py                                           # Task 4 data-scale fit + plot
python src/aggregate_metrics.py                                       # -> results/metrics.json
python src/make_report_numbers.py                                     # -> report/_numbers.json

# Split manifests for data/splits/ (deterministic; needs the graph caches)
python src/dump_splits.py --cache /data/processed/fma_small_graphs.pt --dataset fma_small

# One end-to-end inference example, CPU, from the Task 1 checkpoint
python -m src.infer --text "A gentle piano ballad with soft female vocals."

# 8. Task 4 human evaluation (needs 5 real listeners; see below)
python src/human_eval.py build     # -> results/human_eval/rating_sheet.html
#   ... send the sheet to >=5 listeners, collect their JSON downloads into
#       results/human_eval/ratings/, then:
python src/human_eval.py score     # -> results/metrics_task4_human.json
```

Rebuild the report. `report/_numbers.json` is a build product of
`aggregate_metrics.py` + `make_report_numbers.py`, so the paper's reference
values cannot be older than the runs (run both before recompiling; two
`pdflatex` passes resolve the cross-references):

```bash
cd report && pdflatex final_report.tex && pdflatex final_report.tex
```

## Layout

```
src/            audio_features, graph_builder, bert_encoder, gnn_model,
                fusion_model, contrastive, train*, evaluate, analyze_fusion,
                analyze_scale, attention_viz, human_eval, infer, dump_splits,
                make_plots, aggregate_sweep, aggregate_metrics,
                make_report_numbers
tests/          14 modules / 109 tests, run with pytest
notebooks/      eda.ipynb (dataset findings), demo_context.ipynb (end-to-end demo)
results/        metrics.json (aggregate) + per-task metrics, plots/,
                retrieval_examples/, case studies, t-SNE
data/processed/ 155 committed example graphs (.pt + .json) across FMA and MTAT
data/splits/    train/val/test manifests for all three corpora
report/         final_report.tex / .pdf, _numbers.json
infra/          dataset fetch/extract/validate scripts, GPU box notes
```

## Known gaps

- **Task 4 human evaluation** (spec §6: ≥5 listeners rating retrieved clips
  1–5) has not been run. The instrument is built and ready —
  `python src/human_eval.py build` produces the rating sheet and `score`
  aggregates the returned files — but it needs five real listeners, so the
  result cannot be produced from this repository alone.
- The rating sheet in `results/human_eval/` now embeds all 30 retrieved clips
  as playable YouTube segments, so the study is a listening study. Its examples
  come from `task4_examples_gnnlr.json`, a same-configuration replicate run on
  different hardware: R@10 is identical to the reported run at 0.0339, while
  R@1 and R@5 differ in the third decimal (0.0054/0.0177 vs 0.0058/0.0173).
  The tables in the report and `metrics.json` are the original run throughout,
  which is also what the data-scale ablation is internally consistent with.
- No model checkpoint is committed (253 MB each), so a fresh clone can read
  every result but can only run inference if you supply the weights.
- Single seed everywhere except the Task 3 learning-rate comparison (5 paired
  seeds) and the Task 1 curves. Measured seed spread is ±0.023 Macro-F1, which
  exceeds several of the Task 3 ablation margins.
- Task 1 reports Macro/Micro-F1 but not AUC-PR; the §6 graph-coherence score,
  which the spec marks optional, is not implemented.
