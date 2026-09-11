# GNN-BERT Music Context Understanding

Tanvir Rahman, Student ID 22241134

CSE715 Neural Networks and Fuzzy Systems, BRAC University

This project combines a BERT text encoder with graph neural networks for music
tagging, genre classification, cross-modal context understanding, and
audio-text retrieval.

## Submission Files

- [Final report](report/final_report.pdf)
- [Experiment metrics](results/metrics.json)
- [Demo notebook](notebooks/demo_context.ipynb)
- [Dataset analysis notebook](notebooks/eda.ipynb)
- [Design specification](docs/design-spec.md)
- [Progress summary](PROGRESS.md)

## Results

All values below are taken from the committed experiment artifacts.

### Task 1: BERT Tagging

| Dataset and input | Micro-F1 | Macro-F1 |
|---|---:|---:|
| MagnaTagATune metadata text | 0.275 | 0.192 |
| MusicCaps captions | 0.565 | 0.531 |
| MusicCaps captions with aspect words removed | 0.488 | 0.438 |

The raw-caption MusicCaps result is below the lexical baseline on Macro-F1.
The stripped-caption experiment is the less-leaky language-understanding
comparison.

### Task 2: Graph Classification

FMA-small genre classification uses eight classes and artist-disjoint splits.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| PCA + MLP | 0.379 | 0.371 |
| GraphSAGE | 0.396 | 0.401 |
| GAT | 0.433 | 0.426 |
| CNN on mel-spectrogram | 0.483 | 0.479 |

Both segment-similarity and chord-transition graphs are implemented and
evaluated.

### Task 3: GNN-BERT Fusion

The best reported configuration uses a separate learning rate for non-BERT
parameters.

| Model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---:|---:|---:|
| BERT-only | 0.171 | 0.306 | 0.168 |
| GNN-only | 0.123 | 0.124 | 0.084 |
| Early concatenation | 0.168 | 0.314 | 0.182 |
| Cross-attention | 0.192 | 0.324 | 0.176 |
| Cross-attention, non-BERT learning rate 1e-3 | 0.246 | 0.415 | 0.248 |

The five-seed paired comparison of the two learning-rate settings reports a
Macro-F1 improvement of `0.0674 +/- 0.0226` with `p = 0.0026`. The changed
learning rate applies to all non-BERT parameters, not only the graph encoder.

### Task 4: Contrastive Retrieval

MusicCaps uses the official AudioSet evaluation split with a gallery of 2,773
clips.

| Run and direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Shared learning rate, caption to audio | 0.0029 | 0.0108 | 0.0220 |
| Shared learning rate, audio to caption | 0.0022 | 0.0126 | 0.0224 |
| Separate graph learning rate, caption to audio | 0.0058 | 0.0173 | 0.0339 |
| Separate graph learning rate, audio to caption | 0.0054 | 0.0184 | 0.0317 |

Zero-shot tagging from captions reaches Micro-F1 `0.098`, compared with
Micro-F1 `0.657` for the supervised Task 3 comparator on the same test IDs and
label vocabulary.

The human evaluation aggregate is 174 ratings from six listeners, with a mean
score of `1.79 +/- 1.36` on a five-point scale. Participant-level responses are
not included in this public repository.

## Setup

Python 3.12 is recommended for the pinned audio and machine-learning stack.
Raw datasets are not committed. See [data/README.md](data/README.md) for
sources and download instructions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest -q
```

The current suite contains 165 tests. A clean local run completed with 165
tests passed and two non-failing warnings.

## Reproduction

Validate datasets before preprocessing:

```bash
python infra/validate_datasets.py --raw /data/raw
```

Build graph caches and run the experiments with the scripts in `src/`:

```bash
python src/build_graphs.py --dataset fma --out /data/processed \
  --audio-root /data/raw/fma_small \
  --tracks-csv /data/raw/fma_metadata/tracks.csv
python src/train.py --dataset mtat --save-checkpoint
python src/train_gnn.py --cache /data/processed/fma_small_graphs.pt --model all
python src/train_fusion.py --mtat-cache /data/processed/mtat_graphs.pt --mode all
python src/train_contrastive.py --cache /data/processed/musiccaps_graphs.pt
```

Aggregate metrics and regenerate report values:

```bash
python src/aggregate_metrics.py
python src/make_report_numbers.py
cd report
pdflatex final_report.tex
pdflatex final_report.tex
```

Training requires a GPU and the raw datasets. Evaluation from committed
artifacts runs on CPU.

## Repository Layout

```text
src/                 preprocessing, models, training, evaluation, reporting
tests/               automated tests
data/processed/      committed graph examples
data/splits/         train, validation, and test manifests
results/             metrics, plots, and retrieval examples
notebooks/           analysis and demonstration notebooks
report/              report source, generated values, and PDF
```

## Limitations

- Most experiments use one seed; the Task 3 learning-rate comparison uses five
  paired seeds.
- Retrieval scores are low and are reported without overclaiming.
- Raw datasets and large model checkpoints are not stored in Git.
- Full training reproduction requires the datasets and GPU compute.
