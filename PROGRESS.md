# Progress Log

This is a concise submission-facing record of the project milestones. Detailed
development notes, infrastructure records, and operational logs remain private.

## Scope

- [x] BERT multi-label tag classification.
- [x] GraphSAGE and GAT music-structure classification.
- [x] CNN and PCA+MLP baselines.
- [x] GNN-BERT fusion with ablations and cross-attention.
- [x] Optional DEAM valence/arousal implementation and evaluation artifact.
- [x] MusicCaps graph-text contrastive retrieval.
- [x] Zero-shot tag prediction and retrieval-scale analysis.
- [x] Human evaluation instrument and aggregated evaluation result.
- [x] Required report, plots, graph samples, split manifests, and notebooks.

## Dataset And Preprocessing

- [x] Audio preprocessing implements 22.05 kHz resampling, log-mel/chroma
      extraction, normalization, and fixed-duration segmentation.
- [x] Segment-similarity and chord-transition graph builders are implemented.
- [x] FMA, MagnaTagATune, MusicCaps, and DEAM data loaders are implemented.
- [x] Dataset validation scripts check expected metadata structure and counts.
- [x] Split manifests are committed under `data/splits/`.
- [x] Vocabulary selection and split generation use training-derived provenance.
- [x] Artist-disjoint checks are included where source metadata permits them.

## Task Status

### Task 1: BERT Tagging

- [x] MagnaTagATune top-50 tag experiment.
- [x] MusicCaps caption-to-aspect experiment.
- [x] Leakage-stripped MusicCaps control.
- [x] Best-validation checkpoint selection.
- [x] Macro-F1 and Micro-F1 reporting.
- [x] Example predictions and attention visualizations.

### Task 2: Graph Classification

- [x] GraphSAGE and GAT implementations.
- [x] Segment-similarity graph experiment.
- [x] Chord-transition graph experiment.
- [x] CNN mel-spectrogram comparison.
- [x] PCA+MLP baseline.
- [x] More than 20 committed `.pt`/`.json` graph samples.

### Task 3: GNN-BERT Fusion

- [x] BERT-only, GNN-only, early-concat, and cross-attention ablations.
- [x] Separate non-BERT learning-rate experiment.
- [x] Five-seed paired follow-up for the learning-rate comparison.
- [x] AUC-PR, Macro-F1, and Micro-F1 reporting.
- [x] t-SNE views and qualitative case studies.
- [x] DEAM multi-task loss implementation and documented limitations.

### Task 4: Contrastive Retrieval

- [x] GNN-BERT dual encoder with InfoNCE.
- [x] Caption-to-audio and audio-to-caption R@1/R@5/R@10.
- [x] Qualitative retrieval examples.
- [x] Zero-shot tagging against the supervised Task 3 comparator.
- [x] Training-scale analysis.
- [x] Human-evaluation aggregation.

## Submission Artifacts

- [x] `README.md` with setup, results, and reproduction commands.
- [x] `notebooks/eda.ipynb` and `notebooks/demo_context.ipynb`.
- [x] `results/metrics.json` and per-task metric files.
- [x] `results/plots/` with training, attention, t-SNE, and scale plots.
- [x] `data/processed/graph_samples/` and `data/splits/`.
- [x] `report/final_report.pdf` within the required page range.
- [x] Automated tests for preprocessing, models, evaluation, provenance, and
      report generation.

## Limitations

- Raw datasets and large model checkpoints are not committed.
- GPU retraining is required to reproduce the full training results.
- Task 4 retrieval quality remains modest and is reported without overclaiming.
- Some corrected protocols require refreshed experiments before their results
  can replace older artifacts.
- Participant-level human-evaluation responses are kept private; only the
  aggregate result is included in the submission materials.
