# Model checkpoints

These are the weights that produced the Task 1 numbers in the report. They are
**not committed**: each is 253 MB, over GitHub's 100 MB per-file limit, and the
three together are larger than the rest of the repository by two orders of
magnitude.

`SHA256SUMS` is committed, so a checkpoint obtained by any route can be verified
against the run that produced it.

## Reproducing them

Each is deterministic given the seed recorded in its own `config` block:

```bash
python -m src.train --dataset musiccaps --epochs 6 --seed 42 --save-checkpoint
python -m src.train --dataset musiccaps --epochs 6 --seed 42 --strip-leakage --save-checkpoint
python -m src.train --dataset mtat      --epochs 6 --seed 42 --save-checkpoint
```

Roughly 10 minutes each on one mid-range GPU, after the datasets are fetched
(`infra/fetch_datasets.sh`). Cross-hardware reproduction moves the third decimal;
the report says so where it matters.

## Running inference without training

```bash
python -m src.infer --checkpoint artifacts/checkpoints/task1_musiccaps.pt \
    --text "A gentle piano ballad with soft female vocals."
```

The checkpoint carries its own tag vocabulary and validation-selected threshold,
so nothing about the setup is re-specified at inference time.

`notebooks/demo_context.ipynb` runs the same call and **commits its output**, so
the end-to-end example is inspectable even without the weights. If the file is
missing the notebook prints where to get it rather than failing.

## Asking the author for the weights

If you are grading this and would rather not retrain: request the three files,
verify against `SHA256SUMS`, and place them in this directory.
