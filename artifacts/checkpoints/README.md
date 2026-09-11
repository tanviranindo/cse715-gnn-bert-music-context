# Model Checkpoints

Large checkpoints are not stored in Git. The repository includes checksums and
commands for reproducing the required models.

## Available Checkpoint

The Task 1 MusicCaps checkpoint can be downloaded from the project artifact
store:

```bash
curl -L --retry 3 -C - \
  -o artifacts/checkpoints/task1_musiccaps.pt \
  https://0ixwpqni1gjr0ksc.public.blob.vercel-storage.com/checkpoints/task1_musiccaps.pt
shasum -a 256 -c <(grep task1_musiccaps.pt artifacts/checkpoints/SHA256SUMS)
```

Run CPU inference with:

```bash
python -m src.infer \
  --checkpoint artifacts/checkpoints/task1_musiccaps.pt \
  --text "A middle eastern folk song with an oud and hand percussion."
```

## Reproduce Other Task 1 Checkpoints

After downloading and validating the datasets:

```bash
python -m src.train --dataset musiccaps --epochs 6 \
  --seed 42 --strip-leakage --save-checkpoint
python -m src.train --dataset mtat --epochs 6 \
  --seed 42 --save-checkpoint
```

The other task checkpoints are omitted because they exceed normal Git hosting
file limits. Training and evaluation artifacts remain committed.
