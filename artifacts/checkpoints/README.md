# Model checkpoints

`task1_musiccaps.pt` is downloadable. The other two are regenerable.

## Download and verify

```bash
curl -L --retry 3 -C - -o artifacts/checkpoints/task1_musiccaps.pt \
  https://0ixwpqni1gjr0ksc.public.blob.vercel-storage.com/checkpoints/task1_musiccaps.pt

shasum -a 256 -c <(grep task1_musiccaps.pt artifacts/checkpoints/SHA256SUMS)
```

**Check the checksum.** A first download of this file truncated at 153 MB of
265,648,054 and returned HTTP 200 while doing it; `--retry 3 -C -` is in the
command above for that reason. The file is 253 MB, so a partial transfer is a
realistic outcome, not a hypothetical one.

Then, from a clean checkout:

```bash
python -m src.infer --checkpoint artifacts/checkpoints/task1_musiccaps.pt \
    --text "A middle eastern folk song with an oud and hand percussion."
```

This was verified end to end: fresh download, checksum matched, inference ran.

## The other two

Not hosted, to keep one file inside the storage budget. They are deterministic
given the seed recorded in each one's own `config` block:

```bash
python -m src.train --dataset musiccaps --epochs 6 --seed 42 --strip-leakage --save-checkpoint
python -m src.train --dataset mtat      --epochs 6 --seed 42 --save-checkpoint
```

Roughly 10 minutes each on one mid-range GPU after `infra/fetch_datasets.sh`.
Cross-hardware reproduction moves the third decimal; the report says so where it
matters.

## Why they are not in git

253 MB each, over GitHub's 100 MB per-file limit, and together two orders of
magnitude larger than the rest of the repository. `SHA256SUMS` is committed, so
a checkpoint obtained by any route can be tied to the run that produced it.

`notebooks/demo_context.ipynb` calls the same inference path and **commits its
output**, so the end-to-end example is inspectable even without downloading
anything. Without the file it prints where to get it rather than failing.
