# Dataset Download Instructions

All datasets are downloaded manually (large, some account-gated) and are
git-ignored under `data/raw/`.

## FMA-medium

1. Download `fma_medium.zip` (~22GB) and `fma_metadata.zip` from
   https://github.com/mdeff/fma
2. Extract into `data/raw/fma_medium/` and `data/raw/fma_metadata/`

Task 2's genre classification uses the **FMA-small subset** of this
download, selected via the `set/subset` column in
`fma_metadata/tracks.csv` (values: `small` / `medium` / `large`). GTZAN is
not used and no extra download is needed.

## MagnaTagATune

1. Download the three `.zip` parts and `annotations_final.csv` from
   https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
2. Extract audio into `data/raw/magnatagatune/audio/`, put the CSV at
   `data/raw/magnatagatune/annotations_final.csv`

## DEAM

1. Download audio + annotations from
   https://cvml.unige.ch/databases/DEAM/
2. Extract into `data/raw/deam/`

DEAM is disjoint from FMA/MagnaTagATune — no track carries both tag labels
and valence/arousal targets. Task 3 handles this with masked alternating
batches; see spec §2.1 before wiring up the multi-task loss.

## MusicCaps

1. Clone https://github.com/google-research-datasets/musiccaps for the
   caption CSV (`musiccaps-public.csv`)
2. Audio clips must be downloaded from YouTube per the CSV's video IDs
   (use `yt-dlp`); save into `data/raw/musiccaps/audio/`
3. **Start this early.** A meaningful fraction of the 5,521 YouTube links
   are dead and rate-limiting is likely, so the scrape takes far longer
   than its size suggests. Task 4 (Week 6) depends on it.
4. Record the actual retrieved clip count in `PROGRESS.md` — do not assume
   all 5,521 exist.

## After downloading

Run the dataset-loading CLI (added in Week 2) to generate
`data/splits/*.json` via `src/splits.py`. Do not run graph/feature
extraction against `data/raw/` inside CI or automated tests — those files
are too large and are not present in the repo.
