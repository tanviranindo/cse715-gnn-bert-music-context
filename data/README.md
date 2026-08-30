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

1. Download **three** files from https://cvml.unige.ch/databases/DEAM/
   - `DEAM_audio.zip` (1.3 GB)
   - `DEAM_Annotations.zip` (4.6 MB) — valence/arousal only
   - `metadata.zip` (345 KB) — **easy to miss, and Task 3 needs it**
2. Extract into `data/raw/deam/`

`DEAM_Annotations.zip` contains *no* genre, artist or title. Spec S2.1 needs
a metadata-derived pseudo-caption for the BERT branch, and that metadata
lives only in the separate `metadata.zip`. Without it, S2.1's documented
fallback applies (emotion head on the graph vector alone). With it, 1802/1802
songs get a usable caption and 1744 (96.8%) carry a genre.

Note the three yearly metadata files use different column names
(`song_id`/`Id`/`id`, `Song title`/`Track`/`title`, ...) and contain stray
tabs; `src/deam_data.load_metadata` normalises them.

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
