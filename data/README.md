# Dataset Download Instructions

`infra/fetch_datasets.sh` automates everything below on a fresh GPU box and is
idempotent, and `infra/validate_datasets.py` checks each corpus against its
published shape before any preprocessing runs. The manual steps are documented
here so the sources are traceable, not because you have to follow them by hand.

All datasets are large and some are account-gated, so they are
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

**What this project actually used, and why it is not a YouTube scrape.**
MusicCaps ships captions but not audio, and scraping the 5,521 video IDs from a
rented GPU box does not work: YouTube blocks datacenter IP ranges. We therefore
read audio from a pre-scraped mirror on the Hugging Face Hub, which
`infra/fetch_datasets.sh` does automatically:

    mahendra0203/musiccaps_processed_full   # ~3.2 GB, 5,355 clips with audio

That yields **5,355** of the published 5,521 — the shortfall is dead or
region-locked videos, and `results/metrics_task4*.json` records the count for
every run rather than assuming the full set.

The official caption CSV (`musiccaps-public.csv`) is still needed separately: it
carries the `is_audioset_eval` flag that defines the official test partition,
and the `start_s` / `end_s` window each caption describes. Get it from
https://github.com/google-research-datasets/musiccaps and pass it as
`--musiccaps-csv`. Without it the build refuses to run rather than fall back to
a random split.

Scraping from YouTube with `yt-dlp` remains possible from a residential
connection if you would rather not use the mirror, but expect dead links and
rate-limiting.

## After downloading

Run the dataset-loading CLI (added in Week 2) to generate
`data/splits/*.json` via `src/splits.py`. Do not run graph/feature
extraction against `data/raw/` inside CI or automated tests — those files
are too large and are not present in the repo.
