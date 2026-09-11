# Datasets

Raw datasets are not committed because they are large and some require
separate access. Download them under `data/raw/` and validate them before
preprocessing.

## Download And Validate

The automated scripts are idempotent:

```bash
bash infra/fetch_datasets.sh
bash infra/extract_datasets.sh
python infra/validate_datasets.py --raw /data/raw
```

The scripts support the following datasets:

| Dataset | Use | Required data |
|---|---|---|
| FMA | Audio graphs and genre classification | FMA-small audio and metadata |
| MagnaTagATune | Multi-label tagging and fusion | Audio, annotations, and split files |
| DEAM | Optional valence/arousal regression | Audio, annotations, and metadata |
| MusicCaps | Caption-text experiments and retrieval | Caption CSV and audio clips |

## Dataset Notes

### FMA

The project uses the FMA-small subset selected from FMA metadata. GTZAN is not
used. The metadata file must be available at:

```text
data/raw/fma_metadata/tracks.csv
```

### MagnaTagATune

Place the extracted audio and annotation files under:

```text
data/raw/magnatagatune/
```

### DEAM

DEAM is optional for the main submission. The audio, valence/arousal
annotations, and metadata are needed to reproduce the masked multi-task
experiment.

### MusicCaps

MusicCaps provides captions but not a complete audio distribution. The project
uses a preprocessed mirror with 5,355 audio clips and the official caption CSV
with 5,521 rows. The official CSV provides the `is_audioset_eval` split flag and
the caption time windows.

The caption CSV is required for the official evaluation split. The graph builder
must receive both the audio mirror directory and the CSV; it does not silently
fall back to a random split.

## Generated Data

Graph caches are generated under a separate processed-data directory. The
repository includes representative graph samples under
`data/processed/graph_samples/` and split manifests under `data/splits/`.

Do not run full audio extraction in automated tests. The raw datasets are not
part of the repository and are too large for CI.
