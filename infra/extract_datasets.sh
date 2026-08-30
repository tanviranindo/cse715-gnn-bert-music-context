#!/usr/bin/env bash
# Verify zip integrity, extract, then delete the archives to conserve volume.
# Archives are deleted ONLY after their extraction is verified non-empty.
set -u
RAW=/data/raw
LOG=/data/logs/extract.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== verify integrity ==="
for z in "$RAW/fma_small.zip" "$RAW/fma_metadata.zip" "$RAW/deam/DEAM_audio.zip" "$RAW/deam/DEAM_Annotations.zip"; do
  [ -s "$z" ] || { say "MISSING $z"; continue; }
  if unzip -tq "$z" >/dev/null 2>&1; then say "OK    $(basename "$z")"; else say "CORRUPT $(basename "$z")"; fi
done

say "=== FMA ==="
if [ ! -d "$RAW/fma_small" ]; then
  unzip -q "$RAW/fma_small.zip" -d "$RAW/" 2>>"$LOG"
  n=$(find "$RAW/fma_small" -name '*.mp3' 2>/dev/null | wc -l); say "fma_small: $n mp3 (expect ~8000)"
  [ "$n" -gt 7000 ] && rm -f "$RAW/fma_small.zip" && say "deleted fma_small.zip"
fi
if [ ! -d "$RAW/fma_metadata" ]; then
  unzip -q "$RAW/fma_metadata.zip" -d "$RAW/" 2>>"$LOG"
  [ -f "$RAW/fma_metadata/tracks.csv" ] && rm -f "$RAW/fma_metadata.zip" && say "deleted fma_metadata.zip"
fi

say "=== MagnaTagATune ==="
M="$RAW/magnatagatune"
if [ ! -d "$M/audio" ] && [ -s "$M/mp3.zip" ]; then
  if unzip -tq "$M/mp3.zip" >/dev/null 2>&1; then
    say "OK    mp3.zip"
    mkdir -p "$M/audio"; unzip -q "$M/mp3.zip" -d "$M/audio" 2>>"$LOG"
    n=$(find "$M/audio" -name '*.mp3' | wc -l); say "magnatagatune: $n mp3 (expect ~25863)"
    [ "$n" -gt 20000 ] && rm -f "$M/mp3.zip" && say "deleted mp3.zip"
  else
    say "CORRUPT mp3.zip"
  fi
fi

say "=== DEAM ==="
D="$RAW/deam"
[ ! -d "$D/audio" ] && mkdir -p "$D/audio" && unzip -q "$D/DEAM_audio.zip" -d "$D/audio" 2>>"$LOG"
[ ! -d "$D/annotations" ] && mkdir -p "$D/annotations" && unzip -q "$D/DEAM_Annotations.zip" -d "$D/annotations" 2>>"$LOG"
if [ ! -d "$D/metadata" ] && [ -s "$D/metadata.zip" ]; then
  unzip -q "$D/metadata.zip" -d "$D/" 2>>"$LOG"
  say "deam metadata: $(ls "$D"/metadata/metadata_*.csv 2>/dev/null | wc -l) yearly files"
fi
n=$(find "$D/audio" -name '*.mp3' 2>/dev/null | wc -l); say "deam: $n mp3 (expect ~1802)"
[ "$n" -gt 1500 ] && rm -f "$D/DEAM_audio.zip" && say "deleted DEAM_audio.zip"

say "=== MusicCaps row count ==="
/venv/main/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import glob, pyarrow.parquet as pq
fs = sorted(glob.glob("/data/raw/musiccaps/**/*.parquet", recursive=True))
tot = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
print(f"musiccaps: {len(fs)} shards, {tot} rows (PROGRESS.md records 5355)")
PY

say "=== FINAL ==="
du -sh "$RAW"/* 2>/dev/null | tee -a "$LOG"
du -sh /data | tee -a "$LOG"
say "=== DONE ==="
