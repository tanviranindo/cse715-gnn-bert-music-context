#!/usr/bin/env bash
# Download all four datasets into the persistent volume at /data/raw.
# Idempotent: skips anything already present. Safe to re-run.
# Run inside tmux:  tmux new -s dl "bash infra/fetch_datasets.sh"
set -u
RAW=/data/raw
LOG=/data/logs/fetch.log
mkdir -p "$RAW" /data/logs
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

get(){ # get <url> <dest>
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then say "SKIP  $(basename "$dest") (exists)"; return 0; fi
  say "GET   $(basename "$dest")"
  if wget -q --show-progress --progress=dot:giga -c -O "$dest.part" "$url" 2>>"$LOG"; then
    mv "$dest.part" "$dest"; say "OK    $(basename "$dest") $(du -h "$dest"|cut -f1)"
  else
    say "FAIL  $(basename "$dest")  <- $url"; rm -f "$dest.part"; return 1
  fi
}

say "=== FMA (small subset + metadata) ==="
# Task 2 uses only the FMA-small subset (see data/README.md); fma_medium's
# extra 15GB does not fit the 40GB volume and is not needed.
get https://os.unil.cloud.switch.ch/fma/fma_metadata.zip "$RAW/fma_metadata.zip"
get https://os.unil.cloud.switch.ch/fma/fma_small.zip    "$RAW/fma_small.zip"

say "=== MagnaTagATune (HF mirror) ==="
# The official mirror (mirg.city.ac.uk) rate-limits hard: it fell to 68 KB/s
# (ETA ~8h for the 3 split parts) on 2026-08-29. confit/magnatagatune carries
# the same corpus as a single mp3.zip plus the standard MTT splits.
mkdir -p "$RAW/magnatagatune"
if [ ! -d "$RAW/magnatagatune/audio" ] && [ ! -s "$RAW/magnatagatune/mp3.zip" ]; then
/venv/main/bin/python - <<'MTATPY' 2>&1 | tee -a "$LOG"
from huggingface_hub import hf_hub_download
import shutil, os
for f in ["mp3.zip","annotations_final.csv","clip_info_final.csv",
          "train_gt_mtt.tsv","val_gt_mtt.tsv","test_gt_mtt.tsv"]:
    try:
        p = hf_hub_download("confit/magnatagatune", f, repo_type="dataset")
        dst = os.path.join("/data/raw/magnatagatune", f)
        if not os.path.exists(dst): shutil.copy(p, dst)
        print("OK  ", f, "%.2f GB" % (os.path.getsize(dst)/1e9))
    except Exception as e:
        print("FAIL", f, type(e).__name__, e)
MTATPY
else
  say "SKIP  magnatagatune (already present)"
fi

say "=== DEAM ==="
mkdir -p "$RAW/deam"
get https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip "$RAW/deam/DEAM_audio.zip" \
  || get https://zenodo.org/records/1188976/files/DEAM_audio.zip "$RAW/deam/DEAM_audio.zip" \
  || say "NOTE  DEAM needs manual fetch from https://cvml.unige.ch/databases/DEAM/"
get https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip "$RAW/deam/DEAM_Annotations.zip" \
  || say "NOTE  DEAM annotations need manual fetch"

say "=== MusicCaps (pre-scraped mirror; YouTube blocks datacenter IPs) ==="
mkdir -p "$RAW/musiccaps"
/venv/main/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import os
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download("mahendra0203/musiccaps_processed_full",
                          repo_type="dataset",
                          local_dir="/data/raw/musiccaps",
                          max_workers=8)
    print("MusicCaps OK ->", p)
except Exception as e:
    print("MusicCaps FAILED:", type(e).__name__, e)
PY

say "=== SUMMARY ==="
du -sh "$RAW"/* 2>/dev/null | tee -a "$LOG"
df -h /data | tail -1 | tee -a "$LOG"
say "=== DONE ==="
