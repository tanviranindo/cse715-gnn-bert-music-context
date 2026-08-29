#!/usr/bin/env bash
# Task 1 is text-only: no audio needed, ~30 MB instead of the 17 GB full seed.
set -u
RAW=${1:-/data/raw}
mkdir -p "$RAW/magnatagatune" "$RAW/musiccaps"
HF=https://huggingface.co/datasets
for f in annotations_final.csv clip_info_final.csv train_gt_mtt.tsv val_gt_mtt.tsv test_gt_mtt.tsv; do
  [ -s "$RAW/magnatagatune/$f" ] || curl -sL -o "$RAW/magnatagatune/$f" "$HF/confit/magnatagatune/resolve/main/$f"
  echo "  $f $(du -h "$RAW/magnatagatune/$f" | cut -f1)"
done
[ -s "$RAW/musiccaps/musiccaps-public.csv" ] || \
  curl -sL -o "$RAW/musiccaps/musiccaps-public.csv" "$HF/google/MusicCaps/resolve/main/musiccaps-public.csv"
echo "  musiccaps-public.csv $(du -h "$RAW/musiccaps/musiccaps-public.csv" | cut -f1)"
