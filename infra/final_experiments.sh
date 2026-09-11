#!/usr/bin/env bash
# Leakage-free final experiment matrix. Safe to resume: completed stages have markers.
set -euo pipefail

usage() {
  echo "usage: $0 DATA_ROOT OUTPUT_ROOT [--dry-run]"
  echo "       $0 --list"
}

stages=(validate build_graphs dump_splits task1 task2 task3 task3_sweep task4 analysis aggregate validate_results)
if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${stages[@]}"
  exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then usage >&2; exit 2; fi

DATA_ROOT=$1
OUTPUT_ROOT=$2
DRY_RUN=0
[[ "${3:-}" == "--dry-run" ]] && DRY_RUN=1
[[ $# -eq 2 || "${3:-}" == "--dry-run" ]] || { usage >&2; exit 2; }

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAW_ROOT="$DATA_ROOT/raw"
CACHE_ROOT="$DATA_ROOT/processed"
RESULTS_ROOT="$OUTPUT_ROOT/results"
CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints"
SPLIT_ROOT="$OUTPUT_ROOT/data/splits"
LOG_ROOT="$OUTPUT_ROOT/logs"
DONE_ROOT="$OUTPUT_ROOT/.done"
PYTHON_BIN=${PYTHON_BIN:-/venv/main/bin/python}
GRAPH_WORKERS=${GRAPH_WORKERS:-12}
EXPERIMENT_SOURCE_REVISION=${EXPERIMENT_SOURCE_REVISION:-$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)}
export EXPERIMENT_SOURCE_REVISION
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

if [[ ! "$EXPERIMENT_SOURCE_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "EXPERIMENT_SOURCE_REVISION must be the exact 40-character source commit" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT" "$RESULTS_ROOT" "$CHECKPOINT_ROOT" "$SPLIT_ROOT" "$LOG_ROOT" "$DONE_ROOT"

selected() {
  [[ -z "${ONLY_STAGE:-}" || "$1" == "$ONLY_STAGE" || "$1" == "$ONLY_STAGE"_* ]]
}

run_stage() {
  local name=$1
  shift
  selected "$name" || return 0
  local marker="$DONE_ROOT/$name"
  local logfile="$LOG_ROOT/$name.log"
  local command_file="$LOG_ROOT/$name.command"
  printf '%q ' "$@" >"$command_file"
  printf '\n' >>"$command_file"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run] %s: ' "$name"
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  if [[ -f "$marker" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] $name (completion marker exists)"
    return 0
  fi
  echo "[run ] $name"
  "$@" >"$logfile" 2>&1
  touch "$marker"
  echo "[done] $name"
}

if [[ $DRY_RUN -eq 0 ]]; then
  {
    echo "source_revision=$EXPERIMENT_SOURCE_REVISION"
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
    "$PYTHON_BIN" --version
    "$PYTHON_BIN" -m pip freeze
    nvidia-smi || true
  } >"$LOG_ROOT/environment.txt" 2>&1
fi

run_stage validate_fma "$PYTHON_BIN" infra/validate_datasets.py --raw "$RAW_ROOT" --dataset fma
run_stage validate_mtat "$PYTHON_BIN" infra/validate_datasets.py --raw "$RAW_ROOT" --dataset mtat
run_stage validate_musiccaps "$PYTHON_BIN" infra/validate_datasets.py --raw "$RAW_ROOT" --dataset musiccaps

run_stage build_graphs_fma "$PYTHON_BIN" src/build_graphs.py --dataset fma \
  --out "$CACHE_ROOT" --n-examples 25 --audio-root "$RAW_ROOT/fma_small" \
  --tracks-csv "$RAW_ROOT/fma_metadata/tracks.csv" --workers "$GRAPH_WORKERS"
run_stage build_graphs_mtat "$PYTHON_BIN" src/build_graphs.py --dataset mtat \
  --out "$CACHE_ROOT" --n-examples 0 --audio-root "$RAW_ROOT/magnatagatune/audio" \
  --mtat-dir "$RAW_ROOT/magnatagatune" --no-mel --workers "$GRAPH_WORKERS"
run_stage build_graphs_musiccaps "$PYTHON_BIN" src/build_graphs.py --dataset musiccaps \
  --out "$CACHE_ROOT" --n-examples 0 --musiccaps-dir "$RAW_ROOT/musiccaps" \
  --musiccaps-csv "$RAW_ROOT/musiccaps/musiccaps-public.csv" --no-mel --workers "$GRAPH_WORKERS"
if [[ "${RUN_DEAM:-0}" == "1" ]]; then
  run_stage build_graphs_deam "$PYTHON_BIN" src/build_graphs.py --dataset deam \
    --out "$CACHE_ROOT" --n-examples 0 --audio-root "$RAW_ROOT/deam/audio/MEMD_audio" \
    --deam-annotations "$RAW_ROOT/deam/annotations" --deam-metadata "$RAW_ROOT/deam/metadata" --no-mel --workers "$GRAPH_WORKERS"
fi

run_stage dump_splits_fma "$PYTHON_BIN" src/dump_splits.py --cache "$CACHE_ROOT/fma_small_graphs.pt" --dataset fma_small --out "$SPLIT_ROOT"
run_stage dump_splits_mtat "$PYTHON_BIN" src/dump_splits.py --cache "$CACHE_ROOT/mtat_graphs.pt" --dataset mtat --out "$SPLIT_ROOT"
run_stage dump_splits_musiccaps "$PYTHON_BIN" src/dump_splits.py --cache "$CACHE_ROOT/musiccaps_graphs.pt" --dataset musiccaps --out "$SPLIT_ROOT"

run_stage task1_mtat "$PYTHON_BIN" src/train.py --dataset mtat --data-dir "$RAW_ROOT" --out-dir "$RESULTS_ROOT" --save-checkpoint --checkpoint-dir "$CHECKPOINT_ROOT"
run_stage task1_musiccaps "$PYTHON_BIN" src/train.py --dataset musiccaps --data-dir "$RAW_ROOT" --out-dir "$RESULTS_ROOT" --save-checkpoint --checkpoint-dir "$CHECKPOINT_ROOT"
run_stage task1_musiccaps_stripped "$PYTHON_BIN" src/train.py --dataset musiccaps --data-dir "$RAW_ROOT" --out-dir "$RESULTS_ROOT" --strip-leakage --save-checkpoint --checkpoint-dir "$CHECKPOINT_ROOT"

run_stage task2_segment "$PYTHON_BIN" src/train_gnn.py --cache "$CACHE_ROOT/fma_small_graphs.pt" --out-dir "$RESULTS_ROOT" --model all
run_stage task2_b4 "$PYTHON_BIN" src/train_b4.py --cache "$CACHE_ROOT/fma_small_graphs.pt" --out "$RESULTS_ROOT/metrics_task2_b4.json"
run_stage task2_chord "$PYTHON_BIN" src/train_gnn.py --cache "$CACHE_ROOT/fma_small_graphs.pt" --out-dir "$RESULTS_ROOT" --graph chord --model all

run_stage task3_ablation "$PYTHON_BIN" src/train_fusion.py --mtat-cache "$CACHE_ROOT/mtat_graphs.pt" --out-dir "$RESULTS_ROOT" --mode all
run_stage task3_freeze "$PYTHON_BIN" src/train_fusion.py --mtat-cache "$CACHE_ROOT/mtat_graphs.pt" --out-dir "$RESULTS_ROOT/_freeze" --mode crossattn --freeze-bert
run_stage task3_gnnlr "$PYTHON_BIN" src/train_fusion.py --mtat-cache "$CACHE_ROOT/mtat_graphs.pt" --out-dir "$RESULTS_ROOT/_gnnlr" --mode crossattn --gnn-lr 1e-3
if [[ "${RUN_DEAM:-0}" == "1" ]]; then
  run_stage task3_multitask "$PYTHON_BIN" src/train_fusion.py --mtat-cache "$CACHE_ROOT/mtat_graphs.pt" --deam-cache "$CACHE_ROOT/deam_graphs.pt" --out-dir "$RESULTS_ROOT/_multitask" --mode crossattn --gnn-lr 1e-3
fi

for seed in 42 1 2 3 4; do
  run_stage "task3_sweep_shared_s$seed" "$PYTHON_BIN" src/train_fusion.py --mtat-cache "$CACHE_ROOT/mtat_graphs.pt" --out-dir "$RESULTS_ROOT/_sweep/shared_s$seed" --mode crossattn --seed "$seed"
  run_stage "task3_sweep_gnnlr_s$seed" "$PYTHON_BIN" src/train_fusion.py --mtat-cache "$CACHE_ROOT/mtat_graphs.pt" --out-dir "$RESULTS_ROOT/_sweep/gnnlr_s$seed" --mode crossattn --seed "$seed" --gnn-lr 1e-3
done
run_stage task3_sweep_aggregate "$PYTHON_BIN" src/aggregate_sweep.py "$RESULTS_ROOT/_sweep"

run_stage task4_shared "$PYTHON_BIN" src/train_contrastive.py --cache "$CACHE_ROOT/musiccaps_graphs.pt" --out-dir "$RESULTS_ROOT" --checkpoint-dir "$CHECKPOINT_ROOT"
for item in 025:0.25 050:0.50 075:0.75; do
  suffix=${item%%:*}
  fraction=${item#*:}
  run_stage "task4_scale_$suffix" "$PYTHON_BIN" src/train_contrastive.py --cache "$CACHE_ROOT/musiccaps_graphs.pt" --out-dir "$RESULTS_ROOT" --gnn-lr 1e-3 --batch-size 128 --train-frac "$fraction" --metrics-suffix "_frac$suffix"
done
# Run the full-data headline arm last because the zero-shot artifact has one
# canonical filename; scale ablations must not be allowed to overwrite it.
run_stage task4_gnnlr "$PYTHON_BIN" src/train_contrastive.py --cache "$CACHE_ROOT/musiccaps_graphs.pt" --out-dir "$RESULTS_ROOT" --gnn-lr 1e-3 --batch-size 128 --metrics-suffix _gnnlr --checkpoint-dir "$CHECKPOINT_ROOT"
run_stage task4_supervised_comparison "$PYTHON_BIN" src/train_fusion.py --mtat-cache "$CACHE_ROOT/musiccaps_graphs.pt" --out-dir "$RESULTS_ROOT/_supervised_cmp" --mode crossattn --split official_eval --gnn-lr 1e-3

run_stage analysis_fusion "$PYTHON_BIN" src/analyze_fusion.py --checkpoint "$RESULTS_ROOT/task3_crossattn.pt" --cache "$CACHE_ROOT/mtat_graphs.pt" --out-dir "$RESULTS_ROOT"
run_stage analysis_scale "$PYTHON_BIN" src/analyze_scale.py --results "$RESULTS_ROOT"
run_stage analysis_plots "$PYTHON_BIN" -m src.make_plots --results "$RESULTS_ROOT" --out "$RESULTS_ROOT/plots"
run_stage aggregate_metrics "$PYTHON_BIN" src/aggregate_metrics.py "$RESULTS_ROOT"
run_stage validate_results "$PYTHON_BIN" infra/validate_remote_results.py --results "$RESULTS_ROOT" --expected-revision "$EXPERIMENT_SOURCE_REVISION"

echo "Final experiment matrix complete: $OUTPUT_ROOT"
