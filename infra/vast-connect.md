# Vast.ai — GNN-BERT compute (22241134 / tanviranindo)

## !!! VOLUMES DO NOT WORK ON THIS HOST — DO NOT USE THEM !!!

Tested three ways on machine 143795 (host 377496) on 2026-08-29/30.
**All three lost the data on `destroy instance`:**

| method | result |
|---|---|
| `--env '-v V.<id>:/data'` (what the Vast docs page shows) | LOST 17 GB |
| `--create-volume <ask_id>` | LOST 17 GB |
| standalone `create volume` + `--link-volume` | LOST 17 GB |

A 5 GB / 2-minute marker-file probe of the third method *appeared* to pass,
then the same method lost a 60 GB volume after 1.5 h of uptime. Treat that
probe as a false positive. Verified after each destroy: the machine's free
volume space returned to 500 GB and `show volumes` returned `[]` for 3+ min.

**Do not spend more time on volumes.** Re-downloading is now cheaper and
strictly more reliable — see below.

## Strategy instead of volumes

Full dataset seed is ~22 min / ~$0.05 now that the slow mirrors are replaced:

| dataset | source | time |
|---|---|---|
| FMA-small + metadata | os.unil.cloud.switch.ch | ~6.5 min @ 19 MB/s |
| MagnaTagATune | **HF `confit/magnatagatune`** | ~75 s @ 40 MB/s |
| DEAM | cvml.unige.ch | ~70 s @ 70 MB/s |
| MusicCaps | HF `mahendra0203/musiccaps_processed_full` | ~30 s |
| extraction | | ~12 min |

Cost comparison over the remaining ~12 days:
- volume: $0.133/day = **$1.60**, and it does not survive
- re-download 10x: **$0.50**, and it always works

**So: re-download every session. Never park data on Vast.**

What must NOT be re-derived — pull these DOWN to the Mac before destroying:

    rsync -az -e "ssh -p <port>" root@<host>:/data/checkpoints/ ./artifacts/checkpoints/
    rsync -az -e "ssh -p <port>" root@<host>:/data/results/     ./artifacts/results/
    rsync -az -e "ssh -p <port>" root@<host>:/data/splits/      ./artifacts/splits/

Cached features (~14 GB) are re-computable in ~1 h of CPU ($0.15); pull them
only if the connection makes that worthwhile.

## Example rental (2026-08-29, long since destroyed)

Kept as a worked example of the flags and costs. Instance and offer ids are
single-use and these are dead; look up fresh ones each session.
- Instance `49151882` | offer `45573262` | machine `143795`
- Volume **`49151852`** (`22241134_gnnbert_data`), **60 GB** at `/data`
- 1x RTX 4080 SUPER 16 GB, 64 threads, 62 GB RAM, **$0.143/hr** + volume $0.133/day
- `ssh $(vastai ssh-url 49151882)`
- Python: `/venv/main/bin/python` — 3.10.12, torch 2.5.1+cu121

## Stack note
Image ships Python 3.10 / torch 2.5.1, NOT the `requirements.txt` pins
(3.12 / torch 2.13.0). All 9 Week-1 tests pass unchanged on librosa 0.11.0,
so relax `requirements.txt` rather than rebuilding torch every session.

## Session workflow
    # 1. rent + link
    OFF=$(vastai search offers 'machine_id=143795 num_gpus=1' --storage 90 --raw \
          | python3 -c "import json,sys;d=[o for o in json.load(sys.stdin) if o['rentable']];print(d[0]['id'])")
    vastai create instance $OFF --image vastai/pytorch --disk 50 --ssh --direct \
      --link-volume 49151852 --mount-path /data
    # 2. push code (do NOT put a GitHub key on a rented box)
    rsync -az -e "ssh -p <port>" --exclude='.git' --exclude='.venv' \
      --exclude='data/raw' ./ root@<host>:/workspace/gnn-bert/
    # 3. datasets (idempotent - skips what is already on the volume)
    ssh ... 'cd /workspace/gnn-bert && tmux new -d -s dl "bash infra/fetch_datasets.sh && bash infra/extract_datasets.sh"'
    # 4. when stepping away
    vastai destroy instance <id>        # volume + /data survive

## Rules
1. Only `/data` survives. Container disk `/` is disposable.
2. Always work inside `tmux` — an SSH drop killed a job on 2026-08-24.
3. Offer IDs are single-use; look up a fresh one each session.
4. The volume is convenience, not backup. Push code/results to GitHub.
   At $0 balance volumes can be reclaimed.
5. Idle instance = $3.43/day. Destroy when stepping away.

## Dashboards (local)
    python3 -m http.server 8765 --bind 127.0.0.1     # from repo root
    while true; do bash update_status.sh; sleep 30; done &
- live   http://127.0.0.1:8765/status.html    (status.json + events.json)
- roadmap http://127.0.0.1:8765/dashboard.html (state.json + log.json)
`update_status.sh` resolves host/port from the vastai CLI, so re-renting
needs no edit.

## Budget
Credit $6.62 at rebuild. Volume 60 GB = $0.133/day.
Check: `vastai show user` / `vastai show instances`
