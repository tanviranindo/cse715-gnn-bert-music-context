#!/bin/bash
# Polls the live Vast.ai instance and writes status.json (consumed by status.html)
# plus appends to events.json when something meaningful changes.
# Host/port are resolved from the vastai CLI, so re-renting needs no edit here.
#   loop:  while true; do bash update_status.sh; sleep 20; done
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
STATUS="$DIR/status.json"; EVENTS="$DIR/events.json"; PREV="$DIR/.prev_state.json"
[ -f "$EVENTS" ] || echo "[]" > "$EVENTS"
[ -f "$PREV" ]   || echo "{}" > "$PREV"

INST=$(vastai show instances --raw 2>/dev/null \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(json.dumps(d[0]) if d else '{}')" 2>/dev/null || echo '{}')
IID=$(echo "$INST"   | python3 -c "import json,sys;print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
RATE=$(echo "$INST"  | python3 -c "import json,sys;print(json.load(sys.stdin).get('dph_total',0))" 2>/dev/null)
UPH=$(echo "$INST"   | python3 -c "import json,sys;print(round(json.load(sys.stdin).get('duration',0)/3600,3))" 2>/dev/null)
CREDIT=$(vastai show user --raw 2>/dev/null | python3 -c "import json,sys;print(round(json.load(sys.stdin).get('credit',0),3))" 2>/dev/null || echo 0)

REMOTE=""; SSH_OK=false
if [ -n "$IID" ]; then
  URL=$(vastai ssh-url "$IID" 2>/dev/null)
  H=$(echo "$URL" | sed 's|ssh://root@||;s|:.*||'); P=$(echo "$URL" | sed 's|.*:||')
  if [ -n "$H" ] && [ -n "$P" ]; then
    REMOTE=$(ssh -p "$P" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
      root@"$H" 'for d in fma_small fma_metadata magnatagatune deam musiccaps; do
           echo "SZ_$d=$(du -sb /data/raw/$d* 2>/dev/null | awk "{s+=\$1} END {printf \"%.0f\", s+0}")"
           echo "ZIP_$d=$(ls /data/raw/$d*.zip* /data/raw/$d/*.zip* 2>/dev/null | wc -l)"
         done
         echo "DISK=$(du -sb /data 2>/dev/null | cut -f1)"
         echo "DL=$(tmux has-session -t dl 2>/dev/null && echo 1 || echo 0)"
         echo "PHASE=$(tail -40 /data/logs/extract.log 2>/dev/null | grep -q DONE && echo idle || (test -f /data/logs/extract.log && echo extracting || echo downloading))"
         echo "TESTS=$(cd /workspace/gnn-bert 2>/dev/null && /venv/main/bin/python -m pytest tests/ -q 2>&1 | tail -1)"
         echo "LASTLOG=$(tail -1 /data/logs/fetch.log 2>/dev/null | tr -d "\r" | tail -c 120)"' 2>/dev/null) && SSH_OK=true
  fi
fi

REMOTE_DATA="$REMOTE" REMOTE_OK="$SSH_OK" IID="$IID" RATE="$RATE" UPH="$UPH" CREDIT="$CREDIT" \
python3 - "$STATUS" "$EVENTS" "$PREV" <<'PY'
import json,os,sys,datetime
status_p,events_p,prev_p = sys.argv[1:4]
raw = os.environ.get("REMOTE_DATA","")
kv = dict(l.split('=',1) for l in raw.splitlines() if '=' in l)
def num(k): 
    try: return int(kv.get(k,'0') or 0)
    except: return 0

# name, key, expected final bytes (post-extraction)
# name, key, download-size, extracted-size
SPEC=[("FMA-small","fma_small",7_200_000_000,7_500_000_000),
      ("FMA-metadata","fma_metadata",358_000_000,1_400_000_000),
      ("MagnaTagATune","magnatagatune",2_900_000_000,2_900_000_000),
      ("DEAM","deam",1_310_000_000,1_400_000_000),
      ("MusicCaps","musiccaps",3_200_000_000,3_200_000_000)]
ds=[]
for name,key,dl_t,ex_t in SPEC:
    b=num("SZ_"+key); zips=num("ZIP_"+key)
    target = dl_t if zips>0 else ex_t     # measure against the phase we are in
    pct=min(100.0, b/target*100) if target else 0.0
    if b==0: state="pending"
    elif zips>0: state="downloading" if pct<99 else "downloaded"
    else: state="extracted" if pct>=60 else "downloading"
    ds.append({"name":name,"bytes":b,"total_bytes":target,"pct":round(pct,1),
               "state":state,"detail":("archives removed" if b and not zips else
                                        f"{zips} archive(s) present" if zips else "")})

disk=num("DISK")
st={"generated_at":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "ssh_reachable":os.environ.get("REMOTE_OK")=="true",
    "phase":kv.get("PHASE","unknown").strip(),
    "job_running":kv.get("DL","0").strip()=="1",
    "datasets":ds,
    "disk":{"used_gb":round(disk/1e9,2),"total_gb":60,
            "pct":round(disk/60e9*100,1)},
    "instance":{"id":os.environ.get("IID",""),"rate":float(os.environ.get("RATE") or 0),
                "uptime_h":float(os.environ.get("UPH") or 0),"credit":float(os.environ.get("CREDIT") or 0)},
    "tests_summary":kv.get("TESTS","").strip() or "unknown",
    "latest_commit":kv.get("LASTLOG","").strip()}
json.dump(st,open(status_p,"w"),indent=2)

# event log on meaningful change
prev=json.load(open(prev_p)) if os.path.getsize(prev_p) else {}
now=datetime.datetime.now()
ev=json.load(open(events_p))
def add(t): ev.append({"time":now.strftime("%H:%M"),
    "iso":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"text":t})
for d in ds:
    was=prev.get("ds",{}).get(d["name"])
    if was and was!=d["state"]:
        add(f'{d["name"]}: {was} -> {d["state"]}')
if prev.get("phase") and prev["phase"]!=st["phase"]: add(f'phase: {prev["phase"]} -> {st["phase"]}')
if prev.get("job_running") is not None and prev["job_running"]!=st["job_running"]:
    add("job finished" if not st["job_running"] else "job started")
if prev.get("ssh") is not None and prev["ssh"]!=st["ssh_reachable"]:
    add("instance unreachable" if not st["ssh_reachable"] else "instance reachable again")
json.dump(ev,open(events_p,"w"),indent=1)
json.dump({"ds":{d["name"]:d["state"] for d in ds},"phase":st["phase"],
           "job_running":st["job_running"],"ssh":st["ssh_reachable"]},open(prev_p,"w"))
print(f'{st["phase"]:12s} job={st["job_running"]} disk={st["disk"]["used_gb"]}GB ' +
      ' '.join(f'{d["name"]}:{d["pct"]:.0f}%' for d in ds))
PY
