# Session Log — 2026-08-24: GPU environment setup + dataset downloads

## Goal
Stand up a working GPU training environment for the GNN-BERT project and
download all four datasets (FMA-medium, MagnaTagATune, DEAM, MusicCaps).

## Decisions made

1. **Compute environment: Vast.ai over Deepnote or local machine.**
   - Local Mac (M2 Pro, 16GB RAM, 78GB free disk): too little disk for
     ~60GB+ of raw datasets, no CUDA GPU.
   - Deepnote free tier: CPU-only (confirmed via `nvidia-smi` — not
     found), and its browser terminal is ephemeral — processes die when
     the tab/session disconnects, unsuitable for multi-hour downloads or
     training runs.
   - Vast.ai: rented an on-demand RTX 4090 (24GB VRAM) instance, real SSH
     access, 205GB disk. Real SSH means `tmux` sessions genuinely persist
     across disconnects (unlike Deepnote), since the instance itself keeps
     running as long as it's rented.
   - Cost: ~$0.15-0.36/hr depending on which host got rented; estimated
     total project GPU need is ~25-35 hours (~$5-10), well within an $8
     top-up.

2. **Auth: GitHub fine-grained PAT over SSH keys.**
   Deepnote's `GIT_SSH_COMMAND` was malformed out of the box (unterminated
   quote) and unreliable across ephemeral sessions. Switched to a
   fine-grained PAT scoped to only `gnn-bert-music-context` with
   `Contents: Read and write` — simpler, no key management, low blast
   radius if leaked (single-repo scope only).

3. **MusicCaps audio: pre-scraped HF mirror over direct yt-dlp scrape.**
   Direct `yt-dlp` scraping from the Vast.ai instance's datacenter IP was
   blocked 100% of the time by YouTube's bot detection ("Sign in to
   confirm you're not a bot"), even with `--extractor-args
   "youtube:player_client=android"` and `=tv` workarounds. Rather than
   pursue cookie-based auth (would require exporting the user's personal
   YouTube session cookies onto a rented cloud instance — a real privacy/
   ToS tradeoff), searched Hugging Face and found
   `mahendra0203/musiccaps_processed_full`, a dataset with 5,355 of the
   5,521 MusicCaps clips already scraped as audio + caption + metadata
   parquet shards. Used that instead. Validated row count via `pyarrow`
   (5,355 exact match across 7 shards). The scrape script written for the
   direct approach (`scrape_musiccaps.sh`) was aborted after only 42
   attempts (all failed on bot detection) to avoid falsely marking videos
   as "dead" when they were actually just IP-blocked; those 42 dead/false
   entries were deleted rather than trusted.

## What was actually done (chronological)

1. Rented Vast.ai instance, hit several setup snags along the way:
   - First rental defaulted to 16GB disk despite intending to set 150GB+
     (the search-page slider didn't carry through); had to destroy and
     re-rent twice, once landing on a host with both a broken Docker
     networking port allocation and once successfully on a clean host.
   - Final instance: RTX 4090, 24GB VRAM, 205GB disk (Container Size,
     no separate Volume — kept it simple), $0.356/hr initially quoted,
     billing came out to ~$0.15-0.36/hr band depending on the specific
     host across attempts.
2. Registered a Mac-local SSH public key with the instance, connected via
   `ssh -p <port> root@<ip>`.
3. Cloned the repo using an HTTPS PAT (avoided SSH key hassle entirely for
   the git side), `pip install -r requirements.txt` (torch 2.13.0,
   torch-geometric 2.8.0, transformers 5.15.1, etc.), re-ran the full
   Week 1 test suite (`pytest tests/ -v`) — all 9 tests passed on the new
   environment, confirming no environment drift from local dev.
   `torch.cuda.is_available()` confirmed `True` against the RTX 4090.
4. Started all four dataset downloads inside a `tmux` session
   (`downloads`) so they survive SSH disconnects:
   - FMA-medium (~22GB) + FMA-metadata (342MB) via direct `wget` from
     `os.unil.cloud.switch.ch`.
   - MagnaTagATune: 3 split-zip parts + `annotations_final.csv` from
     `mirg.city.ac.uk`.
   - DEAM: `DEAM_audio.zip` (1.3GB) from `cvml.unige.ch`.
   - MusicCaps: captions CSV first attempted via the original
     `google-research-datasets/musiccaps` GitHub repo — that repo
     returns 404 (removed/renamed), so pulled `musiccaps-public.csv`
     directly from the `google/MusicCaps` dataset on Hugging Face
     instead. Then hit the YouTube-scrape blocker described above and
     pivoted to the pre-scraped HF mirror for the audio itself.
5. Validated everything that finished downloading while FMA-medium (the
   longest download, ~22GB) continued in the background:
   - `unzip -tq` on `fma_metadata.zip` and `deam_audio.zip` — no errors.
   - MagnaTagATune parts inspected with `file` — part 1 is a proper zip
     header, parts 2-3 correctly show as raw split-archive data (expected
     for multi-part zips, not corruption; needs `zip -F` to merge before
     extraction).
   - MusicCaps CSV row count and columns validated with Python's `csv`
     module — 5,521 rows exactly, matching the spec's documented count.
   - MusicCaps audio parquet shards validated with `pyarrow` — 5,355 rows
     total across 7 shards (765 each), columns
     `audio,caption,youtube_id,start_time,end_time,aspect_list`.
6. Also connected the user's Google Drive (5TB free) as a Vast.ai Cloud
   Sync integration, tagged for backup — not yet used for an actual sync,
   but available so datasets/checkpoints aren't lost if the instance is
   destroyed later.

## Open items for next session

- FMA-medium still downloading in background at time of writing (tmux
  session `downloads` on the Vast.ai instance, ~22GB total).
- MagnaTagATune split-zip parts need `zip -F` merge + extraction once all
  parts confirmed complete.
- GitHub Project board still blocked on `gh auth` scope (`project`,
  `read:project`) — unrelated to this session's work, carried over from
  Week 1.
- Week 2 (BERT tag classifier) not yet started — next task once datasets
  are in place and split via `src/splits.py`.
- Remember to **stop (not destroy) the Vast.ai instance** when not
  actively working to conserve the $8 credit — balance depletes at the
  instance's hourly rate regardless of whether you're connected.
