"""Batch audio -> graph preprocessing for Task 2.

Reads FMA-small mp3s, builds one segment graph and one chord graph per track,
and caches them as a single tensor file plus >=20 standalone example graphs
(a hard submission requirement, spec S6).

    python -m src.build_graphs --fma-root /data/raw/fma_small \
        --tracks-csv /data/raw/fma_metadata/tracks.csv --out /data/processed

Work is spread across processes because librosa feature extraction is
CPU-bound and single-threaded; on 32 cores this is the difference between
~10 minutes and ~5 hours.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

# Each worker process otherwise spawns its own BLAS/OpenMP thread pool. With
# 32 workers on 32 cores that is ~1000 threads contending, and measured
# throughput was 2.1 tracks/s where ~3 s/track of real work was expected.
# These must be set before numpy/librosa import in the worker.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from src import fma_data, graph_builder as gb

warnings.filterwarnings("ignore")

_CFG: dict = {}


def _init(cfg: dict) -> None:
    global _CFG
    _CFG = cfg


def process_track(record: dict) -> dict | None:
    """Load one clip, emit its segment graph, chord graph and mel spectrogram."""
    import librosa

    if record.get("audio_bytes") is not None:
        import io
        try:
            y, sr = librosa.load(io.BytesIO(record["audio_bytes"]),
                                 sr=_CFG["sample_rate"], mono=True,
                                 duration=_CFG["duration"])
        except Exception:
            return None
    else:
        if _CFG["dataset"] in ("mtat", "deam"):
            path = Path(_CFG["audio_root"]) / record["mp3_path"]
        else:
            path = fma_data.track_audio_path(_CFG["audio_root"], record["track_id"])
        if not path.exists():
            return None
        try:
            y, sr = librosa.load(path, sr=_CFG["sample_rate"], mono=True,
                                 duration=_CFG["duration"])
        except Exception:
            return None                   # FMA ships a handful of corrupt mp3s
    if y.size < _CFG["sample_rate"]:
        return None

    feats = gb.segment_features(y, sr, _CFG["segment_seconds"])
    if len(feats) < 2:
        return None
    edge_index, edge_weight = gb.build_segment_graph(feats, tau=_CFG["tau"])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chords = gb.estimate_chords(chroma)
    chord_nodes, chord_ei, chord_w = gb.build_chord_graph(chords)

    # mel is only consumed by Task 2's CNN baseline. For 21315 MTAT clips it
    # is 3.49 GB of a 3.6 GB cache, and serialising that killed the writer on
    # a 31 GB box, so Tasks 3 and 4 skip it.
    if _CFG["store_mel"]:
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=_CFG["n_mels"]),
            ref=np.max,
        )
        mel = mel[:, : _CFG["mel_width"]].astype(np.float16)
    else:
        mel = np.zeros((0, 0), dtype=np.float16)
    record.pop("audio_bytes", None)
    return {
        "track_id": record["track_id"],
        "genre": record["genre"],
        "artist": record["artist"],
        "split": record["split"],
        "text": record.get("text", ""),
        "labels": record.get("labels", set()),
        "ytid": record.get("ytid", ""),
        **({"valence_z": record["valence_z"], "arousal_z": record["arousal_z"],
            "valence": record["valence"], "arousal": record["arousal"]}
           if "valence_z" in record else {}),
        "x": feats,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "chord_nodes": chord_nodes,
        "chord_edge_index": chord_ei,
        "chord_edge_weight": chord_w,
        "mel": mel,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fma", "mtat", "deam", "musiccaps"], default="fma")
    p.add_argument("--audio-root", default="/data/raw/fma_small",
                   help="fma: fma_small dir | mtat: magnatagatune/audio dir")
    p.add_argument("--tracks-csv", default="/data/raw/fma_metadata/tracks.csv",
                   help="fma only")
    p.add_argument("--mtat-dir", default="/data/raw/magnatagatune",
                   help="mtat only: holds annotations_final.csv + clip_info_final.csv")
    p.add_argument("--deam-annotations", default="/data/raw/deam/annotations")
    p.add_argument("--deam-metadata", default="/data/raw/deam/metadata")
    p.add_argument("--musiccaps-dir", default="/data/raw/musiccaps",
                   help="musiccaps only: dir of parquet shards with audio+caption")
    p.add_argument("--n-tags", type=int, default=50, help="mtat only")
    p.add_argument("--out", default="/data/processed")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--segment-seconds", type=float, default=1.5)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--tau", type=float, default=0.35,
                   help="cosine threshold AFTER per-track z-scoring; 0.9 was "
                        "correct for raw features and yields a bare chain here")
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--mel-width", type=int, default=640)
    p.add_argument("--no-mel", action="store_true",
                   help="skip log-mel; only Task 2's CNN baseline needs it")
    p.add_argument("--workers", type=int, default=0, help="0 = all cores")
    p.add_argument("--limit", type=int, default=0, help="0 = all tracks")
    p.add_argument("--n-examples", type=int, default=25)
    args = p.parse_args()

    import torch

    if args.dataset == "fma":
        records = fma_data.load_tracks(args.tracks_csv, subset="small")
        for r in records:
            r["labels"] = {r["genre"]}
        print(f"[data] {len(records)} FMA-small tracks")
        print(f"[data] genres: {fma_data.class_balance(records)}")
        vocab = fma_data.genre_vocabulary(records)
    elif args.dataset == "mtat":
        from src import mtat_data
        vocab, mrecs = mtat_data.build_dataset(
            Path(args.mtat_dir) / "annotations_final.csv",
            Path(args.mtat_dir) / "clip_info_final.csv",
            n_tags=args.n_tags,
        )
        records = [
            {"track_id": int(r["clip_id"]), "genre": sorted(r["labels"])[0],
             "artist": r["artist"], "split": "", "mp3_path": r["mp3_path"],
             "text": r["text"], "labels": r["labels"]}
            for r in mrecs
        ]
        print(f"[data] {len(records)} MagnaTagATune clips | {len(vocab)} tags")
    if args.dataset == "musiccaps":
        # MusicCaps audio ships inside parquet shards (the HF mirror), not as
        # files. Shards are processed one at a time so the raw audio of the
        # whole corpus is never resident at once.
        import glob
        import pyarrow.parquet as pq
        shards = sorted(glob.glob(str(Path(args.musiccaps_dir) / "**" / "*.parquet"),
                                  recursive=True))
        print(f"[data] {len(shards)} MusicCaps parquet shards")
        records = []
        for shard in shards:
            table = pq.read_table(shard).to_pylist()
            for row in table:
                audio = row.get("audio")
                blob = audio.get("bytes") if isinstance(audio, dict) else audio
                if not blob:
                    continue
                aspects = row.get("aspect_list") or []
                if isinstance(aspects, str):
                    import ast as _ast
                    try:
                        aspects = _ast.literal_eval(aspects)
                    except (ValueError, SyntaxError):
                        aspects = []
                records.append({
                    "track_id": abs(hash(row.get("youtube_id", ""))) % (10 ** 9),
                    "genre": "", "artist": row.get("youtube_id", ""),
                    "split": "", "mp3_path": "",
                    "text": row.get("caption", "") or "",
                    # aspects are the zero-shot ground truth, NOT training
                    # supervision: the contrastive objective never sees them.
                    "labels": {str(a).strip() for a in aspects if str(a).strip()},
                    "audio_bytes": blob,
                    "ytid": row.get("youtube_id", ""),
                })
        vocab = []
        print(f"[data] {len(records)} MusicCaps clips with audio + caption")
    if args.dataset == "deam":
        from src import deam_data
        drecs = deam_data.build_dataset(args.deam_annotations, args.deam_metadata)
        drecs, stats = deam_data.standardise_targets(drecs)
        print(f"[data] valence/arousal z-scored with {stats}")
        records = [
            {"track_id": int(r["clip_id"]), "genre": "", "artist": r["artist"],
             "split": "", "mp3_path": f"{r['clip_id']}.mp3", "text": r["text"],
             "labels": set(), "valence_z": r["valence_z"],
             "arousal_z": r["arousal_z"], "valence": r["valence"],
             "arousal": r["arousal"]}
            for r in drecs
        ]
        vocab = []
        print(f"[data] {len(records)} DEAM songs with pseudo-captions")
    if args.limit:
        records = records[: args.limit]

    cfg = {
        "dataset": args.dataset,
        "audio_root": args.audio_root,
        "sample_rate": args.sample_rate,
        "segment_seconds": args.segment_seconds,
        "duration": args.duration,
        "tau": args.tau,
        "n_mels": args.n_mels,
        "mel_width": args.mel_width,
        "store_mel": not args.no_mel,
    }
    workers = args.workers or mp.cpu_count()
    print(f"[proc] {workers} workers, {args.segment_seconds}s segments, tau={args.tau}")

    started = time.time()
    out_records = []
    with mp.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, res in enumerate(pool.imap_unordered(process_track, records, chunksize=16), 1):
            if res is not None:
                out_records.append(res)
            if i % 500 == 0:
                rate = i / (time.time() - started)
                print(f"  {i}/{len(records)}  ok={len(out_records)}  "
                      f"{rate:.1f} tracks/s  eta {(len(records)-i)/rate/60:.1f}m",
                      flush=True)

    elapsed = time.time() - started
    print(f"[proc] {len(out_records)}/{len(records)} usable in {elapsed/60:.1f} min")

    out_dir = Path(args.out)
    (out_dir / "graph_samples").mkdir(parents=True, exist_ok=True)
    cache_stem = {"fma": "fma_small", "mtat": "mtat", "deam": "deam",
                  "musiccaps": "musiccaps"}[args.dataset]
    torch.save({"config": vars(args), "vocab": vocab, "records": out_records},
               out_dir / f"{cache_stem}_graphs.pt")

    # >=20 standalone example graphs (spec S6 hard requirement), spread
    # across genres so the sample is not all one class.
    by_genre: dict[str, list[dict]] = {}
    for r in out_records:
        by_genre.setdefault(r["genre"], []).append(r)
    examples, gi = [], 0
    while len(examples) < args.n_examples and any(by_genre.values()):
        for genre in sorted(by_genre):
            if by_genre[genre] and len(examples) < args.n_examples:
                examples.append(by_genre[genre].pop())
        gi += 1
        if gi > args.n_examples:
            break

    index = []
    for r in examples:
        stem = f"{r['track_id']:06d}"
        torch.save(
            {k: r[k] for k in ("x", "edge_index", "edge_weight",
                               "chord_nodes", "chord_edge_index",
                               "chord_edge_weight", "genre", "track_id")},
            out_dir / "graph_samples" / f"{stem}.pt",
        )
        stats = gb.graph_stats(r["edge_index"], len(r["x"]))
        meta = {
            "track_id": r["track_id"], "genre": r["genre"],
            "segment_graph": stats,
            "chord_graph": {
                "n_nodes": len(r["chord_nodes"]),
                "n_edges": int(r["chord_edge_index"].shape[1]),
                "chords": r["chord_nodes"],
            },
        }
        (out_dir / "graph_samples" / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        index.append(meta)
    (out_dir / "graph_samples" / "index.json").write_text(json.dumps(index, indent=2))

    degs = [gb.graph_stats(r["edge_index"], len(r["x"]))["avg_degree"] for r in out_records]
    nodes = [len(r["x"]) for r in out_records]
    chord_nodes = [len(r["chord_nodes"]) for r in out_records]
    print(f"[stat] segment graphs: {np.mean(nodes):.1f} nodes, "
          f"avg degree {np.mean(degs):.2f}")
    print(f"[stat] chord graphs:   {np.mean(chord_nodes):.1f} unique chords")
    print(f"[out ] {out_dir}/{cache_stem}_graphs.pt  +  {len(examples)} examples")


if __name__ == "__main__":
    main()
