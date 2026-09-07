"""Validate every dataset against its published shape before training on it.

A download that returns HTTP 200 is not a dataset. This checks each corpus for
the things that actually break a run downstream: file counts against the
published figure, the columns the loaders index by name, row counts, label
vocabularies, split flags, and whether the audio decodes at all.

    python infra/validate_datasets.py --raw /data/raw
    python infra/validate_datasets.py --raw /data/raw --dataset mtat

Exit status is 0 only if every check passes. Anything short of that prints what
was expected, what was found, and where a replacement copy can be obtained ---
the point is to fail before a 40-minute preprocessing pass, not during it.

Expected values are sourced from the dataset papers and the project brief, not
from a previous run of this pipeline, so a systematically corrupted download
cannot validate itself.
"""

import argparse
import csv
import glob
import json
import os
import pathlib
import sys
import zipfile

# (expected, tolerance) — tolerance covers documented corrupt files, not slack.
EXPECT = {
    "fma_small":   {"mp3": 8000, "tol": 10, "genres": 8, "per_genre": 1000},
    "mtat":        {"mp3": 25863, "tol": 200, "tags": 188, "rows": 25863},
    "deam":        {"mp3": 1802, "tol": 10, "meta_years": 3},
    "musiccaps":   {"csv_rows": 5521, "parquet_rows": 5355, "tol": 20},
}

ALTERNATIVES = {
    "fma_small": [
        "https://os.unil.cloud.switch.ch/fma/fma_small.zip  (canonical, UNIL)",
        "https://huggingface.co/datasets/benjamin-paine/free-music-archive-small",
        "mirror listed at https://github.com/mdeff/fma#data",
    ],
    "mtat": [
        "hf_hub_download('confit/magnatagatune', 'mp3.zip', repo_type='dataset')",
        "https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset  (3 split zips, merge with `zip -F`)",
    ],
    "deam": [
        "https://cvml.unige.ch/databases/DEAM/  (DEAM_audio.zip, DEAM_Annotations.zip, metadata.zip)",
        "hf_hub_download('herrjyj/herrjyj-deam-assets', 'DEAM_audio.zip', repo_type='dataset')",
        "https://zenodo.org/records/11400122  (annotations only)",
    ],
    "musiccaps": [
        "hf_hub_download('google/MusicCaps', 'musiccaps-public.csv', repo_type='dataset')",
        "snapshot_download('mahendra0203/musiccaps_processed_full', repo_type='dataset')  (pre-scraped audio)",
        "yt-dlp against the ytid column — blocked from datacenter IPs, run it from a residential connection",
    ],
}

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))
    print("  %-4s %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def near(found, expected, tol):
    return abs(found - expected) <= tol


def count_files(root, ext):
    return sum(len([f for f in files if f.endswith(ext)]) for _, _, files in os.walk(root))


def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception as e:
        return "%s: %s" % (type(e).__name__, e)


def decodes(paths, n=3):
    """Actually decode a few clips — a truncated mp3 passes a file-count check."""
    try:
        import librosa
    except ImportError:
        return None, "librosa not installed, skipped"
    got = []
    for p in paths[:n]:
        try:
            y, sr = librosa.load(p, sr=22050, duration=5.0)
            got.append(len(y) / sr)
        except Exception as e:
            return False, "%s failed to decode: %s" % (os.path.basename(p), type(e).__name__)
    if not got:
        return False, "no files to decode"
    return all(d > 1.0 for d in got), "decoded %d clips, %.1f-%.1f s" % (
        len(got), min(got), max(got))


# --------------------------------------------------------------------------

def validate_fma(raw):
    print("\nFMA-small — expect %d mp3, %d genres x %d"
          % (EXPECT["fma_small"]["mp3"], EXPECT["fma_small"]["genres"],
             EXPECT["fma_small"]["per_genre"]))
    d = raw / "fma_small"
    z = raw / "fma_small.zip"
    if z.exists():
        check("fma_small.zip integrity", zip_ok(z) is True, "%.2f GB" % (z.stat().st_size / 2**30))
    if not d.exists():
        return check("fma_small extracted", False, "missing %s" % d)

    n = count_files(d, ".mp3")
    check("fma_small mp3 count", near(n, EXPECT["fma_small"]["mp3"], EXPECT["fma_small"]["tol"]),
          "%d found, expected %d (+/-%d)" % (n, EXPECT["fma_small"]["mp3"], EXPECT["fma_small"]["tol"]))

    mp3s = sorted(glob.glob(str(d / "**" / "*.mp3"), recursive=True))
    ok, detail = decodes(mp3s)
    if ok is not None:
        check("fma_small audio decodes", ok, detail)

    tracks = raw / "fma_metadata" / "tracks.csv"
    if not tracks.exists():
        return check("fma tracks.csv", False, "missing — genre labels unavailable")
    with open(tracks, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[:3]
    flat = [",".join(h) for h in header]
    check("fma tracks.csv has subset column", any("subset" in f for f in flat),
          "%d rows total" % (len(rows) - 3))
    # genre balance among the small subset
    try:
        sub_i = header[0].index("set") if "set" in header[0] else None
        idx_subset = [i for i, (a, b) in enumerate(zip(header[0], header[1]))
                      if b.strip() == "subset"]
        idx_genre = [i for i, (a, b) in enumerate(zip(header[0], header[1]))
                     if b.strip() == "genre_top"]
        if idx_subset and idx_genre:
            gi, si = idx_genre[0], idx_subset[0]
            counts = {}
            for r in rows[3:]:
                if len(r) > max(gi, si) and r[si].strip().lower() == "small":
                    counts[r[gi]] = counts.get(r[gi], 0) + 1
            check("fma small genre balance",
                  len(counts) == EXPECT["fma_small"]["genres"]
                  and all(near(v, EXPECT["fma_small"]["per_genre"], 5) for v in counts.values()),
                  "%d genres: %s" % (len(counts), dict(sorted(counts.items()))))
    except Exception as e:
        check("fma small genre balance", False, "parse error: %s" % type(e).__name__)


def validate_mtat(raw):
    e = EXPECT["mtat"]
    print("\nMagnaTagATune — expect %d clips, %d tags" % (e["mp3"], e["tags"]))
    m = raw / "magnatagatune"
    z = m / "mp3.zip"
    if z.exists():
        real = os.path.realpath(z)
        check("mtat mp3.zip integrity", zip_ok(real) is True,
              "%.2f GB" % (os.path.getsize(real) / 2**30))
    audio = m / "audio"
    if audio.exists():
        n = count_files(audio, ".mp3")
        check("mtat mp3 count", near(n, e["mp3"], e["tol"]),
              "%d found, expected %d (+/-%d)" % (n, e["mp3"], e["tol"]))
        mp3s = sorted(glob.glob(str(audio / "**" / "*.mp3"), recursive=True))
        ok, detail = decodes(mp3s)
        if ok is not None:
            check("mtat audio decodes", ok, detail)

    ann = m / "annotations_final.csv"
    if not ann.exists():
        return check("mtat annotations_final.csv", False, "missing — no labels")
    with open(os.path.realpath(ann), newline="", encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t")
        header = next(r)
        rows = sum(1 for _ in r)
    if len(header) == 1:                       # not tab separated after all
        with open(os.path.realpath(ann), newline="", encoding="utf-8") as fh:
            r = csv.reader(fh)
            header = next(r)
            rows = sum(1 for _ in r)
    tags = [h for h in header if h not in ("clip_id", "mp3_path")]
    check("mtat annotation rows", near(rows, e["rows"], e["tol"]),
          "%d rows, expected %d" % (rows, e["rows"]))
    check("mtat tag vocabulary", near(len(tags), e["tags"], 2),
          "%d tag columns, expected %d" % (len(tags), e["tags"]))
    check("mtat clip_id/mp3_path columns", "clip_id" in header and "mp3_path" in header,
          "columns the loader indexes by name")


def validate_deam(raw):
    e = EXPECT["deam"]
    print("\nDEAM — expect %d mp3, valence/arousal annotations, %d metadata years"
          % (e["mp3"], e["meta_years"]))
    d = raw / "deam"
    audio = d / "audio"
    if audio.exists():
        n = count_files(audio, ".mp3")
        check("deam mp3 count", near(n, e["mp3"], e["tol"]),
              "%d found, expected %d (+/-%d)" % (n, e["mp3"], e["tol"]))
        mp3s = sorted(glob.glob(str(audio / "**" / "*.mp3"), recursive=True))
        ok, detail = decodes(mp3s)
        if ok is not None:
            check("deam audio decodes", ok, detail)
    else:
        check("deam audio extracted", False, "missing %s" % audio)

    ann = list(glob.glob(str(d / "annotations" / "**" / "*.csv"), recursive=True))
    static = [p for p in ann if "static" in os.path.basename(p).lower()]
    check("deam annotation csvs", len(ann) > 0, "%d csv files (%d static)" % (len(ann), len(static)))
    if static:
        with open(static[0], newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        has_va = any("valence" in h.lower() for h in header) and \
                 any("arousal" in h.lower() for h in header)
        check("deam valence/arousal columns", has_va, ", ".join(header[:5]))

    metas = sorted(glob.glob(str(d / "metadata" / "metadata_*.csv")))
    ok = check("deam metadata years", len(metas) >= e["meta_years"],
               "%d files: %s" % (len(metas), [os.path.basename(m) for m in metas]))
    if ok:
        with open(metas[0], newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        wanted = ("genre", "artist", "title")
        found = [w for w in wanted if any(w in h.lower() for h in header)]
        check("deam metadata pseudo-caption fields", len(found) >= 2,
              "found %s of %s" % (found, list(wanted)))


def validate_musiccaps(raw):
    e = EXPECT["musiccaps"]
    print("\nMusicCaps — expect %d captions, %d retrievable clips"
          % (e["csv_rows"], e["parquet_rows"]))
    m = raw / "musiccaps"
    pub = m / "musiccaps-public.csv"
    ytids_csv = set()
    if pub.exists():
        with open(pub, newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            rows = list(r)
        ytids_csv = {x.get("ytid", "") for x in rows}
        check("musiccaps caption rows", near(len(rows), e["csv_rows"], e["tol"]),
              "%d rows, expected %d" % (len(rows), e["csv_rows"]))
        check("musiccaps is_audioset_eval flag", "is_audioset_eval" in (rows[0] if rows else {}),
              "official split partition available")
        if rows and "is_audioset_eval" in rows[0]:
            n_eval = sum(1 for x in rows if str(x["is_audioset_eval"]).strip() in ("1", "True", "true"))
            check("musiccaps eval partition non-trivial", 0 < n_eval < len(rows),
                  "%d eval / %d train" % (n_eval, len(rows) - n_eval))
    else:
        check("musiccaps-public.csv", False, "missing — official eval split unavailable")

    shards = sorted(glob.glob(str(m / "**" / "*.parquet"), recursive=True))
    if not shards:
        return check("musiccaps parquet shards", False, "no audio shards found")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return check("musiccaps parquet rows", False, "pyarrow not installed")
    total = 0
    cols = set()
    ytids = set()
    for f in shards:
        pf = pq.ParquetFile(f)
        total += pf.metadata.num_rows
        cols |= set(pf.schema.names)
    check("musiccaps parquet rows", near(total, e["parquet_rows"], e["tol"]),
          "%d rows across %d shards, expected %d" % (total, len(shards), e["parquet_rows"]))
    need = {"caption", "youtube_id"}
    check("musiccaps parquet columns", need <= cols,
          "have %s" % sorted(cols))
    has_audio = any(c in cols for c in ("audio", "audio_bytes", "array", "bytes"))
    check("musiccaps parquet carries audio", has_audio, "audio column present")

    if ytids_csv and "youtube_id" in cols:
        tbl = pq.read_table(shards[0], columns=["youtube_id"])
        ytids = set(tbl.column("youtube_id").to_pylist())
        overlap = len(ytids & ytids_csv)
        check("musiccaps parquet ids match the caption csv", overlap > 0.9 * len(ytids),
              "%d/%d ids in shard 0 found in the csv" % (overlap, len(ytids)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data/raw")
    ap.add_argument("--dataset", default="all",
                    choices=["all", "fma", "mtat", "deam", "musiccaps"])
    args = ap.parse_args()
    raw = pathlib.Path(args.raw)

    print("Validating datasets under %s" % raw)
    runners = {"fma": validate_fma, "mtat": validate_mtat,
               "deam": validate_deam, "musiccaps": validate_musiccaps}
    for name, fn in runners.items():
        if args.dataset in ("all", name):
            try:
                fn(raw)
            except Exception as exc:
                check("%s validator" % name, False,
                      "crashed: %s: %s" % (type(exc).__name__, exc))

    failed = [n for n, ok, _ in results if not ok]
    print("\n%d checks, %d passed, %d failed" % (len(results), len(results) - len(failed), len(failed)))
    if failed:
        print("\nFAILED: %s" % ", ".join(failed))
        print("\nReplacement sources:")
        for key, urls in ALTERNATIVES.items():
            if any(key.split("_")[0] in f for f in failed):
                print("  %s:" % key)
                for u in urls:
                    print("    - %s" % u)
        sys.exit(1)
    print("All datasets validated.")


if __name__ == "__main__":
    main()
