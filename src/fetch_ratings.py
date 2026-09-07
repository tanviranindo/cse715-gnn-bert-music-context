"""Download submitted ratings from the study's Blob store.

The listening study collects one JSON blob per rater. This pulls them into
`results/human_eval/ratings/` where `human_eval.py score` expects them, so the
researcher never has to chase attachments.

    python src/fetch_ratings.py                 # needs BLOB_READ_WRITE_TOKEN

Nothing here interprets a rating: files are written exactly as submitted.
"""

import argparse
import json
import os
import pathlib
import urllib.request

LIST_URL = "https://blob.vercel-storage.com/?prefix=%s&limit=1000"


def fetch(token: str, prefix: str, out_dir: pathlib.Path) -> list[pathlib.Path]:
    req = urllib.request.Request(
        LIST_URL % prefix, headers={"Authorization": "Bearer %s" % token}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        listing = json.load(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    superseded = []
    # A rater who redoes the study submits again rather than overwriting, so the
    # store legitimately holds several files for one person. Keeping them all
    # would inflate the rater count and weight that person's opinion twice, so
    # only their most recent submission is downloaded.
    newest: dict[str, dict] = {}
    for blob in listing.get("blobs", []):
        with urllib.request.urlopen(blob["url"], timeout=60) as r:
            body = r.read()
        doc = json.loads(body)  # refuse anything that is not valid JSON
        who = str(doc.get("rater", "")).strip().lower()
        stamp = blob.get("uploadedAt", "")
        prev = newest.get(who)
        if prev and prev["stamp"] >= stamp:
            superseded.append((doc.get("rater"), blob["pathname"]))
            continue
        if prev:
            superseded.append((prev["doc"].get("rater"), prev["pathname"]))
        newest[who] = {"doc": doc, "body": body, "stamp": stamp,
                       "pathname": blob["pathname"]}

    for entry in newest.values():
        dest = out_dir / pathlib.Path(entry["pathname"]).name
        dest.write_bytes(entry["body"])
        written.append(dest)

    for rater, path in superseded:
        print("superseded, not downloaded: %s (%s)" % (rater, path))
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="ratings/")
    p.add_argument("--out", default="results/human_eval/ratings")
    p.add_argument("--token", default=os.environ.get("BLOB_READ_WRITE_TOKEN", ""))
    args = p.parse_args()

    if not args.token:
        raise SystemExit(
            "No blob token. Run `vercel env pull` in infra/listening-study, or:\n"
            "  BLOB_READ_WRITE_TOKEN=... python src/fetch_ratings.py"
        )

    files = fetch(args.token, args.prefix, pathlib.Path(args.out))
    if not files:
        print("no submissions yet at prefix %r" % args.prefix)
        return
    for f in files:
        doc = json.loads(f.read_text())
        answered = sum(1 for r in doc.get("ratings", []) if r.get("rating") is not None)
        print("%-44s %-20s %d rated" % (f.name, doc.get("rater", "?"), answered))
    print("\n%d submission(s) in %s" % (len(files), args.out))


if __name__ == "__main__":
    main()
