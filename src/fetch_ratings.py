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
    for blob in listing.get("blobs", []):
        name = pathlib.Path(blob["pathname"]).name
        dest = out_dir / name
        with urllib.request.urlopen(blob["url"], timeout=60) as r:
            body = r.read()
        json.loads(body)  # refuse to write anything that is not valid JSON
        dest.write_bytes(body)
        written.append(dest)
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
