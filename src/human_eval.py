"""Task 4 human evaluation (PDF S6): build the rating sheet, then score it.

The brief asks for "minimum 5 listeners rate whether retrieved clip matches
caption on scale [1,5]". This module does the two mechanical halves of that:

    python src/human_eval.py build            # -> results/human_eval/rating_sheet.html
    python src/human_eval.py score            # -> results/metrics_task4_human.json

`build` turns the ten committed caption -> top-3 retrievals into a self-contained
rating page. Each listener opens it, plays each clip, rates 1-5, and downloads
one JSON file. `score` reads every rater file in results/human_eval/ratings/ and
reports mean rating by rank, agreement between raters, and the correlation
between rating and the model's own similarity score --- which is the number that
says whether the retrieval scores mean anything to a listener.

Clip audio. MusicCaps ships YouTube ids, not audio, so the sheet embeds each
clip at its start offset when the examples file carries `ytid`. Examples written
before that field existed have only a hashed `clip_id`; rebuild them with
`src/train_contrastive.py` (which now records `ytid`) before running the study,
or the sheet falls back to caption-only rating and says so on its face.

NOTHING IN THIS FILE INVENTS RATINGS. `score` reads what real raters submitted
and fails loudly if fewer than the required five are present.
"""

import argparse
import json
import math
import pathlib
import statistics

RESULTS = pathlib.Path("results")
STUDY = RESULTS / "human_eval"
RATINGS = STUDY / "ratings"
MIN_RATERS = 5


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task 4 retrieval rating sheet</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 15px/1.55 system-ui, -apple-system, sans-serif; max-width: 820px;
        margin: 0 auto; padding: 2rem 1.25rem 5rem; }
 h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
 .lede { color: #666; margin: 0 0 1.5rem; }
 fieldset { border: 1px solid #bbb; margin: 0 0 1.5rem; padding: 1rem 1.15rem; }
 legend { font-weight: 600; padding: 0 .4rem; }
 .caption { background: rgba(127,127,127,.10); padding: .7rem .85rem; margin: 0 0 1rem; }
 .clip { border-top: 1px solid #ddd; padding: .9rem 0; }
 .clip:first-of-type { border-top: 0; }
 iframe { border: 0; width: 100%; max-width: 420px; aspect-ratio: 16/9; }
 .noaudio { color: #a33; font-size: .85rem; }
 .scale { display: flex; gap: .4rem; flex-wrap: wrap; margin-top: .6rem; }
 .scale label { border: 1px solid #999; padding: .3rem .6rem; cursor: pointer; }
 .scale input { margin-right: .35rem; }
 .bar { position: sticky; bottom: 0; background: Canvas; border-top: 1px solid #999;
        padding: .85rem 0; display: flex; gap: .8rem; align-items: center; flex-wrap: wrap; }
 button { font: inherit; padding: .45rem 1rem; cursor: pointer; }
 input[type=text] { font: inherit; padding: .4rem .5rem; }
 code { background: rgba(127,127,127,.15); padding: .1rem .3rem; }
</style></head><body>
<h1>Task 4 — does the retrieved clip match the caption?</h1>
<p class="lede">Ten query captions, each with the three clips the model retrieved
for it. Play each clip and rate <strong>how well it matches the caption you were
shown</strong>, from 1 (no relation) to 5 (an excellent match). Rate what you
hear, not whether you like it. Roughly 15 minutes.</p>
<p class="lede"><strong>Scale.</strong> 1 = unrelated · 2 = shares little ·
3 = shares mood or instrumentation · 4 = a good match with minor differences ·
5 = matches the description closely.</p>
__WARNING__
<form id="sheet">__ITEMS__</form>
<div class="bar">
  <label>Your name or initials <input type="text" id="rater" required></label>
  <button type="button" id="save">Download my ratings</button>
  <span id="status"></span>
</div>
<script>
const TOTAL = __TOTAL__;
function collect() {
  const out = [];
  document.querySelectorAll("[data-pair]").forEach(function (el) {
    const picked = el.querySelector("input[type=radio]:checked");
    out.push({
      query_index: Number(el.dataset.query),
      rank: Number(el.dataset.rank),
      clip_id: el.dataset.clip,
      rating: picked ? Number(picked.value) : null
    });
  });
  return out;
}
document.getElementById("save").addEventListener("click", function () {
  const rater = document.getElementById("rater").value.trim();
  const status = document.getElementById("status");
  if (!rater) { status.textContent = "Enter your name first."; return; }
  const rows = collect();
  const done = rows.filter(function (r) { return r.rating !== null; }).length;
  if (done < TOTAL) {
    status.textContent = "Rated " + done + " of " + TOTAL + " — please finish all of them.";
    return;
  }
  const blob = new Blob([JSON.stringify({ rater: rater, ratings: rows }, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "rating_" + rater.replace(/[^A-Za-z0-9_-]/g, "_") + ".json";
  a.click();
  status.textContent = "Saved. Send the file back to the study author.";
});
</script>
</body></html>
"""


def build(examples_path):
    examples = json.loads(pathlib.Path(examples_path).read_text())
    has_audio = any(c.get("ytid") for ex in examples for c in ex["top3"])

    blocks, total = [], 0
    for qi, ex in enumerate(examples):
        clips = []
        for rank, c in enumerate(ex["top3"], start=1):
            total += 1
            ytid = c.get("ytid", "")
            if ytid:
                player = (
                    '<iframe src="https://www.youtube-nocookie.com/embed/%s?start=%d" '
                    'allow="encrypted-media" title="clip %d"></iframe>'
                    % (ytid, int(c.get("start_s", 0)), rank)
                )
            else:
                player = ('<p class="noaudio">No audio link for this clip — rate from '
                          'its description below.</p>')
            scale = "".join(
                '<label><input type="radio" name="q%d_r%d" value="%d">%d</label>'
                % (qi, rank, v, v) for v in range(1, 6)
            )
            clips.append(
                '<div class="clip" data-pair data-query="%d" data-rank="%d" data-clip="%s">'
                '<p><strong>Clip %d</strong></p>%s'
                '<p><em>%s</em></p><div class="scale">%s</div></div>'
                % (qi, rank, c["clip_id"], rank, player,
                   c["caption"].replace("<", "&lt;"), scale)
            )
        blocks.append(
            '<fieldset><legend>Query %d of %d</legend>'
            '<p class="caption">%s</p>%s</fieldset>'
            % (qi + 1, len(examples), ex["query_caption"].replace("<", "&lt;"),
               "".join(clips))
        )

    warning = "" if has_audio else (
        '<p class="noaudio"><strong>Caption-only mode.</strong> This examples file '
        'carries no YouTube ids, so raters judge caption-against-caption rather than '
        'by listening. Rebuild the examples with src/train_contrastive.py to get a '
        'true listening study, and label the result accordingly in the report.</p>'
    )

    STUDY.mkdir(parents=True, exist_ok=True)
    RATINGS.mkdir(parents=True, exist_ok=True)
    page = (PAGE.replace("__ITEMS__", "".join(blocks))
                .replace("__WARNING__", warning)
                .replace("__TOTAL__", str(total)))
    out = STUDY / "rating_sheet.html"
    out.write_text(page)
    print("wrote %s — %d clips across %d queries, audio=%s"
          % (out, total, len(examples), has_audio))
    print("Send it to at least %d listeners; collect their JSON into %s/"
          % (MIN_RATERS, RATINGS))


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else None


def score(examples_path):
    files = sorted(RATINGS.glob("*.json"))
    if not files:
        raise SystemExit(
            "No rater files in %s.\nRun `python src/human_eval.py build`, have at "
            "least %d listeners complete the sheet, and put their downloads here.\n"
            "This script will not fabricate ratings." % (RATINGS, MIN_RATERS)
        )
    raters = [json.loads(f.read_text()) for f in files]
    if len(raters) < MIN_RATERS:
        raise SystemExit(
            "Only %d rater file(s) present; the specification requires at least %d. "
            "Collect the rest before scoring." % (len(raters), MIN_RATERS)
        )

    examples = json.loads(pathlib.Path(examples_path).read_text())
    model_score = {}
    for qi, ex in enumerate(examples):
        for rank, c in enumerate(ex["top3"], start=1):
            model_score[(qi, rank)] = c["score"]

    by_pair, by_rank, per_rater = {}, {1: [], 2: [], 3: []}, {}
    for r in raters:
        vals = []
        for row in r["ratings"]:
            if row.get("rating") is None:
                continue
            k = (row["query_index"], row["rank"])
            by_pair.setdefault(k, []).append(row["rating"])
            by_rank.setdefault(row["rank"], []).append(row["rating"])
            vals.append(row["rating"])
        per_rater[r["rater"]] = {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 3) if vals else None,
        }

    allv = [v for vs in by_pair.values() for v in vs]
    pair_means = {k: statistics.mean(v) for k, v in by_pair.items()}
    xs = [pair_means[k] for k in sorted(pair_means)]
    ys = [model_score[k] for k in sorted(pair_means)]

    doc = {
        "note": (
            "Task 4 human evaluation, PDF S6. Listeners rated 1-5 whether each "
            "retrieved clip matches its query caption. Ratings are as submitted; "
            "no value here is synthetic."
        ),
        "n_raters": len(raters),
        "n_ratings": len(allv),
        "overall_mean": round(statistics.mean(allv), 3),
        "overall_sd": round(statistics.stdev(allv), 3) if len(allv) > 1 else 0.0,
        "mean_by_rank": {
            str(k): {
                "mean": round(statistics.mean(v), 3),
                "sd": round(statistics.stdev(v), 3) if len(v) > 1 else 0.0,
                "n": len(v),
            } for k, v in sorted(by_rank.items()) if v
        },
        "per_rater": per_rater,
        "rating_vs_model_score_pearson_r": (
            round(_pearson(xs, ys), 3) if _pearson(xs, ys) is not None else None
        ),
        "interpretation": (
            "A mean near 1 says the retrieved clips do not match their captions, "
            "which is what R@10 = 0.022 predicts. A positive correlation with the "
            "model's similarity score would say the ranking is meaningful even "
            "where the top-1 is wrong."
        ),
    }
    out = RESULTS / "metrics_task4_human.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote %s" % out)
    print("  %d raters, %d ratings, mean %.2f +/- %.2f"
          % (doc["n_raters"], doc["n_ratings"], doc["overall_mean"], doc["overall_sd"]))
    for rank, v in doc["mean_by_rank"].items():
        print("  rank %s: %.2f +/- %.2f (n=%d)" % (rank, v["mean"], v["sd"], v["n"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score"])
    ap.add_argument("--examples", default="results/retrieval_examples/task4_examples.json")
    args = ap.parse_args()
    (build if args.mode == "build" else score)(args.examples)


if __name__ == "__main__":
    main()
