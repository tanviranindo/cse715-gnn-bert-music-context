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

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<meta name="description" content="A short listening study: rate how well each music clip matches its written description.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%231F6156'/%3E%3Crect x='7' y='13' width='2.6' height='6' fill='%23fff'/%3E%3Crect x='12' y='9' width='2.6' height='14' fill='%23fff'/%3E%3Crect x='17' y='11' width='2.6' height='10' fill='%23fff'/%3E%3Crect x='22' y='14' width='2.6' height='4' fill='%23fff'/%3E%3C/svg%3E">
<title>Does the clip match the description?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Inter+Tight:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  color-scheme: light;
  --ground: #E9EDEC;
  --surface: #FFFFFF;
  --ink: #14181B;
  --ink-soft: #5A656C;
  --ink-faint: #8A9399;
  --rule: #D3DAD8;
  --accent: #1F6156;
  --accent-tint: #DCEAE5;
  --flag: #A8471F;
  --serif: Newsreader, Georgia, "Times New Roman", serif;
  --sans: "Inter Tight", system-ui, -apple-system, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.5;
  -webkit-text-size-adjust: 100%;
}
.bar { position: sticky; top: 0; z-index: 20; background: var(--ground);
  border-bottom: 1px solid var(--rule); }
.bar-in { max-width: 42rem; margin: 0 auto; padding: .7rem 1.15rem .55rem;
  display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.bar-name { font-weight: 600; font-size: .95rem; letter-spacing: -.01em; }
.bar-step { font-size: .85rem; color: var(--ink-soft); font-variant-numeric: tabular-nums; }
.track { height: 2px; background: var(--rule); }
.track > i { display: block; height: 100%; width: 0; background: var(--accent);
  transition: width .3s ease; }
main { max-width: 42rem; margin: 0 auto; padding: 1.5rem 1.15rem 6rem; }
h1 { font-family: var(--serif); font-weight: 400; font-size: clamp(1.7rem, 5vw, 2.3rem);
  line-height: 1.2; margin: .4rem 0 .8rem; letter-spacing: -.01em; }
h2 { font-family: var(--sans); font-size: 1rem; font-weight: 600; margin: 0 0 .5rem; }
p { margin: 0 0 1rem; max-width: 34rem; }
.lede { color: var(--ink-soft); }
.quiet { color: var(--ink-soft); font-size: .9rem; }
.caption {
  font-family: var(--serif); font-size: clamp(1.25rem, 3.6vw, 1.6rem);
  line-height: 1.42; margin: 0; color: var(--ink);
}
.caption-wrap { background: var(--surface); border: 1px solid var(--rule);
  border-left: 3px solid var(--accent); padding: 1.15rem 1.25rem; margin: 0 0 1.6rem; }
.ask { font-size: .95rem; color: var(--ink-soft); margin: 0 0 1.75rem; }
.clip { border-top: 1px solid var(--rule); padding: 1.4rem 0 .35rem; }
.clip:first-of-type { border-top: 0; padding-top: .35rem; }
.clip-head { display: flex; align-items: baseline; justify-content: space-between;
  gap: 1rem; margin: 0 0 .7rem; }
.clip-name { font-weight: 600; font-size: .95rem; }
.player { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #0d1113;
  border: 1px solid var(--rule); }
.player iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }
.altlink { font-size: .85rem; margin: .5rem 0 0; }
.altlink a { color: var(--accent); }
.scale { display: grid; grid-template-columns: repeat(5, 1fr); gap: .4rem;
  margin: .9rem 0 .3rem; }
.scale button {
  font: inherit; font-size: .95rem; font-weight: 500;
  min-height: 2.9rem; padding: .3rem;
  background: var(--surface); color: var(--ink);
  border: 1px solid var(--rule); border-radius: 2px; cursor: pointer;
}
.scale button:hover { border-color: var(--accent); }
.scale button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.scale button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent);
  color: #fff; }
.legend { display: flex; justify-content: space-between; font-size: .78rem;
  color: var(--ink-faint); }
.skip { margin: .55rem 0 0; font-size: .85rem; }
.skip button { font: inherit; font-size: .85rem; background: none; border: 0;
  color: var(--ink-soft); text-decoration: underline; cursor: pointer; padding: 0; }
.skip button[aria-pressed="true"] { color: var(--flag); font-weight: 500; }
.nav { position: fixed; left: 0; right: 0; bottom: 0; background: var(--surface);
  border-top: 1px solid var(--rule); }
.nav-in { max-width: 42rem; margin: 0 auto; padding: .75rem 1.15rem;
  display: flex; align-items: center; gap: .8rem;
  padding-bottom: calc(.75rem + env(safe-area-inset-bottom)); }
.nav .spacer { flex: 1; }
button.primary, button.ghost {
  font: inherit; font-weight: 500; font-size: .95rem; cursor: pointer;
  padding: .65rem 1.35rem; border-radius: 2px; min-height: 2.75rem;
}
button.primary { background: var(--accent); color: #fff; border: 1px solid var(--accent); }
button.primary:disabled { background: var(--rule); border-color: var(--rule);
  color: var(--ink-faint); cursor: not-allowed; }
button.ghost { background: transparent; color: var(--ink-soft); border: 1px solid transparent; }
button.ghost:hover { color: var(--ink); }
button.primary:focus-visible, button.ghost:focus-visible { outline: 2px solid var(--ink);
  outline-offset: 2px; }
.note { font-size: .9rem; color: var(--ink-soft); }
label.field { display: block; margin: 0 0 1.5rem; max-width: 22rem; }
label.field span { display: block; font-weight: 500; font-size: .9rem; margin: 0 0 .35rem; }
input[type=text] { font: inherit; width: 100%; padding: .6rem .7rem; min-height: 2.75rem;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 2px; color: var(--ink); }
input[type=text]:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.facts { border-top: 1px solid var(--rule); margin: 1.75rem 0 0; padding: 1rem 0 0; }
.facts div { display: flex; gap: 1rem; padding: .3rem 0; font-size: .9rem; }
.facts dt { color: var(--ink-soft); flex: 0 0 8.5rem; }
.facts dd { margin: 0; }
.status { font-size: .9rem; margin: 1rem 0 0; }
.status.err { color: var(--flag); }
.tick { font-family: var(--serif); font-size: 2.6rem; line-height: 1; color: var(--accent);
  margin: 0 0 .6rem; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head><body>

<header class="bar">
  <div class="bar-in">
    <span class="bar-name">Music description study</span>
    <span class="bar-step" id="step"></span>
  </div>
  <div class="track"><i id="fill"></i></div>
</header>

<main id="view"></main>

<div class="nav"><div class="nav-in" id="nav"></div></div>

<script>
const STUDY = __DATA__;
const ENDPOINT = "__ENDPOINT__";
const state = { rater: "", i: -1, ratings: {}, skipped: {}, sent: false };
const view = document.getElementById("view");
const nav = document.getElementById("nav");
const stepEl = document.getElementById("step");
const fillEl = document.getElementById("fill");

function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
  });
}
function key(qi, rank) { return qi + ":" + rank; }
function mmss(s) { return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }

function answered(qi) {
  return STUDY.queries[qi].clips.every(function (c) {
    return state.ratings[key(qi, c.rank)] != null || state.skipped[key(qi, c.rank)];
  });
}
function totalDone() {
  let n = 0;
  STUDY.queries.forEach(function (q, qi) {
    q.clips.forEach(function (c) {
      if (state.ratings[key(qi, c.rank)] != null || state.skipped[key(qi, c.rank)]) n++;
    });
  });
  return n;
}

function render() {
  if (state.i < 0) return renderWelcome();
  if (state.i >= STUDY.queries.length) return renderDone();
  return renderQuery(state.i);
}

function renderWelcome() {
  stepEl.textContent = "";
  fillEl.style.width = "0%";
  view.innerHTML =
    '<h1>Does the music match the description?</h1>' +
    '<p class="lede">You will see ten short written descriptions of music. Each one ' +
    'comes with three clips that a machine-learning model picked out as matching it. ' +
    'Your job is to say how well each clip actually fits the words.</p>' +
    '<p class="lede">There are no right answers and nothing is being tested about you. ' +
    'Rate what you hear, not whether you like the music.</p>' +
    '<label class="field"><span>Your name or initials</span>' +
    '<input type="text" id="rater" autocomplete="name" placeholder="e.g. A. Rahman"></label>' +
    '<dl class="facts">' +
      '<div><dt>Time needed</dt><dd>About 15 minutes</dd></div>' +
      '<div><dt>What we record</dt><dd>Your ratings and the name you type. Nothing else.</dd></div>' +
      '<div><dt>Purpose</dt><dd>Evaluating a music retrieval model for a university ' +
        'neural networks course project</dd></div>' +
      '<div><dt>Audio</dt><dd>Plays from YouTube, cued to the exact ten seconds each ' +
        'description refers to</dd></div>' +
    '</dl>';
  nav.innerHTML = '<span class="spacer"></span>' +
    '<button class="primary" id="go">Start rating</button>';
  const input = document.getElementById("rater");
  input.value = state.rater;
  input.addEventListener("input", function () { state.rater = input.value; });
  document.getElementById("go").addEventListener("click", function () {
    if (!input.value.trim()) { input.focus(); return; }
    state.rater = input.value.trim();
    state.i = 0; render(); window.scrollTo(0, 0);
  });
}

function renderQuery(qi) {
  const q = STUDY.queries[qi];
  stepEl.textContent = "Description " + (qi + 1) + " of " + STUDY.queries.length;
  fillEl.style.width = ((totalDone() / STUDY.total) * 100).toFixed(1) + "%";

  let html =
    '<div class="caption-wrap"><p class="caption">' + esc(q.caption) + '</p></div>' +
    '<p class="ask">How well does each clip below fit that description?</p>';

  q.clips.forEach(function (c, idx) {
    const k = key(qi, c.rank);
    const chosen = state.ratings[k];
    const skipped = !!state.skipped[k];
    let player;
    if (c.ytid && c.start_s != null) {
      player =
        '<div class="player"><iframe src="https://www.youtube-nocookie.com/embed/' +
        encodeURIComponent(c.ytid) + '?start=' + c.start_s + '&end=' + c.end_s +
        '" allow="encrypted-media" referrerpolicy="strict-origin-when-cross-origin" ' +
        'title="Clip ' + (idx + 1) + '" loading="lazy"></iframe></div>' +
        '<p class="altlink">Not playing? <a href="https://www.youtube.com/watch?v=' +
        encodeURIComponent(c.ytid) + '&t=' + c.start_s + 's" target="_blank" ' +
        'rel="noopener">Open it on YouTube at ' + mmss(c.start_s) + '</a> and listen ' +
        'for ten seconds.</p>';
    } else {
      player = '<p class="altlink">This clip has no audio available. Please skip it.</p>';
    }
    html +=
      '<section class="clip" data-q="' + qi + '" data-rank="' + c.rank + '">' +
        '<div class="clip-head"><span class="clip-name">Clip ' + (idx + 1) + ' of 3</span></div>' +
        player +
        '<div class="scale">' +
          [1, 2, 3, 4, 5].map(function (v) {
            return '<button type="button" data-v="' + v + '" aria-pressed="' +
              (chosen === v) + '">' + v + '</button>';
          }).join("") +
        '</div>' +
        '<div class="legend"><span>1 — unrelated</span><span>5 — matches closely</span></div>' +
        '<p class="skip"><button type="button" data-skip="1" aria-pressed="' + skipped +
          '">' + (skipped ? "Marked as could not listen" : "I could not listen to this clip") +
          '</button></p>' +
      '</section>';
  });
  view.innerHTML = html;

  view.querySelectorAll(".clip").forEach(function (sec) {
    const qq = Number(sec.dataset.q), rank = Number(sec.dataset.rank), k = key(qq, rank);
    sec.querySelectorAll(".scale button").forEach(function (b) {
      b.addEventListener("click", function () {
        state.ratings[k] = Number(b.dataset.v);
        delete state.skipped[k];
        render();
      });
    });
    sec.querySelector("[data-skip]").addEventListener("click", function () {
      if (state.skipped[k]) { delete state.skipped[k]; }
      else { state.skipped[k] = true; delete state.ratings[k]; }
      render();
    });
  });

  const last = qi === STUDY.queries.length - 1;
  nav.innerHTML =
    (qi > 0 ? '<button class="ghost" id="back">Back</button>' : '') +
    '<span class="spacer"></span>' +
    '<button class="primary" id="next"' + (answered(qi) ? '' : ' disabled') + '>' +
    (last ? "Finish and send" : "Continue") + '</button>';
  const back = document.getElementById("back");
  if (back) back.addEventListener("click", function () {
    state.i--; render(); window.scrollTo(0, 0);
  });
  document.getElementById("next").addEventListener("click", function () {
    state.i++; render(); window.scrollTo(0, 0);
  });
}

function payload() {
  const rows = [];
  STUDY.queries.forEach(function (q, qi) {
    q.clips.forEach(function (c) {
      const k = key(qi, c.rank);
      rows.push({
        query_index: qi, rank: c.rank, clip_id: c.clip_id,
        rating: state.ratings[k] != null ? state.ratings[k] : null
      });
    });
  });
  return { rater: state.rater, ratings: rows };
}

function download() {
  const blob = new Blob([JSON.stringify(payload(), null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "rating_" + state.rater.replace(/[^A-Za-z0-9_-]/g, "_") + ".json";
  a.click();
}

function renderDone() {
  stepEl.textContent = "Finished";
  fillEl.style.width = "100%";
  const rated = Object.keys(state.ratings).length;
  view.innerHTML =
    '<p class="tick">Thank you</p>' +
    '<h1>Your ratings are recorded</h1>' +
    '<p class="lede" id="msg">Sending your ' + rated + ' ratings…</p>' +
    '<p class="quiet">You can close this page once it says they are saved. ' +
    'If sending fails you can download the file instead and send it to the researcher.</p>' +
    '<p class="status" id="status"></p>';
  nav.innerHTML = '<span class="spacer"></span>' +
    '<button class="ghost" id="dl">Download a copy</button>' +
    '<button class="primary" id="retry" hidden>Try sending again</button>';
  document.getElementById("dl").addEventListener("click", download);
  document.getElementById("retry").addEventListener("click", send);
  send();
}

function send() {
  const msg = document.getElementById("msg");
  const status = document.getElementById("status");
  const retry = document.getElementById("retry");
  if (!ENDPOINT) {
    msg.textContent = "Your ratings are ready to send.";
    status.className = "status";
    status.textContent = "Download the file and send it to the researcher.";
    return;
  }
  status.className = "status";
  status.textContent = "";
  fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload())
  }).then(function (r) {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }).then(function () {
    state.sent = true;
    msg.textContent = "Saved. Nothing else is needed from you.";
    status.textContent = "";
    retry.hidden = true;
  }).catch(function (e) {
    msg.textContent = "Your ratings could not be sent automatically.";
    status.className = "status err";
    status.textContent = "Download a copy and send it to the researcher. (" + e.message + ")";
    retry.hidden = false;
  });
}

window.addEventListener("beforeunload", function (e) {
  if (state.i >= 0 && !state.sent && totalDone() > 0) { e.preventDefault(); e.returnValue = ""; }
});

render();
</script>
</body></html>
"""


def build(examples_path, endpoint="", out_dir=None):
    """Render the self-contained rating page.

    The stimuli are embedded as JSON and the page renders one description per
    screen, so a rater never scrolls past three clips. `endpoint` is the URL the
    finished ratings POST to; with it empty the page falls back to a download,
    which is what a `file://` copy gets.
    """
    examples = json.loads(pathlib.Path(examples_path).read_text())

    queries, total, no_audio = [], 0, 0
    for ex in examples:
        clips = []
        for rank, c in enumerate(ex["top3"], start=1):
            total += 1
            # A caption describes one ten-second window, usually not at 0:00.
            # Without the offset we cannot point a rater at the right audio, so
            # the clip is offered without a player rather than with a wrong one.
            start_s = c.get("start_s")
            if not c.get("ytid") or start_s is None:
                no_audio += 1
                start_s = None
            clips.append({
                "rank": rank,
                "clip_id": c.get("clip_id", ""),
                "ytid": c.get("ytid", ""),
                "start_s": start_s,
                "end_s": c.get("end_s", (start_s + 10) if start_s is not None else None),
            })
        queries.append({"caption": ex["query_caption"], "clips": clips})

    data = json.dumps({"queries": queries, "total": total}, indent=1)
    page = (PAGE.replace("__DATA__", data)
                .replace("__ENDPOINT__", endpoint))

    STUDY.mkdir(parents=True, exist_ok=True)
    RATINGS.mkdir(parents=True, exist_ok=True)
    written = [STUDY / "rating_sheet.html"]
    if out_dir:
        d = pathlib.Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        written.append(d / "index.html")
    for p in written:
        p.write_text(page)

    print("built %d clips across %d descriptions%s"
          % (total, len(examples),
             "" if not no_audio else " (%d without audio)" % no_audio))
    for p in written:
        print("  wrote %s" % p)
    print("submissions: %s" % (endpoint or "download only (no endpoint set)"))
    print("needs at least %d raters before `score` will run" % MIN_RATERS)


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
    ap.add_argument("--examples",
                    default="results/retrieval_examples/task4_examples_gnnlr.json")
    ap.add_argument("--endpoint", default="",
                    help="URL the finished ratings POST to; empty means the page "
                         "offers a download instead")
    ap.add_argument("--out-dir", default=None,
                    help="also write index.html here (the deployable site)")
    args = ap.parse_args()
    if args.mode == "build":
        build(args.examples, endpoint=args.endpoint, out_dir=args.out_dir)
    else:
        score(args.examples)


if __name__ == "__main__":
    main()
