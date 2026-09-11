"""Task 4 data-scale analysis: is retrieval limited by the model or by the data?

MusicCaps' official AudioSet partition puts 52% of the corpus in the eval set,
which leaves roughly 2.2k training pairs --- small for a contrastive objective.
This fits R@10 against training-set size to say whether the curve has saturated
(the model is the constraint) or is still climbing (the data is).

Only the TRAIN split is subsampled; val, test and the 2,773-clip gallery are
identical across runs, so R@K is directly comparable between points.

    python src/analyze_scale.py

Writes results/metrics_task4_scale.json and results/plots/task4_scale.png.
"""

import argparse
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_points(results: str | pathlib.Path = "results"):
    """(n_train, metrics) for each fraction, read from the run artifacts."""
    results = pathlib.Path(results)
    files = {
        0.25: results / "metrics_task4_frac025.json",
        0.50: results / "metrics_task4_frac050.json",
        0.75: results / "metrics_task4_frac075.json",
        1.00: results / "metrics_task4_gnnlr.json",
    }
    points = []
    for frac, path in sorted(files.items()):
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        points.append({
            "fraction": frac,
            "n_train": d["split_sizes"]["train"],
            "gallery": d["gallery_size"],
            "c2a": d["caption_to_audio"],
            "a2c": d["audio_to_caption"],
            "random_R@10": d["random_baseline_R@10"],
        })
    return points


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return slope, intercept, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", type=pathlib.Path)
    args = parser.parse_args()
    results = args.results
    pts = load_points(results)
    if len(pts) < 3:
        raise SystemExit("need at least 3 scale points, found %d" % len(pts))

    galleries = {p["gallery"] for p in pts}
    assert len(galleries) == 1, "gallery must be identical across points: %s" % galleries

    xs = [math.log(p["n_train"]) for p in pts]
    ys = [p["c2a"]["R@10"] for p in pts]
    slope, intercept, r2 = linfit(xs, ys)

    # What would it take to reach a usable R@10, extrapolating the fit?
    target = 0.10
    need_ln = (target - intercept) / slope if slope > 0 else float("inf")
    need_n = math.exp(need_ln) if need_ln < 30 else float("inf")
    full_n = pts[-1]["n_train"]

    last_gain = (pts[-1]["c2a"]["R@10"] - pts[-2]["c2a"]["R@10"])
    prev_gain = (pts[-2]["c2a"]["R@10"] - pts[-3]["c2a"]["R@10"])

    doc = {
        "note": (
            "Task 4 retrieval against training-set size. Only the train split is "
            "subsampled; val, test and the gallery are held fixed so R@K is "
            "comparable across points. Single seed per point."
        ),
        "gallery_size": pts[0]["gallery"],
        "points": pts,
        "log_linear_fit_R@10_vs_ln_n": {
            "slope_per_ln_unit": round(slope, 5),
            "gain_per_doubling": round(slope * math.log(2), 5),
            "intercept": round(intercept, 5),
            "r_squared": round(r2, 4),
        },
        "extrapolation": {
            "target_R@10": target,
            "pairs_required": None if need_n == float("inf") else int(need_n),
            "multiple_of_available": None if need_n == float("inf")
            else round(need_n / full_n, 1),
            "caveat": (
                "A log-linear extrapolation over one order of magnitude is "
                "indicative, not a prediction; it is reported to characterise the "
                "regime, not to forecast a specific corpus size."
            ),
        },
        "saturation_check": {
            "gain_50_to_75": round(prev_gain, 5),
            "gain_75_to_100": round(last_gain, 5),
            "reading": (
                "still climbing" if last_gain > 0.2 * prev_gain
                else "possible flattening in the last interval, but single-seed "
                     "noise is the same order as this difference"
            ),
        },
    }
    (results / "metrics_task4_scale.json").write_text(json.dumps(doc, indent=2) + "\n")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ns = [p["n_train"] for p in pts]
    r10 = [p["c2a"]["R@10"] for p in pts]
    r5 = [p["c2a"]["R@5"] for p in pts]
    r1 = [p["c2a"]["R@1"] for p in pts]

    ax.plot(ns, r10, "o-", color="#1f3b73", lw=2, label="R@10")
    ax.plot(ns, r5, "s--", color="#4a7ab5", lw=1.5, label="R@5")
    ax.plot(ns, r1, "^:", color="#8fb3d9", lw=1.5, label="R@1")
    ax.axhline(pts[0]["random_R@10"], color="#a5323c", lw=1.2, ls="-.",
               label="random R@10")

    fitx = [min(ns), max(ns)]
    ax.plot(fitx, [slope * math.log(x) + intercept for x in fitx],
            color="#999999", lw=1, ls="-", zorder=0,
            label="log-linear fit (R$^2$=%.2f)" % r2)

    ax.set_xscale("log")
    ax.set_xlabel("training pairs (log scale) — val, test and 2,773-clip gallery fixed")
    ax.set_ylabel("recall on the official eval split")
    ax.set_title("Task 4: retrieval against training-set size", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    (results / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(results / "plots" / "task4_scale.png", dpi=170)

    print("points:")
    for p in pts:
        print("  %5d pairs (%.0f%%)  R@1 %.4f  R@5 %.4f  R@10 %.4f"
              % (p["n_train"], p["fraction"] * 100,
                 p["c2a"]["R@1"], p["c2a"]["R@5"], p["c2a"]["R@10"]))
    print("\nlog-linear fit: R@10 = %.5f*ln(n) %+.5f   (R^2 = %.3f)"
          % (slope, intercept, r2))
    print("gain per doubling of data: %+.4f R@10" % (slope * math.log(2)))
    if need_n != float("inf"):
        print("R@10 = %.2f would need ~%d pairs (%.0fx the %d available)"
              % (target, int(need_n), need_n / full_n, full_n))
    print("saturation: %s" % doc["saturation_check"]["reading"])
    print("\nwrote results/metrics_task4_scale.json and results/plots/task4_scale.png")


if __name__ == "__main__":
    main()
