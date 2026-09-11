"""Regenerate the report's figures from the committed metrics.

The spec (S4.1) requires Macro-F1 / Micro-F1 curves against training epoch for
Task 1. This was previously produced by an ad-hoc script that was never
committed, so the figure could not be rebuilt when the numbers changed. It is a
module now so `python -m src.make_plots` reproduces the figure exactly. The
report includes this canonical results path directly; there is no second copy.

    python -m src.make_plots --results results --out results/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK1 = [
    ("metrics_task1_mtat.json", "MagnaTagATune"),
    ("metrics_task1_musiccaps.json", "MusicCaps (raw captions)"),
    ("metrics_task1_musiccaps_stripped.json", "MusicCaps (stripped)"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/plots")
    args = p.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.0))

    for ax, (fname, title) in zip(axes, TASK1):
        path = res / fname
        if not path.exists():
            ax.set_title(title + "\n(missing)")
            continue
        d = json.loads(path.read_text())
        hist = d["history"]
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["val_macro_f1"] for h in hist], "o-", label="Macro-F1")
        ax.plot(ep, [h["val_micro_f1"] for h in hist], "s-", label="Micro-F1")
        sel = d.get("selected_epoch")
        if sel:
            ax.axvline(sel, ls="--", c="grey", lw=1)
            ax.text(sel, ax.get_ylim()[0], " restored", fontsize=7,
                    va="bottom", color="grey")
        ax.set_title("%s\ntest micro %.3f / macro %.3f"
                     % (title, d["test"]["micro_f1"], d["test"]["macro_f1"]),
                     fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation F1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Task 4: the loss keeps falling long after retrieval stops improving,
    # which is the whole argument for restoring the best validation epoch.
    ax = axes[3]
    t4 = res / "metrics_task4.json"
    if t4.exists():
        d = json.loads(t4.read_text())
        hist = d["history"]
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["loss"] for h in hist], "o-", c="tab:red", label="InfoNCE loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("InfoNCE loss", color="tab:red")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(ep, [h["val_c2a_R@10"] for h in hist], "s-", c="tab:blue",
                 label="val R@10")
        ax2.set_ylabel("validation R@10", color="tab:blue")
        best = d.get("best_epoch")
        if best:
            ax.axvline(best, ls="--", c="grey", lw=1)
        ax.set_title("Task 4 (%s, gallery %s)\nbest epoch %s"
                     % (d.get("split_kind", "?"), d.get("gallery_size"), best),
                     fontsize=9)
    else:
        ax.set_title("Task 4\n(missing)")

    fig.tight_layout()
    canonical = out / "training_curves.png"
    fig.savefig(canonical, dpi=150)
    print("[out ]", canonical)


if __name__ == "__main__":
    main()
