"""Two t-SNE views of the fused space: by genre and by mood (PDF S4.3).

The specification asks for the projection "coloured by genre and mood". A single
panel coloured by whichever tag happens to be most frequent per clip answers
neither question cleanly, because MagnaTagATune's top-50 vocabulary mixes genre
labels, mood labels, instrument labels and production descriptors in one list.
This splits the vocabulary explicitly and draws one panel per axis, leaving
clips with no tag on that axis out of that panel rather than silently colouring
them as something they are not.

    python -m src.plot_tsne_views
"""

import argparse
import json
import pathlib

# MagnaTagATune's top-50 list is not organised by kind, so the split is ours and
# is stated rather than assumed. Instrument and production tags (guitar, piano,
# loud, quiet) belong to neither axis and are excluded from both panels.
GENRE = {
    "classical", "rock", "pop", "jazz", "techno", "electronic", "dance",
    "country", "metal", "hard rock", "opera", "baroque", "new age", "india",
    "eastern", "foreign", "ambient",
}
MOOD = {
    "slow", "fast", "quiet", "loud", "weird", "soft", "modern", "beat",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tsne", default="results/tsne_task3.json")
    p.add_argument("--out", default="results/plots/tsne_task3_views.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(pathlib.Path(args.tsne).read_text())
    coords, tags = d["coords"], d["primary_tag"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, vocab) in zip(axes, (("Genre", GENRE), ("Mood / character", MOOD))):
        shown = sorted({t for t in tags if t in vocab})
        cmap = plt.get_cmap("tab20")
        # Clips with no tag on this axis are drawn once, in grey, behind.
        other = [c for c, t in zip(coords, tags) if t not in vocab]
        if other:
            ax.scatter([c[0] for c in other], [c[1] for c in other], s=4,
                       c="#D9DDDB", linewidths=0, label="_no %s tag" % name.lower())
        for i, t in enumerate(shown):
            pts = [c for c, tag in zip(coords, tags) if tag == t]
            ax.scatter([c[0] for c in pts], [c[1] for c in pts], s=6,
                       color=cmap(i % 20), linewidths=0, label=t)
        ax.set_title("%s (%d of 50 tags, %d clips)"
                     % (name, len(shown), sum(1 for t in tags if t in vocab)),
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, markerscale=2, loc="upper right", frameon=False,
                  ncol=2 if len(shown) > 8 else 1)

    fig.suptitle("t-SNE of the fused representation z, by genre and by mood", fontsize=12)
    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote %s" % out)
    for name, vocab in (("genre", GENRE), ("mood", MOOD)):
        n = sum(1 for t in tags if t in vocab)
        print("  %-6s %d clips across %d tags"
              % (name, n, len({t for t in tags if t in vocab})))


if __name__ == "__main__":
    main()
