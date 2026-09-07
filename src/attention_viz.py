"""Attention visualization for the Task 1 example predictions (PDF S4.1).

The brief asks for "5 example predictions with attention visualization
(optional)". The five predictions were already committed; this reads the same
fine-tuned checkpoint back, re-runs those exact inputs with
`output_attentions=True`, and reports which input tokens the classifier's
pooled representation actually rests on.

What is plotted. DistilBERT pools from the [CLS] position, so the row that
matters is the attention *from* [CLS] *to* every other token, averaged over the
heads of the final layer. That row is the closest thing to "which words did the
tag decision read", and it is what the heatmap shows: one row per example, one
column per token.

    python src/attention_viz.py --dataset mtat

Writes results/attention_task1_<dataset>.json (per-token weights, so the figure
is regenerable and the numbers are auditable) and
results/plots/attention_task1_<dataset>.png.
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["model_state"], ck["vocab"], ck["config"], ck.get("threshold", 0.5)


def build_encoder(state, model_name):
    """Rebuild just the encoder; the head is not needed to read attention."""
    encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    enc_state = {
        k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if unexpected:
        raise RuntimeError("unexpected keys in encoder state: %s" % unexpected[:5])
    if missing:
        print("note: %d encoder tensors not in checkpoint (%s...)" % (len(missing), missing[:2]))
    encoder.eval()
    return encoder


def cls_attention(encoder, tokenizer, text, max_length):
    """Final-layer [CLS] attention row, averaged over heads."""
    enc = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        out = encoder(**enc, output_attentions=True)
    # (batch, heads, query, key) -> take last layer, mean over heads, CLS query row
    last = out.attentions[-1][0]
    row = last.mean(dim=0)[0]
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, row.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mtat", choices=["mtat", "musiccaps", "musiccaps_stripped"])
    ap.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    results = pathlib.Path(args.results)
    ck_path = pathlib.Path(args.checkpoint_dir) / ("task1_%s.pt" % args.dataset)
    ex_path = results / ("examples_task1_%s.json" % args.dataset)

    state, vocab, config, threshold = load_checkpoint(ck_path)
    model_name = config["model"]
    max_length = config.get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = build_encoder(state, model_name)

    examples = json.loads(ex_path.read_text())
    print("%d examples, encoder %s, max_length %d" % (len(examples), model_name, max_length))

    records = []
    for ex in examples:
        tokens, weights = cls_attention(encoder, tokenizer, ex["text"], max_length)
        order = np.argsort(-weights)
        records.append({
            "clip_id": ex["clip_id"],
            "text": ex["text"],
            "true": ex["true"],
            "predicted": ex["predicted"][:5],
            "tokens": tokens,
            "cls_attention": [round(float(w), 5) for w in weights],
            "top_tokens": [
                {"token": tokens[i], "weight": round(float(weights[i]), 5)}
                for i in order[:5]
            ],
        })

    out_json = results / ("attention_task1_%s.json" % args.dataset)
    out_json.write_text(json.dumps({
        "note": (
            "Final-layer [CLS] attention averaged over heads, for the five committed "
            "Task 1 example predictions. Special tokens are retained because they "
            "carry real mass and hiding them would overstate the content words."
        ),
        "dataset": args.dataset,
        "model": model_name,
        "threshold": threshold,
        "examples": records,
    }, indent=2) + "\n")

    # ---- figure: one row per example, tokens on the x axis ----
    width = max(len(r["tokens"]) for r in records)
    grid = np.full((len(records), width), np.nan)
    for i, r in enumerate(records):
        grid[i, :len(r["cls_attention"])] = r["cls_attention"]

    fig, ax = plt.subplots(figsize=(min(1.05 * width, 15), 0.85 * len(records) + 1.9))
    cmap = plt.get_cmap("BuPu").copy()
    cmap.set_bad("#f2f2f0")
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0.0)

    ax.set_yticks(range(len(records)))
    ax.set_yticklabels(
        ["%s\n%s" % (r["clip_id"], ", ".join(r["true"])[:26]) for r in records], fontsize=8
    )
    ax.set_xticks(range(width))
    ax.set_xticklabels([""] * width)
    for i, r in enumerate(records):
        for j, tok in enumerate(r["tokens"]):
            w = r["cls_attention"][j]
            ax.text(j, i, tok.replace("##", ""), ha="center", va="center",
                    fontsize=6.5, rotation=90,
                    color="white" if w > 0.5 * np.nanmax(grid) else "#222222")
    ax.set_xlabel("input tokens (per example)", fontsize=9)
    ax.set_title(
        "Task 1 (%s): final-layer [CLS] attention over the five example inputs" % args.dataset,
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.75, label="attention weight")
    fig.tight_layout()

    out_png = results / "plots" / ("attention_task1_%s.png" % args.dataset)
    fig.savefig(out_png, dpi=170)
    print("wrote %s and %s" % (out_json, out_png))

    for r in records:
        top = ", ".join("%s %.2f" % (t["token"], t["weight"]) for t in r["top_tokens"][:3])
        print("  %-8s %-32s -> %s" % (r["clip_id"], r["text"][:32], top))


if __name__ == "__main__":
    main()
