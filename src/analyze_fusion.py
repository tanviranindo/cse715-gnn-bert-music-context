"""Task 3 qualitative deliverables: t-SNE of z, and case studies (PDF S4.3).

    python -m src.analyze_fusion --checkpoint results/task3_crossattn.pt

Produces:
  results/plots/tsne_task3.png       t-SNE of the fused representation z
  results/tsne_task3.json            raw 2-D coords, so the plot is reproducible
  results/case_studies_task3.json    3 clips with graph structure + the caption
                                     tokens the graph vector attended to
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/task3_crossattn.pt")
    p.add_argument("--cache", default="/data/processed/mtat_graphs.pt")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--n-tsne", type=int, default=1500)
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torch_geometric.loader import DataLoader as GeoLoader
    from src import fusion_model, train_fusion
    from src.bert_encoder import load_tokenizer

    ckpt = torch.load(args.checkpoint, weights_only=False)
    vocab, cfg = ckpt["vocab"], ckpt["config"]
    tok = load_tokenizer()
    _, items = train_fusion.load_cache(args.cache, tok, args.max_length)
    train_fusion.attach_tag_vectors(items, vocab)
    _, _, test = train_fusion.artist_split(items, cfg.get("seed", 42))

    model = fusion_model.GNNBertFusion(
        n_tags=len(vocab), node_dim=items[0].x.shape[1], mode="crossattn",
        hidden_dim=cfg.get("hidden", 128), n_layers=cfg.get("layers", 2),
        conv=cfg.get("conv", "gat"), attn_dim=cfg.get("attn_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- t-SNE of the fused representation ----------------------------------
    subset = test[: args.n_tsne]
    zs, tags = [], []
    with torch.no_grad():
        for batch in GeoLoader(subset, batch_size=64):
            batch = batch.to(device)
            z, _ = model.represent(batch.x, batch.edge_index, batch.batch,
                                   batch.input_ids, batch.attention_mask)
            zs.append(z.cpu().numpy())
    Z = np.concatenate(zs)
    for d in subset:
        tags.append(sorted(d.labels_set)[0] if d.labels_set else "none")

    from sklearn.manifold import TSNE
    coords = TSNE(n_components=2, perplexity=30, init="pca",
                  random_state=args.seed).fit_transform(Z)

    out = Path(args.out_dir); (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "tsne_task3.json").write_text(json.dumps(
        {"coords": coords.tolist(), "primary_tag": tags, "z_dim": int(Z.shape[1])}))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = [t for t, _ in sorted(
        {t: tags.count(t) for t in set(tags)}.items(), key=lambda kv: -kv[1])[:8]]
    fig, ax = plt.subplots(figsize=(8, 7))
    for t in top:
        idx = [i for i, x in enumerate(tags) if x == t]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=6, alpha=0.6, label=t)
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_title("t-SNE of fused representation z (cross-attention), by primary tag")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "plots" / "tsne_task3.png", dpi=150)
    print(f"[tsne] {len(subset)} points, z dim {Z.shape[1]} -> plots/tsne_task3.png")

    # ---- case studies -------------------------------------------------------
    # MTAT clips are 29 s segments of longer tracks, so consecutive clips share
    # a track and a text. Requiring distinct text keeps the three case studies
    # from all being the same song.
    cases = []
    seen_text: set[str] = set()
    with torch.no_grad():
        for d in subset:
            if len(cases) >= args.n_cases:
                break
            if not d.labels_set or not d.text.strip():
                continue
            if d.text in seen_text:
                continue
            seen_text.add(d.text)
            batch = next(iter(GeoLoader([d], batch_size=1))).to(device)
            logits, _, attn = model(batch.x, batch.edge_index, batch.batch,
                                    batch.input_ids, batch.attention_mask)
            probs = torch.sigmoid(logits)[0].cpu()
            ids = batch.input_ids.view(-1).cpu().tolist()
            toks = tok.convert_ids_to_tokens(ids)
            weights = attn[0].cpu().tolist()
            pairs = [(t, w) for t, w in zip(toks, weights)
                     if t not in ("[PAD]", "[CLS]", "[SEP]")]
            pairs.sort(key=lambda p: -p[1])
            ei = np.asarray(d.edge_index)
            deg = np.bincount(ei[0], minlength=len(d.x)) if ei.shape[1] else np.zeros(len(d.x))
            cases.append({
                "clip_id": d.clip_id,
                "text": d.text,
                "true_tags": sorted(d.labels_set),
                "top_predictions": [
                    {"tag": vocab[i], "p": round(float(probs[i]), 3)}
                    for i in probs.argsort(descending=True)[:6].tolist()],
                "graph": {
                    "n_nodes": int(len(d.x)),
                    "n_edges": int(ei.shape[1]),
                    "degree_sequence": deg.tolist(),
                    "busiest_segment": int(deg.argmax()) if len(deg) else -1,
                },
                "attention_top_tokens": [
                    {"token": t, "weight": round(w, 4)} for t, w in pairs[:8]],
            })
    (out / "case_studies_task3.json").write_text(json.dumps(cases, indent=2))
    print(f"[case] {len(cases)} case studies -> case_studies_task3.json")
    for c in cases:
        print(f"  clip {c['clip_id']}: {c['text'][:48]!r}")
        print(f"     attends to: {[t['token'] for t in c['attention_top_tokens'][:5]]}")


if __name__ == "__main__":
    main()
