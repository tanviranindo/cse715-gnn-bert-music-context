"""Task 4 training: contrastive GNN-BERT retrieval on MusicCaps (PDF S4.4).

    python -m src.build_graphs --dataset musiccaps --workers 32
    python -m src.train_contrastive --epochs 30

Deliverables produced:
  results/metrics_task4.json            R@1/5/10 both directions + median rank
  results/retrieval_examples/*.json     10 qualitative caption -> top-3 clips
  results/metrics_task4_zeroshot.json   zero-shot tagging vs Task 1 supervised

Note on comparability: R@K depends on gallery size, so the test-set size is
recorded alongside every number.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src import contrastive, evaluate as ev


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/data/processed/musiccaps_graphs.pt")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--conv", default="gat", choices=["sage", "gat"])
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--freeze-bert", action="store_true")
    p.add_argument("--n-examples", type=int, default=10)
    p.add_argument("--n-zeroshot-tags", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torch_geometric.loader import DataLoader as GeoLoader
    from src.bert_encoder import load_tokenizer
    from src import train_fusion

    tok = load_tokenizer()
    _, items = train_fusion.load_cache(args.cache, tok, args.max_length)
    items = [d for d in items if d.text.strip() and len(d.x) > 1]
    random.shuffle(items)
    n = len(items)
    splits = {"train": items[: int(0.8 * n)],
              "val": items[int(0.8 * n): int(0.9 * n)],
              "test": items[int(0.9 * n):]}
    print(f"[data] {n} clips | " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))

    loaders = {k: GeoLoader(v, batch_size=args.batch_size, shuffle=(k == "train"))
               for k, v in splits.items()}
    model = contrastive.DualEncoder(
        node_dim=items[0].x.shape[1], embed_dim=args.embed_dim,
        hidden_dim=args.hidden, n_layers=args.layers, conv=args.conv,
        temperature=args.temperature, freeze_bert=args.freeze_bert,
    ).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    @torch.no_grad()
    def embed(loader):
        model.eval()
        G, T = [], []
        for b in loader:
            b = b.to(device)
            g, t = model(b.x, b.edge_index, b.batch, b.input_ids, b.attention_mask)
            G.append(g.cpu()); T.append(t.cpu())
        return torch.cat(G), torch.cat(T)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total, steps = time.time(), 0.0, 0
        for b in loaders["train"]:
            b = b.to(device)
            if b.num_graphs < 2:
                continue                     # InfoNCE needs in-batch negatives
            opt.zero_grad()
            g, t = model(b.x, b.edge_index, b.batch, b.input_ids, b.attention_mask)
            loss, _ = contrastive.info_nce(g, t, model.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss); steps += 1
        G, T = embed(loaders["val"])
        r = contrastive.recall_at_k(T, G)     # caption -> audio
        history.append({"epoch": epoch, "loss": total / max(steps, 1),
                        "val_c2a_R@1": r["R@1"], "val_c2a_R@10": r["R@10"],
                        "temperature": float(model.temperature),
                        "seconds": round(time.time() - started, 1)})
        print(f"  [epoch {epoch}] loss {history[-1]['loss']:.4f} "
              f"c2a R@1 {r['R@1']:.4f} R@10 {r['R@10']:.4f} "
              f"tau {float(model.temperature):.4f} ({history[-1]['seconds']}s)",
              flush=True)

    G, T = embed(loaders["test"])
    results = {
        "config": vars(args),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "gallery_size": len(G),
        "history": history,
        "caption_to_audio": contrastive.recall_at_k(T, G),
        "audio_to_caption": contrastive.recall_at_k(G, T),
        "median_rank_caption_to_audio": contrastive.median_rank(T, G),
        "median_rank_audio_to_caption": contrastive.median_rank(G, T),
        "random_baseline_R@1": 1.0 / len(G),
        "random_baseline_R@10": 10.0 / len(G),
    }

    out = Path(args.out_dir)
    (out / "retrieval_examples").mkdir(parents=True, exist_ok=True)
    (out / "metrics_task4.json").write_text(json.dumps(results, indent=2))

    # 10 qualitative caption -> top-3 clip retrievals
    sims = T @ G.t()
    test = splits["test"]
    examples = []
    for i in range(min(args.n_examples, len(test))):
        top = sims[i].topk(3).indices.tolist()
        examples.append({
            "query_caption": test[i].text[:220],
            "true_clip": test[i].clip_id,
            "rank_of_true": int((sims[i] > sims[i, i]).sum()) + 1,
            "top3": [{"clip_id": test[j].clip_id,
                      "caption": test[j].text[:160],
                      "score": round(float(sims[i, j]), 4),
                      "is_correct": bool(j == i)} for j in top],
        })
    (out / "retrieval_examples" / "task4_examples.json").write_text(
        json.dumps(examples, indent=2))

    # zero-shot tagging: embed tag names, score clips, no tag supervision used
    import collections
    counts = collections.Counter()
    for d in test:
        for a in d.labels_set:
            counts[a] += 1
    tags = [t for t, _ in counts.most_common(args.n_zeroshot_tags)]
    if tags:
        enc = tok(tags, truncation=True, padding="max_length",
                  max_length=args.max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            tag_emb = model.encode_text(enc["input_ids"], enc["attention_mask"]).cpu()
        scores = contrastive.zero_shot_tag_scores(tag_emb, G)
        y_true = [[1 if t in d.labels_set else 0 for t in tags] for d in test]
        probs = torch.sigmoid(scores / model.temperature.cpu()).tolist()
        thr, _ = ev.best_threshold(y_true, probs)
        pred = ev.binarize(probs, thr)
        zs = {"n_tags": len(tags), "threshold": thr,
              "micro_f1": ev.micro_f1(y_true, pred),
              "macro_f1": ev.macro_f1(y_true, pred),
              "auc_pr": ev.macro_auc_pr(y_true, probs),
              "note": "no tag supervision; compare against Task 1's supervised "
                      "MusicCaps run in metrics_task1_musiccaps*.json"}
        (out / "metrics_task4_zeroshot.json").write_text(json.dumps(zs, indent=2))
        print(f"[zero-shot] {len(tags)} tags  micro-F1 {zs['micro_f1']:.4f} "
              f"macro-F1 {zs['macro_f1']:.4f} AUC-PR {zs['auc_pr']:.4f}")

    print("\n=== Task 4 retrieval (gallery %d) ===" % len(G))
    for name in ("caption_to_audio", "audio_to_caption"):
        r = results[name]
        print(f"  {name:<18} R@1 {r['R@1']:.4f}  R@5 {r['R@5']:.4f}  R@10 {r['R@10']:.4f}")
    print(f"  random baseline    R@1 {results['random_baseline_R@1']:.4f}  "
          f"R@10 {results['random_baseline_R@10']:.4f}")
    print(f"  median rank        c2a {results['median_rank_caption_to_audio']:.0f}  "
          f"a2c {results['median_rank_audio_to_caption']:.0f}")
    print(f"[out ] {out}/metrics_task4.json")


if __name__ == "__main__":
    main()
