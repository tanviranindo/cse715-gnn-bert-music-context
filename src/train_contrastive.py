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
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="fraction of the TRAIN split to use; val/test/gallery "
                        "stay fixed so R@K remains comparable across runs")
    p.add_argument("--gnn-lr", type=float, default=None,
                   help="separate learning rate for the graph tower "
                        "(Task 3 found this decisive; None = share --lr)")
    p.add_argument("--metrics-suffix", default="",
                   help="appended to metrics/example filenames, for ablation runs")
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
    # MusicCaps metadata identifies the original AudioSet eval partition.
    # Preserve it as test; split only the published train pool for training
    # and validation so retrieval is evaluated on the intended held-out set.
    has_flag = any(getattr(d, "has_eval_flag", False) for d in items)
    eval_items = [d for d in items if getattr(d, "is_eval", False)]
    if has_flag:
        # The cache carries AudioSet provenance, so the official eval clips
        # become test. Never quietly downgrade to a random split here: the
        # resulting R@k would not be comparable with the official partition.
        if not eval_items:
            raise SystemExit("cache has is_eval but it selected 0 clips")
        train_pool = [d for d in items if not getattr(d, "is_eval", False)]
        random.shuffle(train_pool)
        n_val = int(0.15 * len(train_pool))
        splits = {"train": train_pool[n_val:], "val": train_pool[:n_val], "test": eval_items}
        split_kind = "official_audioset_eval"
    else:
        # A cache built before the flag was carried through. Keep the previous
        # seeded random split, but label the result so the report cannot
        # present it as the official partition.
        print("[warn] cache predates is_eval; using the seeded RANDOM split")
        random.shuffle(items)
        n = len(items)
        splits = {"train": items[: int(0.8 * n)],
                  "val": items[int(0.8 * n): int(0.9 * n)],
                  "test": items[int(0.9 * n):]}
        split_kind = "random_synthetic"
    # Data-scale ablation. MusicCaps' official partition leaves only ~2.2k
    # training pairs, which is small for a contrastive objective. Subsampling the
    # TRAIN split only -- val and test, and therefore the gallery, stay fixed --
    # makes R@K comparable across fractions and answers whether retrieval is
    # limited by the model or simply by how little paired data survives the
    # official split.
    if args.train_frac < 1.0:
        keep = max(2, int(round(args.train_frac * len(splits["train"]))))
        splits["train"] = splits["train"][:keep]
        print(f"[data] train subsampled to {args.train_frac:.0%} = {keep} pairs "
              f"(val/test/gallery unchanged)")

    print(f"[data] {len(items)} clips | split={split_kind} | "
          + " ".join(f"{k}={len(v)}" for k, v in splits.items()))

    loaders = {k: GeoLoader(v, batch_size=args.batch_size, shuffle=(k == "train"))
               for k, v in splits.items()}
    model = contrastive.DualEncoder(
        node_dim=items[0].x.shape[1], embed_dim=args.embed_dim,
        hidden_dim=args.hidden, n_layers=args.layers, conv=args.conv,
        temperature=args.temperature, freeze_bert=args.freeze_bert,
    ).to(device)
    # Task 3 established that a single learning rate tuned for BERT leaves the
    # 31.8k-parameter graph branch badly undertrained, and that giving it its own
    # rate was the largest single effect in the project (+0.067 Macro-F1 over five
    # paired seeds). The two towers here have the same imbalance, so the same
    # remedy is worth testing rather than assuming it transfers.
    bert_params, other_params = [], []
    for name, q in model.named_parameters():
        if not q.requires_grad:
            continue
        (bert_params if name.startswith("text_encoder") else other_params).append(q)
    groups = []
    if bert_params:
        groups.append({"params": bert_params, "lr": args.lr})
    if other_params:
        groups.append({"params": other_params, "lr": args.gnn_lr or args.lr})
    params = bert_params + other_params
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    if args.gnn_lr:
        print("[opt ] BERT tower lr=%g | graph tower lr=%g" % (args.lr, args.gnn_lr))

    @torch.no_grad()
    def embed(loader):
        model.eval()
        G, T = [], []
        for b in loader:
            b = b.to(device)
            g, t = model(b.x, b.edge_index, b.batch, b.input_ids, b.attention_mask)
            G.append(g.cpu()); T.append(t.cpu())
        return torch.cat(G), torch.cat(T)

    # Track the best epoch by validation caption->audio R@10. The contrastive
    # objective overfits well before the loss plateaus: on a first run the
    # train loss fell 3.87 -> 0.41 while val R@10 peaked at epoch 6 and then
    # declined, so reporting the final epoch understates the model.
    history, best = [], (-1.0, None, 0)
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
        if r["R@10"] > best[0]:
            best = (r["R@10"],
                    {k: v.detach().clone() for k, v in model.state_dict().items()},
                    epoch)
        print(f"  [epoch {epoch}] loss {history[-1]['loss']:.4f} "
              f"c2a R@1 {r['R@1']:.4f} R@10 {r['R@10']:.4f} "
              f"tau {float(model.temperature):.4f} ({history[-1]['seconds']}s)",
              flush=True)

    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"[best] restored epoch {best[2]} (val c2a R@10 {best[0]:.4f})")
    G, T = embed(loaders["test"])
    results = {
        "config": vars(args),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_kind": split_kind,
        "gallery_size": len(G),
        "history": history,
        "best_epoch": best[2],
        "best_val_c2a_R@10": best[0],
        "caption_to_audio": contrastive.recall_at_k(T, G),
        "audio_to_caption": contrastive.recall_at_k(G, T),
        "median_rank_caption_to_audio": contrastive.median_rank(T, G),
        "median_rank_audio_to_caption": contrastive.median_rank(G, T),
        "random_baseline_R@1": 1.0 / len(G),
        "random_baseline_R@10": 10.0 / len(G),
    }

    out = Path(args.out_dir)
    (out / "retrieval_examples").mkdir(parents=True, exist_ok=True)
    (out / ("metrics_task4%s.json" % args.metrics_suffix)).write_text(json.dumps(results, indent=2))

    # 10 qualitative caption -> top-3 clip retrievals
    sims = T @ G.t()
    test = splits["test"]
    examples = []
    for i in range(min(args.n_examples, len(test))):
        top = sims[i].topk(3).indices.tolist()
        # ytid is carried through so a listening study can resolve each
        # retrieved clip back to its source audio; clip_id alone is a hash.
        examples.append({
            "query_caption": test[i].text[:220],
            "true_clip": test[i].clip_id,
            "true_ytid": getattr(test[i], "ytid", ""),
            "rank_of_true": int((sims[i] > sims[i, i]).sum()) + 1,
            "top3": [{"clip_id": test[j].clip_id,
                      "ytid": getattr(test[j], "ytid", ""),
                      "caption": test[j].text[:160],
                      "score": round(float(sims[i, j]), 4),
                      "is_correct": bool(j == i)} for j in top],
        })
    (out / "retrieval_examples" / ("task4_examples%s.json" % args.metrics_suffix)).write_text(
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
