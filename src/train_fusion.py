"""Task 3 training: GNN-BERT fusion with masked alternating batches (PDF S4.3).

Runs the S4.3 ablation (bert-only / gnn-only / early concat / cross-attention)
against MagnaTagATune tags plus DEAM valence-arousal as the auxiliary term.

    python -m src.train_fusion --mode all --epochs 8

MagnaTagATune and DEAM are disjoint corpora, so no sample carries both a tag
vector and an emotion target. Per spec S2.1 each batch is drawn from a single
corpus and carries a mask; the inactive loss term is zeroed. Batches alternate
proportionally to corpus size.
"""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src import evaluate as ev, fusion_model
from src.evaluation_artifacts import file_digest, write_multilabel_predictions
from src.run_provenance import build_provenance


def load_cache(path, tokenizer, max_length: int):
    from torch_geometric.data import Data

    blob = torch.load(path, weights_only=False)
    recs = blob["records"]
    vocab = blob.get("vocab", [])

    out = []
    for r in recs:
        enc = tokenizer(r.get("text", "") or "[PAD]", truncation=True,
                        padding="max_length", max_length=max_length,
                        return_tensors="pt")
        d = Data(
            x=torch.from_numpy(np.asarray(r["x"], dtype=np.float32)),
            edge_index=torch.from_numpy(np.asarray(r["edge_index"], dtype=np.int64)),
        )
        # The cache stores similarity weights per edge. They were dropped here,
        # which silently turned any weighted analysis downstream into an
        # unweighted one -- the case-study walk called itself "heaviest" while
        # every edge scored 1.0.
        if r.get("edge_weight") is not None:
            d.edge_weight = torch.from_numpy(
                np.asarray(r["edge_weight"], dtype=np.float32))
        d.input_ids = enc["input_ids"]
        d.attention_mask = enc["attention_mask"]
        d.artist = r.get("artist", "")
        d.clip_id = str(r.get("track_id", ""))
        # MusicCaps' YouTube id, carried through so a listening study can
        # resolve a retrieved clip back to its source audio. Without it the
        # Task 4 rating sheet degrades to judging captions against captions.
        d.ytid = str(r.get("ytid", "") or "")
        d.start_s = r.get("start_s")
        d.end_s = r.get("end_s")
        d.genre = r.get("genre", "")
        d.text = r.get("text", "")
        d.is_eval = bool(r.get("is_eval", False))
        # Distinguishes "this cache predates the provenance flag" from
        # "the flag is present and this clip is not an eval clip".
        d.has_eval_flag = "is_eval" in r
        d.labels_set = r.get("labels", set())
        d.valence = float(r.get("valence_z", 0.0))
        d.arousal = float(r.get("arousal_z", 0.0))
        d.has_tags = float(bool(r.get("labels")))
        d.has_emotion = float("valence_z" in r)
        out.append(d)
    return vocab, out


def attach_tag_vectors(items, vocab):
    index = {t: i for i, t in enumerate(vocab)}
    for d in items:
        y = torch.zeros(len(vocab))
        for t in d.labels_set:
            if t in index:
                y[index[t]] = 1.0
        d.y = y.unsqueeze(0)
        d.emotion = torch.tensor([[d.valence, d.arousal]], dtype=torch.float)
        d.tag_mask = torch.tensor([d.has_tags], dtype=torch.float)
        d.emotion_mask = torch.tensor([d.has_emotion], dtype=torch.float)
    return items


def artist_split(items, seed=42, fracs=(0.7, 0.15)):
    from src.splits import artist_grouped_split
    recs = [{"artist": d.artist, "d": d} for d in items]
    tr, va, te = artist_grouped_split(recs, train_frac=fracs[0], val_frac=fracs[1], seed=seed)
    return ([r["d"] for r in tr], [r["d"] for r in va], [r["d"] for r in te])


def prepare_tag_protocol(
    items,
    split_kind: str,
    seed: int,
    n_tags: int = 50,
    cached_vocab: list[str] | None = None,
):
    """Split first, then select the Task 3 vocabulary from training labels."""
    from src.splits import official_eval_split, split_digest

    if split_kind == "official_eval":
        train, val, test = official_eval_split(items, seed=seed)
        policy = "official_audioset_eval"
    elif split_kind == "artist":
        train, val, test = artist_split(items, seed)
        policy = f"artist_grouped_seed{seed}"
    else:
        raise ValueError(f"unsupported Task 3 split {split_kind!r}")

    splits = {"train": train, "val": val, "test": test}
    counts = collections.Counter(
        label for item in train for label in (item.labels_set or ())
    )
    vocab = [label for label, _ in counts.most_common(n_tags)]
    attach_tag_vectors(items, vocab)
    metadata = {
        "split_kind": policy,
        "split_digest": split_digest(splits),
        "vocabulary_from": "train",
        "cache_vocabulary_ignored": bool(cached_vocab),
    }
    return vocab, splits, metadata


@torch.no_grad()
def attention_entropy(model, loader, device, max_batches: int = 20) -> dict:
    """How peaked is the cross-attention, as a fraction of the uniform maximum?

    1.0 means the graph query gives every caption token equal weight, i.e. the
    fusion has degenerated into averaging the text and cross-attention buys
    nothing over early concat. The first ablation measured 0.65-0.88 here.
    """
    if model.mode != "crossattn":
        return {}
    model.eval()
    ratios, peaks = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        _, _, attn = model(batch.x, batch.edge_index, batch.batch,
                           batch.input_ids, batch.attention_mask)
        mask = batch.attention_mask.float()
        n_real = mask.sum(dim=1).clamp(min=1)
        a = attn.clamp(min=1e-12)
        ent = -(a * a.log() * mask).sum(dim=1)
        ratios.extend((ent / n_real.log().clamp(min=1e-9)).cpu().tolist())
        peaks.extend((attn.max(dim=1).values * n_real).cpu().tolist())
    n = max(len(ratios), 1)
    return {
        "entropy_vs_uniform": sum(ratios) / n,
        "peak_over_uniform": sum(peaks) / max(len(peaks), 1),
    }


@torch.no_grad()
def evaluate_split(model, loader, device, mode):
    model.eval()
    tag_ids, tag_true, tag_prob, em_true, em_pred = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits, emotion, _ = model(
            batch.x, batch.edge_index, batch.batch,
            batch.input_ids, batch.attention_mask,
        )
        mask_t = batch.tag_mask.view(-1).bool().cpu()
        mask_e = batch.emotion_mask.view(-1).bool().cpu()
        probs = torch.sigmoid(logits).cpu()
        y = batch.y.view(probs.shape).cpu()
        clip_ids = batch.clip_id
        if isinstance(clip_ids, str):
            clip_ids = [clip_ids]
        for i in range(len(probs)):
            if mask_t[i]:
                tag_ids.append(str(clip_ids[i]))
                tag_true.append(y[i].int().tolist())
                tag_prob.append(probs[i].tolist())
            if mask_e[i]:
                em_true.append(batch.emotion.view(-1, 2)[i].cpu().tolist())
                em_pred.append(emotion[i].cpu().tolist())
    return tag_ids, tag_true, tag_prob, em_true, em_pred


def run_mode(mode, splits, vocab, node_dim, args, device):
    from torch_geometric.loader import DataLoader as GeoLoader

    loaders = {
        k: GeoLoader(v, batch_size=args.batch_size, shuffle=(k == "train"))
        for k, v in splits.items()
    }
    model = fusion_model.GNNBertFusion(
        n_tags=len(vocab), node_dim=node_dim, mode=mode,
        hidden_dim=args.hidden, n_layers=args.layers, conv=args.conv,
        attn_dim=args.attn_dim, dropout=args.dropout,
        freeze_bert=args.freeze_bert,
    ).to(device)
    # Separate learning rates. BERT has 66.4M parameters against the GNN's
    # 31.8k, and a single lr=2e-5 tuned for fine-tuning BERT leaves the graph
    # branch badly undertrained — a likely cause of the cross-attention
    # collapsing to uniform weights in the first ablation.
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

    history = []
    best: tuple[float, int, dict] = (-1.0, 0, {})
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total, seen = time.time(), 0.0, 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            logits, emotion, _ = model(
                batch.x, batch.edge_index, batch.batch,
                batch.input_ids, batch.attention_mask,
            )
            loss, _ = fusion_model.multitask_loss(
                logits, batch.y.view(logits.shape),
                emotion, batch.emotion.view(-1, 2),
                batch.tag_mask.view(-1), batch.emotion_mask.view(-1),
                alpha=args.alpha, beta=args.beta,
            )
            if float(loss) == 0.0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss)
            seen += 1
        _, tt, tp, et, ep = evaluate_split(model, loaders["val"], device, mode)
        pred = ev.binarize(tp, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "val_macro_f1": ev.macro_f1(tt, pred),
            "val_micro_f1": ev.micro_f1(tt, pred),
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        # Restore the best validation epoch before touching test: the fusion
        # ablations are compared against each other, so letting one of them be
        # scored at an unlucky final epoch would corrupt the comparison.
        if row["val_macro_f1"] > best[0]:
            best = (row["val_macro_f1"], epoch,
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  [{mode} epoch {epoch}] loss {row['train_loss']:.4f} "
              f"macroF1 {row['val_macro_f1']:.4f} microF1 {row['val_micro_f1']:.4f} "
              f"({row['seconds']}s)", flush=True)

    if best[2]:
        model.load_state_dict(best[2])
        print(f"  [{mode} select] restored epoch {best[1]} "
              f"(val macro-F1 {best[0]:.4f})", flush=True)
    _, tt, tp, et, ep = evaluate_split(model, loaders["val"], device, mode)
    thr, _ = ev.best_threshold(tt, tp)
    test_ids, tt, tp, et, ep = evaluate_split(model, loaders["test"], device, mode)
    pred = ev.binarize(tp, thr)
    evidence_path = Path(args.out_dir) / "evaluation" / f"task3_{mode}.json.gz"
    write_multilabel_predictions(evidence_path, test_ids, tt, tp, vocab, thr)
    result = {
        "mode": mode,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history,
        "selected_epoch": best[1],
        "selected_val_macro_f1": best[0],
        "threshold": thr,
        "test_macro_f1": ev.macro_f1(tt, pred),
        "test_micro_f1": ev.micro_f1(tt, pred),
        "test_auc_pr": ev.macro_auc_pr(tt, tp),
        "evaluation_artifact": {
            "path": str(evidence_path), "sha256": file_digest(evidence_path)
        },
    }
    if et:
        result["valence"] = ev.regression_metrics([a[0] for a in et], [b[0] for b in ep])
        result["arousal"] = ev.regression_metrics([a[1] for a in et], [b[1] for b in ep])
    att = attention_entropy(model, loaders["test"], device)
    if att:
        result["attention"] = att
        print(f"  [{mode}] attention entropy {att['entropy_vs_uniform']:.3f} of "
              f"uniform, peak {att['peak_over_uniform']:.2f}x uniform")
    return model, result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mtat-cache", default="/data/processed/mtat_graphs.pt")
    p.add_argument("--deam-cache", default="/data/processed/deam_graphs.pt")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--mode", default="all",
                   choices=["bert", "gnn", "concat", "crossattn", "all"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--conv", default="gat", choices=["sage", "gat"])
    p.add_argument("--attn-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--freeze-bert", action="store_true")
    p.add_argument("--gnn-lr", type=float, default=None,
                   help="separate lr for the graph branch, fusion and heads; "
                        "defaults to --lr")
    p.add_argument("--split", default="artist",
                   choices=["artist", "official_eval"],
                   help="'official_eval' keeps the cache's published eval "
                        "partition as test, matching train_contrastive.py, so a "
                        "supervised model can be compared with the zero-shot one "
                        "on identical clips")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.bert_encoder import load_tokenizer
    tok = load_tokenizer()

    cached_vocab, mtat = load_cache(args.mtat_cache, tok, args.max_length)
    deam = []
    if Path(args.deam_cache).exists():
        _, deam = load_cache(args.deam_cache, tok, args.max_length)
    print(f"[data] mtat={len(mtat)} deam={len(deam)} cached-tags={len(cached_vocab)}")
    try:
        vocab, tag_splits, protocol = prepare_tag_protocol(
            mtat, args.split, args.seed, n_tags=50, cached_vocab=cached_vocab
        )
    except ValueError as exc:
        raise SystemExit(f"--split {args.split}: {exc}") from exc
    m_tr, m_va, m_te = (tag_splits[name] for name in ("train", "val", "test"))
    print(f"[data] derived {len(vocab)} tags from the training split")
    if cached_vocab:
        print("[data] ignored cache vocabulary selected before the final split")

    attach_tag_vectors(deam, vocab)

    d_tr, d_va, d_te = artist_split(deam, args.seed) if deam else ([], [], [])
    splits = {"train": m_tr + d_tr, "val": m_va + d_va, "test": m_te + d_te}
    print("[data] splits: " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"[data] emotion-supervised in train: {sum(d.has_emotion for d in splits['train']):.0f}")

    node_dim = mtat[0].x.shape[1]
    modes = ["bert", "gnn", "concat", "crossattn"] if args.mode == "all" else [args.mode]
    results = {"config": vars(args), "n_tags": len(vocab), "vocab": vocab,
               "data_protocol": protocol,
               "provenance": build_provenance(
                   vars(args), tag_splits, vocab, "val", "val_macro_f1"
               ),
               # Written so a reader can verify this run and the zero-shot one
               # scored the same labels on the same clips, rather than trusting
               # that equal split sizes imply equal splits -- they do not.
               "test_clip_ids": sorted(str(getattr(d, "clip_id", "")) for d in m_te),
               "split_sizes": {k: len(v) for k, v in splits.items()}, "runs": {}}

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        print(f"\n=== {mode} ===")
        model, res = run_mode(mode, splits, vocab, node_dim, args, device)
        results["runs"][mode] = res
        (out / "metrics_task3.json").write_text(json.dumps(results, indent=2))
        if mode == "crossattn":
            torch.save({"model_state": model.state_dict(), "vocab": vocab,
                        "config": vars(args)}, out / "task3_crossattn.pt")

    print("\n=== Task 3 ablation ===")
    print(f"{'mode':<12}{'params':>12}{'macro-F1':>10}{'micro-F1':>10}{'AUC-PR':>9}"
          f"{'val MAE':>9}{'aro MAE':>9}")
    for mode, r in results["runs"].items():
        vm = r.get("valence", {}).get("mae", float("nan"))
        am = r.get("arousal", {}).get("mae", float("nan"))
        print(f"{mode:<12}{r['params']:>12,}{r['test_macro_f1']:>10.4f}"
              f"{r['test_micro_f1']:>10.4f}{r['test_auc_pr']:>9.4f}{vm:>9.3f}{am:>9.3f}")
    print(f"[out ] {out}/metrics_task3.json")


if __name__ == "__main__":
    main()
