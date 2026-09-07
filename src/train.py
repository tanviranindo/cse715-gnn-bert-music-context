"""Task 1 training: BERT multi-label tag classifier (PDF S4.1).

Two dataset variants, both sanctioned by the PDF ("MagnaTagATune tag subset
(top-50 tags) or MusicCaps caption -> tag proxy task"). We run both because
each has a defect the other does not — see the module docstrings of
`mtat_data` and `musiccaps_data`.

    python -m src.train --dataset mtat      --epochs 6
    python -m src.train --dataset musiccaps --epochs 6 --strip-leakage
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src import bert_encoder, evaluate as ev, mtat_data, musiccaps_data
from src.splits import artist_grouped_split


def build_records(args) -> tuple[list[str], dict[str, list[dict]]]:
    raw = Path(args.data_dir)
    if args.dataset == "mtat":
        vocab, records = mtat_data.build_dataset(
            raw / "magnatagatune" / "annotations_final.csv",
            raw / "magnatagatune" / "clip_info_final.csv",
            n_tags=args.n_tags,
            use_artist=args.use_artist,
        )
        # Artist-grouped, NOT the official split: the official MTT split shares
        # 45 artists between train and test, covering 61.6% of test clips.
        train, val, test = artist_grouped_split(records, seed=args.seed)
    else:
        vocab, records = musiccaps_data.build_dataset(
            raw / "musiccaps" / "musiccaps-public.csv",
            n_aspects=args.n_tags,
            strip_leakage=args.strip_leakage,
        )
        # MusicCaps publishes the AudioSet provenance split in the metadata:
        # keep its eval examples as test and only split the train pool.
        # Treating all rows as one random pool makes the reported test score
        # incomparable with the dataset's intended evaluation partition.
        rng = random.Random(args.seed)
        if not any("is_eval" in r for r in records):
            raise SystemExit(
                "musiccaps records carry no is_eval flag: the metadata CSV is "
                "missing is_audioset_eval. Refusing to fall back to a random "
                "split, which would not be comparable with the official one."
            )
        train_pool = [r for r in records if not r.get("is_eval", False)]
        test = [r for r in records if r.get("is_eval", False)]
        if not test:
            raise SystemExit("musiccaps is_audioset_eval selected 0 rows")
        rng.shuffle(train_pool)
        n_val = int(0.15 * len(train_pool))
        val = train_pool[:n_val]
        train = train_pool[n_val:]
    return vocab, {"train": train, "val": val, "test": test}


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[list[int]], list[list[float]]]:
    model.eval()
    y_true, probs = [], []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        probs.extend(torch.sigmoid(logits).cpu().tolist())
        y_true.extend(batch["labels"].int().tolist())
    return y_true, probs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mtat", "musiccaps"], default="mtat")
    p.add_argument("--data-dir", default="/data/raw")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--model", default=bert_encoder.DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-tags", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--use-artist", action="store_true",
                   help="mtat: include artist in the text (memorisation risk)")
    p.add_argument("--strip-leakage", action="store_true",
                   help="musiccaps: delete aspect words from the caption")
    p.add_argument("--save-checkpoint", action="store_true",
                   help="persist encoder weights for Task 3/4 reuse")
    p.add_argument("--checkpoint-dir", default="/data/checkpoints")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab, splits = build_records(args)
    print(f"[data] {args.dataset}: {len(vocab)} tags | "
          + " ".join(f"{k}={len(v)}" for k, v in splits.items()))
    if args.dataset == "mtat":
        print(f"[data] artist leakage: {mtat_data.artist_overlap(splits)}")

    tok = bert_encoder.load_tokenizer(args.model)
    loaders = {
        k: DataLoader(
            bert_encoder.TagDataset(v, vocab, tok, args.max_length),
            batch_size=args.batch_size,
            shuffle=(k == "train"),
            num_workers=4,
        )
        for k, v in splits.items()
    }

    model = bert_encoder.BertTagger(
        len(vocab), args.model, args.freeze_encoder, pooling=args.pooling
    ).to(device)
    opt = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=args.lr
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=bert_encoder.pos_weight_from_records(splits["train"], vocab).to(device)
    )

    history = []
    best: tuple[float, int, dict] = (-1.0, 0, {})
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for batch in loaders["train"]:
            opt.zero_grad()
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            opt.step()
            total += loss.item()
        y, probs = predict(model, loaders["val"], device)
        pred = ev.binarize(probs, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": total / max(len(loaders["train"]), 1),
            "val_micro_f1": ev.micro_f1(y, pred),
            "val_macro_f1": ev.macro_f1(y, pred),
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        # Select on validation macro-F1 rather than trusting the final epoch:
        # these runs are short and the last epoch is not reliably the best.
        if row["val_macro_f1"] > best[0]:
            best = (row["val_macro_f1"], epoch,
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"[epoch {epoch}] loss {row['train_loss']:.4f} "
              f"micro-F1 {row['val_micro_f1']:.4f} macro-F1 {row['val_macro_f1']:.4f} "
              f"({row['seconds']}s)")

    if best[2]:
        model.load_state_dict(best[2])
        print(f"[select] restored epoch {best[1]} (val macro-F1 {best[0]:.4f})")

    # threshold tuned on val, then applied once to test
    y_val, p_val = predict(model, loaders["val"], device)
    thr, _ = ev.best_threshold(y_val, p_val)
    y_test, p_test = predict(model, loaders["test"], device)
    pred_test = ev.binarize(p_test, thr)

    y_train = [
        mtat_data.labels_to_vector(r["labels"], vocab) for r in splits["train"]
    ]
    train_rates = [sum(row[i] for row in y_train) / max(len(y_train), 1)
                   for i in range(len(vocab))]
    results = {
        "config": vars(args),
        "n_tags": len(vocab),
        "vocab": vocab,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "history": history,
        "selected_epoch": best[1],
        "selected_val_macro_f1": best[0],
        "threshold": thr,
        "test": {
            "micro_f1": ev.micro_f1(y_test, pred_test),
            "macro_f1": ev.macro_f1(y_test, pred_test),
        },
        "baselines": {
            "B1_random_prevalence": {
                # Estimate each tag's prevalence from training labels only.
                "micro_f1": ev.micro_f1(y_test, ev.baseline_random(y_test, args.seed, rate=train_rates)),
                "macro_f1": ev.macro_f1(y_test, ev.baseline_random(y_test, args.seed, rate=train_rates)),
            },
            "B1_majority": {
                "micro_f1": ev.micro_f1(y_test, ev.baseline_majority(y_train, len(y_test))),
                "macro_f1": ev.macro_f1(y_test, ev.baseline_majority(y_train, len(y_test))),
            },
        },
        "per_tag_f1": ev.per_tag_f1(y_test, pred_test, vocab),
    }
    if args.dataset == "musiccaps":
        lex = [
            mtat_data.labels_to_vector(
                musiccaps_data.lexical_match_predict(r["text"], vocab), vocab
            )
            for r in splits["test"]
        ]
        results["baselines"]["B5_lexical_match"] = {
            "micro_f1": ev.micro_f1(y_test, lex),
            "macro_f1": ev.macro_f1(y_test, lex),
        }

    out = Path(args.out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    tag = f"task1_{args.dataset}" + ("_stripped" if args.strip_leakage else "")
    (out / f"metrics_{tag}.json").write_text(json.dumps(results, indent=2))

    # Persist the encoder: Task 3 fuses this text branch with the GNN, and
    # Task 4 reuses it as the text tower. Runs are seeded so this is a
    # convenience, not the only route back to the weights.
    if args.save_checkpoint:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "vocab": vocab,
             "config": vars(args), "threshold": thr,
             "test_metrics": results["test"]},
            ckpt_dir / f"{tag}.pt",
        )
        print(f"[ckpt] {ckpt_dir}/{tag}.pt")

    examples = []
    for r, probs_row in list(zip(splits["test"], p_test))[:5]:
        ranked = sorted(zip(vocab, probs_row), key=lambda x: -x[1])[:8]
        examples.append({
            "clip_id": r["clip_id"],
            "text": r["text"][:200],
            "true": sorted(r["labels"]),
            "predicted": [{"tag": t, "p": round(v, 3)} for t, v in ranked],
        })
    (out / f"examples_{tag}.json").write_text(json.dumps(examples, indent=2))

    print(f"\n[test] micro-F1 {results['test']['micro_f1']:.4f} "
          f"macro-F1 {results['test']['macro_f1']:.4f} (threshold {thr})")
    for name, b in results["baselines"].items():
        print(f"[base] {name:<22} micro-F1 {b['micro_f1']:.4f} macro-F1 {b['macro_f1']:.4f}")
    print(f"[out ] {out}/metrics_{tag}.json, examples_{tag}.json")


if __name__ == "__main__":
    main()
