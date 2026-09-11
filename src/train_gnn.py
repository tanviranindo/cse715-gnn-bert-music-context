"""Task 2 training: GNN on music structure graphs vs CNN baseline (PDF S4.2).

Trains GraphSAGE and GAT on the cached FMA-small segment graphs, plus the B2
CNN baseline on log-mel spectrograms, and reports all three side by side.

    python -m src.build_graphs                 # once, produces the cache
    python -m src.train_gnn --model sage
    python -m src.train_gnn --model gat
    python -m src.train_gnn --model cnn        # B2 baseline
    python -m src.train_gnn --model all        # all three, one metrics file
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import evaluate as ev, fma_data, gnn_model, graph_builder as gb
from src.evaluation_artifacts import file_digest, write_multiclass_predictions
from src.run_provenance import build_provenance


def load_cache(path: str | Path) -> tuple[list[dict], list[str]]:
    blob = torch.load(path, weights_only=False)
    records = blob["records"]
    genres = sorted({r["genre"] for r in records})
    return records, genres


def chord_node_features(chord_names: list[str]) -> np.ndarray:
    """Node features for a chord-transition graph.

    A chord node carries no audio vector of its own, so it is described by what
    the chord *is*: its 12-dim binary pitch-class template (the same templates
    the estimator matched against), plus a quality flag separating major from
    minor. That keeps the representation faithful to the symbolic graph rather
    than smuggling spectral features back in through the node table.
    """
    templates, names = gb._chord_templates()
    lookup = {n: templates[i] for i, n in enumerate(names)}
    rows = []
    for name in chord_names:
        pitch = lookup.get(name)
        if pitch is None:                       # "N" (no chord) or unknown label
            rows.append(np.zeros(13, dtype=np.float32))
            continue
        is_minor = 1.0 if name.endswith(":min") else 0.0
        rows.append(np.concatenate([np.asarray(pitch, dtype=np.float32), [is_minor]]))
    # float32 throughout: the chord templates are float64 and a Double node
    # table against Float weights fails inside the first linear layer.
    return (np.stack(rows).astype(np.float32) if rows
            else np.zeros((0, 13), dtype=np.float32))


def to_pyg(records: list[dict], genres: list[str], graph: str = "segment"):
    """Wrap cached arrays as PyTorch Geometric Data objects.

    `graph` selects which of the two structures the specification names as the
    GNN input: the segment-similarity graph, or the chord-transition graph.
    Both are built by the same preprocessing pass and stored in the cache, so
    switching between them re-uses identical data, splits and labels --- the
    comparison isolates the graph, not the pipeline.
    """
    from torch_geometric.data import Data

    index = {g: i for i, g in enumerate(genres)}
    out = []
    for r in records:
        if graph == "chord":
            names = list(r["chord_nodes"])
            if len(names) < 2:
                continue          # a single-chord track has no transition to learn
            x = chord_node_features(names)
            ei = np.asarray(r["chord_edge_index"], dtype=np.int64)
            w = np.asarray(r["chord_edge_weight"], dtype=np.float32)
            if len(w):
                w = w / max(float(w.mean()), 1e-12)
            d = Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(ei),
                edge_attr=torch.from_numpy(w),
                y=torch.tensor([index[r["genre"]]], dtype=torch.long),
            )
        else:
            d = Data(
                x=torch.from_numpy(np.asarray(r["x"], dtype=np.float32)),
                edge_index=torch.from_numpy(np.asarray(r["edge_index"], dtype=np.int64)),
                y=torch.tensor([index[r["genre"]]], dtype=torch.long),
            )
        d.track_id = r["track_id"]
        out.append(d)
    return out


def standardise(splits: dict[str, list], device) -> None:
    """Z-score node features using TRAIN statistics only."""
    train_x = torch.cat([d.x for d in splits["train"]], dim=0)
    mean, std = train_x.mean(0), train_x.std(0).clamp(min=1e-6)
    for part in splits.values():
        for d in part:
            d.x = (d.x - mean) / std


def pad_mel(mel: np.ndarray, width: int) -> np.ndarray:
    if mel.shape[1] >= width:
        return mel[:, :width]
    return np.pad(mel, ((0, 0), (0, width - mel.shape[1])), mode="constant")


@torch.no_grad()
def eval_graph(model, loader, device, use_edge_weight=False) -> tuple[list[str], list[int], list[int]]:
    model.eval()
    ids, y_true, y_pred = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = (
            model(batch.x, batch.edge_index, batch.edge_attr.view(-1), batch.batch)
            if use_edge_weight
            else model(batch.x, batch.edge_index, batch.batch)
        )
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(batch.y.cpu().tolist())
        track_ids = batch.track_id
        ids.extend(str(value) for value in track_ids.cpu().tolist())
    return ids, y_true, y_pred


@torch.no_grad()
def eval_cnn(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return y_true, y_pred


def run_graph_model(kind, splits, genres, args, device) -> dict:
    from torch_geometric.loader import DataLoader as GeoLoader

    loaders = {
        k: GeoLoader(v, batch_size=args.batch_size, shuffle=(k == "train"))
        for k, v in splits.items()
    }
    use_edge_weight = kind == "weighted_chord"
    if use_edge_weight:
        model = gnn_model.WeightedGNNClassifier(
            splits["train"][0].x.shape[1], len(genres),
            hidden_dim=args.hidden, n_layers=args.layers, dropout=args.dropout,
        ).to(device)
    else:
        model = gnn_model.GNNClassifier(
            splits["train"][0].x.shape[1], len(genres),
            hidden_dim=args.hidden, n_layers=args.layers, conv=kind,
            dropout=args.dropout,
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history, best = [], (-1.0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            logits = (
                model(batch.x, batch.edge_index, batch.edge_attr.view(-1), batch.batch)
                if use_edge_weight
                else model(batch.x, batch.edge_index, batch.batch)
            )
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
            total += loss.item()
        _, yt, yp = eval_graph(model, loaders["val"], device, use_edge_weight)
        acc = ev.accuracy(yt, yp)
        row = {"epoch": epoch, "train_loss": total / len(loaders["train"]),
               "val_acc": acc,
               "val_macro_f1": ev.multiclass_f1(yt, yp, len(genres)),
               "seconds": round(time.time() - started, 1)}
        history.append(row)
        if acc > best[0]:
            best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"  [{kind} epoch {epoch}] loss {row['train_loss']:.4f} "
              f"val-acc {acc:.4f} val-macroF1 {row['val_macro_f1']:.4f} "
              f"({row['seconds']}s)", flush=True)

    if best[1]:
        model.load_state_dict(best[1])
    ids, yt, yp = eval_graph(model, loaders["test"], device, use_edge_weight)
    evidence_path = Path(args.out_dir) / "evaluation" / f"task2_{args.graph}_{kind}.json.gz"
    write_multiclass_predictions(evidence_path, ids, yt, yp, genres)
    return {
        "model": kind,
        "uses_edge_weight": use_edge_weight,
        "params": gnn_model.count_parameters(model),
        "history": history,
        "test_accuracy": ev.accuracy(yt, yp),
        "test_macro_f1": ev.multiclass_f1(yt, yp, len(genres)),
        "per_class_f1": ev.per_class_f1(yt, yp, genres),
        "confusion": ev.confusion_matrix(yt, yp, len(genres)),
        "evaluation_artifact": {
            "path": str(evidence_path), "sha256": file_digest(evidence_path)
        },
    }


def run_cnn(records_by_split, genres, args, device) -> dict:
    index = {g: i for i, g in enumerate(genres)}
    tensors = {}
    for name, recs in records_by_split.items():
        X = np.stack([pad_mel(np.asarray(r["mel"], dtype=np.float32), args.mel_width) for r in recs])
        Y = np.asarray([index[r["genre"]] for r in recs], dtype=np.int64)
        tensors[name] = torch.utils.data.TensorDataset(
            torch.from_numpy(X).unsqueeze(1), torch.from_numpy(Y)
        )
    mean = tensors["train"].tensors[0].mean()
    std = tensors["train"].tensors[0].std().clamp(min=1e-6)
    for name in tensors:
        x, y = tensors[name].tensors
        tensors[name] = torch.utils.data.TensorDataset((x - mean) / std, y)

    loaders = {
        k: torch.utils.data.DataLoader(v, batch_size=args.batch_size,
                                       shuffle=(k == "train"))
        for k, v in tensors.items()
    }
    model = gnn_model.CNNBaseline(len(genres), dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history, best = [], (-1.0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            total += loss.item()
        yt, yp = eval_cnn(model, loaders["val"], device)
        acc = ev.accuracy(yt, yp)
        row = {"epoch": epoch, "train_loss": total / len(loaders["train"]),
               "val_acc": acc, "val_macro_f1": ev.multiclass_f1(yt, yp, len(genres)),
               "seconds": round(time.time() - started, 1)}
        history.append(row)
        if acc > best[0]:
            best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"  [cnn epoch {epoch}] loss {row['train_loss']:.4f} "
              f"val-acc {acc:.4f} ({row['seconds']}s)", flush=True)

    if best[1]:
        model.load_state_dict(best[1])
    yt, yp = eval_cnn(model, loaders["test"], device)
    ids = [str(record["track_id"]) for record in records_by_split["test"]]
    evidence_path = Path(args.out_dir) / "evaluation" / "task2_cnn.json.gz"
    write_multiclass_predictions(evidence_path, ids, yt, yp, genres)
    return {
        "model": "cnn_b2",
        "params": gnn_model.count_parameters(model),
        "history": history,
        "test_accuracy": ev.accuracy(yt, yp),
        "test_macro_f1": ev.multiclass_f1(yt, yp, len(genres)),
        "per_class_f1": ev.per_class_f1(yt, yp, genres),
        "confusion": ev.confusion_matrix(yt, yp, len(genres)),
        "evaluation_artifact": {
            "path": str(evidence_path), "sha256": file_digest(evidence_path)
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/data/processed/fma_small_graphs.pt")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--model", default="all",
                   choices=["sage", "gat", "cnn", "weighted_chord", "all"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--mel-width", type=int, default=640)
    p.add_argument("--graph", default="segment", choices=["segment", "chord"],
                   help="which cached structure feeds the GNN (PDF S3.3 names both)")
    p.add_argument("--metrics-name", default=None,
                   help="output filename; defaults to metrics_task2[_<graph>].json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.model == "weighted_chord" and args.graph != "chord":
        p.error("--model weighted_chord requires --graph chord")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records, genres = load_cache(args.cache)
    by_split = fma_data.official_splits(records)
    leakage = fma_data.artist_leakage(by_split)
    print(f"[data] {len(records)} tracks | {len(genres)} genres: {genres}")
    print(f"[data] splits: " + " ".join(f"{k}={len(v)}" for k, v in by_split.items()))
    print(f"[data] artist leakage (verified, not assumed): {leakage}")

    graph_splits = {k: to_pyg(v, genres, args.graph) for k, v in by_split.items()}
    if args.graph == "chord":
        dropped = {k: len(by_split[k]) - len(graph_splits[k]) for k in by_split}
        sizes = " ".join(f"{k}={len(v)}" for k, v in graph_splits.items())
        print(f"[data] chord graphs: {sizes} (dropped for <2 chords: {dropped})")
    standardise(graph_splits, device)

    # The CNN baseline reads spectrograms, not graphs, so it is unaffected by
    # --graph; running it again under chord mode would duplicate a number.
    wanted = ["sage", "gat", "cnn"] if args.model == "all" else [args.model]
    if args.graph == "chord" and "cnn" in wanted:
        wanted = [k for k in wanted if k != "cnn"]
        print("[note] skipping the CNN baseline: it does not consume a graph")
    if args.graph == "chord" and args.model == "all":
        wanted.append("weighted_chord")
    results = {"config": vars(args), "graph": args.graph, "genres": genres,
               "split_sizes": {k: len(v) for k, v in by_split.items()},
               "artist_leakage": leakage,
               "provenance": build_provenance(
                   vars(args), by_split, genres, "not_applicable", "val_accuracy",
                   vocabulary_from="fma_small_definition",
               ),
               "runs": {}}
    for kind in wanted:
        print(f"\n=== {kind} ===")
        results["runs"][kind] = (
            run_cnn(by_split, genres, args, device) if kind == "cnn"
            else run_graph_model(kind, graph_splits, genres, args, device)
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = args.metrics_name or (
        "metrics_task2.json" if args.graph == "segment"
        else "metrics_task2_%s.json" % args.graph
    )
    (out / name).write_text(json.dumps(results, indent=2))

    print("\n=== Task 2 summary ===")
    print(f"{'model':<10}{'params':>10}{'test-acc':>10}{'macro-F1':>10}")
    for kind, r in results["runs"].items():
        print(f"{r['model']:<10}{r['params']:>10,}{r['test_accuracy']:>10.4f}"
              f"{r['test_macro_f1']:>10.4f}")
    print(f"[out ] {out}/{name}")


if __name__ == "__main__":
    main()
