"""B4 PCA+MLP audio-only baseline for Task 2.

The baseline uses per-track mean/std summaries of the cached MFCC/chroma
segment features, fits PCA on the training split only, then trains a small MLP
on the reduced vectors.  It is deliberately independent of the graph edges and
the CNN so it provides a third, hand-crafted-audio reference point.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

from src import evaluate as ev, fma_data


def track_features(record: dict) -> np.ndarray:
    x = np.asarray(record["x"], dtype=np.float32)
    return np.concatenate([x.mean(axis=0), x.std(axis=0)], axis=0)


class MLP(nn.Module):
    def __init__(self, width: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/data/processed/fma_small_graphs.pt")
    p.add_argument("--out", default="results/metrics_task2_b4.json")
    p.add_argument("--components", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(args.cache, weights_only=False)
    records = blob["records"]
    splits = fma_data.official_splits(records)
    genres = sorted({r["genre"] for r in records}); index = {g: i for i, g in enumerate(genres)}
    X = {k: np.stack([track_features(r) for r in v]) for k, v in splits.items()}
    y = {k: np.asarray([index[r["genre"]] for r in v], dtype=np.int64) for k, v in splits.items()}

    scaler = StandardScaler().fit(X["train"])
    Xs = {k: scaler.transform(v) for k, v in X.items()}
    n_components = min(args.components, Xs["train"].shape[0], Xs["train"].shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed).fit(Xs["train"])
    Z = {k: pca.transform(v).astype(np.float32) for k, v in Xs.items()}
    loaders = {k: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Z[k]), torch.from_numpy(y[k])),
        batch_size=128, shuffle=(k == "train")) for k in Z}
    model = MLP(n_components, len(genres)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(); history = []; best = (-1.0, None)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for xb, yb in loaders["train"]:
            opt.zero_grad(); loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward(); opt.step(); total += float(loss)
        model.eval(); yt = []; yp = []
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                yp.extend(model(xb.to(device)).argmax(1).cpu().tolist()); yt.extend(yb.tolist())
        row = {"epoch": epoch, "train_loss": total / len(loaders["train"]),
               "val_accuracy": ev.accuracy(yt, yp),
               "val_macro_f1": ev.multiclass_f1(yt, yp, len(genres))}
        history.append(row)
        if row["val_accuracy"] > best[0]:
            best = (row["val_accuracy"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval(); yt = []; yp = []
    with torch.no_grad():
        for xb, yb in loaders["test"]:
            yp.extend(model(xb.to(device)).argmax(1).cpu().tolist()); yt.extend(yb.tolist())
    result = {"model": "pca_mlp_b4", "params": sum(p.numel() for p in model.parameters()),
              "components": n_components, "split_sizes": {k: len(v) for k, v in splits.items()},
              "artist_leakage": fma_data.artist_leakage(splits), "history": history,
              "test_accuracy": ev.accuracy(yt, yp), "test_macro_f1": ev.multiclass_f1(yt, yp, len(genres)),
              "per_class_f1": ev.per_class_f1(yt, yp, genres), "genres": genres}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["test_accuracy", "test_macro_f1", "components"]}, indent=2))


if __name__ == "__main__":
    main()
