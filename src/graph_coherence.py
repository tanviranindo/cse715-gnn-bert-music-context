"""Graph coherence score, PDF S6.

    S_graph = (1/|E|) * sum over (i,j) in E of  1[ cos(h_i, h_j) > tau ]

The fraction of graph edges whose endpoints the trained GNN maps to similar
representations. The specification frames it as asking whether high-attention
edges align with repeated musical structure; concretely it asks whether the
edges we built are the edges the network ends up agreeing with.

A number on its own means nothing here, because cosine similarity between
ReLU activations is positive by construction and will look high whatever the
graph says. So every score is reported against two controls on the same nodes:

  * rewired  -- the same number of edges drawn at random between the same
    nodes. If the real graph does not beat this, the edges carry no signal
    the encoder uses.
  * untrained -- the same architecture at initialisation. Separates what the
    encoder learned from what its shape imposes.

Node embeddings are taken before pooling, from the graph tower of the trained
Task 3 checkpoint.

    python -m src.graph_coherence --checkpoint results/task3_crossattn.pt
"""

import argparse
import glob
import json
import pathlib
import random

import torch


def node_embeddings(encoder, x, edge_index):
    """Per-node representations: the encoder's stack without the pooling step."""
    encoder.eval()
    with torch.no_grad():
        h = x
        for layer, norm in zip(encoder.layers, encoder.norms):
            h = layer(h, edge_index)
            h = norm(h)
            h = torch.relu(h)
    return h


def coherence(h: torch.Tensor, edge_index: torch.Tensor, tau: float) -> float:
    """Fraction of edges whose endpoints exceed cosine similarity `tau`."""
    if edge_index.numel() == 0:
        return float("nan")
    hn = torch.nn.functional.normalize(h, dim=-1)
    sim = (hn[edge_index[0]] * hn[edge_index[1]]).sum(-1)
    return float((sim > tau).float().mean())


def rewired(edge_index: torch.Tensor, n_nodes: int, rng: random.Random) -> torch.Tensor:
    """Same edge count, same nodes, no structure -- the null this is measured against."""
    m = edge_index.shape[1]
    src = [rng.randrange(n_nodes) for _ in range(m)]
    dst = [rng.randrange(n_nodes) for _ in range(m)]
    return torch.tensor([src, dst], dtype=torch.long)


def _reading(best: dict, separates: bool) -> str:
    """State what the numbers support, including where training is not the cause."""
    if not separates:
        return ("At no threshold do the built edges separate from random edges "
                "between the same nodes (best margin %+.3f at tau = %.3f), so the "
                "segment graph's specific wiring carries little the encoder "
                "distinguishes." % (best["real_minus_rewired"], best["tau"]))

    head = ("Real edges are markedly more coherent than rewired ones -- at tau = "
            "%.3f, %.3f against %.3f (%+.3f). The graph's wiring is not arbitrary."
            % (best["tau"], best["real"], best["rewired"], best["real_minus_rewired"]))

    # The honest qualifier: the segment graph is *built* by thresholding cosine
    # similarity of node features, so its endpoints start similar. If the
    # untrained encoder scores at least as high, the coherence is a property of
    # the construction rather than something the network learned.
    if best["untrained"] >= best["real"]:
        return (head + " But an untrained encoder of the same shape scores %.3f "
                "on those same edges -- higher than the trained one -- so this "
                "coherence is a property of how the graph was built (edges are "
                "thresholded on feature cosine similarity) rather than something "
                "training produced. Training slightly *reduces* it, which is what "
                "an encoder learning to tell adjacent segments apart would do."
                % best["untrained"])
    return (head + " The untrained control scores %.3f, so part of the margin is "
            "attributable to training rather than to construction."
            % best["untrained"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/task3_crossattn.pt")
    p.add_argument("--samples", default="data/processed/graph_samples")
    p.add_argument("--tau", type=float, default=0.5,
                   help="headline threshold; a sweep is reported either way")
    p.add_argument("--out", default="results/metrics_graph_coherence.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from src.gnn_model import GraphEncoder

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    cfg = ckpt.get("config", {})
    gstate = {k[len("graph_encoder."):]: v for k, v in state.items()
              if k.startswith("graph_encoder.")}
    if not gstate:
        raise SystemExit("no graph_encoder weights in %s" % args.checkpoint)

    files = sorted(glob.glob(str(pathlib.Path(args.samples) / "*.pt")))
    if not files:
        raise SystemExit("no sample graphs in %s" % args.samples)
    in_dim = torch.load(files[0], weights_only=False)["x"].shape[1]

    kw = dict(in_dim=in_dim,
              hidden_dim=cfg.get("hidden", 128),
              n_layers=cfg.get("layers", 2),
              conv=cfg.get("conv", "gat"))
    trained = GraphEncoder(**kw)
    trained.load_state_dict(gstate)
    torch.manual_seed(args.seed)
    untrained = GraphEncoder(**kw)

    import statistics

    # Post-ReLU embeddings are non-negative, so cosine similarity between any
    # two of them is high by construction and a single low threshold marks
    # every edge coherent regardless of the graph. The score is therefore
    # reported as a sweep, and the honest headline is where -- if anywhere --
    # the real edges separate from rewired ones.
    taus = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
    rng = random.Random(args.seed)

    graphs = []
    for f in files:
        g = torch.load(f, weights_only=False)
        # The committed samples store plain arrays so they can be read without
        # torch_geometric; the encoder needs tensors.
        x = torch.as_tensor(g["x"], dtype=torch.float32)
        ei = torch.as_tensor(g["edge_index"], dtype=torch.long)
        graphs.append((x, ei, node_embeddings(trained, x, ei),
                       node_embeddings(untrained, x, ei),
                       rewired(ei, x.shape[0], rng)))

    sweep = []
    for tau in taus:
        row = {"tau": tau}
        for name, fn in (("real", lambda x, ei, h, hu, rw: coherence(h, ei, tau)),
                         ("rewired", lambda x, ei, h, hu, rw: coherence(h, rw, tau)),
                         ("untrained", lambda x, ei, h, hu, rw: coherence(hu, ei, tau))):
            vals = [fn(*g) for g in graphs]
            row[name] = round(statistics.mean(vals), 4)
        row["real_minus_rewired"] = round(row["real"] - row["rewired"], 4)
        sweep.append(row)

    best = max(sweep, key=lambda r: r["real_minus_rewired"])
    separates = best["real_minus_rewired"] > 0.05

    doc = {
        "note": ("Graph coherence S_graph (PDF S6): the fraction of edges whose "
                 "endpoints the trained GNN maps to cosine similarity above tau. "
                 "Swept over tau, because post-ReLU embeddings are non-negative "
                 "and any single low threshold marks every edge coherent."),
        "n_graphs": len(files),
        "checkpoint": args.checkpoint,
        "conv": kw["conv"],
        "controls": {
            "rewired": "same edge count between the same nodes, drawn at random",
            "untrained": "same architecture at initialisation",
        },
        "sweep": sweep,
        "most_discriminating": best,
        "reading": _reading(best, separates),
    }
    pathlib.Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote %s" % args.out)
    print("  %-7s %-8s %-8s %-9s %s" % ("tau", "real", "rewired", "untrained", "real-rewired"))
    for r in sweep:
        print("  %-7.3f %-8.4f %-8.4f %-9.4f %+.4f"
              % (r["tau"], r["real"], r["rewired"], r["untrained"], r["real_minus_rewired"]))
    print("\n  %s" % doc["reading"])


if __name__ == "__main__":
    main()
