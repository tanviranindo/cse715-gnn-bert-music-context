"""Task 2's chord-transition graph as a GNN input (PDF S3.3).

The segment graph and the chord graph are built by the same preprocessing pass,
so the risk here is not the graph construction (covered in
test_graph_builder.py) but the adapter that turns cached chord records into
PyTorch Geometric objects: chord *names* have to become node features, and a
track with a single chord has no transition to learn from.
"""

import numpy as np
import pytest
import torch

from src import graph_builder as gb, train_gnn


def _record(chords, genre="Rock", track_id=1):
    """A cache record shaped like the one build_graphs.py writes."""
    nodes, ei, w = gb.build_chord_graph(chords)
    return {
        "x": np.zeros((4, 3), dtype=np.float32),
        "edge_index": np.zeros((2, 2), dtype=np.int64),
        "chord_nodes": nodes,
        "chord_edge_index": ei,
        "chord_edge_weight": w,
        "genre": genre,
        "track_id": track_id,
    }


def test_chord_features_encode_pitch_classes_and_quality():
    feats = train_gnn.chord_node_features(["C:maj", "C:min", "A:maj"])
    assert feats.shape == (3, 13), "12 pitch classes plus a major/minor flag"
    # C major is C, E, G -> pitch classes 0, 4, 7
    assert feats[0][:12].tolist() == [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    # C minor flattens the third: C, Eb, G -> 0, 3, 7
    assert feats[1][:12].tolist() == [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    assert feats[0][12] == 0.0 and feats[1][12] == 1.0, "quality flag separates maj/min"
    assert feats.dtype == np.float32, "Double features fail against Float weights"


def test_unknown_chord_label_is_a_zero_row_not_a_crash():
    """`estimate_chords` can emit labels outside the 24 triad templates."""
    feats = train_gnn.chord_node_features(["N", "C:maj"])
    assert feats.shape == (2, 13)
    assert feats[0].sum() == 0.0


def test_to_pyg_chord_mode_uses_the_chord_graph_not_the_segment_graph():
    recs = [_record(["C:maj", "G:maj", "A:min", "F:maj", "C:maj"])]
    data = train_gnn.to_pyg(recs, ["Rock"], graph="chord")
    assert len(data) == 1
    d = data[0]
    assert d.x.shape[1] == 13, "chord features, not the 3-column segment features"
    assert d.x.shape[0] == len(recs[0]["chord_nodes"])
    assert d.edge_index.shape[1] == recs[0]["chord_edge_index"].shape[1]
    assert d.edge_attr.shape[0] == d.edge_index.shape[1], "transition counts kept"
    assert float(d.edge_attr.mean()) == pytest.approx(1.0), "counts are scale-normalised per graph"
    assert d.x.dtype == torch.float32, "must match the model's parameter dtype"


def test_to_pyg_segment_mode_is_unchanged_by_the_new_flag():
    recs = [_record(["C:maj", "G:maj"])]
    data = train_gnn.to_pyg(recs, ["Rock"], graph="segment")
    assert data[0].x.shape == (4, 3)
    assert not hasattr(data[0], "edge_attr") or data[0].edge_attr is None


def test_single_chord_tracks_are_dropped_rather_than_yielding_empty_graphs():
    """One chord means no transition: message passing has nothing to pass."""
    recs = [_record(["C:maj"] * 8, track_id=1), _record(["C:maj", "G:maj"], track_id=2)]
    assert len(recs[0]["chord_nodes"]) == 1, "repeats collapse to a single node"
    data = train_gnn.to_pyg(recs, ["Rock"], graph="chord")
    assert len(data) == 1, "the degenerate track is dropped, the real one survives"
    assert data[0].track_id == 2


def test_labels_survive_the_chord_path():
    recs = [_record(["C:maj", "G:maj"], genre="Jazz", track_id=7)]
    genres = ["Electronic", "Jazz", "Rock"]
    data = train_gnn.to_pyg(recs, genres, graph="chord")
    assert int(data[0].y) == genres.index("Jazz")


def test_weighted_chord_classifier_changes_when_transition_counts_change():
    from src.gnn_model import WeightedGNNClassifier

    torch.manual_seed(7)
    model = WeightedGNNClassifier(
        in_dim=13, n_classes=3, hidden_dim=8, n_layers=2, dropout=0
    ).eval()
    x = torch.randn(3, 13)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    batch = torch.zeros(3, dtype=torch.long)

    out_a = model(x, edge_index, torch.tensor([1.0, 1.0, 1.0, 1.0]), batch)
    out_b = model(x, edge_index, torch.tensor([9.0, 1.0, 1.0, 1.0]), batch)

    assert not torch.allclose(out_a, out_b)


@pytest.mark.parametrize("mode", ["segment", "chord"])
def test_both_modes_produce_batchable_objects(mode):
    from torch_geometric.loader import DataLoader

    recs = [_record(["C:maj", "G:maj", "A:min"], track_id=i) for i in range(4)]
    data = train_gnn.to_pyg(recs, ["Rock"], graph=mode)
    batch = next(iter(DataLoader(data, batch_size=4)))
    assert batch.num_graphs == 4
    assert torch.isfinite(batch.x).all()
