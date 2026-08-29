"""Task 2 graph construction: structure, thresholds, chord transitions."""

import numpy as np

from src import graph_builder as gb


def test_cosine_similarity_is_one_on_the_diagonal():
    x = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    sim = gb.cosine_similarity_matrix(x)
    assert np.allclose(np.diag(sim), 1.0)
    assert np.isclose(sim[0, 1], 0.0)


def test_cosine_handles_zero_vector_without_nan():
    sim = gb.cosine_similarity_matrix(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert not np.isnan(sim).any()


def test_adjacency_edges_exist_even_below_threshold():
    # three mutually dissimilar segments; only the temporal chain should survive
    x = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    ei, ew = gb.build_segment_graph(x, tau=0.99)
    # chain 0-1 and 1-2, stored both directions => 4 directed edges
    assert ei.shape == (2, 4)
    assert len(ew) == 4


def test_similarity_edges_appear_above_threshold():
    # nodes 0 and 2 identical -> a similarity edge on top of the chain
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ei, _ = gb.build_segment_graph(x, tau=0.9)
    edges = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert (0, 2) in edges and (2, 0) in edges


def test_graph_is_undirected_and_selfloop_free():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 5))
    ei, _ = gb.build_segment_graph(x, tau=0.2)
    edges = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert all((b, a) in edges for a, b in edges)
    assert all(a != b for a, b in edges)


def test_empty_features_produce_an_empty_graph():
    ei, ew = gb.build_segment_graph(np.zeros((0, 4)))
    assert ei.shape == (2, 0) and len(ew) == 0


def test_chord_templates_recover_a_pure_triad():
    chroma = np.zeros((12, 4))
    chroma[[0, 4, 7], :] = 1.0          # C major triad
    assert gb.estimate_chords(chroma) == ["C:maj"] * 4


def test_minor_triad_is_distinguished_from_major():
    chroma = np.zeros((12, 2))
    chroma[[0, 3, 7], :] = 1.0          # C minor
    assert gb.estimate_chords(chroma)[0] == "C:min"


def test_collapse_repeats_keeps_only_changes():
    assert gb.collapse_repeats(["C", "C", "G", "G", "C"]) == ["C", "G", "C"]
    assert gb.collapse_repeats([]) == []


def test_chord_graph_counts_repeated_transitions():
    nodes, ei, w = gb.build_chord_graph(["C", "G", "C", "G", "C"])
    assert nodes == ["C", "G"]
    weights = {(int(a), int(b)): float(x) for a, b, x in zip(ei[0], ei[1], w)}
    ci, gi = nodes.index("C"), nodes.index("G")
    assert weights[(ci, gi)] == 2.0    # C->G twice
    assert weights[(gi, ci)] == 2.0    # G->C twice


def test_chord_graph_with_a_single_chord_has_no_edges():
    nodes, ei, w = gb.build_chord_graph(["C", "C", "C"])
    assert nodes == ["C"] and ei.shape == (2, 0) and len(w) == 0


def test_graph_stats_reports_degree():
    ei, _ = gb.build_segment_graph(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]), tau=0.9
    )
    stats = gb.graph_stats(ei, 3)
    assert stats["n_nodes"] == 3 and stats["n_edges"] == ei.shape[1]
    assert stats["avg_degree"] > 0
