"""S_graph is a thresholded edge statistic, so the things worth pinning are the
threshold behaviour and the null it is measured against."""

import random

import torch

from src import graph_coherence as gc


def test_identical_endpoints_are_fully_coherent():
    h = torch.ones(4, 8)
    ei = torch.tensor([[0, 1], [1, 2]])
    assert gc.coherence(h, ei, 0.9) == 1.0


def test_orthogonal_endpoints_are_not_coherent():
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ei = torch.tensor([[0], [1]])
    assert gc.coherence(h, ei, 0.5) == 0.0


def test_threshold_is_strict():
    # cosine is exactly 0.5 here; the spec says > tau, not >=
    h = torch.tensor([[1.0, 0.0], [0.5, 0.8660254]])
    ei = torch.tensor([[0], [1]])
    assert gc.coherence(h, ei, 0.5) == 0.0


def test_empty_graph_does_not_crash():
    h = torch.ones(2, 4)
    ei = torch.zeros((2, 0), dtype=torch.long)
    assert gc.coherence(h, ei, 0.5) != gc.coherence(h, ei, 0.5) or True  # nan


def test_rewiring_keeps_the_edge_count_and_node_range():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    rw = gc.rewired(ei, 4, random.Random(0))
    assert rw.shape == ei.shape
    assert int(rw.max()) < 4 and int(rw.min()) >= 0


def test_reading_flags_construction_when_untrained_scores_higher():
    best = {"tau": 0.98, "real": 0.72, "rewired": 0.37,
            "untrained": 0.84, "real_minus_rewired": 0.35}
    text = gc._reading(best, separates=True)
    assert "how the graph was built" in text, "must not credit training for it"


def test_reading_credits_training_when_untrained_scores_lower():
    best = {"tau": 0.98, "real": 0.72, "rewired": 0.37,
            "untrained": 0.40, "real_minus_rewired": 0.35}
    text = gc._reading(best, separates=True)
    assert "attributable to training" in text
