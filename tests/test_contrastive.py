"""Task 4 contrastive: InfoNCE behaviour and retrieval metrics."""

import pytest

torch = pytest.importorskip("torch")

from src.contrastive import info_nce, median_rank, recall_at_k


def _normed(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_perfect_alignment_gives_near_zero_loss():
    e = _normed(torch.eye(6))
    loss, parts = info_nce(e, e, torch.tensor(0.05))
    assert float(loss) < 0.01
    assert parts["audio_to_caption"] == pytest.approx(parts["caption_to_audio"])


def test_misaligned_pairs_cost_more_than_aligned():
    e = _normed(torch.eye(6))
    shuffled = e[torch.tensor([1, 2, 3, 4, 5, 0])]
    good, _ = info_nce(e, e, torch.tensor(0.05))
    bad, _ = info_nce(e, shuffled, torch.tensor(0.05))
    assert float(bad) > float(good)


def test_loss_is_symmetric_in_its_two_directions():
    torch.manual_seed(0)
    g = _normed(torch.randn(8, 16))
    t = _normed(torch.randn(8, 16))
    a, _ = info_nce(g, t, torch.tensor(0.07))
    b, _ = info_nce(t, g, torch.tensor(0.07))
    assert float(a) == pytest.approx(float(b), abs=1e-5)


def test_lower_temperature_sharpens_the_loss():
    torch.manual_seed(0)
    g = _normed(torch.randn(8, 16))
    t = _normed(torch.randn(8, 16))
    sharp, _ = info_nce(g, t, torch.tensor(0.01))
    soft, _ = info_nce(g, t, torch.tensor(0.5))
    assert float(sharp) > float(soft)


def test_recall_is_one_for_perfect_retrieval():
    e = _normed(torch.eye(10))
    r = recall_at_k(e, e)
    assert r["R@1"] == 1.0 and r["R@5"] == 1.0 and r["R@10"] == 1.0


def test_recall_at_larger_k_is_never_smaller():
    torch.manual_seed(1)
    g = _normed(torch.randn(30, 8))
    t = _normed(torch.randn(30, 8))
    r = recall_at_k(g, t)
    assert r["R@1"] <= r["R@5"] <= r["R@10"]


def test_recall_matches_a_hand_built_ranking():
    # query 0 matches gallery 0 exactly (sim 1.0 vs 0.0).
    # query 1 leans toward gallery 0 (sim 0.8) over its true match gallery 1
    # (sim 0.6), so its true match ranks 2nd.
    g = _normed(torch.tensor([[1.0, 0.0], [0.8, 0.6]]))
    t = _normed(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    r = recall_at_k(g, t, ks=(1, 2))
    assert r["R@1"] == 0.5      # only query 0 is rank-1
    assert r["R@2"] == 1.0


def test_median_rank_is_one_for_perfect_retrieval():
    e = _normed(torch.eye(9))
    assert median_rank(e, e) == 1.0


def test_median_rank_degrades_for_random_embeddings():
    torch.manual_seed(2)
    g = _normed(torch.randn(50, 8))
    t = _normed(torch.randn(50, 8))
    assert median_rank(g, t) > 1.0


def test_temperature_is_clamped_away_from_zero():
    from src.contrastive import DualEncoder
    enc = DualEncoder.__new__(DualEncoder)
    torch.nn.Module.__init__(enc)
    enc.log_temperature = torch.nn.Parameter(torch.tensor(-50.0))
    assert float(enc.temperature) >= 1e-3, "unclamped tau would produce inf logits"
