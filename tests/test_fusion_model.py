"""Task 3 fusion: attention shape/masking and the masked multi-task loss.

The loss tests matter most. DEAM is disjoint from the tag corpora, so a batch
never carries both targets; a silently unmasked term would train against
fabricated zeros and look like it was working.
"""

import pytest

torch = pytest.importorskip("torch")

from src.fusion_model import CrossAttentionFusion, multitask_loss


def test_cross_attention_output_shape():
    fuse = CrossAttentionFusion(graph_dim=16, text_dim=32, attn_dim=8)
    g = torch.randn(4, 16)
    h = torch.randn(4, 7, 32)
    z, attn = fuse(g, h)
    assert z.shape == (4, 16 + 8) == (4, fuse.out_dim)
    assert attn.shape == (4, 7)


def test_attention_weights_sum_to_one():
    fuse = CrossAttentionFusion(8, 8, 4)
    _, attn = fuse(torch.randn(3, 8), torch.randn(3, 5, 8))
    assert torch.allclose(attn.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_padding_tokens_receive_no_attention():
    fuse = CrossAttentionFusion(8, 8, 4)
    mask = torch.tensor([[1, 1, 0, 0]])
    _, attn = fuse(torch.randn(1, 8), torch.randn(1, 4, 8), attention_mask=mask)
    assert attn[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert attn[0, 3].item() == pytest.approx(0.0, abs=1e-6)
    assert attn[0, :2].sum().item() == pytest.approx(1.0, abs=1e-5)


def _dummy(batch=4, n_tags=6):
    return (torch.randn(batch, n_tags), torch.randint(0, 2, (batch, n_tags)).float(),
            torch.randn(batch, 2), torch.randn(batch, 2))


def test_tag_only_batch_ignores_the_emotion_term():
    tl, tt, ep, et = _dummy()
    loss, parts = multitask_loss(tl, tt, ep, et,
                                 tag_mask=torch.ones(4), emotion_mask=torch.zeros(4))
    assert "tag" in parts
    assert "valence_mse" not in parts and "arousal_mse" not in parts


def test_emotion_only_batch_ignores_the_tag_term():
    tl, tt, ep, et = _dummy()
    loss, parts = multitask_loss(tl, tt, ep, et,
                                 tag_mask=torch.zeros(4), emotion_mask=torch.ones(4))
    assert "tag" not in parts
    assert "valence_mse" in parts and "arousal_mse" in parts


def test_fully_unsupervised_batch_gives_zero_loss():
    tl, tt, ep, et = _dummy()
    loss, parts = multitask_loss(tl, tt, ep, et,
                                 tag_mask=torch.zeros(4), emotion_mask=torch.zeros(4))
    assert float(loss) == 0.0 and parts == {}


def test_masked_samples_do_not_affect_the_tag_loss():
    """Only sample 0 is supervised, so garbage in sample 1 must not matter."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5)
    targets = torch.randint(0, 2, (2, 5)).float()
    mask = torch.tensor([1.0, 0.0])
    ep, et = torch.zeros(2, 2), torch.zeros(2, 2)

    a, _ = multitask_loss(logits, targets, ep, et, mask, torch.zeros(2))
    corrupted = targets.clone()
    corrupted[1] = 1 - corrupted[1]          # flip the unsupervised sample
    b, _ = multitask_loss(logits, corrupted, ep, et, mask, torch.zeros(2))
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_alpha_beta_scale_the_emotion_terms():
    tl, tt, ep, et = _dummy()
    lo, _ = multitask_loss(tl, tt, ep, et, torch.zeros(4), torch.ones(4),
                           alpha=1.0, beta=1.0)
    hi, _ = multitask_loss(tl, tt, ep, et, torch.zeros(4), torch.ones(4),
                           alpha=3.0, beta=3.0)
    assert float(hi) == pytest.approx(3 * float(lo), rel=1e-5)


def test_invalid_mode_is_rejected():
    from src.fusion_model import GNNBertFusion
    with pytest.raises(ValueError, match="mode must be one of"):
        GNNBertFusion(n_tags=5, node_dim=8, mode="nonsense")
