"""The demo notebook's end-to-end example (spec S10.5) runs through
`src.infer.predict`, so its contract is worth pinning without downloading a
2xx MB checkpoint in CI."""

import torch

from src import infer


class _StubTokenizer:
    def __call__(self, text, **kwargs):
        n = kwargs.get("max_length", 8)
        return {
            "input_ids": torch.ones(1, n, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        }


class _StubBatch(dict):
    def to(self, device):
        return self


class _StubTokenizerWithTo(_StubTokenizer):
    def __call__(self, text, **kwargs):
        return _StubBatch(super().__call__(text, **kwargs))


class _StubModel:
    """Returns fixed logits so the ordering is checkable."""

    def __init__(self, logits):
        self.logits = torch.tensor([logits])

    def __call__(self, input_ids, attention_mask):
        return self.logits


def test_predict_returns_every_tag_ranked_by_probability():
    ranked = infer.predict(
        _StubModel([-2.0, 3.0, 0.0]), _StubTokenizerWithTo(),
        ["quiet", "loud", "piano"], "some caption",
    )
    assert [t for t, _ in ranked] == ["loud", "piano", "quiet"]


def test_predict_applies_sigmoid_to_the_logits():
    ranked = infer.predict(
        _StubModel([0.0]), _StubTokenizerWithTo(), ["piano"], "x",
    )
    assert ranked[0][1] == 0.5


def test_predict_probabilities_are_descending():
    ranked = infer.predict(
        _StubModel([1.0, -1.0, 0.5, 2.0]), _StubTokenizerWithTo(),
        list("abcd"), "x",
    )
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
