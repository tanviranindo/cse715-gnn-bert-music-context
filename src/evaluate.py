"""Metrics and non-neural baselines for Task 1.

Micro/Macro-F1 are implemented directly rather than pulled from sklearn so
that the metric tests run in any environment, including one without the
scientific stack installed.
"""

import random


def _counts(y_true: list[list[int]], y_pred: list[list[int]], k: int) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t[k] and p[k]:
            tp += 1
        elif p[k]:
            fp += 1
        elif t[k]:
            fn += 1
    return tp, fp, fn


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def micro_f1(y_true: list[list[int]], y_pred: list[list[int]]) -> float:
    """F1 over all (clip, tag) decisions pooled together."""
    if not y_true:
        return 0.0
    tp = fp = fn = 0
    for k in range(len(y_true[0])):
        a, b, c = _counts(y_true, y_pred, k)
        tp += a
        fp += b
        fn += c
    return f1(tp, fp, fn)


def macro_f1(y_true: list[list[int]], y_pred: list[list[int]]) -> float:
    """Mean of per-tag F1 — dominated by rare tags, so much harsher."""
    if not y_true:
        return 0.0
    scores = [f1(*_counts(y_true, y_pred, k)) for k in range(len(y_true[0]))]
    return sum(scores) / len(scores)


def per_tag_f1(
    y_true: list[list[int]], y_pred: list[list[int]], vocab: list[str]
) -> dict[str, float]:
    """Per-tag F1, for finding which tags the model cannot learn."""
    return {t: f1(*_counts(y_true, y_pred, k)) for k, t in enumerate(vocab)}


def binarize(probs: list[list[float]], threshold: float = 0.5) -> list[list[int]]:
    return [[1 if p >= threshold else 0 for p in row] for row in probs]


def best_threshold(
    y_true: list[list[int]], probs: list[list[float]], grid: tuple[float, ...] = ()
) -> tuple[float, float]:
    """Threshold maximising Micro-F1. Multi-label BCE rarely peaks at 0.5."""
    grid = grid or tuple(i / 100 for i in range(5, 100, 5))
    best = (0.5, -1.0)
    for t in grid:
        score = micro_f1(y_true, binarize(probs, t))
        if score > best[1]:
            best = (t, score)
    return best


# ---------------------------------------------------------------- baselines

def baseline_random(
    y_true: list[list[int]], seed: int = 42, rate: float | None = None
) -> list[list[int]]:
    """B1: random predictor.

    With `rate=None` each tag fires at its training prevalence, which is a
    stronger and fairer B1 than a uniform coin flip.
    """
    if not y_true:
        return []
    k = len(y_true[0])
    rng = random.Random(seed)
    if rate is None:
        rates = [sum(row[i] for row in y_true) / len(y_true) for i in range(k)]
    else:
        rates = [rate] * k
    return [[1 if rng.random() < rates[i] else 0 for i in range(k)] for _ in y_true]


def baseline_majority(y_true_train: list[list[int]], n: int) -> list[list[int]]:
    """B1 variant: always predict every tag that is positive in >=50% of train."""
    if not y_true_train:
        return []
    k = len(y_true_train[0])
    m = len(y_true_train)
    const = [1 if sum(r[i] for r in y_true_train) * 2 >= m else 0 for i in range(k)]
    return [list(const) for _ in range(n)]
