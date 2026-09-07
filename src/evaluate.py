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
    y_true: list[list[int]], seed: int = 42,
    rate: float | list[float] | None = None,
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
    elif isinstance(rate, list):
        if len(rate) != k:
            raise ValueError("rate list must have one prevalence per tag")
        rates = rate
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


# ------------------------------------------------- multiclass (Task 2 genre)

def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def multiclass_f1(
    y_true: list[int], y_pred: list[int], n_classes: int, average: str = "macro"
) -> float:
    """Macro or micro F1 for single-label multiclass.

    Note micro-F1 equals accuracy when every sample has exactly one predicted
    and one true label, so macro is the informative number for genre.
    """
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    if average == "micro":
        return f1(sum(tp), sum(fp), sum(fn))
    return sum(f1(tp[k], fp[k], fn[k]) for k in range(n_classes)) / n_classes


def per_class_f1(
    y_true: list[int], y_pred: list[int], classes: list[str]
) -> dict[str, float]:
    n = len(classes)
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    return {classes[k]: f1(tp[k], fp[k], fn[k]) for k in range(n)}


def confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int) -> list[list[int]]:
    m = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


# ------------------------------------------- ranking metrics (Task 3 AUC-PR)

def average_precision(y_true: list[int], scores: list[float]) -> float:
    """Area under the precision-recall curve for one tag, by interpolation
    over the ranked list (the standard 'average precision' definition).

    Reported instead of ROC-AUC for multi-label tagging because the positive
    class is rare (2-25% prevalence here) and ROC-AUC is optimistic under
    class imbalance.
    """
    pairs = sorted(zip(scores, y_true), key=lambda p: -p[0])
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    tp = 0
    total = 0.0
    for i, (_, label) in enumerate(pairs, start=1):
        if label:
            tp += 1
            total += tp / i
    return total / n_pos


def macro_auc_pr(y_true: list[list[int]], probs: list[list[float]]) -> float:
    """Mean average precision across tags. Tags with no positives are skipped."""
    if not y_true:
        return 0.0
    n_tags = len(y_true[0])
    scores = []
    for k in range(n_tags):
        column = [row[k] for row in y_true]
        if sum(column) == 0:
            continue
        scores.append(average_precision(column, [row[k] for row in probs]))
    return sum(scores) / len(scores) if scores else 0.0


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """MAE and R^2 for the valence/arousal heads."""
    if not y_true:
        return {"mae": 0.0, "r2": 0.0}
    n = len(y_true)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    mean = sum(y_true) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean) ** 2 for a in y_true)
    return {"mae": mae, "r2": 1 - ss_res / ss_tot if ss_tot else 0.0}
