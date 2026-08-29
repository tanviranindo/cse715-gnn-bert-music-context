"""Task 1 metrics: hand-checked values, not just self-consistency."""

import pytest

from src import evaluate as ev


def test_perfect_and_empty_predictions():
    y = [[1, 0, 1], [0, 1, 0]]
    assert ev.micro_f1(y, y) == 1.0
    assert ev.macro_f1(y, y) == 1.0
    zeros = [[0, 0, 0], [0, 0, 0]]
    assert ev.micro_f1(y, zeros) == 0.0


def test_micro_f1_matches_hand_computation():
    #            tag0      tag1      tag2
    y_true = [[1, 1, 0], [1, 0, 0]]
    y_pred = [[1, 0, 1], [1, 0, 0]]
    # tp = tag0 twice = 2 ; fp = tag2 once = 1 ; fn = tag1 once = 1
    assert ev.micro_f1(y_true, y_pred) == 2 * 2 / (2 * 2 + 1 + 1)


def test_macro_f1_is_penalised_by_an_unlearned_tag():
    y_true = [[1, 1], [1, 1]]
    y_pred = [[1, 0], [1, 0]]   # tag0 perfect, tag1 never predicted
    assert ev.macro_f1(y_true, y_pred) == 0.5
    assert ev.micro_f1(y_true, y_pred) > ev.macro_f1(y_true, y_pred)


def test_binarize_respects_threshold():
    assert ev.binarize([[0.4, 0.6]], 0.5) == [[0, 1]]
    assert ev.binarize([[0.4, 0.6]], 0.3) == [[1, 1]]


def test_best_threshold_beats_naive_half():
    # every label positive, but probabilities all sit below 0.5
    y_true = [[1, 1], [1, 1]]
    probs = [[0.3, 0.35], [0.31, 0.36]]
    t, score = ev.best_threshold(y_true, probs)
    assert score == 1.0 and t <= 0.3
    assert ev.micro_f1(y_true, ev.binarize(probs, 0.5)) == 0.0


def test_random_baseline_uses_prevalence_by_default():
    # tag0 always on, tag1 always off -> baseline should mirror that
    y_true = [[1, 0]] * 200
    pred = ev.baseline_random(y_true, seed=0)
    assert all(r[0] == 1 for r in pred)
    assert all(r[1] == 0 for r in pred)


def test_random_baseline_is_deterministic_under_seed():
    y = [[1, 0, 1]] * 50
    assert ev.baseline_random(y, seed=7) == ev.baseline_random(y, seed=7)


def test_majority_baseline_predicts_only_frequent_tags():
    train = [[1, 0], [1, 0], [1, 1]]   # tag0 in 3/3, tag1 in 1/3
    out = ev.baseline_majority(train, n=2)
    assert out == [[1, 0], [1, 0]]


def test_per_tag_f1_exposes_the_dead_tag():
    y_true = [[1, 1], [1, 1]]
    y_pred = [[1, 0], [1, 0]]
    scores = ev.per_tag_f1(y_true, y_pred, ["good", "dead"])
    assert scores["good"] == 1.0 and scores["dead"] == 0.0


def test_accuracy_counts_exact_matches():
    assert ev.accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert ev.accuracy([0, 1, 2], [0, 1, 0]) == pytest.approx(2 / 3)
    assert ev.accuracy([], []) == 0.0


def test_multiclass_micro_f1_equals_accuracy():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 1, 1, 1]
    assert ev.multiclass_f1(y_true, y_pred, 3, "micro") == pytest.approx(
        ev.accuracy(y_true, y_pred)
    )


def test_multiclass_macro_f1_penalises_an_ignored_class():
    # class 2 is never predicted, so its F1 is 0 and drags macro down
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]
    assert ev.multiclass_f1(y_true, y_pred, 3) < ev.accuracy(y_true, y_pred)


def test_per_class_f1_names_the_classes():
    scores = ev.per_class_f1([0, 1], [0, 0], ["rock", "folk"])
    assert scores["rock"] > 0 and scores["folk"] == 0.0


def test_confusion_matrix_places_counts_correctly():
    m = ev.confusion_matrix([0, 0, 1], [0, 1, 1], 2)
    assert m == [[1, 1], [0, 1]]
