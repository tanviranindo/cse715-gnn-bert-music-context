"""Two scripts compare models "on the same data", so the split they derive
independently has to be provably identical -- an earlier version differed only
in whether the pool was shuffled, which no printed number revealed."""

import random

from src.splits import official_eval_split


class _Item:
    def __init__(self, cid, is_eval):
        self.clip_id, self.is_eval = cid, is_eval


def _items(n=100, n_eval=30):
    return [_Item(f"c{i:03d}", i < n_eval) for i in range(n)]


def test_eval_partition_is_held_out_whole():
    tr, va, te = official_eval_split(_items())
    assert len(te) == 30
    assert all(d.is_eval for d in te)
    assert not any(d.is_eval for d in tr + va)


def test_two_independent_calls_agree_exactly():
    a = official_eval_split(_items())
    b = official_eval_split(_items())
    for x, y in zip(a, b):
        assert [d.clip_id for d in x] == [d.clip_id for d in y]


def test_result_does_not_depend_on_prior_rng_use():
    # The old bug: one caller had drawn random numbers first, so a shared global
    # seed produced a different shuffle.
    a = official_eval_split(_items())
    random.random(); random.random(); random.random()
    b = official_eval_split(_items())
    assert [d.clip_id for d in a[0]] == [d.clip_id for d in b[0]]


def test_input_order_does_not_change_the_split():
    items = _items()
    a = official_eval_split(items)
    shuffled = items[:]
    random.Random(7).shuffle(shuffled)
    b = official_eval_split(shuffled)
    assert sorted(d.clip_id for d in a[0]) == sorted(d.clip_id for d in b[0])
    assert sorted(d.clip_id for d in a[1]) == sorted(d.clip_id for d in b[1])


def test_different_seeds_give_different_train_val_membership():
    a = official_eval_split(_items(), seed=1)
    b = official_eval_split(_items(), seed=2)
    assert [d.clip_id for d in a[1]] != [d.clip_id for d in b[1]]


def test_train_val_test_are_disjoint_and_complete():
    tr, va, te = official_eval_split(_items())
    ids = [d.clip_id for d in tr + va + te]
    assert len(ids) == len(set(ids)) == 100
