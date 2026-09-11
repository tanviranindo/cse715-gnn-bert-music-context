import pytest

from src.evaluation_artifacts import (
    reconstruct_multiclass,
    reconstruct_multilabel,
    reconstruct_retrieval,
    write_multiclass_predictions,
    write_multilabel_predictions,
    write_retrieval_ranks,
)


def test_multilabel_artifact_reconstructs_hand_checked_scores(tmp_path):
    path = tmp_path / "predictions.json.gz"
    write_multilabel_predictions(
        path,
        ["a", "b"],
        [[1, 0], [0, 1]],
        [[0.9, 0.2], [0.6, 0.8]],
        ["x", "y"],
        0.5,
    )
    got = reconstruct_multilabel(path)
    assert got["micro_f1"] == pytest.approx(0.8)
    assert got["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert 0.0 <= got["auc_pr"] <= 1.0


def test_multiclass_artifact_reconstructs_accuracy_and_confusion(tmp_path):
    path = tmp_path / "classes.json.gz"
    write_multiclass_predictions(
        path, ["a", "b", "c"], [0, 1, 1], [0, 0, 1], ["rock", "folk"]
    )
    got = reconstruct_multiclass(path)
    assert got["accuracy"] == pytest.approx(2 / 3)
    assert got["confusion"] == [[1, 0], [1, 1]]


def test_retrieval_artifact_reconstructs_bidirectional_recall(tmp_path):
    path = tmp_path / "ranks.json.gz"
    write_retrieval_ranks(
        path,
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d"],
        [1, 2, 7, 11],
        [10, 5, 1, 20],
    )
    got = reconstruct_retrieval(path)
    assert got["caption_to_audio"] == {"R@1": 0.25, "R@5": 0.5, "R@10": 0.75}
    assert got["audio_to_caption"] == {"R@1": 0.25, "R@5": 0.5, "R@10": 0.75}


def test_prediction_writer_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_multilabel_predictions(
            tmp_path / "bad.json.gz",
            ["a", "a"],
            [[1], [0]],
            [[0.9], [0.1]],
            ["x"],
            0.5,
        )
