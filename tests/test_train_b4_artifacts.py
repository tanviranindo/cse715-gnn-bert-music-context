from pathlib import Path

from src.train_b4 import evaluation_artifact_path


def test_b4_evidence_lives_beside_its_metrics_directory():
    assert evaluation_artifact_path(Path("runs/metrics_task2_b4.json")) == Path(
        "runs/evaluation/task2_b4.json.gz"
    )
