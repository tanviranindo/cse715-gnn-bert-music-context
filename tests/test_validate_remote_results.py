import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infra import validate_remote_results
from src.evaluation_artifacts import file_digest, write_multilabel_predictions
from src.run_provenance import canonical_digest


REVISION = "a" * 40


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    evidence = results / "evaluation" / "task1.json.gz"
    ids = ["clip-a", "clip-b"]
    vocab = ["rock", "quiet"]
    write_multilabel_predictions(
        evidence,
        ids,
        [[1, 0], [0, 1]],
        [[0.9, 0.1], [0.2, 0.8]],
        vocab,
        0.5,
    )
    metrics = {
        "provenance": {
            "source_revision": REVISION,
            "split_digest": "b" * 64,
            "split_id_digests": {"test": canonical_digest(sorted(ids))},
            "vocabulary_digest": canonical_digest(vocab),
            "vocabulary_from": "train",
            "threshold_from": "val",
            "checkpoint_metric": "val_macro_f1",
        },
        "test": {"micro_f1": 1.0, "macro_f1": 1.0},
        "evaluation_artifact": {
            "path": "/remote/results/evaluation/task1.json.gz",
            "sha256": file_digest(evidence),
        },
    }
    metrics_path = results / "metrics_task1.json"
    metrics_path.write_text(json.dumps(metrics))
    return results, metrics_path


def _rewrite_evidence(results: Path, mutate) -> None:
    path = results / "evaluation" / "task1.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    mutate(payload)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(encoded)
    metrics_path = results / "metrics_task1.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["evaluation_artifact"]["sha256"] = file_digest(path)
    metrics_path.write_text(json.dumps(metrics))


def test_validator_accepts_reconstructable_result_bundle(tmp_path):
    results, _ = _fixture(tmp_path)
    summary = validate_remote_results.validate(results, REVISION)
    assert summary == {"metric_files": 1, "evaluation_artifacts": 1}


def test_validator_documented_direct_cli_entrypoint(tmp_path):
    results, _ = _fixture(tmp_path)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "infra/validate_remote_results.py",
            "--results",
            str(results),
            "--expected-revision",
            REVISION,
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_validator_resolves_repeated_evidence_names_by_results_relative_path(tmp_path):
    results, metrics_path = _fixture(tmp_path)
    original_metrics = json.loads(metrics_path.read_text())
    original_evidence = results / "evaluation" / "task1.json.gz"
    metrics_path.unlink()
    for run in ("run-a", "run-b"):
        run_dir = results / run
        evidence = run_dir / "evaluation" / "task1.json.gz"
        evidence.parent.mkdir(parents=True)
        shutil.copyfile(original_evidence, evidence)
        document = dict(original_metrics)
        document["evaluation_artifact"] = dict(original_metrics["evaluation_artifact"])
        document["evaluation_artifact"]["path"] = f"/remote/results/{run}/evaluation/task1.json.gz"
        (run_dir / "metrics_task1.json").write_text(json.dumps(document))
    original_evidence.unlink()

    summary = validate_remote_results.validate(results, REVISION)
    assert summary == {"metric_files": 2, "evaluation_artifacts": 2}


@pytest.mark.parametrize("tamper", ["test_id", "vocabulary", "prediction", "metric"])
def test_validator_rejects_tampered_bundle(tmp_path, tamper):
    results, metrics_path = _fixture(tmp_path)
    if tamper == "test_id":
        _rewrite_evidence(results, lambda payload: payload["ids"].__setitem__(0, "other"))
    elif tamper == "vocabulary":
        _rewrite_evidence(
            results, lambda payload: payload["vocabulary"].__setitem__(0, "jazz")
        )
    elif tamper == "prediction":
        _rewrite_evidence(
            results, lambda payload: payload["probabilities"][0].__setitem__(0, 0.1)
        )
    else:
        metrics = json.loads(metrics_path.read_text())
        metrics["test"]["micro_f1"] = 0.5
        metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(validate_remote_results.ValidationError):
        validate_remote_results.validate(results, REVISION)
