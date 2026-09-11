#!/usr/bin/env python3
"""Validate remote experiment outputs before promoting them into the repository."""

import argparse
import gzip
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation_artifacts import (
    file_digest,
    reconstruct_multiclass,
    reconstruct_multilabel,
    reconstruct_retrieval,
)
from src.run_provenance import canonical_digest


class ValidationError(RuntimeError):
    """An artifact cannot support the metric or provenance it reports."""


def _nodes_with_evidence(value):
    if isinstance(value, dict):
        if "evaluation_artifact" in value:
            yield value
        for child in value.values():
            yield from _nodes_with_evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nodes_with_evidence(child)


def _resolve_evidence(results: Path, declared: str) -> Path:
    candidate = Path(declared)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute() and (results / candidate).exists():
        return results / candidate
    result_indexes = [
        index for index, part in enumerate(candidate.parts) if part == "results"
    ]
    if result_indexes:
        relative = Path(*candidate.parts[result_indexes[-1] + 1 :])
        relocated = results / relative
        if relocated.exists():
            return relocated
    matches = list(results.rglob(candidate.name))
    if len(matches) != 1:
        raise ValidationError(
            f"cannot uniquely resolve evidence {declared!r}: {len(matches)} matches"
        )
    return matches[0]


def _assert_close(label: str, reported, reconstructed) -> None:
    if not math.isclose(float(reported), float(reconstructed), rel_tol=1e-7, abs_tol=1e-9):
        raise ValidationError(
            f"{label} mismatch: reported={reported!r} reconstructed={reconstructed!r}"
        )


def _reported_metrics(node: dict, kind: str) -> dict:
    if kind == "multilabel":
        if "test" in node:
            return node["test"]
        if "test_micro_f1" in node:
            return {
                "micro_f1": node["test_micro_f1"],
                "macro_f1": node["test_macro_f1"],
                "auc_pr": node.get("test_auc_pr"),
            }
        return {key: node.get(key) for key in ("micro_f1", "macro_f1", "auc_pr")}
    if kind == "multiclass":
        return {
            "accuracy": node.get("test_accuracy"),
            "macro_f1": node.get("test_macro_f1"),
        }
    if kind == "retrieval":
        return {
            "caption_to_audio": node.get("caption_to_audio"),
            "audio_to_caption": node.get("audio_to_caption"),
        }
    raise ValidationError(f"unknown evaluation artifact kind {kind!r}")


def _validate_evidence(results: Path, node: dict, provenance: dict) -> None:
    artifact = node["evaluation_artifact"]
    path = _resolve_evidence(results, artifact["path"])
    actual_digest = file_digest(path)
    if actual_digest != artifact.get("sha256"):
        raise ValidationError(f"SHA-256 mismatch for {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)

    ids = payload.get("ids", payload.get("query_ids"))
    expected_id_digest = provenance.get("split_id_digests", {}).get("test")
    if not ids or canonical_digest(sorted(str(value) for value in ids)) != expected_id_digest:
        raise ValidationError(f"test ID digest mismatch for {path}")
    if payload["kind"] == "retrieval":
        target_ids = payload.get("target_ids", [])
        if sorted(map(str, ids)) != sorted(map(str, target_ids)):
            raise ValidationError(f"retrieval query/target IDs differ for {path}")
    else:
        vocab = payload.get("vocabulary", payload.get("classes"))
        if canonical_digest(vocab) != provenance.get("vocabulary_digest"):
            raise ValidationError(f"vocabulary/class digest mismatch for {path}")

    kind = payload["kind"]
    if kind == "multilabel":
        rebuilt = reconstruct_multilabel(path)
    elif kind == "multiclass":
        rebuilt = reconstruct_multiclass(path)
    elif kind == "retrieval":
        rebuilt = reconstruct_retrieval(path)
    else:
        raise ValidationError(f"unknown evaluation artifact kind {kind!r}")
    reported = _reported_metrics(node, kind)
    if kind == "retrieval":
        for direction in ("caption_to_audio", "audio_to_caption"):
            if not isinstance(reported[direction], dict):
                raise ValidationError(f"missing reported {direction} metrics for {path}")
            for metric, value in rebuilt[direction].items():
                _assert_close(f"{path}:{direction}:{metric}", reported[direction][metric], value)
    else:
        for metric, value in rebuilt.items():
            if metric == "confusion" or reported.get(metric) is None:
                continue
            _assert_close(f"{path}:{metric}", reported[metric], value)


def validate(results_dir: str | Path, expected_revision: str) -> dict[str, int]:
    results = Path(results_dir)
    metric_paths = sorted(results.rglob("metrics*.json"))
    if not metric_paths:
        raise ValidationError(f"no metrics JSON files under {results}")
    artifact_count = 0
    for metrics_path in metric_paths:
        try:
            document = json.loads(metrics_path.read_text())
            nodes = list(_nodes_with_evidence(document))
            for node in nodes:
                provenance = node.get("provenance", document.get("provenance"))
                if not provenance:
                    raise ValidationError(f"missing provenance in {metrics_path}")
                if provenance.get("source_revision") != expected_revision:
                    raise ValidationError(
                        f"source revision mismatch in {metrics_path}: "
                        f"{provenance.get('source_revision')!r} != {expected_revision!r}"
                    )
                for required in (
                    "split_digest",
                    "split_id_digests",
                    "vocabulary_digest",
                    "vocabulary_from",
                    "threshold_from",
                    "checkpoint_metric",
                ):
                    if required not in provenance:
                        raise ValidationError(f"missing provenance field {required} in {metrics_path}")
                _validate_evidence(results, node, provenance)
                artifact_count += 1
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"invalid result file {metrics_path}: {exc}") from exc
    if artifact_count == 0:
        raise ValidationError(f"no reconstructable evaluation artifacts under {results}")
    return {"metric_files": len(metric_paths), "evaluation_artifacts": artifact_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    summary = validate(args.results, args.expected_revision)
    print(
        f"validated {summary['evaluation_artifacts']} evaluation artifacts "
        f"across {summary['metric_files']} metric files"
    )


if __name__ == "__main__":
    main()
