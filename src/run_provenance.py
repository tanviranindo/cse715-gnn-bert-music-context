"""Stable data and configuration provenance for experimental artifacts."""

import hashlib
import json
import os
import subprocess

from src.splits import record_id, split_digest


def canonical_digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_revision() -> str | None:
    explicit = os.environ.get("EXPERIMENT_SOURCE_REVISION")
    if explicit:
        if len(explicit) != 40 or any(ch not in "0123456789abcdef" for ch in explicit.lower()):
            raise ValueError("EXPERIMENT_SOURCE_REVISION must be a 40-character git SHA")
        return explicit.lower()
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_provenance(
    config: dict,
    by_split: dict[str, list],
    vocab: list[str],
    threshold_from: str,
    checkpoint_metric: str,
    vocabulary_from: str = "train",
) -> dict:
    """Describe the decisions that can otherwise silently leak held-out data."""
    split_ids = {
        name: sorted(record_id(record) for record in by_split.get(name, []))
        for name in ("train", "val", "test")
    }
    return {
        "source_revision": _source_revision(),
        "config_digest": canonical_digest(config),
        "split_digest": split_digest(by_split),
        "split_sizes": {name: len(ids) for name, ids in split_ids.items()},
        "split_id_digests": {
            name: canonical_digest(ids) for name, ids in split_ids.items()
        },
        "vocabulary_from": vocabulary_from,
        "vocabulary_digest": canonical_digest(vocab),
        "threshold_from": threshold_from,
        "checkpoint_metric": checkpoint_metric,
    }
