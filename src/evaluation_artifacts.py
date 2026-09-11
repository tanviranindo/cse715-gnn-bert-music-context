"""Write compact evidence from which reported evaluation metrics can be rebuilt."""

import gzip
import hashlib
import json
from pathlib import Path

from src import evaluate as ev


def _validate_ids(ids: list[str]) -> list[str]:
    normalized = [str(value) for value in ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate example IDs are not allowed")
    return normalized


def _write(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(encoded)


def _read(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def file_digest(path: str | Path) -> str:
    """SHA-256 for tying a metrics summary to its exact evidence file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_multilabel_predictions(
    path: str | Path,
    ids: list[str],
    y_true: list[list[int]],
    probabilities: list[list[float]],
    vocab: list[str],
    threshold: float,
) -> None:
    ids = _validate_ids(ids)
    if not (len(ids) == len(y_true) == len(probabilities)):
        raise ValueError("IDs, labels, and probabilities must have equal row counts")
    width = len(vocab)
    if any(len(row) != width for row in y_true + probabilities):
        raise ValueError("every label and probability row must match the vocabulary")
    _write(
        path,
        {
            "kind": "multilabel",
            "ids": ids,
            "vocabulary": list(vocab),
            "threshold": float(threshold),
            "y_true": y_true,
            "probabilities": probabilities,
        },
    )


def reconstruct_multilabel(path: str | Path) -> dict[str, float]:
    payload = _read(path)
    predicted = ev.binarize(payload["probabilities"], payload["threshold"])
    return {
        "micro_f1": ev.micro_f1(payload["y_true"], predicted),
        "macro_f1": ev.macro_f1(payload["y_true"], predicted),
        "auc_pr": ev.macro_auc_pr(payload["y_true"], payload["probabilities"]),
    }


def write_multiclass_predictions(
    path: str | Path,
    ids: list[str],
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
) -> None:
    ids = _validate_ids(ids)
    if not (len(ids) == len(y_true) == len(y_pred)):
        raise ValueError("IDs, true labels, and predictions must have equal row counts")
    _write(
        path,
        {
            "kind": "multiclass",
            "ids": ids,
            "classes": list(classes),
            "y_true": y_true,
            "y_pred": y_pred,
        },
    )


def reconstruct_multiclass(path: str | Path) -> dict:
    payload = _read(path)
    n_classes = len(payload["classes"])
    return {
        "accuracy": ev.accuracy(payload["y_true"], payload["y_pred"]),
        "macro_f1": ev.multiclass_f1(
            payload["y_true"], payload["y_pred"], n_classes
        ),
        "confusion": ev.confusion_matrix(
            payload["y_true"], payload["y_pred"], n_classes
        ),
    }


def write_retrieval_ranks(
    path: str | Path,
    query_ids: list[str],
    target_ids: list[str],
    caption_to_audio_ranks: list[int],
    audio_to_caption_ranks: list[int],
) -> None:
    query_ids = _validate_ids(query_ids)
    target_ids = _validate_ids(target_ids)
    n = len(query_ids)
    if not (
        n == len(target_ids)
        == len(caption_to_audio_ranks)
        == len(audio_to_caption_ranks)
    ):
        raise ValueError("retrieval IDs and ranks must have equal row counts")
    if any(rank < 1 for rank in caption_to_audio_ranks + audio_to_caption_ranks):
        raise ValueError("retrieval ranks are one-based positive integers")
    _write(
        path,
        {
            "kind": "retrieval",
            "query_ids": query_ids,
            "target_ids": target_ids,
            "caption_to_audio_ranks": caption_to_audio_ranks,
            "audio_to_caption_ranks": audio_to_caption_ranks,
        },
    )


def _recalls(ranks: list[int]) -> dict[str, float]:
    n = max(len(ranks), 1)
    return {f"R@{k}": sum(rank <= k for rank in ranks) / n for k in (1, 5, 10)}


def reconstruct_retrieval(path: str | Path) -> dict[str, dict[str, float]]:
    payload = _read(path)
    return {
        "caption_to_audio": _recalls(payload["caption_to_audio_ranks"]),
        "audio_to_caption": _recalls(payload["audio_to_caption_ranks"]),
    }
