"""Task 1 data layer: synonym merging, vocabulary, text, leakage checks."""

import csv

import pytest

from src import mtat_data
from src.splits import artist_grouped_split


def _write_tsv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def corpus(tmp_path):
    """Tiny MTAT-shaped corpus: 3 tags (2 of them synonyms) over 4 clips."""
    ann = tmp_path / "annotations_final.csv"
    _write_tsv(
        ann,
        ["clip_id", "classical", "clasical", "guitar", "mp3_path"],
        [
            ["1", "1", "0", "0", "a/one.mp3"],
            ["2", "0", "1", "0", "a/two.mp3"],
            ["3", "0", "0", "1", "b/three.mp3"],
            ["4", "0", "0", "0", "b/four.mp3"],
        ],
    )
    info = tmp_path / "clip_info_final.csv"
    _write_tsv(
        info,
        ["clip_id", "title", "artist", "album", "url"],
        [
            ["1", "Aria", "Bach Soloists", "Cantatas", "u"],
            ["2", "Fugue", "Bach Soloists", "Cantatas", "u"],
            ["3", "Riff", "Some Band", "Loud", "u"],
            ["4", "Silence", "Some Band", "Loud", "u"],
        ],
    )
    return ann, info


def test_synonyms_merge_into_canonical_tag(corpus):
    ann, info = corpus
    vocab, records = mtat_data.build_dataset(ann, info, n_tags=10)
    assert "clasical" not in vocab, "misspelling must merge away"
    assert "classical" in vocab
    by_id = {r["clip_id"]: r for r in records}
    # clip 2 was tagged only 'clasical' and must still count as classical
    assert by_id["2"]["labels"] == {"classical"}


def test_untagged_clip_is_dropped(corpus):
    ann, info = corpus
    _, records = mtat_data.build_dataset(ann, info, n_tags=10)
    assert "4" not in {r["clip_id"] for r in records}


def test_vocabulary_is_capped_and_frequency_ordered(corpus):
    ann, info = corpus
    vocab, _ = mtat_data.build_dataset(ann, info, n_tags=1)
    # classical (2 clips after merge) outranks guitar (1 clip)
    assert vocab == ["classical"]


def test_text_excludes_artist_by_default(corpus):
    ann, info = corpus
    _, records = mtat_data.build_dataset(ann, info, n_tags=10)
    text = next(r["text"] for r in records if r["clip_id"] == "1")
    assert "Bach Soloists" not in text, "artist leaks identity into the input"
    assert "Aria" in text and "Cantatas" in text


def test_text_includes_artist_when_requested(corpus):
    ann, info = corpus
    _, records = mtat_data.build_dataset(ann, info, n_tags=10, use_artist=True)
    text = next(r["text"] for r in records if r["clip_id"] == "1")
    assert "Bach Soloists" in text


def test_labels_to_vector_is_multi_hot():
    vocab = ["a", "b", "c"]
    assert mtat_data.labels_to_vector({"a", "c"}, vocab) == [1, 0, 1]
    assert mtat_data.labels_to_vector(set(), vocab) == [0, 0, 0]
    assert mtat_data.labels_to_vector({"zzz"}, vocab) == [0, 0, 0]


def test_artist_grouped_split_has_no_leakage(corpus):
    ann, info = corpus
    _, records = mtat_data.build_dataset(ann, info, n_tags=10)
    train, val, test = artist_grouped_split(records, train_frac=0.5, val_frac=0.0)
    overlap = mtat_data.artist_overlap(
        {"train": train, "val": val, "test": test}
    )
    assert overlap == {"train_test": 0, "train_val": 0, "val_test": 0}
