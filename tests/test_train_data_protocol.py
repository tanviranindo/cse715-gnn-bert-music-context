"""Task 1 must choose its evaluated labels without seeing held-out labels."""

import csv
from types import SimpleNamespace

from src import train


def _args(data_dir, dataset):
    return SimpleNamespace(
        data_dir=str(data_dir),
        dataset=dataset,
        n_tags=1,
        seed=42,
        use_artist=False,
        strip_leakage=False,
    )


def test_musiccaps_test_frequency_cannot_select_a_vocabulary_label(tmp_path):
    root = tmp_path / "musiccaps"
    root.mkdir()
    path = root / "musiccaps-public.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ytid", "caption", "aspect_list", "is_audioset_eval"])
        # 130 leaves 111 training rows after the fixed 15% validation split,
        # still above MusicCaps' production min_count=100 threshold.
        for i in range(130):
            writer.writerow([f"train-{i}", "training caption", "['train_only']", "False"])
        for i in range(140):
            writer.writerow([f"test-{i}", "held out caption", "['test_only']", "True"])

    vocab, splits = train.build_records(_args(tmp_path, "musiccaps"))

    assert vocab == ["train_only"]
    assert "test_only" not in vocab
    assert splits["train"]
    assert all(record["labels"] <= set(vocab) for part in splits.values() for record in part)


def test_mtat_held_out_artist_cannot_select_a_vocabulary_label(tmp_path):
    root = tmp_path / "magnatagatune"
    root.mkdir()
    annotations = root / "annotations_final.csv"
    clip_info = root / "clip_info_final.csv"

    rows = []
    info_rows = []
    clip_id = 0
    for artist_id in range(10):
        repeats = 20 if artist_id == 1 else 1
        for _ in range(repeats):
            clip_id += 1
            held_out = artist_id == 1
            rows.append([str(clip_id), "0" if held_out else "1", "1" if held_out else "0", f"{clip_id}.mp3"])
            info_rows.append([str(clip_id), f"title {clip_id}", f"artist_{artist_id}", "album", "url"])

    with annotations.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["clip_id", "train_only", "test_only", "mp3_path"])
        writer.writerows(rows)
    with clip_info.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["clip_id", "title", "artist", "album", "url"])
        writer.writerows(info_rows)

    vocab, splits = train.build_records(_args(tmp_path, "mtat"))

    assert vocab == ["train_only"]
    assert "test_only" not in vocab
    assert {record["artist"] for record in splits["train"]}.isdisjoint(
        {record["artist"] for record in splits["test"]}
    )
