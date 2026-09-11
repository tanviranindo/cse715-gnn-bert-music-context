import json

from src import analyze_scale


def test_load_points_consumes_training_commands_canonical_outputs(tmp_path):
    for suffix, fraction in (("frac025", 0.25), ("frac050", 0.5), ("frac075", 0.75), ("gnnlr", 1.0)):
        document = {
            "split_sizes": {"train": int(100 * fraction)},
            "gallery_size": 20,
            "caption_to_audio": {"R@1": 0.1, "R@5": 0.2, "R@10": fraction / 10},
            "audio_to_caption": {"R@1": 0.1, "R@5": 0.2, "R@10": fraction / 10},
            "random_baseline_R@10": 0.5,
        }
        (tmp_path / f"metrics_task4_{suffix}.json").write_text(json.dumps(document))

    points = analyze_scale.load_points(tmp_path)
    assert [point["fraction"] for point in points] == [0.25, 0.5, 0.75, 1.0]
    assert points[-1]["n_train"] == 100
