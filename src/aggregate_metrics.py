"""Collect every headline number into the single `results/metrics.json` the
specification names as a submission artifact.

The per-task files stay the source of truth; this only lifts the test-set
numbers out of them so a reader (or a grader) has one file to open. Nothing is
typed by hand, so the aggregate cannot drift from the runs.

    python src/aggregate_metrics.py [results_dir]
"""

import json
import pathlib
import sys

import aggregate_sweep

RESULTS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def read(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def task1():
    variants = {
        "mtat_metadata_text": "metrics_task1_mtat.json",
        "musiccaps_caption": "metrics_task1_musiccaps.json",
        "musiccaps_caption_stripped": "metrics_task1_musiccaps_stripped.json",
    }
    out = {}
    for label, fname in variants.items():
        d = read(fname)
        if not d:
            continue
        out[label] = {
            "model": d["config"]["model"],
            "n_tags": d["n_tags"],
            "split_sizes": d["split_sizes"],
            "selected_epoch": d["selected_epoch"],
            "test": d["test"],
            "baselines": d["baselines"],
        }
    return out


def task2():
    d = read("metrics_task2.json")
    if not d:
        return None
    models = {
        name: {
            "params": run["params"],
            "test_accuracy": run["test_accuracy"],
            "test_macro_f1": run["test_macro_f1"],
        }
        for name, run in d["runs"].items()
    }
    b4 = read("metrics_task2_b4.json")
    if b4:
        models["b4_pca_mlp"] = {
            "params": b4["params"],
            "test_accuracy": b4["test_accuracy"],
            "test_macro_f1": b4["test_macro_f1"],
        }
    return {
        "dataset": "fma_small",
        "genres": d["genres"],
        "split_sizes": d["split_sizes"],
        "artist_leakage": d["artist_leakage"],
        "chance_accuracy": 1.0 / len(d["genres"]),
        "models": models,
    }


def _t3_run(run):
    return {
        "params": run["params"],
        "test_macro_f1": run["test_macro_f1"],
        "test_micro_f1": run["test_micro_f1"],
        "test_auc_pr": run["test_auc_pr"],
        "attention": run.get("attention"),
    }


def task3():
    d = read("metrics_task3.json")
    if not d:
        return None
    ablation = {name: _t3_run(run) for name, run in d["runs"].items()}
    for label, sub in (("crossattn_freeze_bert", "_freeze"), ("crossattn_gnn_lr_1e-3", "_gnnlr")):
        p = RESULTS / sub / "metrics_task3.json"
        if p.exists():
            ablation[label] = _t3_run(json.loads(p.read_text())["runs"]["crossattn"])

    out = {
        "dataset": "magnatagatune_top50",
        "split_sizes": d["split_sizes"],
        "ablation": ablation,
        "seed_sweep": aggregate_sweep.summarize(RESULTS / "_sweep"),
    }

    mt = read("metrics_task3_multitask.json")
    if mt:
        run = mt["runs"]["crossattn"]
        out["multitask_deam"] = {
            "note": "final-epoch scoring; not re-run under the best-val checkpoint policy",
            "tags": {
                "test_macro_f1": run["test_macro_f1"],
                "test_micro_f1": run["test_micro_f1"],
                "test_auc_pr": run["test_auc_pr"],
            },
            "valence": run["valence"],
            "arousal": run["arousal"],
        }
    return out


def task4():
    d = read("metrics_task4.json")
    if not d:
        return None
    out = {
        "dataset": "musiccaps",
        "split_kind": d["split_kind"],
        "split_sizes": d["split_sizes"],
        "gallery_size": d["gallery_size"],
        "best_epoch": d["best_epoch"],
        "caption_to_audio": d["caption_to_audio"],
        "audio_to_caption": d["audio_to_caption"],
        "median_rank": {
            "caption_to_audio": d["median_rank_caption_to_audio"],
            "audio_to_caption": d["median_rank_audio_to_caption"],
        },
        "random_baseline": {
            "R@1": d["random_baseline_R@1"],
            "R@10": d["random_baseline_R@10"],
        },
    }
    zs = read("metrics_task4_zeroshot.json")
    if zs:
        out["zero_shot_tagging"] = zs
    return out


def main():
    doc = {
        "project": "GNN-BERT Music Context Understanding (CSE715)",
        "generated_by": "src/aggregate_metrics.py",
        "note": (
            "Headline test-set numbers for all four tasks. Per-task files in this "
            "directory hold the full histories, per-class breakdowns and configs."
        ),
        "task1_bert_tags": task1(),
        "task2_gnn_structure": task2(),
        "task3_gnn_bert_fusion": task3(),
        "task4_contrastive_retrieval": task4(),
    }
    out = RESULTS / "metrics.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote %s (%d bytes)" % (out, out.stat().st_size))


if __name__ == "__main__":
    main()
