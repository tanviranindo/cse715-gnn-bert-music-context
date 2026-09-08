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
    out = {
        "dataset": "fma_small",
        "genres": d["genres"],
        "split_sizes": d["split_sizes"],
        "artist_leakage": d["artist_leakage"],
        "chance_accuracy": 1.0 / len(d["genres"]),
        "graph_input": d.get("graph", "segment"),
        "models": models,
    }

    # The spec (S3.3) names two graph structures. Both are trained on identical
    # splits and labels so the comparison is like-for-like.
    coh = read("metrics_graph_coherence.json")
    if coh:
        out["graph_coherence"] = coh

    chord = read("metrics_task2_chord.json")
    if chord:
        out["chord_graph_input"] = {
            "note": (
                "Chord-transition graph as the GNN input, same splits/labels as "
                "the segment-similarity graphs above. Nodes are chords from a "
                "24-triad vocabulary, features are the chord's pitch-class "
                "template, so all timbre is discarded."
            ),
            "artist_leakage": chord["artist_leakage"],
            "models": {
                name: {
                    "params": run["params"],
                    "test_accuracy": run["test_accuracy"],
                    "test_macro_f1": run["test_macro_f1"],
                }
                for name, run in chord["runs"].items()
            },
        }
    return out


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

    def _multitask(run, note):
        return {
            "note": note,
            "tags": {
                "test_macro_f1": run["test_macro_f1"],
                "test_micro_f1": run["test_micro_f1"],
                "test_auc_pr": run["test_auc_pr"],
            },
            "valence": run["valence"],
            "arousal": run["arousal"],
        }

    # The re-run under best-validation checkpointing is the headline; the
    # original final-epoch run is kept beside it because the report's earlier
    # draft quoted it and a reader may be holding that version.
    # A control run: the same architecture on MusicCaps, where the text is a
    # real description rather than track metadata. Kept separate from the
    # ablation because the corpus and label set differ.
    mc = RESULTS / "_musiccaps" / "metrics_task3.json"
    if mc.exists():
        run = json.loads(mc.read_text())["runs"]["crossattn"]
        out["text_control_musiccaps"] = {
            "note": ("Same cross-attention fusion trained on MusicCaps captions "
                     "instead of MagnaTagATune metadata. The F1 is not "
                     "comparable with the ablation above (different corpus and "
                     "labels, and MusicCaps aspects leak into their captions); "
                     "the attention statistics are."),
            "split_sizes": json.loads(mc.read_text())["split_sizes"],
            "test_macro_f1": run["test_macro_f1"],
            "test_micro_f1": run["test_micro_f1"],
            "test_auc_pr": run["test_auc_pr"],
            "attention": run.get("attention"),
        }

    rerun = RESULTS / "_multitask" / "metrics_task3.json"
    if rerun.exists():
        out["multitask_deam"] = _multitask(
            json.loads(rerun.read_text())["runs"]["crossattn"],
            "best-validation checkpointing, same policy as the ablation above; "
            "directly comparable with the crossattn arm",
        )
    mt = read("metrics_task3_multitask.json")
    if mt:
        key = "multitask_deam_superseded_final_epoch" if "multitask_deam" in out \
            else "multitask_deam"
        out[key] = _multitask(
            mt["runs"]["crossattn"],
            "final-epoch scoring; superseded by the best-val re-run and kept "
            "only for continuity with the first report draft",
        )
    return out


def _t4_run(d):
    return {
        "best_epoch": d["best_epoch"],
        "caption_to_audio": d["caption_to_audio"],
        "audio_to_caption": d["audio_to_caption"],
        "median_rank": {
            "caption_to_audio": d["median_rank_caption_to_audio"],
            "audio_to_caption": d["median_rank_audio_to_caption"],
        },
    }


def task4():
    d = read("metrics_task4.json")
    if not d:
        return None
    out = {
        "dataset": "musiccaps",
        "split_kind": d["split_kind"],
        "split_sizes": d["split_sizes"],
        "gallery_size": d["gallery_size"],
        "random_baseline": {
            "R@1": d["random_baseline_R@1"],
            "R@10": d["random_baseline_R@10"],
        },
    }

    # Task 3 found that the graph tower needs its own learning rate. Testing
    # whether that transfers is the headline run; the shared-lr run stays as
    # the arm it is compared against.
    runs = {"shared_lr": _t4_run(d)}
    gnnlr = read("metrics_task4_gnnlr.json")
    if gnnlr:
        runs["gnn_lr_1e-3"] = _t4_run(gnnlr)
        out["headline"] = "gnn_lr_1e-3"
    else:
        out["headline"] = "shared_lr"
    out["runs"] = runs

    head = runs[out["headline"]]
    out["caption_to_audio"] = head["caption_to_audio"]
    out["audio_to_caption"] = head["audio_to_caption"]
    out["median_rank"] = head["median_rank"]
    out["best_epoch"] = head["best_epoch"]

    scale = read("metrics_task4_scale.json")
    if scale:
        out["data_scale_ablation"] = {
            "note": scale["note"],
            "points": [
                {"fraction": pt["fraction"], "n_train": pt["n_train"],
                 "c2a_R@10": pt["c2a"]["R@10"], "a2c_R@10": pt["a2c"]["R@10"]}
                for pt in scale["points"]
            ],
            "log_linear_fit_R@10_vs_ln_n": scale["log_linear_fit_R@10_vs_ln_n"],
            "extrapolation": scale["extrapolation"],
            "saturation_check": scale["saturation_check"],
        }

    zs = read("metrics_task4_zeroshot.json")
    if zs:
        out["zero_shot_tagging"] = zs

    # The supervised counterpart the specification asks zero-shot to be
    # compared against, trained on the same corpus, the same official eval
    # split and the same train-derived vocabulary.
    cmp_path = RESULTS / "_supervised_cmp" / "metrics_task3.json"
    if cmp_path.exists():
        doc = json.loads(cmp_path.read_text())
        run = doc["runs"]["crossattn"]
        out["supervised_comparison"] = {
            "note": ("Task 3 cross-attention fusion on MusicCaps under the "
                     "official AudioSet eval split, scored on the same 50 "
                     "train-derived tags as the zero-shot run, so the two are "
                     "directly comparable."),
            "split_sizes": doc["split_sizes"],
            "n_tags": doc["n_tags"],
            "test_micro_f1": run["test_micro_f1"],
            "test_macro_f1": run["test_macro_f1"],
            "test_auc_pr": run["test_auc_pr"],
        }

    human = read("metrics_task4_human.json")
    if human:
        out["human_evaluation"] = human
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
