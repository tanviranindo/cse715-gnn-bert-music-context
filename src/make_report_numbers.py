"""Regenerate report data products from `results/metrics.json`.

The report used to carry a hand-curated copy of the numbers, which is how it
came to state that the DEAM re-run was impossible while the re-run sat in
`results/_multitask/`. This makes that copy a build product: run
`aggregate_metrics.py` then this, and the report's reference numbers cannot be
older than the runs.

    python src/make_report_numbers.py [results_dir] [report_dir]
"""

import json
import pathlib
import sys

try:
    from src.make_report_tex import generate as generate_tex
except ModuleNotFoundError:  # documented direct-script invocation
    from make_report_tex import generate as generate_tex

RESULTS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
REPORT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "report")


def main():
    m = json.loads((RESULTS / "metrics.json").read_text())
    t1, t2 = m["task1_bert_tags"], m["task2_gnn_structure"]
    t3, t4 = m["task3_gnn_bert_fusion"], m["task4_contrastive_retrieval"]

    doc = {
        "_generated_from": "results/metrics.json via src/make_report_numbers.py",
        "_warning": "build product - do not edit by hand; run the two scripts instead",
        "task1": {
            key: {
                "micro": v["test"]["micro_f1"],
                "macro": v["test"]["macro_f1"],
                "splits": v["split_sizes"],
                "selected_epoch": v["selected_epoch"],
                "baselines": v["baselines"],
            }
            for key, v in t1.items()
        },
        "task2": {
            "splits": t2["split_sizes"],
            "artist_leakage": t2["artist_leakage"],
            "chance": t2["chance_accuracy"],
            "segment_graph": t2["models"],
            "chord_graph": t2.get("chord_graph_input", {}).get("models"),
        },
        "task3": {
            "splits": t3["split_sizes"],
            "ablation": t3["ablation"],
            "seed_sweep": t3["seed_sweep"]["metrics"],
            "multitask_deam": t3.get("multitask_deam"),
            "multitask_deam_superseded": t3.get("multitask_deam_superseded_final_epoch"),
        },
        "task4": {
            "split_kind": t4["split_kind"],
            "splits": t4["split_sizes"],
            "gallery": t4["gallery_size"],
            "headline": t4["headline"],
            "runs": t4["runs"],
            "random": t4["random_baseline"],
            "data_scale": t4.get("data_scale_ablation"),
            "zeroshot": t4.get("zero_shot_tagging"),
            "human_evaluation": t4.get("human_evaluation"),
        },
    }
    out = REPORT / "_numbers.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote %s (%d bytes)" % (out, out.stat().st_size))
    tex = REPORT / "generated_metrics.tex"
    generate_tex(RESULTS / "metrics.json", tex)
    print("wrote %s (%d bytes)" % (tex, tex.stat().st_size))


if __name__ == "__main__":
    main()
