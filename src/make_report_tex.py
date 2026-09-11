"""Generate TeX-safe report macros from the canonical aggregate metrics."""

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


FIELDS = {
    "TaskOneMtatMicro": (("task1_bert_tags", "mtat_metadata_text", "test", "micro_f1"), "f3"),
    "TaskOneMtatMacro": (("task1_bert_tags", "mtat_metadata_text", "test", "macro_f1"), "f3"),
    "TaskOneMtatRandomMicro": (("task1_bert_tags", "mtat_metadata_text", "baselines", "B1_random_prevalence", "micro_f1"), "f3"),
    "TaskOneMtatRandomMacro": (("task1_bert_tags", "mtat_metadata_text", "baselines", "B1_random_prevalence", "macro_f1"), "f3"),
    "TaskOneMusicCapsMicro": (("task1_bert_tags", "musiccaps_caption", "test", "micro_f1"), "f3"),
    "TaskOneMusicCapsMacro": (("task1_bert_tags", "musiccaps_caption", "test", "macro_f1"), "f3"),
    "TaskOneMusicCapsLexicalMicro": (("task1_bert_tags", "musiccaps_caption", "baselines", "B5_lexical_match", "micro_f1"), "f3"),
    "TaskOneMusicCapsLexicalMacro": (("task1_bert_tags", "musiccaps_caption", "baselines", "B5_lexical_match", "macro_f1"), "f3"),
    "TaskOneStrippedMicro": (("task1_bert_tags", "musiccaps_caption_stripped", "test", "micro_f1"), "f3"),
    "TaskOneStrippedMacro": (("task1_bert_tags", "musiccaps_caption_stripped", "test", "macro_f1"), "f3"),
    "TaskTwoChance": (("task2_gnn_structure", "chance_accuracy"), "f3"),
    "TaskTwoBFourParams": (("task2_gnn_structure", "models", "b4_pca_mlp", "params"), "integer"),
    "TaskTwoBFourAcc": (("task2_gnn_structure", "models", "b4_pca_mlp", "test_accuracy"), "f3"),
    "TaskTwoBFourMacro": (("task2_gnn_structure", "models", "b4_pca_mlp", "test_macro_f1"), "f3"),
    "TaskTwoSageParams": (("task2_gnn_structure", "models", "sage", "params"), "integer"),
    "TaskTwoSageAcc": (("task2_gnn_structure", "models", "sage", "test_accuracy"), "f3"),
    "TaskTwoSageMacro": (("task2_gnn_structure", "models", "sage", "test_macro_f1"), "f3"),
    "TaskTwoGatParams": (("task2_gnn_structure", "models", "gat", "params"), "integer"),
    "TaskTwoGatAcc": (("task2_gnn_structure", "models", "gat", "test_accuracy"), "f3"),
    "TaskTwoGatMacro": (("task2_gnn_structure", "models", "gat", "test_macro_f1"), "f3"),
    "TaskTwoCnnParams": (("task2_gnn_structure", "models", "cnn", "params"), "integer"),
    "TaskTwoCnnAcc": (("task2_gnn_structure", "models", "cnn", "test_accuracy"), "f3"),
    "TaskTwoCnnMacro": (("task2_gnn_structure", "models", "cnn", "test_macro_f1"), "f3"),
    "TaskTwoChordSageAcc": (("task2_gnn_structure", "chord_graph_input", "models", "sage", "test_accuracy"), "f3"),
    "TaskTwoChordSageMacro": (("task2_gnn_structure", "chord_graph_input", "models", "sage", "test_macro_f1"), "f3"),
    "TaskTwoChordGatAcc": (("task2_gnn_structure", "chord_graph_input", "models", "gat", "test_accuracy"), "f3"),
    "TaskTwoChordGatMacro": (("task2_gnn_structure", "chord_graph_input", "models", "gat", "test_macro_f1"), "f3"),
    "TaskThreeBertMacro": (("task3_gnn_bert_fusion", "ablation", "bert", "test_macro_f1"), "f3"),
    "TaskThreeBertMicro": (("task3_gnn_bert_fusion", "ablation", "bert", "test_micro_f1"), "f3"),
    "TaskThreeBertAuc": (("task3_gnn_bert_fusion", "ablation", "bert", "test_auc_pr"), "f3"),
    "TaskThreeGnnMacro": (("task3_gnn_bert_fusion", "ablation", "gnn", "test_macro_f1"), "f3"),
    "TaskThreeGnnMicro": (("task3_gnn_bert_fusion", "ablation", "gnn", "test_micro_f1"), "f3"),
    "TaskThreeGnnAuc": (("task3_gnn_bert_fusion", "ablation", "gnn", "test_auc_pr"), "f3"),
    "TaskThreeConcatMacro": (("task3_gnn_bert_fusion", "ablation", "concat", "test_macro_f1"), "f3"),
    "TaskThreeConcatMicro": (("task3_gnn_bert_fusion", "ablation", "concat", "test_micro_f1"), "f3"),
    "TaskThreeConcatAuc": (("task3_gnn_bert_fusion", "ablation", "concat", "test_auc_pr"), "f3"),
    "TaskThreeCrossMacro": (("task3_gnn_bert_fusion", "ablation", "crossattn", "test_macro_f1"), "f3"),
    "TaskThreeCrossMicro": (("task3_gnn_bert_fusion", "ablation", "crossattn", "test_micro_f1"), "f3"),
    "TaskThreeCrossAuc": (("task3_gnn_bert_fusion", "ablation", "crossattn", "test_auc_pr"), "f3"),
    "TaskThreeSeparateMacro": (("task3_gnn_bert_fusion", "ablation", "crossattn_gnn_lr_1e-3", "test_macro_f1"), "f3"),
    "TaskThreeSeparateMicro": (("task3_gnn_bert_fusion", "ablation", "crossattn_gnn_lr_1e-3", "test_micro_f1"), "f3"),
    "TaskThreeSeparateAuc": (("task3_gnn_bert_fusion", "ablation", "crossattn_gnn_lr_1e-3", "test_auc_pr"), "f3"),
    "TaskThreeSeedDelta": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "macro_f1", "paired_delta_mean"), "signed3"),
    "TaskThreeSeedDeltaSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "macro_f1", "paired_delta_sd"), "f3"),
    "TaskThreeSeedP": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "macro_f1", "p_value"), "p3"),
    "TaskThreeSeedSharedMacro": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "macro_f1", "shared_lr_mean"), "f3"),
    "TaskThreeSeedSharedMacroSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "macro_f1", "shared_lr_sd"), "f3"),
    "TaskThreeSeedSeparateMacro": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "macro_f1", "gnn_lr_mean"), "f3"),
    "TaskThreeSeedSeparateMacroSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "macro_f1", "gnn_lr_sd"), "f3"),
    "TaskThreeSeedSharedMicro": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "micro_f1", "shared_lr_mean"), "f3"),
    "TaskThreeSeedSharedMicroSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "micro_f1", "shared_lr_sd"), "f3"),
    "TaskThreeSeedSeparateMicro": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "micro_f1", "gnn_lr_mean"), "f3"),
    "TaskThreeSeedSeparateMicroSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "micro_f1", "gnn_lr_sd"), "f3"),
    "TaskThreeSeedMicroDelta": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "micro_f1", "paired_delta_mean"), "signed3"),
    "TaskThreeSeedMicroDeltaSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "micro_f1", "paired_delta_sd"), "f3"),
    "TaskThreeSeedMicroP": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "micro_f1", "p_value"), "p3"),
    "TaskThreeSeedSharedAuc": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "auc_pr", "shared_lr_mean"), "f3"),
    "TaskThreeSeedSharedAucSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "auc_pr", "shared_lr_sd"), "f3"),
    "TaskThreeSeedSeparateAuc": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "auc_pr", "gnn_lr_mean"), "f3"),
    "TaskThreeSeedSeparateAucSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "auc_pr", "gnn_lr_sd"), "f3"),
    "TaskThreeSeedAucDelta": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "auc_pr", "paired_delta_mean"), "signed3"),
    "TaskThreeSeedAucDeltaSd": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "auc_pr", "paired_delta_sd"), "f3"),
    "TaskThreeSeedAucP": (("task3_gnn_bert_fusion", "seed_sweep", "metrics", "auc_pr", "p_value"), "p3"),
    "TaskFourGallery": (("task4_contrastive_retrieval", "gallery_size"), "integer"),
    "TaskFourRandomOne": (("task4_contrastive_retrieval", "random_baseline", "R@1"), "f4"),
    "TaskFourRandomTen": (("task4_contrastive_retrieval", "random_baseline", "R@10"), "f4"),
    "TaskFourSharedCtwoaOne": (("task4_contrastive_retrieval", "runs", "shared_lr", "caption_to_audio", "R@1"), "f4"),
    "TaskFourSharedCtwoaFive": (("task4_contrastive_retrieval", "runs", "shared_lr", "caption_to_audio", "R@5"), "f4"),
    "TaskFourSharedCtwoaTen": (("task4_contrastive_retrieval", "runs", "shared_lr", "caption_to_audio", "R@10"), "f4"),
    "TaskFourSharedAtocOne": (("task4_contrastive_retrieval", "runs", "shared_lr", "audio_to_caption", "R@1"), "f4"),
    "TaskFourSharedAtocFive": (("task4_contrastive_retrieval", "runs", "shared_lr", "audio_to_caption", "R@5"), "f4"),
    "TaskFourSharedAtocTen": (("task4_contrastive_retrieval", "runs", "shared_lr", "audio_to_caption", "R@10"), "f4"),
    "TaskFourSeparateCtwoaOne": (("task4_contrastive_retrieval", "runs", "gnn_lr_1e-3", "caption_to_audio", "R@1"), "f4"),
    "TaskFourSeparateCtwoaFive": (("task4_contrastive_retrieval", "runs", "gnn_lr_1e-3", "caption_to_audio", "R@5"), "f4"),
    "TaskFourSeparateCtwoaTen": (("task4_contrastive_retrieval", "runs", "gnn_lr_1e-3", "caption_to_audio", "R@10"), "f4"),
    "TaskFourSeparateAtocOne": (("task4_contrastive_retrieval", "runs", "gnn_lr_1e-3", "audio_to_caption", "R@1"), "f4"),
    "TaskFourSeparateAtocFive": (("task4_contrastive_retrieval", "runs", "gnn_lr_1e-3", "audio_to_caption", "R@5"), "f4"),
    "TaskFourSeparateAtocTen": (("task4_contrastive_retrieval", "runs", "gnn_lr_1e-3", "audio_to_caption", "R@10"), "f4"),
    "TaskFourRandomOne": (("task4_contrastive_retrieval", "random_baseline", "R@1"), "f4"),
    "TaskFourRandomTen": (("task4_contrastive_retrieval", "random_baseline", "R@10"), "f4"),
    "TaskFourZeroMicro": (("task4_contrastive_retrieval", "zero_shot_tagging", "micro_f1"), "f3"),
    "TaskFourZeroMacro": (("task4_contrastive_retrieval", "zero_shot_tagging", "macro_f1"), "f3"),
    "TaskFourZeroAuc": (("task4_contrastive_retrieval", "zero_shot_tagging", "auc_pr"), "f3"),
    "TaskFourSupervisedMicro": (("task4_contrastive_retrieval", "supervised_comparison", "test_micro_f1"), "f3"),
    "TaskFourSupervisedMacro": (("task4_contrastive_retrieval", "supervised_comparison", "test_macro_f1"), "f3"),
    "TaskFourSupervisedAuc": (("task4_contrastive_retrieval", "supervised_comparison", "test_auc_pr"), "f3"),
    "TaskFourHumanMean": (("task4_contrastive_retrieval", "human_evaluation", "overall_mean"), "f2"),
}


def _lookup(document: dict, path: tuple[str, ...]):
    value = document
    for key in path:
        value = value[key]
    return value


def _format(value, style: str) -> str:
    if style == "integer":
        return f"{int(value):,}"
    if style == "p3" and Decimal(str(value)) < Decimal("0.001"):
        return "<0.001"
    digits = {"f2": 2, "f3": 3, "f4": 4, "signed3": 3, "p3": 3}
    if style in digits:
        places = digits[style]
        quantum = Decimal(1).scaleb(-places)
        rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
        prefix = "+" if style == "signed3" and rounded >= 0 else ""
        return prefix + f"{rounded:.{places}f}"
    raise ValueError(f"unknown report format {style!r}")


def render_macros(values: dict[str, tuple[object, str]]) -> str:
    lines = ["% Generated by src/make_report_tex.py; do not edit by hand."]
    for name, (value, style) in values.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{_format(value, style)}}}")
    return "\n".join(lines) + "\n"


def derived_values(document: dict) -> dict[str, tuple[object, str]]:
    """Return headline quantities derived from canonical scalar metrics."""
    ablation = document["task3_gnn_bert_fusion"]["ablation"]
    separate = Decimal(str(ablation["crossattn_gnn_lr_1e-3"]["test_macro_f1"]))
    bert = Decimal(str(ablation["bert"]["test_macro_f1"]))
    return {"TaskThreeSeparateVsBert": (separate - bert, "signed3")}


def generate(
    metrics_path: str | Path = "results/metrics.json",
    output_path: str | Path = "report/generated_metrics.tex",
    fields=FIELDS,
) -> None:
    document = json.loads(Path(metrics_path).read_text())
    values = {}
    for name, (path, style) in fields.items():
        try:
            values[name] = (_lookup(document, path), style)
        except KeyError as exc:
            raise KeyError(f"required report metric {name} missing at {'.'.join(path)}") from exc
    if fields is FIELDS:
        values.update(derived_values(document))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_macros(values))


if __name__ == "__main__":
    generate()
