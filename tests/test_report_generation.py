import json
from decimal import Decimal
import subprocess
import sys
from pathlib import Path

import pytest

from src import make_report_tex


def test_render_macros_formats_metrics_and_parameter_counts():
    rendered = make_report_tex.render_macros(
        {
            "TaskOneMtatMacro": (0.123456, "f3"),
            "TaskTwoBFourParams": (5256, "integer"),
        }
    )
    assert r"\newcommand{\TaskOneMtatMacro}{0.123}" in rendered
    assert r"\newcommand{\TaskTwoBFourParams}{5,256}" in rendered


def test_render_macros_uses_conventional_half_up_rounding():
    rendered = make_report_tex.render_macros(
        {
            "TaskTwoGatAcc": (0.4325, "f3"),
            "TaskTwoCnnAcc": (0.4825, "f3"),
        }
    )
    assert r"\newcommand{\TaskTwoGatAcc}{0.433}" in rendered
    assert r"\newcommand{\TaskTwoCnnAcc}{0.483}" in rendered


def test_p_value_format_never_reports_zero():
    rendered = make_report_tex.render_macros(
        {"TaskThreeSeedAucP": (0.000071, "p3")}
    )
    assert r"\newcommand{\TaskThreeSeedAucP}{<0.001}" in rendered


def test_generate_changes_when_canonical_metric_changes(tmp_path):
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "generated_metrics.tex"
    document = {
        "task1_bert_tags": {
            "mtat_metadata_text": {
                "test": {"micro_f1": 0.2, "macro_f1": 0.123456}
            }
        }
    }
    metrics.write_text(json.dumps(document))
    make_report_tex.generate(
        metrics,
        output,
        fields={
            "TaskOneMtatMacro": (
                ("task1_bert_tags", "mtat_metadata_text", "test", "macro_f1"),
                "f3",
            )
        },
    )
    first = output.read_bytes()
    document["task1_bert_tags"]["mtat_metadata_text"]["test"]["macro_f1"] = 0.654321
    metrics.write_text(json.dumps(document))
    make_report_tex.generate(
        metrics,
        output,
        fields={
            "TaskOneMtatMacro": (
                ("task1_bert_tags", "mtat_metadata_text", "test", "macro_f1"),
                "f3",
            )
        },
    )
    assert output.read_bytes() != first


def test_generate_rejects_missing_required_metric(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    with pytest.raises(KeyError, match="TaskOneMtatMacro"):
        make_report_tex.generate(
            metrics,
            tmp_path / "out.tex",
            fields={"TaskOneMtatMacro": (("missing",), "f3")},
        )


def test_derived_values_compute_fusion_gain_from_canonical_metrics():
    document = {
        "task3_gnn_bert_fusion": {
            "ablation": {
                "bert": {"test_macro_f1": 0.1714},
                "crossattn_gnn_lr_1e-3": {"test_macro_f1": 0.2461},
            }
        }
    }
    values = make_report_tex.derived_values(document)
    assert values["TaskThreeSeparateVsBert"] == (Decimal("0.0747"), "signed3")


def test_make_report_numbers_documented_script_entrypoint_imports(tmp_path):
    result = subprocess.run(
        [sys.executable, "src/make_report_numbers.py", str(tmp_path), str(tmp_path)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert "ModuleNotFoundError" not in result.stderr
