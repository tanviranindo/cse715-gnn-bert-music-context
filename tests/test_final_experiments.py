import os
from pathlib import Path
import subprocess


def test_dry_run_caps_graph_workers_and_runs_full_zeroshot_last(tmp_path):
    environment = dict(os.environ)
    environment["EXPERIMENT_SOURCE_REVISION"] = "a" * 40
    environment["PYTHON_BIN"] = "python"
    completed = subprocess.run(
        [
            "bash",
            "infra/final_experiments.sh",
            str(tmp_path / "data"),
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    graph_lines = [line for line in completed.stdout.splitlines() if "build_graphs.py" in line]
    assert len(graph_lines) == 3
    assert all("--workers 12" in line for line in graph_lines)
    assert completed.stdout.index("task4_scale_075") < completed.stdout.index("task4_gnnlr")
