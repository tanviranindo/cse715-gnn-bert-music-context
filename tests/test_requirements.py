from pathlib import Path


def test_musiccaps_parquet_runtime_is_declared():
    requirements = Path("requirements.txt").read_text().lower().splitlines()
    assert any(line.startswith("pyarrow") for line in requirements)
