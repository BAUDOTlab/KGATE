from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd
import pytest
import tomli_w


import subprocess
from kgate import Architect


# Always (re)generate the synthetic dataset before running E2E tests
def _generate_synthetic_dataset(repo_root: Path) -> None:
    script = repo_root / "tests" / "make_synthetic_test_graph.py"
    result = subprocess.run(
        ["python", str(script)], cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Synthetic dataset generation failed: {result.stderr}")


E2E_CONFIGS = [
    "configs/examples/lp_default_transe.toml",
    "configs/examples/lp_gat_transe.toml",
    "configs/examples/lp_gcn_distmult.toml",
    "configs/examples/tc_default_transe.toml",
    "configs/examples/tc_gat_transe.toml",
    "configs/examples/tc_gcn_distmult.toml",
]


@pytest.mark.e2e
@pytest.mark.parametrize("base_config", E2E_CONFIGS)
def test_end_to_end_pipeline_creates_artifacts(
    repo_root: Path, tmp_path: Path, base_config: str
) -> None:
    """Test that the end-to-end pipeline runs and creates all expected artifacts for a given config."""
    # Always generate the synthetic dataset before running the E2E test
    _generate_synthetic_dataset(repo_root)

    config_path = repo_root / base_config
    assert config_path.exists(), f"Missing e2e config: {config_path}"

    with open(config_path, "rb") as fh:
        config = tomllib.load(fh)

    scenario_name = config_path.stem
    scenario_output = tmp_path / scenario_name

    config["output_directory"] = str(scenario_output)

    tmp_config_path = tmp_path / f"{scenario_name}.toml"
    with open(tmp_config_path, "wb") as fh:
        tomli_w.dump(config, fh)

    metadata_df = pd.read_csv(repo_root / config["metadata_csv"])
    if config["model"]["encoder"]["name"] == "Default":
        metadata_df["type"] = "entity"

    model = Architect(config_path=str(tmp_config_path), metadata=metadata_df)
    model.train_model()
    results = model.test()

    metrics_file = scenario_output / "evaluation_metrics.yaml"
    training_file = scenario_output / "training_metrics.csv"
    checkpoints_dir = scenario_output / "checkpoints"

    assert isinstance(results, dict)
    assert "Global_metrics" in results
    assert metrics_file.exists() and metrics_file.stat().st_size > 0
    assert training_file.exists() and training_file.stat().st_size > 0
    assert checkpoints_dir.exists()
    assert any(
        checkpoints_dir.glob("best_model_checkpoint_validation_metric_value=*.pt")
    )
