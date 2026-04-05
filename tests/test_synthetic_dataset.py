from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_synthetic_dataset_invariants(repo_root: Path) -> None:
    graph_path = repo_root / "data" / "synthetic_test_graph.csv"
    metadata_path = repo_root / "data" / "synthetic_test_metadata.csv"

    assert graph_path.exists(), f"Missing graph file: {graph_path}"
    assert metadata_path.exists(), f"Missing metadata file: {metadata_path}"

    graph_df = pd.read_csv(graph_path)
    metadata_df = pd.read_csv(metadata_path)

    assert len(graph_df) == 500
    assert graph_df["edge"].nunique() == 5
    assert metadata_df["type"].nunique() == 3

    graph_nodes = set(graph_df["head"]).union(set(graph_df["tail"]))
    metadata_nodes = set(metadata_df["id"])

    assert graph_nodes.issubset(metadata_nodes)
    assert not graph_df[["head", "tail", "edge"]].isna().any().any()
    assert not metadata_df[["id", "type"]].isna().any().any()
