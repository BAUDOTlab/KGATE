from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd
import pytest

from kgate.preprocessing import (
    NODE_CLASS_COLUMN,
    NODE_CLASS_EDGE_NAME,
    prepare_knowledge_graph,
)
from tests.make_synthetic_test_graph import make_nodes, sample_edges


TC_CONFIGS = [
    "configs/examples/tc_default_transe.toml",
    "configs/examples/tc_gat_transe.toml",
    "configs/examples/tc_gcn_distmult.toml",
]


def _load_tc_config(repo_root: Path, base_config: str, out_dir: Path) -> dict:
    with open(repo_root / base_config, "rb") as fh:
        config = tomllib.load(fh)

    out_dir.mkdir(parents=True, exist_ok=True)
    config["output_directory"] = str(out_dir)
    config["kg_pkl"] = ""

    # Keep preprocessing focused on verifying triplet injection behavior.
    config["preprocessing"]["remove_duplicate_triples"] = False
    config["preprocessing"]["flag_near_duplicate_edges"] = False
    config["preprocessing"]["make_directed"] = False
    config["preprocessing"]["make_directed_edges"] = []
    config["preprocessing"]["clean_train_set"] = False

    return config


def _concat_triples(kg_train, kg_validation, kg_test) -> pd.DataFrame:
    return pd.concat(
        [
            kg_train.get_dataframe(),
            kg_validation.get_dataframe(),
            kg_test.get_dataframe(),
        ],
        ignore_index=True,
    )


def _expected_node_class_pairs(
    metadata: pd.DataFrame, graph_df: pd.DataFrame
) -> set[tuple[str, str]]:
    nodes_in_graph = set(graph_df["head"]).union(graph_df["tail"])
    rows = metadata.loc[
        metadata["id"].isin(nodes_in_graph) & metadata[NODE_CLASS_COLUMN].notna(),
        ["id", NODE_CLASS_COLUMN],
    ]

    expected_pairs: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        for cls in str(row[NODE_CLASS_COLUMN]).split("|"):
            cls = cls.strip()
            if cls:
                expected_pairs.add((row["id"], cls))

    return expected_pairs


def _preprocess(
    add_node_class: bool, repo_root: Path, base_config: str, tmp_path: Path
) -> pd.DataFrame:
    metadata = make_nodes(add_node_class=add_node_class)
    dataframe = sample_edges(metadata)
    config = _load_tc_config(repo_root, base_config, tmp_path / Path(base_config).stem)

    kg_train, kg_validation, kg_test = prepare_knowledge_graph(
        config=config,
        dataframe=dataframe,
        metadata=metadata,
    )
    all_triples = _concat_triples(kg_train, kg_validation, kg_test)
    return all_triples, kg_train, metadata, dataframe


@pytest.mark.parametrize("base_config", TC_CONFIGS)
def test_node_class_triple_injection(
    repo_root: Path, tmp_path: Path, base_config: str
) -> None:
    """Test that node_class triples are injected when node_class column is present in metadata."""
    all_triples, kg_train, metadata, dataframe = _preprocess(
        add_node_class=True,
        repo_root=repo_root,
        base_config=base_config,
        tmp_path=tmp_path,
    )

    expected_pairs = _expected_node_class_pairs(metadata, dataframe)
    node_class_triples = all_triples.loc[
        all_triples["edge"] == NODE_CLASS_EDGE_NAME, ["head", "tail"]
    ]
    actual_pairs = set(zip(node_class_triples["head"], node_class_triples["tail"]))

    assert expected_pairs
    assert expected_pairs == actual_pairs

    # For each node with multiple classes, check that all class triples exist
    nodes_with_multiclass = metadata[metadata[NODE_CLASS_COLUMN].notna()]
    nodes_with_multiclass = nodes_with_multiclass[
        nodes_with_multiclass[NODE_CLASS_COLUMN].str.contains("\|")
    ]
    for _, row in nodes_with_multiclass.iterrows():
        node_id = row["id"]
        classes = [
            cls.strip() for cls in str(row[NODE_CLASS_COLUMN]).split("|") if cls.strip()
        ]
        for cls in classes:
            assert (
                node_id,
                cls,
            ) in actual_pairs, f"Missing triple for node {node_id} and class {cls}"

    # Injected class labels should become known nodes in the resulting graph mapping.
    class_nodes = {tail for _, tail in expected_pairs}
    assert class_nodes.issubset(set(kg_train.node_to_index))


@pytest.mark.parametrize("base_config", TC_CONFIGS)
def test_triplet_classification_without_class_triple_injection(
    repo_root: Path, tmp_path: Path, base_config: str
) -> None:
    """Test that no node_class triples are injected when node_class column is absent in metadata."""
    all_triples, kg_train, _, _ = _preprocess(
        add_node_class=False,
        repo_root=repo_root,
        base_config=base_config,
        tmp_path=tmp_path,
    )

    assert not (all_triples["edge"] == NODE_CLASS_EDGE_NAME).any()
    assert not any(node_id.startswith("Class") for node_id in kg_train.node_to_index)


if __name__ == "__main__":
    pytest.main([__file__])
