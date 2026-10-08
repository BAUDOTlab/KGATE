"""
Shared fixtures for the KGATE test suite.

All fixtures build small, deterministic mock knowledge graphs so that
tests run fast and without out-of-memory issues.
"""

import pandas as pd
import pytest
import torch
import torch.nn as nn

from kgate.config import Configuration
from kgate.knowledgegraph import KnowledgeGraph


def make_kg_dataframe() -> pd.DataFrame:
    """Single-node-type KG: 16 triplets, 8 nodes, 2 edge types."""
    rows = []
    for i in range(8):
        rows.append({"head": f"n{i}", "tail": f"n{(i + 1) % 8}", "edge": "E1"})
        rows.append({"head": f"n{i}", "tail": f"n{(i + 3) % 8}", "edge": "E2"})
    return pd.DataFrame(rows)


def make_hetero_dataframe() -> pd.DataFrame:
    """Two node types: n0-n3 ('A'), n4-n7 ('B'), 8 triplets, 2 edge types."""
    rows = []
    for i in range(4):
        rows.append({"head": f"n{i}", "tail": f"n{4 + i}", "edge": "E1"})
        rows.append({"head": f"n{4 + i}", "tail": f"n{i}", "edge": "E2"})
    return pd.DataFrame(rows)


def make_hetero_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [f"n{i}" for i in range(8)], "type": ["A"] * 4 + ["B"] * 4}
    )


def add_embeddings(kg: KnowledgeGraph, node_dim: int = 4, edge_dim: int = 4):
    """Attach random embeddings (one per node type, one per edge type)."""
    node_embeddings = nn.ParameterList(
        nn.Parameter(torch.randn(len(ids), node_dim))
        for ids in kg.node_type_to_global.values()
    )
    edge_embeddings = nn.Parameter(torch.randn(kg.edge_count, edge_dim))
    kg.embeddings = node_embeddings, edge_embeddings
    return node_embeddings, edge_embeddings


@pytest.fixture
def kg_dataframe() -> pd.DataFrame:
    return make_kg_dataframe()


@pytest.fixture
def kg(kg_dataframe: pd.DataFrame) -> KnowledgeGraph:
    return KnowledgeGraph(dataframe=kg_dataframe)


@pytest.fixture
def embedded_kg(kg: KnowledgeGraph) -> KnowledgeGraph:
    """Small KG with masks and random embeddings already set."""
    kg.generate_masks(split_proportions=(0.5, 0.25, 0.25), sizes=(8, 4, 4))
    add_embeddings(kg)
    return kg


@pytest.fixture
def hetero_kg() -> KnowledgeGraph:
    return KnowledgeGraph(
        dataframe=make_hetero_dataframe(), metadata=make_hetero_metadata()
    )


@pytest.fixture
def small_config(tmp_path) -> Configuration:
    """Configuration with tiny dimensions and a temp output directory."""
    return Configuration(
        config_dict={
            "seed": 42,
            "output_directory": str(tmp_path / "out"),
            "preprocessing": {"run_preprocessing": False},
            "model": {
                "node_embedding_dimensions": 4,
                "edge_embedding_dimensions": 4,
                "decoder": {"name": "TransE", "dissimilarity": "L2"},
                "loss": {"name": "Margin", "margin": 1.0},
            },
            "training": {"batch_size": 4, "evaluation_batch_size": 4},
        }
    )
