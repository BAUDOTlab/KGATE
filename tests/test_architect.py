"""
Tests for kgate.architect (Architect class).

Only the construction path is tested here:
- ``train_model`` / ``test`` require a full training setup and checkpoints
  and are not unit-testable.
"""

from pathlib import Path

import pytest
import torch

from kgate.architect import Architect
from kgate.knowledgegraph import KnowledgeGraph

from conftest import add_embeddings


class TestArchitectInit:
    def test_init_with_preprocessed_kg(self, kg, tmp_path):
        architect = Architect(
            knowledge_graph=kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"node_embedding_dimensions": 4},
        )
        assert architect.knowledge_graph is kg
        assert architect.node_embedding_dimensions == 4
        # edge_embedding_dimensions defaults to -1 -> equal to node dimensions
        assert architect.edge_embedding_dimensions == 4
        # The device is picked from torch.cuda availability at runtime.
        assert isinstance(architect.device, torch.device)
        assert architect.device.type in ("cpu", "cuda")
        assert architect.checkpoints_directory == Path(
            tmp_path / "out" / "checkpoints"
        )
        assert (tmp_path / "out").is_dir()
        # Model components are not initialized until train_model()
        assert architect.encoder is None
        assert architect.decoder is None
        assert architect.optimizer is None

    def test_init_with_explicit_edge_dimensions(self, kg, tmp_path):
        architect = Architect(
            knowledge_graph=kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={
                "node_embedding_dimensions": 4,
                "edge_embedding_dimensions": 2,
            },
        )
        assert architect.node_embedding_dimensions == 4
        assert architect.edge_embedding_dimensions == 2

    def test_init_with_kg_built_from_metadata(self, kg_dataframe, tmp_path):
        import pandas as pd

        metadata = pd.DataFrame(
            {"id": [f"n{i}" for i in range(8)], "type": ["A"] * 4 + ["B"] * 4}
        )
        # The metadata is passed to the KnowledgeGraph constructor, so
        # Architect.set_metadata receives None and is a no-op.
        architect = Architect(
            knowledge_graph=KnowledgeGraph(
                dataframe=kg_dataframe, metadata=metadata
            ),
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"node_embedding_dimensions": 4},
        )
        assert architect.metadata is None
        assert set(architect.knowledge_graph.node_type_to_global) == {"A", "B"}

    def test_init_with_metadata_argument(self, kg_dataframe, tmp_path):
        import pandas as pd

        metadata = pd.DataFrame(
            {"id": [f"n{i}" for i in range(8)], "type": ["A"] * 4 + ["B"] * 4}
        )
        architect = Architect(
            knowledge_graph=KnowledgeGraph(dataframe=kg_dataframe),
            metadata=metadata,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"node_embedding_dimensions": 4},
        )
        assert architect.metadata is not None
        assert set(architect.knowledge_graph.node_type_to_global) == {"A", "B"}

    def test_init_with_dataframe_triggers_preprocessing(self, kg_dataframe, tmp_path):
        Architect(
            dataframe=kg_dataframe,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": True},
            model={"node_embedding_dimensions": 4},
        )


class TestArchitectGetEmbeddings:
    def test_get_embeddings_returns_node_and_edge_mappings(self, kg, tmp_path):
        add_embeddings(kg)
        architect = Architect(
            knowledge_graph=kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"node_embedding_dimensions": 4},
        )
        embeddings = architect.get_embeddings()
        assert "nodes" in embeddings
        assert "edges" in embeddings
        assert embeddings["node_mapping"] == kg.node_to_index
        assert embeddings["edge_mapping"] == kg.edge_to_index
