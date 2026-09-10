"""
Tests for kgate.architect (Architect class).

Only the construction path is tested here:
- ``train_model`` / ``test`` require a full training setup and checkpoints
  and are not unit-testable.
- The ``run_preprocessing=True`` path is covered by an xfail: the
  preprocessing pipeline itself is broken (see fixes/preprocessing.py.txt).

Known bug (documented in fixes/architect.txt and exposed with
xfail(strict=True)):
- ``Architect.get_embeddings`` references ``self.knowledge_grpah`` (a
  typo of ``self.knowledge_graph``) -> NameError.
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

    @pytest.mark.xfail(
        strict=True,
        reason="`set_metadata` is called in __init__ before "
               "`self.knowledge_graph` is assigned, so passing the `metadata` "
               "argument together with a knowledge graph raises "
               "AttributeError: 'Architect' object has no attribute "
               "'knowledge_graph'. See fixes/architect.txt.",
    )
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

    @pytest.mark.xfail(
        strict=True,
        reason="With `run_preprocessing=True`, the preprocessing pipeline "
               "crashes: `remove_duplicate_triplets` passes "
               "`~indices_to_keep` (bitwise-NOT of long indices, i.e. "
               "negative positions) to remove_triplets_from_training, and "
               "`clean_knowledge_graph` then references the nonexistent "
               "`configuration.preprocessing.split` attribute. "
               "See fixes/preprocessing.py.txt and fixes/knowledgegraph.py.txt.",
    )
    def test_init_with_dataframe_triggers_preprocessing(self, kg_dataframe, tmp_path):
        Architect(
            dataframe=kg_dataframe,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": True},
            model={"node_embedding_dimensions": 4},
        )


class TestArchitectGetEmbeddings:
    @pytest.mark.xfail(
        strict=True,
        reason="`get_embeddings` references `self.knowledge_grpah` "
               "(typo of `self.knowledge_graph`) -> NameError. "
               "See fixes/architect.txt.",
    )
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
