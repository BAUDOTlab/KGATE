"""
Tests for kgate.initializers (Initializer, FeatureInitializer,
Node2VecInitializer).

Known bugs (documented in fixes/initializers.txt and exposed with
xfail(strict=True)):
- ``FeatureInitializer.initialize_all_embeddings`` calls
  ``self.initialize_embeddings`` (plural) instead of
  ``self.initialize_embedding`` (singular) -> AttributeError.
- ``Node2VecInitializer.initialize_all_embeddings`` never returns the
  embeddings and never sets ``knowledge_graph.embeddings`` (only a
  ``#TODO``/``pass`` in the ``inplace`` branch).

``Node2VecInitializer.initialize_all_embeddings`` trains for a hardcoded
100 epochs and is therefore not run in the test suite; only
``__init__`` is tested.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from kgate.initializers import (
    FeatureInitializer,
    Initializer,
    Node2VecInitializer,
)


class TestInitializer:
    def test_initialize_embedding_shape_and_device(self):
        embedding = Initializer().initialize_embedding(5, 3, "cpu")
        assert isinstance(embedding, nn.Parameter)
        assert embedding.shape == (5, 3)
        assert embedding.device.type == "cpu"
        assert torch.isfinite(embedding).all()

    def test_initialize_embedding_is_xavier_uniform(self):
        torch.manual_seed(0)
        embedding = Initializer().initialize_embedding(500, 3, "cpu")
        # xavier_uniform_ bound = sqrt(6 / (fan_in + fan_out))
        bound = (6 / (500 + 3)) ** 0.5
        assert embedding.abs().max() <= bound + 1e-6
        # The tensor should actually be filled, not left empty/zero.
        assert embedding.abs().max() > 0.5 * bound

    def test_initialize_all_embeddings_single_node_type(self, kg):
        node_embeddings, edge_embeddings = Initializer().initialize_all_embeddings(
            kg,
            node_embedding_dimensions=4,
            edge_embedding_dimensions=4,
            device="cpu",
        )
        assert isinstance(node_embeddings, nn.ParameterList)
        assert len(node_embeddings) == 1
        assert node_embeddings[0].shape == (kg.node_count, 4)
        assert isinstance(edge_embeddings, nn.Parameter)
        assert edge_embeddings.shape == (kg.edge_count, 4)

    def test_initialize_all_embeddings_inplace(self, kg):
        result = Initializer().initialize_all_embeddings(
            kg,
            node_embedding_dimensions=4,
            edge_embedding_dimensions=4,
            device="cpu",
            inplace=True,
        )
        assert result is None
        node_embeddings = kg.node_embeddings
        edge_embeddings = kg.edge_embeddings
        assert node_embeddings is not None
        assert node_embeddings[0].shape == (kg.node_count, 4)
        assert edge_embeddings.shape == (kg.edge_count, 4)

    def test_initialize_all_embeddings_heterogeneous(self, hetero_kg):
        node_embeddings, edge_embeddings = Initializer().initialize_all_embeddings(
            hetero_kg,
            node_embedding_dimensions=3,
            edge_embedding_dimensions=3,
            device="cpu",
        )
        assert len(node_embeddings) == 2  # one per node type ("A" and "B")
        assert node_embeddings[0].shape[0] == 4
        assert node_embeddings[1].shape[0] == 4
        assert edge_embeddings.shape[0] == hetero_kg.edge_count


class TestFeatureInitializer:
    def test_initialize_embedding_uses_feature_values(self, kg):
        features = pd.DataFrame(
            [[float(i), -i, 0.5] for i in range(8)],
            index=[f"n{i}" for i in range(8)],
        )
        initializer = FeatureInitializer(
            node_features={"Node": features}, edge_features=pd.DataFrame()
        )
        embedding = initializer.initialize_embedding(features, kg, "Node", "cpu")
        assert embedding.shape == (8, 3)
        # First node (n0) has features (0, 0, 0.5)
        torch.testing.assert_close(
            embedding[0], torch.tensor([0.0, 0.0, 0.5])
        )
        # Last node (n7) has features (7, -7, 0.5)
        torch.testing.assert_close(
            embedding[7], torch.tensor([7.0, -7.0, 0.5])
        )

    def test_initialize_embedding_wrong_feature_count_raises(self, kg):
        features = pd.DataFrame(
            [[1.0, 2.0] for _ in range(3)],
            index=[f"n{i}" for i in range(3)],
        )
        initializer = FeatureInitializer(
            node_features={"Node": features}, edge_features=pd.DataFrame()
        )
        with pytest.raises(AssertionError, match="must match the number of nodes"):
            initializer.initialize_embedding(features, kg, "Node", "cpu")

    @pytest.mark.xfail(
        strict=True,
        reason="`initialize_all_embeddings` calls the nonexistent "
               "`self.initialize_embeddings` (plural) instead of "
               "`self.initialize_embedding` (singular) -> AttributeError. "
               "See fixes/initializers.txt.",
    )
    def test_initialize_all_embeddings_with_node_features(self, kg):
        features = pd.DataFrame(
            [[float(i)] for i in range(8)],
            index=[f"n{i}" for i in range(8)],
        )
        initializer = FeatureInitializer(
            node_features={"Node": features}, edge_features=pd.DataFrame()
        )
        node_embeddings, edge_embeddings = initializer.initialize_all_embeddings(
            kg,
            node_embedding_dimensions=4,
            edge_embedding_dimensions=4,
            device="cpu",
        )
        assert node_embeddings[0].shape == (8, 1)

    def test_initialize_all_embeddings_edge_features_only(self, kg):
        # No node features: node embeddings are randomly initialized, while
        # edge embeddings come from the given edge features.
        edge_features = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=["E1", "E2"],
        )
        initializer = FeatureInitializer(
            node_features={}, edge_features=edge_features
        )
        node_embeddings, edge_embeddings = initializer.initialize_all_embeddings(
            kg,
            node_embedding_dimensions=4,
            edge_embedding_dimensions=2,
            device="cpu",
        )
        assert node_embeddings[0].shape == (8, 4)
        assert edge_embeddings.shape == (2, 2)
        # Edge order: E1 -> index 0, E2 -> index 1
        torch.testing.assert_close(edge_embeddings[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(edge_embeddings[1], torch.tensor([3.0, 4.0]))


class TestNode2VecInitializer:
    def test_init(self, tmp_path):
        # PyG's Node2Vec requires the pyg-lib or torch-cluster backend for
        # its random walks. Neither is installed (nor declared in
        # pyproject.toml), so skip instead of failing (see
        # fixes/initializers.txt).
        from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER

        if not (WITH_PYG_LIB or WITH_TORCH_CLUSTER):
            pytest.skip("Node2Vec requires the pyg-lib or torch-cluster package")

        edge_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
        initializer = Node2VecInitializer(
            edge_indices,
            embedding_dimensions=4,
            walk_length=5,
            context_size=3,
            output_directory=tmp_path,
            device="cpu",
        )
        assert initializer.model is not None
        assert isinstance(initializer.loader, torch.utils.data.DataLoader)
        assert isinstance(initializer.optimizer, torch.optim.SparseAdam)
        assert initializer.device == "cpu"
        assert initializer.output_directory == Path(tmp_path)
