"""
Tests for kgate.normalizers and its Configuration / Architect wiring.

The Normalizer is the module that gathers what the decoders used to do in
their own `score` method (e.g. RESCAL, DistMult, TransE, TransH, TransR and
TransD L2-normalizing their head and tail embeddings). It is initialized by
the Architect after the decoder (see `kgate.modules.initialize_normalizer`)
and is applied:
- to the initial embeddings, in place, right after the initializer runs
  (`Architect.initialize_model` and `Normalizer.initialize`);
- batchwise to the (encoder output) embeddings between the encoder and the
  decoder step during the training loop, when there is an encoder
  (`Architect.scoring_function`);
- to the whole-graph embeddings, in place, before training starts and before
  an export of the embeddings, when there is no encoder
  (`Architect.normalize_parameters` / `Architect.apply_normalizer`).
"""

import pytest
import torch
import torch.nn as nn

from kgate.architect import Architect
from kgate.config import Configuration
from kgate.normalizers import NORMALIZER_FUNCTIONS, Normalizer, l1_normalize, l2_normalize, normalize_embeddings
from kgate.modules import initialize_normalizer


class TestNormalizer:
    def test_defaults(self):
        normalizer = Normalizer()
        assert normalizer.initial_targets == "all"
        assert normalizer.training_targets == "all"
        assert normalizer.nodes is True
        assert normalizer.edges is True
        # Both functions default to the identity: no normalization
        x = torch.randn(3, 4)
        assert torch.equal(normalizer.initial_normalization(x), x)
        assert torch.equal(normalizer.training_normalization(x), x)

    def test_non_callable_training_normalization_raises(self):
        with pytest.raises(TypeError, match="must be callable"):
            Normalizer(training_normalization="L2")

    def test_initial_falls_back_to_training(self):
        normalizer = Normalizer(training_normalization=l2_normalize)
        assert normalizer.initial_normalization is l2_normalize

        normalizer = Normalizer(initial_normalization=None, training_normalization=l1_normalize)
        assert normalizer.initial_normalization is l1_normalize

    def test_batchwise_nodes_only(self):
        head = torch.randn(5, 8)
        tail = torch.randn(5, 8)
        edge = torch.randn(5, 8)

        normalizer = Normalizer(training_normalization=l2_normalize, training_targets="node")
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)

        assert new_head is not head
        assert new_tail is not tail
        assert new_edge is edge  # Edges are not normalized: same object returned
        assert torch.allclose(new_head.norm(dim=1), torch.ones(5))
        assert torch.allclose(new_tail.norm(dim=1), torch.ones(5))
        assert torch.allclose(head.norm(dim=1), torch.norm(head, dim=1))  # Inputs are not modified in place

    def test_batchwise_edges_only(self):
        head = torch.randn(5, 8)
        tail = torch.randn(5, 8)
        edge = torch.randn(5, 8)

        normalizer = Normalizer(training_normalization=l1_normalize, training_targets="edge")
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)

        assert new_head is head
        assert new_tail is tail
        assert new_edge is not edge
        assert torch.allclose(new_edge.norm(dim=1, p=1), torch.ones(5))

    def test_batchwise_all(self):
        head = torch.randn(5, 8)
        tail = torch.randn(5, 8)
        edge = torch.randn(5, 8)

        normalizer = Normalizer(training_normalization=l2_normalize, training_targets="all")
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)

        assert new_head is not head
        assert new_tail is not tail
        assert new_edge is not edge
        assert torch.allclose(new_head.norm(dim=1), torch.ones(5))
        assert torch.allclose(new_tail.norm(dim=1), torch.ones(5))
        assert torch.allclose(new_edge.norm(dim=1), torch.ones(5))

    def test_batchwise_custom_function(self):
        head = torch.randn(5, 8)
        tail = torch.randn(5, 8)
        edge = torch.randn(5, 8)

        scale = lambda x: x * 0.5
        normalizer = Normalizer(training_normalization=scale, training_targets="all")
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)

        assert torch.allclose(new_head, head * 0.5)
        assert torch.allclose(new_tail, tail * 0.5)
        assert torch.allclose(new_edge, edge * 0.5)

    def test_initialize_single_node_parameter(self):
        node_embeddings = nn.Parameter(torch.randn(10, 4))
        edge_embeddings = nn.Parameter(torch.randn(6, 4))

        normalizer = Normalizer(initial_normalization=l2_normalize, initial_targets="all")
        returned_nodes, returned_edges = normalizer.initialize(node_embeddings, edge_embeddings)

        assert returned_nodes is node_embeddings  # Normalized in place, same object
        assert returned_edges is edge_embeddings
        assert torch.allclose(node_embeddings.norm(dim=1), torch.ones(10))
        assert torch.allclose(edge_embeddings.norm(dim=1), torch.ones(6))

    def test_initialize_node_parameter_list(self):
        # As stored in `KnowledgeGraph.node_embeddings`: one Parameter per node type
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(4, 4)), nn.Parameter(torch.randn(3, 4))])
        edge_embeddings = nn.Parameter(torch.randn(6, 4))

        normalizer = Normalizer(initial_normalization=l2_normalize, initial_targets="all")
        normalizer.initialize(node_embeddings, edge_embeddings)

        assert torch.allclose(node_embeddings[0].norm(dim=1), torch.ones(4))
        assert torch.allclose(node_embeddings[1].norm(dim=1), torch.ones(3))
        assert torch.allclose(edge_embeddings.norm(dim=1), torch.ones(6))

    def test_initialize_targets_node(self):
        node_embeddings = nn.Parameter(torch.randn(10, 4))
        edge_embeddings = nn.Parameter(torch.randn(6, 4))
        original_edge = edge_embeddings.clone()

        normalizer = Normalizer(initial_normalization=l2_normalize, initial_targets="node")
        normalizer.initialize(node_embeddings, edge_embeddings)

        assert torch.allclose(node_embeddings.norm(dim=1), torch.ones(10))
        assert torch.allclose(edge_embeddings, original_edge)  # Edges left untouched

    def test_initialize_targets_edge(self):
        node_embeddings = nn.Parameter(torch.randn(10, 4))
        edge_embeddings = nn.Parameter(torch.randn(6, 4))
        original_node = node_embeddings.clone()

        normalizer = Normalizer(initial_normalization=l2_normalize, initial_targets="edge")
        normalizer.initialize(node_embeddings, edge_embeddings)

        assert torch.allclose(node_embeddings, original_node)  # Nodes left untouched
        assert torch.allclose(edge_embeddings.norm(dim=1), torch.ones(6))

    def test_properties(self):
        normalizer = Normalizer(
            initial_normalization=l1_normalize,
            training_normalization=l2_normalize,
            initial_targets="edge",
            training_targets="node"
        )
        assert normalizer.initial_normalization is l1_normalize
        assert normalizer.training_normalization is l2_normalize
        assert normalizer.initial_targets == "edge"
        assert normalizer.training_targets == "node"
        assert normalizer.nodes is True
        assert normalizer.edges is False

    def test_repr(self):
        normalizer = Normalizer(initial_normalization=l2_normalize, training_normalization=l1_normalize)
        r = repr(normalizer)
        assert "Normalizer" in r
        assert "l2_normalize" in r
        assert "l1_normalize" in r


class TestBuiltinFunctions:
    def test_registered_functions(self):
        assert set(NORMALIZER_FUNCTIONS.keys()) == {"L1", "L2"}

    def test_l2_normalize_matches_torch(self):
        x = torch.randn(4, 5)
        assert torch.allclose(l2_normalize(x), torch.nn.functional.normalize(x, p=2, dim=1))

    def test_l1_normalize(self):
        x = torch.randn(4, 5)
        assert torch.allclose(l1_normalize(x), x / x.norm(dim=1, p=1, keepdim=True))

    def test_normalize_embeddings(self):
        x = torch.randn(4, 5)
        assert torch.allclose(normalize_embeddings(x, p=2), torch.nn.functional.normalize(x, p=2, dim=1))
        assert torch.allclose(normalize_embeddings(x, p=1), x / x.norm(dim=1, p=1, keepdim=True))


class TestConfiguration:
    def test_defaults(self):
        normalizer_config = Configuration().normalizer
        assert normalizer_config.initial_normalization == "L2"
        assert normalizer_config.training_normalization == "L2"
        assert normalizer_config.initial_parameters == "all"
        assert normalizer_config.training_parameters == "all"

    def test_inline_override(self):
        normalizer_config = Configuration(
            config_dict = {
                "model": {
                    "normalizer": {
                        "initial_normalization": "L1",
                        "training_normalization": "L2",
                        "initial_parameters": "node",
                        "training_parameters": "edge"
                    }
                }
            }
        ).normalizer
        assert normalizer_config.initial_normalization == "L1"
        assert normalizer_config.training_normalization == "L2"
        assert normalizer_config.initial_parameters == "node"
        assert normalizer_config.training_parameters == "edge"

    def test_none_scopes(self):
        normalizer_config = Configuration(
            config_dict = {
                "model": {
                    "normalizer": {
                        "initial_normalization": "None",
                        "training_normalization": "None"
                    }
                }
            }
        ).normalizer
        assert normalizer_config.initial_normalization == "None"
        assert normalizer_config.training_normalization == "None"

    def test_setters_validate(self):
        normalizer_config = Configuration().normalizer

        with pytest.raises(AssertionError, match="Unsupported normalizer given"):
            normalizer_config.initial_normalization = "L9"
        with pytest.raises(AssertionError, match="Unsupported normalizer given"):
            normalizer_config.training_normalization = "L9"

        normalizer_config.initial_normalization = "L1"
        normalizer_config.training_normalization = "L1"
        assert normalizer_config.initial_normalization == "L1"
        assert normalizer_config.training_normalization == "L1"

        with pytest.raises(AssertionError, match="Unsupported normalizer parameters given"):
            normalizer_config.initial_parameters = "banana"
        with pytest.raises(AssertionError, match="Unsupported normalizer parameters given"):
            normalizer_config.training_parameters = "None"

        normalizer_config.initial_parameters = "node"
        normalizer_config.training_parameters = "edge"
        assert normalizer_config.initial_parameters == "node"
        assert normalizer_config.training_parameters == "edge"

    def test_register_name(self):
        normalizer_config = Configuration().normalizer

        with pytest.raises(AssertionError):
            normalizer_config.initial_normalization = "custom_norm"

        normalizer_config.register_name("custom_norm")
        normalizer_config.initial_normalization = "custom_norm"
        assert normalizer_config.initial_normalization == "custom_norm"

    def test_repr(self):
        normalizer_config = Configuration().normalizer
        r = repr(normalizer_config)
        assert "initial_normalization" in r
        assert "training_normalization" in r
        assert "initial_parameters" in r
        assert "training_parameters" in r


class TestModules:
    def test_default(self, embedded_kg):
        normalizer = initialize_normalizer(Configuration().normalizer, embedded_kg)

        assert isinstance(normalizer, Normalizer)
        assert normalizer.initial_normalization is l2_normalize
        assert normalizer.training_normalization is l2_normalize
        assert normalizer.initial_targets == "all"
        assert normalizer.training_targets == "all"
        assert normalizer.nodes is True
        assert normalizer.edges is True

    def test_mixed_scopes(self, embedded_kg):
        config = Configuration(
            config_dict = {
                "model": {
                    "normalizer": {
                        "initial_normalization": "L1",
                        "training_normalization": "L2",
                        "initial_parameters": "node",
                        "training_parameters": "edge"
                    }
                }
            }
        ).normalizer

        normalizer = initialize_normalizer(config, embedded_kg)

        assert normalizer.initial_normalization is l1_normalize
        assert normalizer.training_normalization is l2_normalize
        assert normalizer.initial_targets == "node"
        assert normalizer.training_targets == "edge"
        assert normalizer.nodes is False
        assert normalizer.edges is True

    def test_both_none_is_noop(self, embedded_kg):
        config = Configuration(
            config_dict = {
                "model": {
                    "normalizer": {
                        "initial_normalization": "None",
                        "training_normalization": "None"
                    }
                }
            }
        ).normalizer

        normalizer = initialize_normalizer(config, embedded_kg)

        x = torch.randn(3, 4)
        # Both scopes are the identity function
        assert torch.equal(normalizer.initial_normalization(x), x)
        assert torch.equal(normalizer.training_normalization(x), x)
        # And the initial function falls back to the (identity) training function
        h, t, e = normalizer(head_embeddings=x, tail_embeddings=x, edge_embeddings=x)
        assert h is x and t is x and e is x

    def test_initial_none_falls_back_to_training(self, embedded_kg):
        config = Configuration(
            config_dict = {
                "model": {
                    "normalizer": {
                        "initial_normalization": "None",
                        "training_normalization": "L2"
                    }
                }
            }
        ).normalizer

        normalizer = initialize_normalizer(config, embedded_kg)

        assert normalizer.initial_normalization is l2_normalize
        assert normalizer.training_normalization is l2_normalize

    def test_unknown_name_raises(self, embedded_kg):
        config = Configuration(
            config_dict = {
                "model": {
                    "normalizer": {"initial_normalization": "banana", "training_normalization": "banana"}
                }
            }
        ).normalizer

        with pytest.raises(KeyError):
            initialize_normalizer(config, embedded_kg)


class TestArchitectWiring:
    def test_default_normalizer(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
        )

        architect.initialize_model()

        assert isinstance(architect.normalizer, Normalizer)
        # The initial normalization is applied in place to the initial parameters
        node_embeddings = embedded_kg.node_embeddings[0]
        edge_embeddings = embedded_kg.edge_embeddings
        assert torch.allclose(node_embeddings.norm(dim=1), torch.ones(node_embeddings.shape[0]))
        assert torch.allclose(edge_embeddings.norm(dim=1), torch.ones(edge_embeddings.shape[0]))

    def test_apply_normalizer_renormalizes(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
        )
        architect.initialize_model()

        # Perturb the parameters
        node_embeddings = embedded_kg.node_embeddings[0]
        edge_embeddings = embedded_kg.edge_embeddings
        node_embeddings.data.add_(torch.randn_like(node_embeddings.data) * 0.1)
        edge_embeddings.data.add_(torch.randn_like(edge_embeddings.data) * 0.1)
        assert not torch.allclose(node_embeddings.norm(dim=1), torch.ones(node_embeddings.shape[0]))

        # Without an encoder, the normalizer is applied to the whole graph, in place
        architect.apply_normalizer()

        assert torch.allclose(node_embeddings.norm(dim=1), torch.ones(node_embeddings.shape[0]))
        assert torch.allclose(edge_embeddings.norm(dim=1), torch.ones(edge_embeddings.shape[0]))

    def test_apply_normalizer_noop_with_encoder(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
            model = {"encoder": {"name": "GCN", "gnn_layer_number": 1}},
        )
        architect.initialize_model()

        # Perturb the parameters
        node_embeddings = embedded_kg.node_embeddings[0]
        edge_embeddings = embedded_kg.edge_embeddings
        node_embeddings.data.add_(torch.randn_like(node_embeddings.data) * 0.1)
        edge_embeddings.data.add_(torch.randn_like(edge_embeddings.data) * 0.1)
        perturbed_node = node_embeddings.clone()
        perturbed_edge = edge_embeddings.clone()

        # With an encoder, the decoder sees the encoder output, not the
        # parameters: the whole-graph application is skipped
        architect.apply_normalizer()

        assert torch.allclose(node_embeddings, perturbed_node)
        assert torch.allclose(edge_embeddings, perturbed_edge)

    def test_scoring_function_with_encoder_normalizes(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
            model = {"encoder": {"name": "GCN", "gnn_layer_number": 1}},
        )
        architect.initialize_model()

        # Simulate encoder output: arbitrary (unnormalized) embeddings
        embedded_kg.node_embeddings[0].data.add_(torch.randn(8, 4) * 0.3)
        embedded_kg.edge_embeddings.data.add_(torch.randn(2, 4) * 0.3)
        node_embeddings = torch.cat([emb for emb in embedded_kg.node_embeddings])
        train_indices = architect.knowledge_graph.train_mask.nonzero(as_tuple=True)[0]
        head_embeddings = node_embeddings[architect.knowledge_graph.graphindices[0][train_indices]]
        tail_embeddings = node_embeddings[architect.knowledge_graph.graphindices[1][train_indices]]
        edge_indices = architect.knowledge_graph.graphindices[2][train_indices]
        edge_embeddings = embedded_kg.edge_embeddings[edge_indices]

        # Hand-computed TransE score on L2-normalized embeddings
        normalized_head = head_embeddings / head_embeddings.norm(dim=1, keepdim=True)
        normalized_tail = tail_embeddings / tail_embeddings.norm(dim=1, keepdim=True)
        normalized_edge = edge_embeddings / edge_embeddings.norm(dim=1, keepdim=True)
        expected_score = - (normalized_head + normalized_edge - normalized_tail).norm(dim=1)**2

        # The normalizer is applied batchwise between the encoder and the decoder step
        batch = architect.knowledge_graph.graphindices[:, train_indices]
        scores = architect.scoring_function(batch, node_embeddings)

        torch.testing.assert_close(scores, expected_score)

    def test_scoring_function_without_encoder(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
        )
        architect.initialize_model()

        # Without an encoder, the decoder sees the parameters directly:
        # no batchwise normalization is applied in `scoring_function`
        # (perturb the parameters to make the check non-degenerate)
        embedded_kg.node_embeddings[0].data.add_(torch.randn(8, 4) * 0.3)
        embedded_kg.edge_embeddings.data.add_(torch.randn(2, 4) * 0.3)
        node_embeddings = torch.cat([emb for emb in embedded_kg.node_embeddings])
        train_indices = architect.knowledge_graph.train_mask.nonzero(as_tuple=True)[0]
        head_embeddings = node_embeddings[architect.knowledge_graph.graphindices[0][train_indices]]
        tail_embeddings = node_embeddings[architect.knowledge_graph.graphindices[1][train_indices]]
        edge_indices = architect.knowledge_graph.graphindices[2][train_indices]
        edge_embeddings = embedded_kg.edge_embeddings[edge_indices]

        raw_score = - (head_embeddings + edge_embeddings - tail_embeddings).norm(dim=1)**2

        batch = architect.knowledge_graph.graphindices[:, train_indices]
        scores = architect.scoring_function(batch, node_embeddings)

        torch.testing.assert_close(scores, raw_score)

    def test_no_normalizer(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
            model = {
                "decoder": {"name": "DistMult"},
                "normalizer": {"initial_normalization": "None", "training_normalization": "None"}
            },
        )

        architect.initialize_model()

        # Perturb the parameters to make the check non-degenerate
        embedded_kg.node_embeddings[0].data.add_(torch.randn(8, 4) * 0.3)
        embedded_kg.edge_embeddings.data.add_(torch.randn(2, 4) * 0.3)
        node_embeddings = torch.cat([emb for emb in embedded_kg.node_embeddings])
        train_indices = architect.knowledge_graph.train_mask.nonzero(as_tuple=True)[0]
        head_embeddings = node_embeddings[architect.knowledge_graph.graphindices[0][train_indices]]
        tail_embeddings = node_embeddings[architect.knowledge_graph.graphindices[1][train_indices]]
        edge_indices = architect.knowledge_graph.graphindices[2][train_indices]
        edge_embeddings = embedded_kg.edge_embeddings[edge_indices]

        # DistMult is a bilinear model: raw score, no normalization
        # (the perturbation above makes this check non-degenerate)
        raw_score = (head_embeddings * edge_embeddings * tail_embeddings).sum(dim=1)

        batch = architect.knowledge_graph.graphindices[:, train_indices]
        scores = architect.scoring_function(batch, node_embeddings)

        torch.testing.assert_close(scores, raw_score)

    def test_normalize_parameters(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
        )
        architect.initialize_model()

        # Perturb the parameters
        node_embeddings = embedded_kg.node_embeddings[0]
        edge_embeddings = embedded_kg.edge_embeddings
        node_embeddings.data.add_(torch.randn_like(node_embeddings.data) * 0.1)
        edge_embeddings.data.add_(torch.randn_like(edge_embeddings.data) * 0.1)

        # `normalize_parameters` is called before training and before an export
        architect.normalize_parameters()

        assert torch.allclose(node_embeddings.norm(dim=1), torch.ones(node_embeddings.shape[0]))
        assert torch.allclose(edge_embeddings.norm(dim=1), torch.ones(edge_embeddings.shape[0]))

    def test_normalize_parameters_noop_with_encoder(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing = {"run_preprocessing": False},
            model = {"encoder": {"name": "GCN", "gnn_layer_number": 1}},
        )
        architect.initialize_model()

        node_embeddings = embedded_kg.node_embeddings[0]
        edge_embeddings = embedded_kg.edge_embeddings
        node_embeddings.data.add_(torch.randn_like(node_embeddings.data) * 0.1)
        edge_embeddings.data.add_(torch.randn_like(edge_embeddings.data) * 0.1)
        perturbed_node = node_embeddings.clone()
        perturbed_edge = edge_embeddings.clone()

        architect.normalize_parameters()

        assert torch.allclose(node_embeddings, perturbed_node)
        assert torch.allclose(edge_embeddings, perturbed_edge)


class TestDecoderConsolidation:
    def test_distmult_no_longer_normalizes(self):
        from kgate.decoders.bilinear import DistMult
        decoder = DistMult(node_count=4, edge_count=2, embedding_dimensions=3)
        head = torch.randn(2, 3)
        tail = torch.randn(2, 3)
        edge = torch.randn(2, 3)
        scores = decoder.score(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge,
                               head_indices=torch.tensor([0, 1]),
                               tail_indices=torch.tensor([2, 3]),
                               edge_indices=torch.tensor([0, 1]))
        # DistMult is a bilinear model: element-wise product of the three, summed
        torch.testing.assert_close(scores, (head * edge * tail).sum(dim=1))

    def test_transE_no_longer_normalizes(self):
        from kgate.decoders.translational import TransE
        decoder = TransE(dissimilarity_type="L2")
        head = torch.randn(2, 4)
        tail = torch.randn(2, 4)
        edge = torch.randn(2, 4)
        scores = decoder.score(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge,
                               head_indices=torch.tensor([0, 1]),
                               tail_indices=torch.tensor([2, 3]),
                               edge_indices=torch.tensor([0, 1]))
        raw_score = - (head + edge - tail).norm(dim=1)**2
        torch.testing.assert_close(scores, raw_score)
