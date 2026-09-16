"""
Tests for kgate.normalizers and its Configuration / Architect wiring.

The Normalizer is the module that gathers what the decoders used to do in
their own `score` method (e.g. RESCAL, DistMult, TransE, TransH, TransR and
TransD L2-normalizing their head and tail embeddings). It is initialized by
the Architect after the decoder, is given a set of embeddings to normalize
and the function to apply to them, and is applied by the Architect between
the encoder and the decoder step: batchwise when there is an encoder (see
`Architect.scoring_function`), or once over the whole graph at the beginning
of each epoch when there is not (see `Architect.apply_normalizer`).
"""

import pytest
import torch
import torch.nn as nn

from kgate.architect import Architect
from kgate.config import Configuration
from kgate.normalizers import NORMALIZER_FUNCTIONS, Normalizer, l1_normalize, l2_normalize, normalize_embeddings
from kgate.modules import initialize_normalizer


class TestNormalizer:
    def test_l2_batchwise_call(self):
        head = torch.randn(5, 8)
        tail = torch.randn(5, 8)
        edge = torch.randn(5, 8)
        head_original = head.clone()
        tail_original = tail.clone()
        normalizer = Normalizer(func=l2_normalize, node=True, edge=False)
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)
        # Functional: new tensors for the selected embeddings, the inputs (and
        # the parameters they are built from) are left untouched, so the
        # gradients flow through the normalization
        assert new_head is not head
        assert new_tail is not tail
        torch.testing.assert_close(head, head_original)
        torch.testing.assert_close(tail, tail_original)
        # Edge is not selected: the same tensor is returned, unchanged
        assert new_edge is edge
        # Node embeddings are row-wise L2-normalized
        torch.testing.assert_close(torch.norm(new_head, dim=1), torch.ones(5))
        torch.testing.assert_close(torch.norm(new_tail, dim=1), torch.ones(5))

    def test_l1_batchwise_call(self):
        head = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
        tail = head.clone()
        edge = torch.ones(2, 2)
        normalizer = Normalizer(func=l1_normalize, node=True, edge=True)
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)
        torch.testing.assert_close(torch.norm(new_head, p=1, dim=1), torch.ones(2))
        torch.testing.assert_close(new_head[0], torch.tensor([3.0 / 7.0, 4.0 / 7.0]))
        torch.testing.assert_close(torch.norm(new_edge, p=1, dim=1), torch.ones(2))

    def test_custom_function(self):
        head = torch.ones(3, 4)
        tail = torch.ones(3, 4)
        edge = torch.ones(3, 4)
        normalizer = Normalizer(func=lambda x: x * 0.5, node=True, edge=True)
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)
        torch.testing.assert_close(new_head, torch.full_like(head, 0.5))
        torch.testing.assert_close(new_tail, torch.full_like(tail, 0.5))
        torch.testing.assert_close(new_edge, torch.full_like(edge, 0.5))

    def test_node_only_selection(self):
        head = torch.randn(2, 3)
        tail = torch.randn(2, 3)
        edge = torch.randn(2, 3)
        normalizer = Normalizer(func=l2_normalize, node=True, edge=False)
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)
        torch.testing.assert_close(torch.norm(new_head, dim=1), torch.ones(2))
        torch.testing.assert_close(torch.norm(new_tail, dim=1), torch.ones(2))
        torch.testing.assert_close(new_edge, edge)

    def test_edge_only_selection(self):
        head = torch.randn(2, 3)
        tail = torch.randn(2, 3)
        edge = torch.randn(2, 3)
        normalizer = Normalizer(func=l2_normalize, node=False, edge=True)
        new_head, new_tail, new_edge = normalizer(head_embeddings=head, tail_embeddings=tail, edge_embeddings=edge)
        torch.testing.assert_close(new_head, head)
        torch.testing.assert_close(new_tail, tail)
        torch.testing.assert_close(torch.norm(new_edge, dim=1), torch.ones(2))

    def test_apply_whole_graph_in_place(self):
        param = nn.Parameter(torch.randn(5, 8))
        original = param
        normalizer = Normalizer(func=l2_normalize, node=True, edge=False, params=[param])
        normalizer.apply_whole_graph()
        # In place: the parameter object is preserved, so the optimizer keeps
        # its references to the same parameters
        assert param is original
        torch.testing.assert_close(torch.norm(param.data, dim=1), torch.ones(5))

    def test_apply_whole_graph_multiple_params(self):
        params = [nn.Parameter(torch.randn(3, 4)), nn.Parameter(torch.randn(2, 4))]
        normalizer = Normalizer(func=l2_normalize, node=True, edge=True, params=params)
        normalizer.apply_whole_graph()
        for param in params:
            torch.testing.assert_close(torch.norm(param.data, dim=1), torch.ones(param.data.shape[0]))

    def test_apply_whole_graph_idempotent_for_l1_l2(self):
        # Applying the normalization twice must not change the result, so the
        # epoch-start application and the evaluation-time application do not
        # interfere
        param = nn.Parameter(torch.randn(5, 8))
        normalizer = Normalizer(func=l2_normalize, node=True, edge=False, params=[param])
        normalizer.apply_whole_graph()
        first = param.data.clone()
        normalizer.apply_whole_graph()
        torch.testing.assert_close(param.data, first)

    def test_non_callable_function_raises(self):
        with pytest.raises(TypeError):
            Normalizer(func="L2")

    def test_non_tensor_params_raise(self):
        with pytest.raises(TypeError):
            Normalizer(func=l2_normalize, params=[torch.ones(3, 4).numpy()])

    def test_no_selection_raises(self):
        with pytest.raises(ValueError):
            Normalizer(func=l2_normalize, node=False, edge=False)

    def test_properties(self):
        param = nn.Parameter(torch.randn(2, 3))
        normalizer = Normalizer(func=l2_normalize, node=True, edge=False, params=[param])
        assert normalizer.params == [param]
        assert normalizer.func is l2_normalize
        assert normalizer.node is True
        assert normalizer.edge is False

    def test_repr(self):
        normalizer = Normalizer(func=l2_normalize, node=True, edge=False, params=[nn.Parameter(torch.randn(2, 3))])
        assert "Normalizer" in repr(normalizer)
        assert "l2_normalize" in repr(normalizer)
        assert "1 parameter(s)" in repr(normalizer)


class TestBuiltinFunctions:
    def test_registered_functions(self):
        assert set(NORMALIZER_FUNCTIONS) == {"L1", "L2"}

    def test_l2_matches_decoder_behavior(self):
        # This must be the same operation the RESCAL, DistMult, TransE,
        # TransH, TransR and TransD decoders applied to their head and tail
        # embeddings in their `score` method
        x = torch.randn(6, 8)
        expected = torch.nn.functional.normalize(x, p=2, dim=1)
        torch.testing.assert_close(l2_normalize(x), expected)

    def test_l1(self):
        x = torch.randn(6, 8)
        expected = torch.nn.functional.normalize(x, p=1, dim=1)
        torch.testing.assert_close(l1_normalize(x), expected)

    def test_normalize_embeddings(self):
        x = torch.randn(6, 8)
        torch.testing.assert_close(normalize_embeddings(x, 2), torch.nn.functional.normalize(x, p=2, dim=1))
        torch.testing.assert_close(normalize_embeddings(x, 1), torch.nn.functional.normalize(x, p=1, dim=1))
        torch.testing.assert_close(
            normalize_embeddings(x, 2, squared=True),
            torch.nn.functional.normalize(x, p=2, dim=1)**2
        )

    def test_utils_backward_compatibility(self):
        # kgate.utils.normalize_embeddings must still work and delegate to
        # the normalizer module
        from kgate.utils import normalize_embeddings as utils_normalize_embeddings
        x = torch.randn(4, 5)
        torch.testing.assert_close(utils_normalize_embeddings(x, 2, False), torch.nn.functional.normalize(x, p=2, dim=1))
        torch.testing.assert_close(utils_normalize_embeddings(x, 1, True), torch.nn.functional.normalize(x, p=1, dim=1)**2)


class TestConfiguration:
    def test_defaults(self):
        # L2 on the node embeddings: the historical behavior of the decoders
        # that used to normalize in their own `score` method
        config = Configuration()
        assert config.normalizer.name == "L2"
        assert config.normalizer.params == "node"

    def test_inline_override(self):
        config = Configuration(config_dict={"model": {"normalizer": {"name": "L1", "params": "all"}}})
        assert config.normalizer.name == "L1"
        assert config.normalizer.params == "all"

    def test_none(self):
        config = Configuration(config_dict={"model": {"normalizer": {"name": "None"}}})
        assert config.normalizer.name == "None"

    def test_invalid_name_raises(self):
        config = Configuration()
        with pytest.raises(AssertionError):
            config.normalizer.name = "L9"

    def test_invalid_params_raises(self):
        config = Configuration()
        with pytest.raises(AssertionError):
            config.normalizer.params = "banana"

    def test_register_name(self):
        config = Configuration()
        config.normalizer.register_name("MyCustom")
        assert config.normalizer.name == "MyCustom"
        assert "MyCustom" in config.normalizer.supported_normalizers

    def test_repr(self):
        assert "Normalizer_Configuration" in repr(Configuration().normalizer)


class TestModules:
    def test_default_normalizer(self, embedded_kg):
        # The default configuration normalizes the node embeddings with L2
        configuration = Configuration(config_dict={"model": {"normalizer": {}}})
        normalizer = initialize_normalizer(configuration.normalizer, embedded_kg)
        assert isinstance(normalizer, Normalizer)
        assert normalizer.func is l2_normalize
        assert normalizer.node is True
        assert normalizer.edge is False
        # Only the node embeddings are normalized
        assert normalizer.params == list(embedded_kg.node_embeddings)

    def test_normalizer_edge(self, embedded_kg):
        configuration = Configuration(config_dict={"model": {"normalizer": {"name": "L1", "params": "edge"}}})
        normalizer = initialize_normalizer(configuration.normalizer, embedded_kg)
        assert normalizer.func is l1_normalize
        assert normalizer.edge is True
        assert normalizer.node is False
        assert normalizer.params == [embedded_kg.edge_embeddings]

    def test_normalizer_all(self, embedded_kg):
        configuration = Configuration(config_dict={"model": {"normalizer": {"name": "L2", "params": "all"}}})
        normalizer = initialize_normalizer(configuration.normalizer, embedded_kg)
        assert normalizer.node is True
        assert normalizer.edge is True
        assert len(normalizer.params) == len(embedded_kg.node_embeddings) + 1

    def test_unknown_name_raises(self, embedded_kg):
        configuration = Configuration(config_dict={"model": {"normalizer": {"name": "MyRegistered"}}})
        configuration.normalizer.register_name("MyRegistered")
        with pytest.raises(KeyError):
            initialize_normalizer(configuration.normalizer, embedded_kg)


class TestArchitectWiring:
    def test_default_normalizer(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
        )
        architect.initialize_model()
        # The default configuration initializes a normalizer
        assert isinstance(architect.normalizer, Normalizer)
        assert architect.normalizer.func is l2_normalize
        assert architect.normalizer.node is True
        assert architect.normalizer.edge is False
        # No encoder: the whole-graph application is a no-op-free in-place
        # normalization of the node embeddings
        node_embedding = architect.knowledge_graph.node_embeddings[0]
        assert not torch.allclose(torch.norm(node_embedding.data, dim=1), torch.ones(node_embedding.data.shape[0]))
        architect.apply_normalizer()
        torch.testing.assert_close(torch.norm(node_embedding.data, dim=1), torch.ones(node_embedding.data.shape[0]))

    def test_none_normalizer(self, embedded_kg, tmp_path):
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"normalizer": {"name": "None"}},
        )
        architect.initialize_model()
        assert architect.normalizer is None
        architect.apply_normalizer()  # no-op, no error

    def test_apply_normalizer_skipped_with_encoder(self, embedded_kg, tmp_path):
        # With an encoder, the normalization is applied batchwise in
        # `scoring_function`, not at the epoch start: `apply_normalizer` is a
        # no-op
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={
                "encoder": {"name": "GCN", "layer_count": 1},
                "normalizer": {"name": "L2", "params": "all"},
            },
        )
        architect.initialize_model()
        assert architect.encoder is not None
        assert isinstance(architect.normalizer, Normalizer)
        node_embedding = architect.knowledge_graph.node_embeddings[0]
        edge_embedding = architect.knowledge_graph.edge_embeddings
        assert not torch.allclose(torch.norm(node_embedding.data, dim=1), torch.ones(node_embedding.data.shape[0]))
        architect.apply_normalizer()  # no-op: encoder case
        assert not torch.allclose(torch.norm(node_embedding.data, dim=1), torch.ones(node_embedding.data.shape[0]))
        assert not torch.allclose(torch.norm(edge_embedding.data, dim=1), torch.ones(edge_embedding.data.shape[0]))

    def test_scoring_function_applies_normalizer_with_encoder(self, embedded_kg, tmp_path):
        # With an encoder, the normalizer is applied between the encoder and
        # the decoder step, batchwise
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={
                "encoder": {"name": "GCN", "layer_count": 1},
                "normalizer": {"name": "L2", "params": "all"},
            },
        )
        architect.initialize_model()

        batch = architect.knowledge_graph.graphindices[:, architect.knowledge_graph.train_mask.nonzero(as_tuple=True)[0]]

        node_embeddings = torch.cat(list(architect.knowledge_graph.node_embeddings), dim=0)
        edge_embeddings = architect.knowledge_graph.edge_embeddings

        scores = architect.scoring_function(batch, node_embeddings)

        # The decoder must have received normalized embeddings: recompute the
        # (default, TransE) score by hand on the normalized embeddings and
        # check that it differs from the score on the raw (unnormalized)
        # ones, i.e. that the normalization actually happened between the
        # encoder and the decoder step
        head, tail, edge = batch[0], batch[1], batch[2]
        h = l2_normalize(node_embeddings[head])
        t = l2_normalize(node_embeddings[tail])
        e = l2_normalize(edge_embeddings[edge])
        normalized_score = - (h + e - t).norm(dim=1)**2
        raw_score = - (node_embeddings[head] + edge_embeddings[edge] - node_embeddings[tail]).norm(dim=1)**2
        assert not torch.allclose(scores, raw_score)
        assert torch.allclose(scores, normalized_score)

    def test_scoring_function_without_normalizer(self, embedded_kg, tmp_path):
        # With no normalizer, the scoring function is unchanged: the decoder
        # sees the raw embeddings
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={
                "decoder": {"name": "DistMult"},
                "normalizer": {"name": "None"},
            },
        )
        architect.initialize_model()
        assert architect.normalizer is None

        batch = architect.knowledge_graph.graphindices[:, architect.knowledge_graph.train_mask.nonzero(as_tuple=True)[0]]

        node_embeddings = torch.cat(list(architect.knowledge_graph.node_embeddings), dim=0)
        edge_embeddings = architect.knowledge_graph.edge_embeddings

        scores = architect.scoring_function(batch, node_embeddings)
        head, tail, edge = batch[0], batch[1], batch[2]
        raw_score = (node_embeddings[head] * edge_embeddings[edge] * node_embeddings[tail]).sum(dim=1)
        torch.testing.assert_close(scores, raw_score)

    def test_normalize_parameters_applies_normalizer(self, embedded_kg, tmp_path):
        # `normalize_parameters` (used by `get_embeddings`) applies the
        # configured normalizer when there is no encoder
        architect = Architect(
            knowledge_graph=embedded_kg,
            output_directory=str(tmp_path / "out"),
            preprocessing={"run_preprocessing": False},
            model={"normalizer": {"name": "L2", "params": "all"}},
        )
        architect.initialize_model()
        assert isinstance(architect.normalizer, Normalizer)
        assert len(architect.normalizer.params) == len(embedded_kg.node_embeddings) + 1
        architect.normalize_parameters()
        for param in architect.normalizer.params:
            expected = torch.ones(param.data.shape[0], device=param.data.device)
            torch.testing.assert_close(torch.norm(param.data, dim=1), expected)


class TestDecoderConsolidation:
    def test_distmult_score_no_longer_normalizes(self):
        # The DistMult `score` method must not normalize its inputs anymore:
        # it is expected to receive embeddings already normalized between the
        # encoder and the decoder step (by the Architect's normalizer)
        from kgate.decoders import DistMult

        embedding_dimensions = 4
        decoder = DistMult(embedding_dimensions = embedding_dimensions,
                           node_count = 8,
                           edge_count = 2)

        head = torch.randn(5, embedding_dimensions)
        tail = torch.randn(5, embedding_dimensions)
        edge = torch.randn(5, embedding_dimensions)

        scores = decoder.score(head_embeddings = head,
                               tail_embeddings = tail,
                               edge_embeddings = edge)

        # Raw bilinear score: no internal normalization
        raw_score = (head * edge * tail).sum(dim = 1)
        torch.testing.assert_close(scores, raw_score)

    def test_transE_score_no_longer_normalizes(self):
        # The TransE `score` method must not normalize its inputs anymore
        from kgate.decoders import TransE

        embedding_dimensions = 4
        decoder = TransE(dissimilarity_type = "L2")

        head = torch.randn(5, embedding_dimensions)
        tail = torch.randn(5, embedding_dimensions)
        edge = torch.randn(5, embedding_dimensions)

        scores = decoder.score(head_embeddings = head,
                               tail_embeddings = tail,
                               edge_embeddings = edge)

        # Raw translational score: no internal normalization
        raw_score = - (head + edge - tail).norm(dim = 1)**2
        torch.testing.assert_close(scores, raw_score)
