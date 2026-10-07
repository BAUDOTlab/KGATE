"""
Tests for the KGATE decoders.

This file was rewritten to match the current decoder API (see
``src/kgate/decoders/``). The previous version tested an older API with
different constructor signatures (``node_count``/``edge_count`` for TransE,
positional ``score(head, tail, edge, h_norm=...)`` calls, ``get_embeddigs``
...).

All tests use minimal mock embeddings so they run quickly and without
out-of-memory issues.
"""

import pytest
import torch
import torch.nn as nn

from kgate.decoders import (
    TranslationalDecoder,
    BilinearDecoder,
    ConvolutionalDecoder,
    TransE,
    TransH,
    TransR,
    TransD,
    TorusE,
    RESCAL,
    DistMult,
    ComplEx,
    ConvKB,
)


def make_embeddings(node_count=4, edge_count=2, dim=4):
    node_embeddings = nn.ParameterList(
        [nn.Parameter(torch.randn(node_count, dim))]
    )
    edge_embeddings = nn.Parameter(torch.randn(edge_count, dim))
    return node_embeddings, edge_embeddings


class TestInterfaces:
    def test_translational_interface_score_raises(self):
        decoder = TranslationalDecoder()
        with pytest.raises(NotImplementedError):
            decoder.score(head_embeddings=None, tail_embeddings=None,
                          edge_embeddings=None, head_indices=None,
                          tail_indices=None, edge_indices=None)

    def test_translational_get_embeddings_none(self):
        decoder = TranslationalDecoder()
        assert decoder.get_embeddings() is None

    def test_translational_inference_prepare_candidates_raises(self):
        decoder = TranslationalDecoder()
        with pytest.raises(NotImplementedError):
            decoder.inference_prepare_candidates(head_indices=None,
                                                 tail_indices=None,
                                                 edge_indices=None,
                                                 node_embeddings=None,
                                                 edge_embeddings=None)

    def test_bilinear_interface_score_raises(self):
        decoder = BilinearDecoder()
        with pytest.raises(NotImplementedError):
            decoder.score(head_embeddings=None, tail_embeddings=None,
                          edge_embeddings=None, head_indices=None,
                          tail_indices=None, edge_indices=None)

    def test_bilinear_inference_prepare_candidates_raises(self):
        decoder = BilinearDecoder()
        with pytest.raises(NotImplementedError):
            decoder.inference_prepare_candidates(head_indices=None,
                                                 tail_indices=None,
                                                 edge_indices=None,
                                                 node_embeddings=None,
                                                 edge_embeddings=None)

    def test_bilinear_inference_score_raises(self):
        decoder = BilinearDecoder()
        with pytest.raises(NotImplementedError):
            decoder.inference_score(head_embeddings=None,
                                    tail_embeddings=None,
                                    edge_embeddings=None)

    def test_convolutional_interface_score_raises(self):
        decoder = ConvolutionalDecoder()
        with pytest.raises(NotImplementedError):
            decoder.score(head_embeddings=None, tail_embeddings=None,
                          edge_embeddings=None, head_indices=None,
                          tail_indices=None, edge_indices=None)

    def test_convolutional_inference_prepare_candidates_raises(self):
        decoder = ConvolutionalDecoder()
        with pytest.raises(NotImplementedError):
            decoder.inference_prepare_candidates(head_indices=None,
                                                 tail_indices=None,
                                                 edge_indices=None,
                                                 node_embeddings=None,
                                                 edge_embeddings=None)

    def test_convolutional_inference_score_raises(self):
        decoder = ConvolutionalDecoder()
        with pytest.raises(NotImplementedError):
            decoder.inference_score(head_embeddings=None,
                                    tail_embeddings=None,
                                    edge_embeddings=None)


class TestTranslationalDecoders:
    def test_transE_L2_score_shape(self):
        decoder = TransE(dissimilarity_type="L2")
        h = torch.randn(5, 4)
        t = torch.randn(5, 4)
        e = torch.randn(5, 4)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.shape == (5,)
        assert scores.dtype == torch.float32
        assert torch.isfinite(scores).all()

    def test_transE_L1_score_shape(self):
        decoder = TransE(dissimilarity_type="L1")
        h = torch.randn(3, 3)
        t = torch.randn(3, 3)
        e = torch.randn(3, 3)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_transE_zero_translation_scores_zero(self):
        decoder = TransE(dissimilarity_type="L2")
        h = torch.zeros(2, 3)
        t = torch.zeros(2, 3)
        e = torch.zeros(2, 3)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.tolist() == [pytest.approx(0.0), pytest.approx(0.0)]

    def test_transE_unknown_dissimilarity_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            TransE(dissimilarity_type="L3")

    def test_transE_get_embeddings_none(self):
        decoder = TransE(dissimilarity_type="L2")
        assert decoder.get_embeddings() is None

    def test_transE_normalize_parameters(self):
        decoder = TransE(dissimilarity_type="L2")
        node_embeddings, edge_embeddings = make_embeddings(node_count=3, edge_count=2, dim=3)
        new_nodes, new_edges = decoder.normalize_parameters(node_embeddings, edge_embeddings)
        assert len(new_nodes) == 1
        assert new_nodes[0].shape == (3, 3)
        assert new_edges.shape == (2, 3)

    def test_transH_init(self):
        decoder = TransH(embedding_dimensions=4, node_count=5, edge_count=2,
                         device=torch.device("cpu"))
        # One normal vector per edge
        assert decoder.normal_vector.shape == (2, 4)
        # Projections are allocated per (edge, node) pair
        assert decoder.projected_nodes.shape == (2, 5, 4)
        assert decoder.evaluated_projections is False

    def test_transH_score_shape(self):
        decoder = TransH(embedding_dimensions=3, node_count=4, edge_count=2,
                         device=torch.device("cpu"))
        h = torch.randn(4, 3)
        t = torch.randn(4, 3)
        e = torch.randn(4, 3)
        edge_indices = torch.tensor([0, 1, 0, 1])
        scores = decoder.score(head_embeddings=h, tail_embeddings=t,
                               edge_embeddings=e, edge_indices=edge_indices)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_transH_get_embeddings(self):
        decoder = TransH(embedding_dimensions=3, node_count=4, edge_count=2,
                         device=torch.device("cpu"))
        embeddings = decoder.get_embeddings()
        assert isinstance(embeddings, dict)

    def test_transR_init(self):
        decoder = TransR(node_count=5, edge_count=2,
                         node_embedding_dimensions=3, edge_embedding_dimensions=3,
                         device=torch.device("cpu"))
        assert decoder.node_count == 5
        assert decoder.edge_count == 2

    def test_transR_score_shape(self):
        decoder = TransR(node_count=4, edge_count=2,
                         node_embedding_dimensions=3, edge_embedding_dimensions=3,
                         device=torch.device("cpu"))
        h = torch.randn(4, 3)
        t = torch.randn(4, 3)
        e = torch.randn(4, 3)
        edge_indices = torch.tensor([0, 1, 0, 1])
        scores = decoder.score(head_embeddings=h, tail_embeddings=t,
                               edge_embeddings=e, edge_indices=edge_indices)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    @pytest.mark.xfail(
        strict=True,
        reason="`TransR.__init__` initializes `projection_matrix` with "
               "`node_count` rows instead of `edge_count` rows, although "
               "`score` indexes it per edge (see fixes/translational.txt). "
               "The bug is silent whenever edge_count <= node_count.",
    )
    def test_transR_projection_matrix_shape(self):
        decoder = TransR(node_count=5, edge_count=2,
                         node_embedding_dimensions=3, edge_embedding_dimensions=3,
                         device=torch.device("cpu"))
        # One projection matrix per EDGE, of size dim x dim
        assert decoder.projection_matrix.shape == (2, 9)

    def test_transD_init(self):
        decoder = TransD(node_count=5, edge_count=2,
                         node_embedding_dimensions=3, edge_embedding_dimensions=3,
                         device=torch.device("cpu"))
        assert decoder.node_count == 5
        assert decoder.edge_count == 2

    def test_transD_score_shape(self):
        decoder = TransD(node_count=4, edge_count=2,
                         node_embedding_dimensions=3, edge_embedding_dimensions=3,
                         device=torch.device("cpu"))
        h = torch.randn(4, 3)
        t = torch.randn(4, 3)
        e = torch.randn(4, 3)
        head_indices = torch.tensor([0, 1, 2, 3])
        tail_indices = torch.tensor([1, 2, 3, 0])
        edge_indices = torch.tensor([0, 1, 0, 1])
        scores = decoder.score(head_embeddings=h, tail_embeddings=t,
                               edge_embeddings=e, head_indices=head_indices,
                               tail_indices=tail_indices,
                               edge_indices=edge_indices)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_torusE_score_shape(self):
        decoder = TorusE(dissimilarity_type="torus_L1")
        h = torch.randn(3, 4)
        t = torch.randn(3, 4)
        e = torch.randn(3, 4)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_torusE_L2(self):
        decoder = TorusE(dissimilarity_type="torus_L2")
        h = torch.randn(2, 3)
        t = torch.randn(2, 3)
        e = torch.randn(2, 3)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.shape == (2,)


class TestBilinearDecoders:
    def test_rescal_init_and_score(self):
        decoder = RESCAL(node_count=4, edge_count=2, embedding_dimensions=3,
                         device=torch.device("cpu"))
        h = torch.randn(4, 3)
        t = torch.randn(4, 3)
        edge_indices = torch.tensor([0, 1, 0, 1])
        scores = decoder.score(head_embeddings=h, tail_embeddings=t,
                               edge_indices=edge_indices)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_rescal_get_embeddings(self):
        decoder = RESCAL(node_count=4, edge_count=2, embedding_dimensions=3,
                         device=torch.device("cpu"))
        embeddings = decoder.get_embeddings()
        assert "edge_embeddings_matrix" in embeddings

    def test_distmult_init_and_score(self):
        decoder = DistMult(node_count=4, edge_count=2, embedding_dimensions=3)
        h = torch.randn(4, 3)
        t = torch.randn(4, 3)
        e = torch.randn(4, 3)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_distmult_score_manual_computation(self):
        torch.manual_seed(0)
        decoder = DistMult(node_count=4, edge_count=2, embedding_dimensions=3)
        h = torch.randn(2, 3)
        t = torch.randn(2, 3)
        e = torch.randn(2, 3)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        # The `score` method no longer normalizes its inputs: the row-wise L2
        # normalization used to be done here, and is now gathered in the
        # `Normalizer` module (see `kgate.normalizers`), applied by the
        # Architect between the encoder and the decoder step
        expected = (h * e * t).sum(dim=1)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_complex_init_and_score(self):
        decoder = ComplEx(embedding_dimensions=2)
        # ComplEx works on 2 * embedding_dimensions dimensions
        h = torch.randn(3, 4)
        t = torch.randn(3, 4)
        e = torch.randn(3, 4)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()


class TestConvolutionalDecoders:
    def test_convkb_init(self):
        decoder = ConvKB(node_count=4, edge_count=2, embedding_dimensions=3,
                         filter_count=2)
        assert decoder.node_count == 4
        assert decoder.embedding_dimensions == 3

    @pytest.mark.xfail(
        strict=True,
        reason="ConvKB stores the edge count under the misspelled "
               "attribute `edge_cont`; the documented `edge_count` "
               "attribute does not exist (AttributeError). "
               "See fixes/convolutional.txt.",
    )
    def test_convkb_edge_count_attribute(self):
        decoder = ConvKB(node_count=4, edge_count=2, embedding_dimensions=3,
                         filter_count=2)
        assert decoder.edge_count == 2

    def test_convkb_score_shape(self):
        decoder = ConvKB(node_count=4, edge_count=2, embedding_dimensions=3,
                         filter_count=2)
        h = torch.randn(4, 3)
        t = torch.randn(4, 3)
        e = torch.randn(4, 3)
        scores = decoder.score(head_embeddings=h, tail_embeddings=t, edge_embeddings=e)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()
        # The output is a softmax probability, hence in [0, 1]
        assert (scores >= 0).all() and (scores <= 1).all()
