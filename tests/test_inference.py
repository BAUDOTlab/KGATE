"""
Tests for kgate.inference (Inference_KG, EdgeInference, NodeInference).

Known bugs (documented in fixes/inference.py.txt and exposed with
xfail(strict=True)):
- ``EdgeInference.evaluate`` / ``NodeInference.evaluate`` call
  ``decoder.inference_score(...)`` with positional arguments, but every
  decoder's ``inference_score`` is keyword-only -> TypeError.
- The encoder branches of both methods are broken independently:
  ``encoder.n_layers`` (should be ``layer_count``),
  ``input.mapping`` (should be ``input.node_mapping``) and a positional
  call to the keyword-only ``KnowledgeGraph.get_encoder_input``.
- ``EdgeInference.evaluate`` also ends with a two-argument index on a
  2D scores tensor: ``scores[i * batch_size, (i + 1) * batch_size]``.
- ``NodeInference.evaluate`` allocates ``scores`` as an integer tensor
  (``torch.empty(...).long()``), silently truncating float scores.
"""

import pytest
import torch
import torch.nn as nn

from kgate.decoders import TransE
from kgate.inference import EdgeInference, Inference_KG, NodeInference


class TestInferenceKG:
    def test_init_stores_tensors(self):
        first = torch.tensor([0, 1])
        second = torch.tensor([2, 3])
        dataset = Inference_KG(first, second)
        assert torch.equal(dataset.first_tensor_index, first)
        assert torch.equal(dataset.second_tensor_index, second)

    def test_len_and_getitem(self):
        dataset = Inference_KG(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
        assert len(dataset) == 3
        assert dataset[0] == (0, 3)
        assert dataset[2] == (2, 5)

    def test_size_mismatch_raises(self):
        with pytest.raises(AssertionError, match="same size"):
            Inference_KG(torch.tensor([0, 1]), torch.tensor([0]))

    def test_is_a_torch_dataset(self):
        dataset = Inference_KG(torch.tensor([0]), torch.tensor([1]))
        assert isinstance(dataset, torch.utils.data.Dataset)


class TestEdgeInference:
    def test_init(self, kg):
        inference = EdgeInference(kg)
        assert inference.kg is kg

    @pytest.mark.xfail(
        strict=True,
        reason="`EdgeInference.evaluate` calls `decoder.inference_score` "
               "with positional arguments although it is keyword-only "
               "(TypeError); the encoder branch also uses `encoder.n_layers` "
               "instead of `layer_count`, `input.mapping` instead of "
               "`input.node_mapping`, calls the keyword-only "
               "`get_encoder_input` positionally and finally indexes the "
               "2D scores tensor with two scalar indices. "
               "See fixes/inference.py.txt.",
    )
    def test_evaluate_without_encoder(self, kg):
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        inference = EdgeInference(kg)
        predictions, scores = inference.evaluate(
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
            top_k=2,
            batch_size=2,
            encoder=None,
            decoder=TransE(),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )
        assert predictions.shape == (2, 2)


class TestNodeInference:
    def test_init(self, kg):
        inference = NodeInference(kg)
        assert inference.kg is kg

    @pytest.mark.xfail(
        strict=True,
        reason="`NodeInference.evaluate` calls `decoder.inference_score` "
               "with positional arguments although it is keyword-only "
               "(TypeError); it also stores scores in an integer tensor "
               "(`.long()`), truncating float values, and its encoder "
               "branch has the same `n_layers`/`mapping` bugs as "
               "`EdgeInference.evaluate`. "
               "See fixes/inference.py.txt.",
    )
    def test_evaluate_missing_tail_without_encoder(self, kg):
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        inference = NodeInference(kg)
        predictions, scores = inference.evaluate(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            top_k=2,
            missing_triplet_part="tail",
            batch_size=2,
            encoder=None,
            decoder=TransE(),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )
        assert predictions.shape == (2, 2)
