"""
Tests for kgate.inference (Inference_KG, EdgeInference, NodeInference).
"""

import pytest
import torch
import torch.nn as nn

from kgate.decoders import TransE
from kgate.encoders import GATEncoder
from kgate.inference import EdgeInference, Inference_KG, NodeInference


def _make_encoder():
    return GATEncoder(
        edge_types=[("Node", "E1", "Node"), ("Node", "E2", "Node")],
        embedding_dimensions=4,
        gat_layer_count=2,
        device="cpu",
    )


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
        assert scores.shape == (2, 2)
        assert predictions.dtype == torch.long
        assert scores.dtype == torch.float
        # Valid edge indices, ranked from best to worst score
        assert ((predictions >= 0) & (predictions < kg.edge_count)).all()
        assert (scores[:, 0] >= scores[:, 1]).all()

    def test_evaluate_with_encoder(self, kg):
        torch.manual_seed(0)
        encoder = _make_encoder()
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        # The encoder path reads the embeddings from the knowledge graph itself.
        kg.embeddings = node_embeddings, edge_embeddings
        inference = EdgeInference(kg)
        predictions, scores = inference.evaluate(
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
            top_k=2,
            batch_size=2,
            encoder=encoder,
            decoder=TransE(),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )
        assert predictions.shape == (2, 2)
        assert scores.shape == (2, 2)
        assert ((predictions >= 0) & (predictions < kg.edge_count)).all()
        assert (scores[:, 0] >= scores[:, 1]).all()

    def test_evaluate_hetero_kg(self, hetero_kg):
        # Two node types: the node embeddings must be flattened over all types,
        # so that global node indices and the node candidates work.
        kg = hetero_kg
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25), sizes=(4, 2, 2))
        node_embeddings = nn.ParameterList(
            nn.Parameter(torch.randn(len(ids), 4)) for ids in kg.node_type_to_global.values()
        )
        edge_embeddings = nn.Parameter(torch.randn(kg.edge_count, 4))
        inference = EdgeInference(kg)
        predictions, scores = inference.evaluate(
            torch.tensor([0, 1]),
            torch.tensor([4, 5]),
            top_k=2,
            batch_size=2,
            encoder=None,
            decoder=TransE(),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )
        assert predictions.shape == (2, 2)
        assert scores.shape == (2, 2)
        assert ((predictions >= 0) & (predictions < kg.edge_count)).all()

    def test_top_k_larger_than_edge_count_raises(self, kg):
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        inference = EdgeInference(kg)
        with pytest.raises(AssertionError, match="top_k"):
            inference.evaluate(
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
                top_k=3,
                batch_size=2,
                encoder=None,
                decoder=TransE(),
                node_embeddings=node_embeddings,
                edge_embeddings=edge_embeddings,
                verbose=False,
            )


class TestNodeInference:
    def test_init(self, kg):
        inference = NodeInference(kg)
        assert inference.knowledge_graph is kg

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
        assert scores.shape == (2, 2)
        assert predictions.dtype == torch.long
        assert scores.dtype == torch.float
        # Valid node indices, ranked from best to worst score
        assert ((predictions >= 0) & (predictions < kg.node_count)).all()
        assert (scores[:, 0] >= scores[:, 1]).all()

    def test_evaluate_missing_head_without_encoder(self, kg):
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        inference = NodeInference(kg)
        predictions, scores = inference.evaluate(
            torch.tensor([2, 3]),
            torch.tensor([0, 1]),
            top_k=2,
            missing_triplet_part="head",
            batch_size=2,
            encoder=None,
            decoder=TransE(),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )
        assert predictions.shape == (2, 2)
        assert scores.shape == (2, 2)
        assert ((predictions >= 0) & (predictions < kg.node_count)).all()
        assert (scores[:, 0] >= scores[:, 1]).all()

    def test_evaluate_with_encoder(self, kg):
        torch.manual_seed(0)
        encoder = _make_encoder()
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        # The encoder path reads the embeddings from the knowledge graph itself.
        kg.embeddings = node_embeddings, edge_embeddings
        inference = NodeInference(kg)
        predictions, scores = inference.evaluate(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            top_k=2,
            missing_triplet_part="tail",
            batch_size=2,
            encoder=encoder,
            decoder=TransE(),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )
        assert predictions.shape == (2, 2)
        assert scores.shape == (2, 2)
        assert ((predictions >= 0) & (predictions < kg.node_count)).all()
        assert (scores[:, 0] >= scores[:, 1]).all()

    def test_last_smaller_batch(self, kg):
        # A batch count that doesn't divide the input length: the last batch
        # is smaller and must still be written at the correct position.
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        inference = NodeInference(kg)
        predictions, scores = inference.evaluate(
            torch.tensor([0, 1, 2, 3, 4]),
            torch.tensor([0, 1, 0, 1, 0]),
            top_k=2,
            missing_triplet_part="tail",
            batch_size=2,
            encoder=None,
            decoder=TransE(),
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )
        assert predictions.shape == (5, 2)
        assert scores.shape == (5, 2)
        assert torch.isfinite(scores).all()
