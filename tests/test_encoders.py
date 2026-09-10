"""
Tests for kgate.encoders (GNN interface, GATEncoder, GCNEncoder).

The encoders operate on PyTorch Geometric heterogeneous inputs:
- ``x_dict``: {node_type: [node_count, embedding_dimensions]}
- ``edge_index_dict``: {(head_type, edge_type, tail_type): [2, edge_count]}

Note: GNN.__init__ mutates the given edge_types list in place by appending
a "self" edge type for every node type. Tests therefore pass a fresh list
each time.
"""

import pytest
import torch

from kgate.encoders import GNN, GATEncoder, GCNEncoder


def _edge_types():
    """Return a fresh edge_types list (GNN mutates it in place)."""
    return [("Node", "E1", "Node"), ("Node", "E2", "Node")]


def _hetero_input(dim=4, node_count=3):
    """Minimal heterogeneous graph: 3 nodes, two real edge types + self-loops."""
    x_dict = {"Node": torch.randn(node_count, dim)}
    edge_index_dict = {
        ("Node", "E1", "Node"): torch.tensor([[0, 1, 2], [1, 2, 0]]),
        ("Node", "E2", "Node"): torch.tensor([[0, 1], [1, 0]]),
        ("Node", "self", "Node"): torch.arange(node_count).repeat_interleave(1).unsqueeze(0).repeat(2, 1),
    }
    return x_dict, edge_index_dict


class TestGNN:
    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Aggregation must be within"):
            GNN(_edge_types(), aggregation="product")

    def test_default_aggregation_is_sum(self):
        gnn = GNN(_edge_types())
        assert gnn.aggregation == "sum"

    def test_aggregation_none_is_allowed(self):
        gnn = GNN(_edge_types(), aggregation=None)
        assert gnn.aggregation is None

    def test_self_loop_edge_types_are_appended(self):
        edge_types = [("A", "E1", "B"), ("B", "E2", "C")]
        gnn = GNN(edge_types)
        assert ("A", "self", "A") in gnn.edge_types
        assert ("B", "self", "B") in gnn.edge_types
        assert ("C", "self", "C") in gnn.edge_types

    def test_forward_without_convolutions_is_identity(self):
        gnn = GNN(_edge_types())
        x_dict, edge_index_dict = _hetero_input()
        out = gnn(x_dict, edge_index_dict)
        # No convolution layer means the input is returned unchanged.
        assert torch.equal(out["Node"], x_dict["Node"])

    def test_no_convolutions_by_default(self):
        gnn = GNN(_edge_types())
        assert len(gnn.convolutions) == 0


class TestGATEncoder:
    def test_init(self):
        encoder = GATEncoder(_edge_types(),
                             embedding_dimensions=4,
                             gat_layer_count=2,
                             aggregation="sum",
                             device="cpu")
        assert encoder.layer_count == 2
        assert len(encoder.convolutions) == 2
        # NB: the `device` argument only moves the parameters; the inherited
        # `GNN.device` attribute stays hardcoded to "cuda" (see
        # fixes/encoders.txt). Check the actual parameter device instead.
        assert next(encoder.parameters()).device.type == "cpu"

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Aggregation must be within"):
            GATEncoder(_edge_types(), 4, aggregation="median")

    def test_forward_output_shape(self):
        encoder = GATEncoder(_edge_types(),
                             embedding_dimensions=4,
                             gat_layer_count=1,
                             device="cpu")
        x_dict, edge_index_dict = _hetero_input(dim=4)
        out = encoder(x_dict, edge_index_dict)
        assert "Node" in out
        assert out["Node"].shape == (3, 4)

    def test_forward_two_layers(self):
        encoder = GATEncoder(_edge_types(),
                             embedding_dimensions=4,
                             gat_layer_count=2,
                             device="cpu")
        x_dict, edge_index_dict = _hetero_input(dim=4)
        out = encoder(x_dict, edge_index_dict)
        assert out["Node"].shape == (3, 4)
        assert torch.isfinite(out["Node"]).all()


class TestGCNEncoder:
    def test_init(self):
        encoder = GCNEncoder(_edge_types(),
                             embedding_dimensions=4,
                             gcn_layer_count=2,
                             aggregation="sum",
                             device="cpu")
        assert encoder.layer_count == 2
        assert len(encoder.convolutions) == 2
        # See fixes/encoders.txt: `GNN.device` is hardcoded to "cuda".
        assert next(encoder.parameters()).device.type == "cpu"

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Aggregation must be within"):
            GCNEncoder(_edge_types(), 4, aggregation="median")

    def test_forward_output_shape(self):
        encoder = GCNEncoder(_edge_types(),
                             embedding_dimensions=4,
                             gcn_layer_count=1,
                             device="cpu")
        x_dict, edge_index_dict = _hetero_input(dim=4)
        out = encoder(x_dict, edge_index_dict)
        assert "Node" in out
        assert out["Node"].shape == (3, 4)

    def test_forward_two_layers(self):
        encoder = GCNEncoder(_edge_types(),
                             embedding_dimensions=4,
                             gcn_layer_count=2,
                             device="cpu")
        x_dict, edge_index_dict = _hetero_input(dim=4)
        out = encoder(x_dict, edge_index_dict)
        assert out["Node"].shape == (3, 4)
        assert torch.isfinite(out["Node"]).all()
