"""
Tests for kgate.knowledgegraph.

Uses small deterministic mock dataframes (see conftest.py) to keep tests
fast and memory-light.
"""

import pandas as pd
import pytest
import torch

from kgate.knowledgegraph import KnowledgeGraph

from conftest import (
    make_kg_dataframe,
    make_hetero_dataframe,
    make_hetero_metadata,
    add_embeddings,
)


class TestConstruction:
    def test_from_dataframe(self, kg):
        assert len(kg) == 16
        assert kg.node_count == 8
        assert kg.edge_count == 2
        assert kg.triplet_count == 16
        assert set(kg.node_type_to_index) == {"Node"}
        assert kg.node_type_to_index["Node"] == 0

    def test_graphindices(self, kg):
        assert kg.graphindices.shape == (4, 16)
        assert kg.graphindices.dtype == torch.long

    def test_getitem(self, kg):
        triplet = kg[0]
        assert triplet.shape == (4,)
        # First row of the dataframe: (n0, n1, E1)
        assert triplet.tolist() == [0, 1, 0, 0]

    def test_index_properties(self, kg):
        assert len(kg.head_indices) == 16
        assert len(kg.tail_indices) == 16
        assert len(kg.edge_indices) == 16
        assert len(kg.triplets) == 16
        # TorchKGE aliases
        torch.testing.assert_close(kg.tail_idx, kg.tail_indices)
        torch.testing.assert_close(kg.relations, kg.edge_indices)
        torch.testing.assert_close(kg.edge_list, kg.graphindices[:2])

    def test_from_hetero_dataframe(self, hetero_kg):
        kg = hetero_kg
        assert kg.node_count == 8
        assert len(kg) == 8
        assert set(kg.node_type_to_index) == {"A", "B"}
        assert kg.triplet_types[0] == ("A", "E1", "B")
        assert kg.triplet_types[1] == ("B", "E2", "A")
        # One node type per half of the graph
        assert len(kg.node_type_to_global["A"]) == 4
        assert len(kg.node_type_to_global["B"]) == 4

    def _make_torchkge_kg(self):
        # torchkge.KnowledgeGraph expects the columns "from", "to" and "rel"
        import torchkge
        df = pd.DataFrame({
            "from": ["a", "b", "c"],
            "to": ["b", "c", "a"],
            "rel": ["R0", "R1", "R0"],
        })
        return torchkge.KnowledgeGraph(
            df=df, ent2ix={"a": 0, "b": 1, "c": 2}, rel2ix={"R0": 0, "R1": 1}
        )

    def test_from_torchkge(self):
        tk_kg = self._make_torchkge_kg()
        kg = KnowledgeGraph.from_torchkge(tk_kg)
        assert len(kg) == 3
        assert kg.node_count == 3
        assert kg.edge_count == 2
        assert set(kg.node_type_to_index) == {"Node"}

    @pytest.mark.xfail(
        strict=True,
        reason="from_torchkge with metadata passes torchkge's `get_df()` "
               "(columns 'from'/'to'/'rel') directly to the KnowledgeGraph "
               "constructor, which requires 'head'/'tail'/'edge' columns "
               "(KeyError: 'head'). See fixes/knowledgegraph.py.txt.",
    )
    def test_from_torchkge_with_metadata(self):
        tk_kg = self._make_torchkge_kg()
        metadata = pd.DataFrame({"id": ["a", "b", "c"], "type": ["X", "Y", "X"]})
        kg = KnowledgeGraph.from_torchkge(tk_kg, metadata=metadata)
        assert set(kg.node_type_to_index) == {"X", "Y"}


class TestMasks:
    def test_generate_masks_by_proportions(self, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25))
        total = (kg.train_mask.sum().item()
                 + kg.validation_mask.sum().item()
                 + kg.test_mask.sum().item())
        assert total == 16
        # No triplet is in more than one split
        assert (kg.train_mask & kg.validation_mask).sum().item() == 0
        assert (kg.train_mask & kg.test_mask).sum().item() == 0
        assert (kg.validation_mask & kg.test_mask).sum().item() == 0
        # generate_masks guarantees that every node appears in the training set
        train_nodes = torch.cat([kg.head_indices[kg.train_mask],
                                 kg.tail_indices[kg.train_mask]]).unique()
        assert train_nodes.numel() == kg.node_count

    def test_generate_masks_by_sizes(self, kg):
        kg.generate_masks(split_proportions=(1, 0, 0), sizes=(10, 3, 3))
        assert kg.train_mask.sum().item() == 10
        assert kg.validation_mask.sum().item() == 3
        assert kg.test_mask.sum().item() == 3

    def test_generate_masks_invalid_proportions_raise(self, kg):
        with pytest.raises(AssertionError):
            kg.generate_masks(split_proportions=(0.5, 0.3, 0.1))

    def test_masks_are_mutually_exclusive(self, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25))
        combined = (kg.train_mask | kg.validation_mask | kg.test_mask)
        assert combined.sum().item() == 16
        assert (kg.train_mask & kg.validation_mask).sum().item() == 0
        assert (kg.train_mask & kg.test_mask).sum().item() == 0
        assert (kg.validation_mask & kg.test_mask).sum().item() == 0

    def test_get_mask(self, kg):
        train, validation, test = kg.get_mask((0.5, 0.25, 0.25))
        assert train.sum().item() == 8
        assert validation.sum().item() == 4
        assert test.sum().item() == 4

    def test_split_properties(self, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25))
        # The sets are graphindices-like tensors of shape [4, n]
        assert kg.train_set.shape[0] == 4
        assert kg.validation_set.shape[0] == 4
        assert kg.test_set.shape[0] == 4
        total = (kg.train_set.size(1) + kg.validation_set.size(1) + kg.test_set.size(1))
        assert total == 16


class TestEdgeManipulation:
    @pytest.mark.xfail(
        strict=True,
        reason="remove_duplicate_triplets passes `~indices_to_keep` to "
               "remove_triplets_from_training. `indices_to_keep` is a long tensor of "
               "index values, so `~` is a bitwise NOT producing negative long values "
               "(-1, -2, ...) that are interpreted as (negative) triplet positions "
               "instead of a boolean mask. See fixes/knowledgegraph.py.txt.",
    )
    def test_remove_duplicate_triplets(self):
        rows = [
            {"head": "n0", "tail": "n1", "edge": "E1"},
            {"head": "n0", "tail": "n1", "edge": "E1"},  # duplicate
            {"head": "n1", "tail": "n0", "edge": "E1"},  # reverse pair: same sorted pair
            {"head": "n0", "tail": "n2", "edge": "E2"},
            {"head": "n0", "tail": "n2", "edge": "E2"},  # duplicate
        ]
        kg = KnowledgeGraph(dataframe=pd.DataFrame(rows))
        kg.generate_masks((1, 0, 0))
        kg.remove_duplicate_triplets()
        # The duplicate triplets are removed from the training set
        # (one triplet per unique head/tail pair per edge)
        assert kg.train_mask.sum().item() == 2
        # The triplets themselves are still present in the knowledge graph
        assert len(kg) == 5

    def test_add_reverse_edges(self, kg):
        kg.generate_masks((1, 0, 0))
        original_count = len(kg)
        reverse_list = kg.add_reverse_edges(undirected_edges=[0])
        # The reverse edge is appended after the existing edges (index 2)
        assert (0, 2) in reverse_list
        # New reverse edge type exists
        assert kg.edge_count == 3
        # Triplets added: for each E1 triplet, (A, E1_rev, B) and (B, E1, A)
        assert len(kg) > original_count
        assert ("Node", "E1_rev", "Node") in kg.triplet_types

    def test_get_pairs(self, kg):
        pairs = kg.get_pairs(0)
        assert pairs.shape == (2, 8)  # 8 triplets of edge type E1
        # Check first pair: (n0, n1)
        assert pairs[:, 0].tolist() == [0, 1]

    def test_get_pairs_with_split(self, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25))
        pairs = kg.get_pairs(0, split="train")
        assert pairs.shape[0] == 2
        expected = kg.train_mask & (kg.edge_indices == 0)
        assert pairs.shape[1] == expected.sum().item()

    def test_duplicates_identical_edges(self):
        # Two edges with exactly the same pairs
        rows = [
            {"head": f"n{i}", "tail": f"n{(i + 1) % 4}", "edge": "E1"} for i in range(4)
        ] + [
            {"head": f"n{i}", "tail": f"n{(i + 1) % 4}", "edge": "E2"} for i in range(4)
        ]
        kg = KnowledgeGraph(dataframe=pd.DataFrame(rows))
        duplicates, reverse_duplicates = kg.duplicates(0.8, 0.8)
        assert (0, 1) in duplicates

    def test_duplicates_reverse_edges(self):
        # E2 pairs are the reverse of E1 pairs
        rows = [
            {"head": f"n{i}", "tail": f"n{(i + 1) % 4}", "edge": "E1"} for i in range(4)
        ] + [
            {"head": f"n{(i + 1) % 4}", "tail": f"n{i}", "edge": "E2"} for i in range(4)
        ]
        kg = KnowledgeGraph(dataframe=pd.DataFrame(rows))
        duplicates, reverse_duplicates = kg.duplicates(0.8, 0.8)
        assert (0, 1) in reverse_duplicates

    def test_cartesian_product_edges(self):
        # E1 is a complete bipartite between {n0, n1} and {n2, n3}
        rows = [
            {"head": h, "tail": t, "edge": "E1"}
            for h in ["n0", "n1"]
            for t in ["n2", "n3"]
        ]
        # E2 is sparse: 2 triplets over a 2x2 node set
        rows.append({"head": "n0", "tail": "n2", "edge": "E2"})
        rows.append({"head": "n1", "tail": "n3", "edge": "E2"})
        kg = KnowledgeGraph(dataframe=pd.DataFrame(rows))
        selected = kg.cartesian_product_edges(theta=0.8)
        # 4 triplets / (2 heads * 2 tails) = 1.0 > 0.8
        assert 0 in selected
        # 2 triplets / (2 heads * 2 tails) = 0.5, not > 0.8
        assert 1 not in selected


class TestMetadataAndIdentity:
    def test_identity_default_is_node_id(self, hetero_kg):
        # Note: the property docstring says DataFrame, but it returns a Series
        # (a single metadata column).
        identity = hetero_kg.identity
        assert isinstance(identity, pd.Series)
        assert identity.tolist() == [f"n{i}" for i in range(8)]

    def test_set_identity_requires_metadata(self, kg):
        with pytest.raises(AssertionError):
            kg.set_identity("type")

    def test_set_identity_invalid_column(self, hetero_kg):
        with pytest.raises(AssertionError):
            hetero_kg.set_identity("nonexistent")

    def test_set_identity(self, hetero_kg):
        metadata = make_hetero_metadata()
        metadata["name"] = [f"name{i}" for i in range(8)]
        hetero_kg.add_metadata(metadata)
        hetero_kg.set_identity("name")
        assert hetero_kg.identity.tolist() == [f"name{i}" for i in range(8)]

    def test_add_metadata_merge(self, hetero_kg):
        extra = pd.DataFrame({
            "id": [f"n{i}" for i in range(8)],
            "color": ["red"] * 8,
        })
        hetero_kg.add_metadata(extra)
        assert "color" in hetero_kg.metadata.columns

    def test_get_dataframe(self, kg):
        dataframe = kg.get_dataframe()
        assert list(dataframe.columns) == ["head", "tail", "edge"]
        assert len(dataframe) == 16

    @pytest.mark.xfail(
        strict=True,
        reason="get_dataframe(include_splits=True) uses chained assignment "
               "`dataframe['split'][self.train_mask] = 'train'` with a raw torch "
               "boolean tensor, which modern pandas rejects (KeyError). The "
               "assignment should use `dataframe.loc[mask, 'split']`. "
               "See fixes/knowledgegraph.py.txt.",
    )
    def test_get_dataframe_with_splits(self, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25))
        dataframe = kg.get_dataframe(include_splits=True)
        assert "split" in dataframe.columns
        assert set(dataframe["split"].unique()) == {"train", "validation", "test"}


class TestTripletModification:
    @pytest.mark.xfail(
        strict=True,
        reason="delete_triplets applies the boolean mask `graphindices != -1` "
               "(shape [4, n]) to the 2-D graphindices tensor, collapsing it to a "
               "1-D tensor. Any subsequent `len(kg)` call raises IndexError. "
               "See fixes/knowledgegraph.py.txt.",
    )
    def test_delete_triplets(self, kg):
        kg.generate_masks((1, 0, 0))
        original = len(kg)
        kg.delete_triplets([0, 1])
        assert len(kg) == original - 2

    def test_remove_triplets_from_training(self, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25))
        train_before = kg.train_mask.sum().item()
        # Remove two triplets that are actually in the training split
        to_remove = kg.train_mask.nonzero(as_tuple=False)[:, 0][:2].tolist()
        kg.remove_triplets_from_training(to_remove)
        assert kg.train_mask.sum().item() == train_before - 2
        assert len(kg) == 16  # triplets still exist in the KG

    def test_add_triplets(self, kg):
        kg.generate_masks((1, 0, 0))
        new_triplets = torch.tensor([
            [0, 2, 0, 0],
            [1, 3, 1, 1],
        ]).T
        original = len(kg)
        kg.add_triplets(new_triplets, split="train")
        assert len(kg) == original + 2
        assert kg.train_mask[-2:].sum().item() == 2

    def test_add_triplets_invalid_shape(self, kg):
        with pytest.raises(AssertionError):
            kg.add_triplets(torch.randn(3, 4))

    def test_add_triplets_unknown_node(self, kg):
        with pytest.raises(ValueError):
            kg.add_triplets(torch.tensor([[0, 99, 0, 0]]).T)


class TestEncoderInput:
    def test_get_encoder_input(self, embedded_kg):
        kg = embedded_kg
        input = kg.get_encoder_input(seed_nodes=torch.tensor([0, 1]), hop_count=1)
        assert "Node" in input.x_dict
        # All 8 nodes are reachable in 1 hop (both rings are connected)
        assert input.x_dict["Node"].shape[0] == 8

    def test_get_encoder_input_requires_embeddings(self, kg):
        with pytest.raises((AssertionError, AttributeError, IndexError)):
            kg.get_encoder_input(seed_nodes=torch.tensor([0]), hop_count=1)

    def test_get_encoder_input_with_mask(self, embedded_kg):
        kg = embedded_kg
        mask = torch.zeros(16, dtype=torch.bool)
        mask[0] = True  # only triplet (n0, n1, E1)
        input = kg.get_encoder_input(seed_nodes=torch.tensor([0]), hop_count=1, mask=mask)
        assert "Node" in input.x_dict


class TestCleaning:
    def test_flatten_embeddings(self, embedded_kg):
        embeddings = embedded_kg.flatten_embeddings()
        assert embeddings.shape == (8, 4)

    def test_clean_removes_self_loop_edge_type(self, embedded_kg):
        kg = embedded_kg
        # Simulate an added self-loop triplet type
        kg.triplet_types.append(("Node", "self", "Node"))
        kg.clean()
        assert ("Node", "self", "Node") not in kg.triplet_types
