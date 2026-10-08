"""
Tests for kgate.data_leakage.

`permute_tails` mutates the knowledge graph in place and returns it.
The multi-node-type path contains a known off-by-one bug, exposed with
xfail(strict=True) and documented in fixes/data_leakage.txt.
"""

import random
from collections import Counter

import pandas as pd
import pytest
import torch

from kgate.data_leakage import permute_tails
from kgate.knowledgegraph import KnowledgeGraph


class TestPermuteTails:
    def test_single_type_preserves_degree(self, kg):
        random.seed(0)
        mask = kg.edge_indices == 0
        original_heads = kg.head_indices[mask].clone()
        original_tail_counts = Counter(kg.tail_indices[mask].tolist())
        original_e2_tails = kg.tail_indices[~mask].clone()

        result = permute_tails(kg, "E1", preserve_node_degree=True)

        assert result is kg
        # Heads are untouched
        torch.testing.assert_close(kg.head_indices[mask], original_heads)
        # Tail degree distribution is preserved for the permuted edge
        assert Counter(kg.tail_indices[mask].tolist()) == original_tail_counts
        # Other edges are untouched
        torch.testing.assert_close(kg.tail_indices[~mask], original_e2_tails)
        # No self-loop anywhere
        assert all(h != t for h, t in zip(kg.head_indices.tolist(),
                                          kg.tail_indices.tolist()))

    def test_single_type_no_self_loops_across_seeds(self, kg):
        # The swap-based self-loop correction must work for a variety of shuffles
        for seed in range(5):
            kg2 = KnowledgeGraph(dataframe=pd.DataFrame({
                "head": [f"n{i}" for i in range(8)],
                "tail": [f"n{(i + 1) % 8}" for i in range(8)],
                "edge": ["E1"] * 8,
            }))
            random.seed(seed)
            permute_tails(kg2, "E1", preserve_node_degree=True)
            assert all(h != t for h, t in zip(kg2.head_indices.tolist(),
                                              kg2.tail_indices.tolist())), f"self-loop with seed {seed}"

    def test_preserve_node_degree_false(self, kg):
        random.seed(0)
        result = permute_tails(kg, "E1", preserve_node_degree=False)
        assert result is kg
        # Tails are re-sampled, so degree is not necessarily preserved,
        # but the number of triplets per edge must stay the same
        assert result.triplet_count == kg.triplet_count

    def test_returns_same_object(self, kg):
        random.seed(0)
        result = permute_tails(kg, "E2", preserve_node_degree=True)
        assert result is kg

    @staticmethod
    def _make_multi_type_kg():
        """
        4 node types (A, C heads; B, D tails) with edge E1 carrying
        (A->B) and (C->D) triplets only. A shuffle that crosses the
        type blocks creates brand-new triplet types.
        """
        df = pd.DataFrame({
            "head": ["a0", "a1", "c0", "c1"],
            "tail": ["b0", "b1", "d0", "d1"],
            "edge": ["E1"] * 4,
        })
        metadata = pd.DataFrame({
            "id": ["a0", "a1", "c0", "c1", "b0", "b1", "d0", "d1"],
            "type": ["A", "A", "C", "C", "B", "B", "D", "D"],
        })
        return KnowledgeGraph(dataframe=df, metadata=metadata)

    def test_multi_type_existing_combinations(self):
        # A seed whose shuffle keeps every head inside its original tail type
        kg = self._make_multi_type_kg()
        random.seed(9)  # safe shuffle [4, 5, 6, 7]
        result = permute_tails(kg, "E1", preserve_node_degree=True)
        assert result is kg
        assert (result.graphindices[3] >= 0).all()
        assert (result.graphindices[3] < len(result.triplet_types)).all()

    @pytest.mark.xfail(
        strict=True,
        reason="Off-by-one: newly created triplet types are indexed with "
               "`len(triplets_types)` after the `append` instead of "
               "`len(triplets_types) - 1`, producing out-of-range triplet "
               "type indices. See fixes/data_leakage.txt.",
    )
    def test_multi_type_new_combination_off_by_one(self):
        kg = self._make_multi_type_kg()
        assert kg.triplet_types == [("A", "E1", "B"), ("C", "E1", "D")]
        random.seed(0)  # shuffle [6, 4, 5, 7] -> creates (A,E1,D) and (C,E1,B)
        result = permute_tails(kg, "E1", preserve_node_degree=True)
        # All triplet type indices must stay within the valid range
        assert (result.graphindices[3] >= 0).all()
        assert (result.graphindices[3] < len(result.triplet_types)).all()
