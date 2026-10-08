"""
Tests for kgate.preprocessing.

Known bugs (documented in fixes/preprocessing.py.txt and exposed with
xfail(strict=True)):
- `save_knowledge_graph` with the default pickle path: the
  `knowledge_graph_pickle_file` property returns a Path, so the
  `== ""` check never matches and `Path(".")` is used as filename
  (IsADirectoryError).
- `clean_knowledge_graph` references the nonexistent configuration
  attribute `preprocessing.split` (should be `split_proportions`)
  -> AttributeError, breaking `prepare_knowledge_graph`.
- The second direction of `clean_datasets` computes the triplets to
  remove but never removes them (only logs them).
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from kgate.config import Configuration
from kgate.knowledgegraph import KnowledgeGraph
from kgate.preprocessing import (
    prepare_knowledge_graph,
    save_knowledge_graph,
    load_knowledge_graph,
    clean_knowledge_graph,
    verify_node_coverage,
    ensure_node_coverage,
    clean_datasets,
    clean_cartesians,
)


def make_three_node_kg():
    df = pd.DataFrame({
        "head": ["a", "b", "a"],
        "tail": ["b", "c", "c"],
        "edge": ["E", "E", "E"],
    })
    return KnowledgeGraph(dataframe=df)


class TestSaveLoadKnowledgeGraph:
    @pytest.mark.xfail(
        strict=True,
        reason="`knowledge_graph_pickle_file` returns a Path, so the "
               "`== ''` check never matches and `Path('.')` is used as the "
               "pickle filename (IsADirectoryError). "
               "See fixes/preprocessing.py.txt.",
    )
    def test_save_default_path(self, small_config, kg):
        save_knowledge_graph(small_config, kg)
        expected = Path(small_config.output_directory) / "kg.pkl"
        assert expected.exists()

    def test_save_explicit_path(self, tmp_path, kg):
        custom = tmp_path / "custom.pkl"
        custom.touch()
        config = Configuration(config_dict={
            "output_directory": str(tmp_path / "out"),
            "kg_pkl": str(custom),
        })
        save_knowledge_graph(config, kg)
        loaded = load_knowledge_graph(custom)
        assert isinstance(loaded, KnowledgeGraph)
        torch.testing.assert_close(loaded.graphindices, kg.graphindices)
        assert loaded.node_to_index == kg.node_to_index
        assert loaded.edge_to_index == kg.edge_to_index


class TestVerifyNodeCoverage:
    def test_missing_node_in_train(self):
        kg = make_three_node_kg()
        kg.train_mask = torch.tensor([True, False, False])
        kg.validation_mask = torch.tensor([False, True, False])
        kg.test_mask = torch.tensor([False, False, True])

        ok, missing = verify_node_coverage(kg)
        assert ok is False
        assert missing == ["c"]

    def test_all_nodes_covered(self):
        kg = make_three_node_kg()
        kg.train_mask = torch.ones(3, dtype=torch.bool)
        kg.validation_mask = torch.zeros(3, dtype=torch.bool)
        kg.test_mask = torch.zeros(3, dtype=torch.bool)

        ok, missing = verify_node_coverage(kg)
        assert ok is True
        assert missing == []


class TestEnsureNodeCoverage:
    def test_moves_triplet_to_train(self):
        kg = make_three_node_kg()
        # Node "c" is missing from the training set
        kg.train_mask = torch.tensor([True, False, False])
        kg.validation_mask = torch.tensor([False, True, False])
        kg.test_mask = torch.tensor([False, False, True])

        ensure_node_coverage(kg)

        # Triplet 1 (b, c) is moved to the training set
        assert kg.train_mask.tolist() == [True, True, False]
        assert kg.validation_mask.tolist() == [False, False, False]
        assert kg.test_mask.tolist() == [False, False, True]

        ok, missing = verify_node_coverage(kg)
        assert ok is True
        assert missing == []


class TestCleanDatasets:
    @pytest.fixture
    def leaking_kg(self):
        """
        DataFrame rows:
        t0: (a, b, E0) validation
        t1: (a, b, E1) train  -> leaks (matches t0 pair), must be removed
        t2: (c, a, E0) train  -> leaks in the reverse direction (matches t3 pair)
        t3: (c, a, E1) test

        Note: the KnowledgeGraph constructor groups triplets by edge, so the
        internal column order is [t0, t2, t1, t3]. Assertions are therefore
        made on the (head, tail, edge) contents of the training set, not on
        raw mask positions.
        """
        df = pd.DataFrame({
            "head": ["a", "a", "c", "c"],
            "tail": ["b", "b", "a", "a"],
            "edge": ["E0", "E1", "E0", "E1"],
        })
        kg = KnowledgeGraph(dataframe=df)
        kg.train_mask = torch.tensor([False, True, True, False])
        kg.validation_mask = torch.tensor([True, False, False, False])
        kg.test_mask = torch.tensor([False, False, False, True])
        return kg

    def test_removes_leaking_train_triplets(self, leaking_kg):
        clean_datasets(leaking_kg, known_reverses=[(0, 1)])
        train_triplets = leaking_kg.graphindices[:3,
                                                 leaking_kg.train_mask].T.tolist()
        # (a, b, E1) is removed from training (node order a=0, b=1, c=2)
        assert [0, 1, 1] not in train_triplets
        # The reverse-direction leak (c, a, E0) stays in training (see the
        # xfail test below for the missing removal)
        assert [2, 0, 0] in train_triplets
        # Other splits are untouched
        assert leaking_kg.validation_mask.tolist() == [True, False, False, False]
        assert leaking_kg.test_mask.tolist() == [False, False, False, True]

    @pytest.mark.xfail(
        strict=True,
        reason="The second direction of `clean_datasets` computes the "
               "triplets to remove but never calls "
               "`remove_triplets_from_training` (it only logs them). "
               "See fixes/preprocessing.py.txt.",
    )
    def test_removes_reverse_leaking_triplets(self, leaking_kg):
        clean_datasets(leaking_kg, known_reverses=[(0, 1)])
        # t2 (c, a, E0) should also be removed from training
        assert leaking_kg.train_mask.tolist() == [False, False, False, False]

    def test_no_known_reverses_is_noop(self, leaking_kg):
        clean_datasets(leaking_kg, known_reverses=[])
        assert leaking_kg.train_mask.tolist() == [False, True, True, False]


class TestCleanCartesians:
    @pytest.mark.xfail(
        strict=True,
        reason="clean_cartesians computes the indices of the train triplets "
               "to remove relative to `train_set` (the filtered columns), but "
               "passes them to remove_triplets_from_training, which interprets "
               "them as GLOBAL graphindices positions. When the training set "
               "does not occupy the first global positions, the wrong "
               "triplets are removed (or nothing at all). "
               "See fixes/preprocessing.py.txt.",
    )
    def test_head_position(self):
        df = pd.DataFrame({
            "head": ["a", "a", "b"],
            "tail": ["x", "y", "y"],
            "edge": ["E0", "E0", "E0"],
        })
        kg = KnowledgeGraph(dataframe=df)
        # t0: (a, x) test; t1: (a, y) train; t2: (b, y) train
        kg.train_mask = torch.tensor([False, True, True])
        kg.validation_mask = torch.zeros(3, dtype=torch.bool)
        kg.test_mask = torch.tensor([True, False, False])

        clean_cartesians(kg, known_cartesian=[0], node_position="head")

        # t1 shares its head with a test triplet -> removed from training
        assert kg.train_mask.tolist() == [False, False, True]
        # The test split is untouched
        assert kg.test_mask.tolist() == [True, False, False]

    @pytest.mark.xfail(
        strict=True,
        reason="Same local-vs-global index bug as test_head_position. "
               "See fixes/preprocessing.py.txt.",
    )
    def test_tail_position(self):
        df = pd.DataFrame({
            "head": ["x", "y", "w"],
            "tail": ["z", "z", "v"],
            "edge": ["E0", "E0", "E0"],
        })
        kg = KnowledgeGraph(dataframe=df)
        # t0: (x, z) test; t1: (y, z) train; t2: (w, v) train
        kg.train_mask = torch.tensor([False, True, True])
        kg.validation_mask = torch.zeros(3, dtype=torch.bool)
        kg.test_mask = torch.tensor([True, False, False])

        clean_cartesians(kg, known_cartesian=[0], node_position="tail")

        assert kg.train_mask.tolist() == [False, False, True]
        assert kg.test_mask.tolist() == [True, False, False]

    def test_head_position_aligned(self):
        # When the training set occupies the first global positions, the
        # local and global indices coincide and the function works.
        df = pd.DataFrame({
            "head": ["b", "a", "a"],
            "tail": ["y", "y", "x"],
            "edge": ["E0", "E0", "E0"],
        })
        kg = KnowledgeGraph(dataframe=df)
        # t0: (b, y) train; t1: (a, y) train; t2: (a, x) test
        kg.train_mask = torch.tensor([True, True, False])
        kg.validation_mask = torch.zeros(3, dtype=torch.bool)
        kg.test_mask = torch.tensor([False, False, True])

        clean_cartesians(kg, known_cartesian=[0], node_position="head")

        # t1 shares its head (a) with the test triplet -> removed
        assert kg.train_mask.tolist() == [True, False, False]
        assert kg.test_mask.tolist() == [False, False, True]

    def test_invalid_node_position(self, kg):
        kg.train_mask = torch.ones(kg.triplet_count, dtype=torch.bool)
        kg.validation_mask = torch.zeros(kg.triplet_count, dtype=torch.bool)
        kg.test_mask = torch.zeros(kg.triplet_count, dtype=torch.bool)
        with pytest.raises(AssertionError):
            clean_cartesians(kg, known_cartesian=[0], node_position="side")


class TestPrepareKnowledgeGraph:
    @pytest.mark.xfail(
        strict=True,
        reason="`clean_knowledge_graph` references the nonexistent "
               "configuration attribute `preprocessing.split` (should be "
               "`split_proportions`) -> AttributeError. "
               "See fixes/preprocessing.py.txt.",
    )
    def test_prepare_from_dataframe(self, small_config, kg_dataframe):
        knowledge_graph = prepare_knowledge_graph(small_config, dataframe=kg_dataframe)
        assert isinstance(knowledge_graph, KnowledgeGraph)
        assert knowledge_graph.train_mask.sum() > 0
        assert knowledge_graph.validation_mask.sum() > 0
        assert knowledge_graph.test_mask.sum() > 0

    @pytest.mark.xfail(
        strict=True,
        reason="`clean_knowledge_graph` crashes with AttributeError on the "
               "nonexistent `preprocessing.split` attribute. "
               "See fixes/preprocessing.py.txt.",
    )
    def test_clean_knowledge_graph(self, small_config, kg):
        clean_knowledge_graph(small_config, kg)
        assert kg.train_mask.sum() > 0
