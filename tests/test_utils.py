"""
Tests for kgate.utils.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import random
import torch

from kgate.knowledgegraph import KnowledgeGraph
from kgate import utils


class TestSetRandomSeeds:
    def test_reproducibility(self):
        utils.set_random_seeds(42)
        torch_value = torch.rand(4).tolist()
        python_value = random.random()
        numpy_value = np.random.rand()

        utils.set_random_seeds(42)
        assert torch.rand(4).tolist() == torch_value
        assert random.random() == python_value
        assert np.random.rand() == numpy_value


class TestComputeTripletProportions:
    def test_proportions_per_edge(self, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25))
        proportions = utils.compute_triplet_proportions(kg)
        # Both edges are present
        assert set(proportions.keys()) == {0, 1}
        for edge in proportions.values():
            assert set(edge.keys()) == {"train", "test", "validation"}
            for value in edge.values():
                assert 0.0 <= value <= 1.0
            # Every triplet belongs to exactly one split
            assert pytest.approx(sum(edge.values()), abs=1e-5) == 1.0


class TestFindBestModel:
    def test_finds_best_by_validation_metric(self, tmp_path):
        (tmp_path / "best_model_checkpoint_validation_metric_value=0.42.pt").touch()
        (tmp_path / "best_model_checkpoint_validation_metric_value=0.87.pt").touch()
        (tmp_path / "best_model_checkpoint_validation_metric_value=0.63.pt").touch()
        result = utils.find_best_model(tmp_path)
        assert result is not None
        assert result.name == "best_model_checkpoint_validation_metric_value=0.87.pt"

    def test_fallback_to_latest_checkpoint(self, tmp_path):
        (tmp_path / "checkpoint_3.pt").touch()
        (tmp_path / "checkpoint_10.pt").touch()
        (tmp_path / "checkpoint_7.pt").touch()
        result = utils.find_best_model(tmp_path)
        assert result is not None
        assert result.name == "checkpoint_10.pt"

    def test_empty_directory_returns_none(self, tmp_path):
        result = utils.find_best_model(tmp_path)
        assert result is None


class TestReadTrainMetrics:
    def test_filters_restarts_and_deduplicates(self, tmp_path):
        file = tmp_path / "train_metrics.csv"
        file.write_text(
            "Epoch,Training Loss,Validation Mean rank\n"
            "0,1.0,0.2\n"
            "CHECKPOINT RESTART,,\n"
            "1,0.8,0.3\n"
            "1,0.7,0.35\n"
            "0,1.1,0.1\n"
        )
        df = utils.read_train_metrics(file)
        assert list(df["Epoch"]) == [0, 1]
        # Duplicates keep the last occurrence
        assert df.loc[df["Epoch"] == 0, "Training Loss"].iloc[0] == pytest.approx(1.1)
        assert df.loc[df["Epoch"] == 1, "Training Loss"].iloc[0] == pytest.approx(0.7)


class TestPlotLearningCurves:
    def test_creates_plots(self, tmp_path):
        metrics_file = tmp_path / "train_metrics.csv"
        metrics_file.write_text(
            "Epoch,Training Loss,Validation Mean rank\n"
            "0,1.0,0.2\n"
            "1,0.8,0.3\n"
            "2,0.6,0.4\n"
        )
        utils.plot_learning_curves(metrics_file, tmp_path, "Mean rank")
        assert (tmp_path / "training_loss_curve.png").exists()
        assert (tmp_path / "validation_metric_curve.png").exists()


class TestFilterScores:
    @pytest.fixture
    def shared_tail_graphindices(self):
        # Two triplets sharing a tail and an edge: (h0, t0, e0), (h1, t0, e0)
        return torch.tensor([[0, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.long)

    @pytest.fixture
    def shared_head_graphindices(self):
        # Two triplets sharing a head and an edge: (h0, t0, e0), (h0, t1, e0)
        return torch.tensor([[0, 0], [0, 1], [0, 0], [0, 0]], dtype=torch.long)

    def test_filter_head_prediction(self, shared_tail_graphindices):
        scores = torch.ones(1, 2)
        filtered = utils.filter_scores(
            scores, shared_tail_graphindices,
            missing="head",
            first_index=torch.tensor([0]),   # tail being predicted
            second_index=torch.tensor([0]),  # edge being predicted
            true_index=torch.tensor([0])     # true head
        )
        # The other true head (h1) is filtered out, the true head is kept
        assert torch.isinf(filtered[0, 1]) and filtered[0, 1] < 0
        assert filtered[0, 0] == pytest.approx(1.0)

    def test_filter_tail_prediction(self, shared_head_graphindices):
        # Predicting the tail of (h0, t?, e0)
        scores = torch.ones(1, 2)
        filtered = utils.filter_scores(
            scores, shared_head_graphindices,
            missing="tail",
            first_index=torch.tensor([0]),   # head being predicted
            second_index=torch.tensor([0]),  # edge being predicted
            true_index=torch.tensor([0])     # true tail
        )
        assert torch.isinf(filtered[0, 1]) and filtered[0, 1] < 0
        assert filtered[0, 0] == pytest.approx(1.0)

    def test_filter_without_true_index_filters_all(self, shared_tail_graphindices):
        scores = torch.ones(1, 2)
        filtered = utils.filter_scores(
            scores, shared_tail_graphindices,
            missing="head",
            first_index=torch.tensor([0]),
            second_index=torch.tensor([0]),
            true_index=None
        )
        assert torch.isinf(filtered).all()

    def test_no_matching_triplets_leaves_scores_untouched(self, shared_tail_graphindices):
        scores = torch.ones(1, 2)
        filtered = utils.filter_scores(
            scores, shared_tail_graphindices,
            missing="head",
            first_index=torch.tensor([5]),   # unknown tail
            second_index=torch.tensor([0]),
            true_index=torch.tensor([0])
        )
        torch.testing.assert_close(filtered, scores)


class TestMergeKg:
    @pytest.fixture
    def shared_mappings(self):
        return {
            "node_to_index": {"A": 0, "B": 1, "C": 2, "D": 3},
            "edge_to_index": {"E1": 0},
            "node_type_to_index": {"Node": 0},
            "triplet_types": [("Node", "E1", "Node")],
        }

    def test_merge_two_kgs(self, shared_mappings):
        kg1 = KnowledgeGraph(
            graphindices=torch.tensor([[0, 1], [1, 2], [0, 0], [0, 0]]),
            **shared_mappings
        )
        kg2 = KnowledgeGraph(
            graphindices=torch.tensor([[2, 3], [3, 0], [0, 0], [0, 0]]),
            **shared_mappings
        )
        merged = utils.merge_kg([kg1, kg2])
        assert merged.triplet_count == 4
        assert merged.node_to_index == shared_mappings["node_to_index"]
        assert merged.edge_to_index == shared_mappings["edge_to_index"]

    def test_merge_asserts_same_node_mapping(self, shared_mappings):
        kg1 = KnowledgeGraph(
            graphindices=torch.tensor([[0, 1], [1, 2], [0, 0], [0, 0]]),
            **shared_mappings
        )
        other_mappings = dict(shared_mappings)
        other_mappings["node_to_index"] = {"A": 0, "B": 1, "C": 2, "X": 3}
        kg2 = KnowledgeGraph(
            graphindices=torch.tensor([[2, 3], [3, 0], [0, 0], [0, 0]]),
            **other_mappings
        )
        with pytest.raises(AssertionError):
            utils.merge_kg([kg1, kg2])


class TestGetDictionaryMapping:
    def test_node_mapping(self):
        df = pd.DataFrame({
            "head": ["B", "A", "C"],
            "tail": ["A", "C", "B"],
            "edge": ["E2", "E1", "E1"],
        })
        mapping = utils.get_dictionary_mapping(df, nodes=True)
        assert mapping == {"A": 0, "B": 1, "C": 2}

    def test_edge_mapping(self):
        df = pd.DataFrame({
            "head": ["B", "A", "C"],
            "tail": ["A", "C", "B"],
            "edge": ["E2", "E1", "E1"],
        })
        mapping = utils.get_dictionary_mapping(df, nodes=False)
        assert mapping == {"E1": 0, "E2": 1}


class TestDegreeStatistics:
    def test_average_heads_per_tail_ring(self, kg):
        result = utils.get_average_heads_per_tail(kg.graphindices)
        assert result == {0.0: 1.0, 1.0: 1.0}

    def test_average_tails_per_head_ring(self, kg):
        result = utils.get_average_tails_per_head(kg.graphindices)
        assert result == {0.0: 1.0, 1.0: 1.0}

    def test_asymmetric_graph(self):
        # Tail t0 has two heads (h0, h1); tail t1 has one head (h0)
        graphindices = torch.tensor(
            [[0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.long
        )
        result = utils.get_average_heads_per_tail(graphindices)
        assert result == {0.0: 1.5}
        result = utils.get_average_tails_per_head(graphindices)
        # Head h0 has two tails, head h1 has one tail
        assert result == {0.0: 1.5}

    def test_bernoulli_probabilities_ring(self, kg):
        result = utils.get_bernoulli_probabilities(kg)
        assert result == {0.0: 0.5, 1.0: 0.5}


class TestLoadKnowledgeGraph:
    def test_pickle_roundtrip(self, kg, tmp_path):
        pkl = tmp_path / "kg.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(kg, f)
        loaded = utils.load_knowledge_graph(pkl)
        assert isinstance(loaded, KnowledgeGraph)
        torch.testing.assert_close(loaded.graphindices, kg.graphindices)
        assert loaded.node_to_index == kg.node_to_index
        assert loaded.edge_to_index == kg.edge_to_index
