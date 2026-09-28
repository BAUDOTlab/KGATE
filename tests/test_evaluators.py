"""
Tests for kgate.evaluators.

`Predictions`, `LinkPredictionEvaluator` and `TripletClassificationEvaluator`
are fully tested.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Subset

from kgate.decoders import TransE
from kgate.encoders import GATEncoder
from kgate.evaluators import (
    Predictions,
    LinkPredictionEvaluator,
    TripletClassificationEvaluator,
    TripletClassificationResults,
)
from kgate.samplers import PositionalNegativeSampler


class TestPredictions:
    def test_mean_rank(self):
        predictions = Predictions(
            torch.tensor([1., 3., 5.]),
            torch.tensor([1., 1., 3.]),
        )
        assert predictions.mean_rank == (pytest.approx(3.0), pytest.approx(5.0 / 3))

    def test_hit_at_k(self):
        predictions = Predictions(
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 1, 1, 2]),
        )
        true_hit, filtered_hit = predictions.hit_at_k(2)
        assert true_hit == pytest.approx(0.5)
        assert filtered_hit == pytest.approx(1.0)

    def test_mrr(self):
        predictions = Predictions(
            torch.tensor([1, 2, 4]),
            torch.tensor([1, 1, 2]),
        )
        mrr, filtered_mrr = predictions.mrr
        assert mrr == pytest.approx((1 + 0.5 + 0.25) / 3)
        assert filtered_mrr == pytest.approx((1 + 1 + 0.5) / 3)

    def test_str(self):
        predictions = Predictions(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 1, 2]),
        )
        message = str(predictions)
        assert "Hit@10" in message
        assert "MRR" in message
        assert "Mean Rank" in message

    def test_median_rank(self):
        predictions = Predictions(
            torch.tensor([1., 3., 5., 7.]),
            torch.tensor([1., 2., 3., 4.]),
        )
        median, filtered_median = predictions.median_rank
        # torch.median picks the lower of the two middle values for even counts:
        # ranks [1,3,5,7] -> 3; [1,2,3,4] -> 2
        assert median == pytest.approx(3.0)
        assert filtered_median == pytest.approx(2.0)

    def test_mean_reciprocal_rank_at_k(self):
        # ranks: 1, 2, 4, 8  ->  for k=3: only 1 and 2 count (1/1, 1/2)
        # filtered: 1, 1, 2, 8 -> for k=3: 1, 1, 1/2 count
        predictions = Predictions(
            torch.tensor([1, 2, 4, 8]),
            torch.tensor([1, 1, 2, 8]),
        )
        mrr3, filtered_mrr3 = predictions.mean_reciprocal_rank_at_k(3)
        assert mrr3 == pytest.approx((1 + 0.5 + 0 + 0) / 4)
        assert filtered_mrr3 == pytest.approx((1 + 1 + 0.5 + 0) / 4)
        # For k >= max rank, MRR@k equals plain MRR
        mrr10, filtered_mrr10 = predictions.mean_reciprocal_rank_at_k(10)
        assert mrr10 == pytest.approx(predictions.mrr[0])
        assert filtered_mrr10 == pytest.approx(predictions.mrr[1])

    def test_score_gap(self):
        predictions = Predictions(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 1, 2]),
        )
        true_scores = torch.tensor([10.0, 20.0, 30.0])
        best_other_unfiltered = torch.tensor([9.0, 8.0, 25.0])
        best_other_filtered = torch.tensor([5.0, 6.0, 20.0])
        gap, filtered_gap = predictions.score_gap(
            true_scores, best_other_unfiltered, best_other_filtered)
        assert gap == pytest.approx(((10 - 9) + (20 - 8) + (30 - 25)) / 3)
        assert filtered_gap == pytest.approx(((10 - 5) + (20 - 6) + (30 - 20)) / 3)

    def test_score_gap_shape_mismatch(self):
        predictions = Predictions(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 1, 2]),
        )
        with pytest.raises(ValueError):
            predictions.score_gap(
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([1.0]),
                torch.tensor([1.0, 2.0, 3.0]),
            )

    def test_relative_rank(self):
        predictions = Predictions(
            torch.tensor([1, 2, 4]),
            torch.tensor([1, 1, 2]),
        )
        rel, filtered_rel = predictions.relative_rank(10)
        assert rel == pytest.approx((0.1 + 0.2 + 0.4) / 3)
        assert filtered_rel == pytest.approx((0.1 + 0.1 + 0.2) / 3)
        with pytest.raises(ValueError):
            predictions.relative_rank(0)

    def test_to_dict(self):
        predictions = Predictions(
            torch.tensor([1, 2, 4]),
            torch.tensor([1, 1, 2]),
        )
        true_scores = torch.tensor([1.0, 2.0, 3.0])
        best_other_u = torch.tensor([0.5, 1.0, 2.0])
        best_other_f = torch.tensor([0.2, 0.8, 1.5])
        d = predictions.to_dict(
            k_values=(1, 3, 10),
            true_scores=true_scores,
            best_other_unfiltered=best_other_u,
            best_other_filtered=best_other_f,
            candidate_count=10,
        )
        # All expected keys present
        for key in ("mean_rank", "filtered_mean_rank", "median_rank",
                    "mrr", "filtered_mrr", "hit_at_1", "hit_at_3", "hit_at_10",
                    "mrr_at_1", "mrr_at_3", "mrr_at_10",
                    "score_gap", "filtered_score_gap",
                    "relative_rank", "filtered_relative_rank"):
            assert key in d, f"missing key {key}"
        # Values are plain floats
        assert all(isinstance(v, float) for v in d.values())
        # spot-check a couple of values
        assert d["mrr"] == pytest.approx(predictions.mrr[0])
        assert d["relative_rank"] == pytest.approx(predictions.relative_rank(10)[0])

    def test_to_dict_without_optional_metrics(self):
        predictions = Predictions(
            torch.tensor([1, 2, 4]),
            torch.tensor([1, 1, 2]),
        )
        d = predictions.to_dict()
        assert "mean_rank" in d
        assert "mrr" in d
        # optional metrics should be absent
        assert "score_gap" not in d
        assert "relative_rank" not in d


class TestLinkPredictionEvaluator:
    def test_init(self, kg):
        evaluator = LinkPredictionEvaluator(kg.graphindices, embedding_dimensions=4)
        assert evaluator.evaluated is False
        assert evaluator.embedding_dimensions == 4
        torch.testing.assert_close(evaluator.graphindices, kg.graphindices)

    def test_evaluate_without_encoder(self, kg):
        torch.manual_seed(0)
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        subset = Subset(kg, list(range(4)))

        evaluator = LinkPredictionEvaluator(kg.graphindices, embedding_dimensions=4)
        decoder = TransE(dissimilarity_type="L2")
        head_predictions, tail_predictions = evaluator.evaluate(
            batch_size=2,
            encoder=None,
            decoder=decoder,
            evaluated_subset=subset,
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )

        assert evaluator.evaluated is True
        assert evaluator.rank_true_heads.shape == (4,)
        assert evaluator.rank_true_tails.shape == (4,)
        # Ranks are 1-based and bounded by the node count
        assert (evaluator.rank_true_heads >= 1).all()
        assert (evaluator.rank_true_heads <= 8).all()
        assert (evaluator.rank_true_tails >= 1).all()
        assert (evaluator.rank_true_tails <= 8).all()
        # Filtering can only improve ranks
        assert (evaluator.filtered_rank_true_heads <= evaluator.rank_true_heads).all()
        assert (evaluator.filtered_rank_true_tails <= evaluator.rank_true_tails).all()
        # The returned predictions are usable
        assert isinstance(head_predictions, Predictions)
        assert isinstance(tail_predictions, Predictions)
        assert head_predictions.mean_rank[0] >= 1.0
        assert tail_predictions.mrr[1] > 0.0

    def test_evaluate_with_encoder(self, kg):
        torch.manual_seed(0)
        encoder = GATEncoder(
            edge_types=[("Node", "E1", "Node"), ("Node", "E2", "Node")],
            embedding_dimensions=4,
            gat_layer_count=2,
            device="cpu",
        )
        node_embeddings = nn.ParameterList([nn.Parameter(torch.randn(8, 4))])
        edge_embeddings = nn.Parameter(torch.randn(2, 4))
        # The encoder path calls knowledge_graph.get_encoder_input, which
        # reads the embeddings from the knowledge graph itself.
        kg.embeddings = node_embeddings, edge_embeddings
        subset = Subset(kg, list(range(4)))

        evaluator = LinkPredictionEvaluator(kg.graphindices, embedding_dimensions=4)
        decoder = TransE(dissimilarity_type="L2")
        head_predictions, tail_predictions = evaluator.evaluate(
            batch_size=2,
            encoder=encoder,
            decoder=decoder,
            evaluated_subset=subset,
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            verbose=False,
        )

        assert evaluator.evaluated is True
        assert (evaluator.rank_true_heads >= 1).all()
        assert (evaluator.rank_true_heads <= 8).all()
        assert (evaluator.filtered_rank_true_heads <= evaluator.rank_true_heads).all()


class TestTripletClassificationResults:
    def _make(self, pc=None, pi=None, nc=None, ni=None, n=4):
        return TripletClassificationResults(
            positive_correct=pc if pc is not None else torch.tensor([True, True, True, True]),
            positive_incorrect=pi if pi is not None else torch.tensor([False, False, False, False]),
            negative_correct=nc if nc is not None else torch.tensor([True, True, False, False]),
            negative_incorrect=ni if ni is not None else torch.tensor([False, False, True, True]),
        )

    def test_perfect_classification(self):
        r = self._make(
            pc=torch.tensor([True, True, True, True]),
            pi=torch.tensor([False, False, False, False]),
            nc=torch.tensor([True, True, True, True]),
            ni=torch.tensor([False, False, False, False]),
        )
        assert r.accuracy == pytest.approx(1.0)
        assert r.precision == pytest.approx(1.0)
        assert r.recall == pytest.approx(1.0)
        assert r.specificity == pytest.approx(1.0)
        assert r.f1 == pytest.approx(1.0)
        assert r.balanced_accuracy == pytest.approx(1.0)
        assert r.false_positive_rate == pytest.approx(0.0)
        assert r.false_negative_rate == pytest.approx(0.0)

    def test_perfect_rejection(self):
        # All rejected: 0 accepted, 0 true positives, all negative rejected
        r = self._make(
            pc=torch.tensor([False, False, False, False]),
            pi=torch.tensor([True, True, True, True]),
            nc=torch.tensor([True, True, True, True]),
            ni=torch.tensor([False, False, False, False]),
        )
        assert r.recall == pytest.approx(0.0)
        assert r.specificity == pytest.approx(1.0)
        assert r.precision == 0.0  # no accepted
        assert r.f1 == 0.0

    def test_mixed(self):
        # 4 pos: 3 correct, 1 incorrect
        # 4 neg: 2 correct, 2 incorrect
        r = self._make(
            pc=torch.tensor([True, True, True, False]),
            pi=torch.tensor([False, False, False, True]),
            nc=torch.tensor([True, True, False, False]),
            ni=torch.tensor([False, False, True, True]),
        )
        # accuracy = (3+2)/(4+4) = 0.625
        assert r.accuracy == pytest.approx(0.625)
        # precision = 3/(3+2) = 0.6
        assert r.precision == pytest.approx(3/5)
        # recall = 3/4
        assert r.recall == pytest.approx(0.75)
        # specificity = 2/4 = 0.5
        assert r.specificity == pytest.approx(0.5)
        # f1 = 2*(0.6*0.75)/(0.6+0.75)
        expected_f1 = 2 * (3/5) * 0.75 / ((3/5) + 0.75)
        assert r.f1 == pytest.approx(expected_f1)
        # balanced accuracy = (0.75 + 0.5)/2 = 0.625
        assert r.balanced_accuracy == pytest.approx(0.625)
        # FPR = 1 - 0.5 = 0.5
        assert r.false_positive_rate == pytest.approx(0.5)
        # FNR = 1 - 0.75 = 0.25
        assert r.false_negative_rate == pytest.approx(0.25)
        # counts
        assert r.positive_count == 4
        assert r.negative_count == 4

    def test_to_dict(self):
        r = self._make()
        d = r.to_dict()
        for key in ("accuracy", "precision", "recall", "specificity",
                    "f1", "balanced_accuracy", "false_positive_rate",
                    "false_negative_rate", "positive_count", "negative_count"):
            assert key in d
        assert all(isinstance(v, (float, int)) for v in d.values())

    def test_str(self):
        r = self._make()
        s = str(r)
        assert "Accuracy" in s
        assert "Precision" in s
        assert "F1" in s

    def test_empty(self):
        r = TripletClassificationResults(
            positive_correct=torch.tensor([], dtype=torch.bool),
            positive_incorrect=torch.tensor([], dtype=torch.bool),
            negative_correct=torch.tensor([], dtype=torch.bool),
            negative_incorrect=torch.tensor([], dtype=torch.bool),
        )
        assert r.accuracy == 0.0
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0


class TestTripletClassificationEvaluator:
    @pytest.fixture
    def tc_evaluator(self, kg):
        """Evaluator with embeddings and a TransE decoder, no encoder."""
        torch.manual_seed(42)
        from conftest import add_embeddings
        add_embeddings(kg)
        decoder = TransE(dissimilarity_type="L2")
        return TripletClassificationEvaluator(
            knowledge_graph=kg,
            decoder=decoder,
        )

    def test_init(self, tc_evaluator, kg):
        assert tc_evaluator.evaluated is False
        assert tc_evaluator.thresholds is None
        assert isinstance(tc_evaluator.sampler, PositionalNegativeSampler)
        assert tc_evaluator.device == torch.device("cpu")

    def test_init_without_decoder(self, kg):
        ev = TripletClassificationEvaluator(knowledge_graph=kg)
        assert ev.decoder is None

    def test_reset(self, tc_evaluator):
        tc_evaluator.evaluated = True
        tc_evaluator._cached_node_embeddings = torch.zeros(1, 4)
        tc_evaluator.reset()
        assert tc_evaluator.evaluated is False
        assert tc_evaluator._cached_node_embeddings is None

    def test_evaluate_computes_thresholds(self, tc_evaluator, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25), sizes=(8, 4, 4))
        validation_subset = Subset(kg, kg.validation_mask.nonzero(as_tuple=True)[0])
        tc_evaluator.evaluate(batch_size=2, knowledge_graph_subset=validation_subset)
        assert tc_evaluator.evaluated is True
        assert tc_evaluator.thresholds is not None
        assert tc_evaluator.thresholds.shape == (kg.edge_count,)

    def test_accuracy_before_evaluate_raises(self, tc_evaluator, kg):
        subset = Subset(kg, list(range(4)))
        with pytest.raises(RuntimeError, match="has not been evaluated"):
            tc_evaluator.accuracy(batch_size=2, kg_to_evaluate=subset)

    def test_accuracy_returns_results(self, tc_evaluator, kg):
        kg.generate_masks(split_proportions=(0.5, 0.25, 0.25), sizes=(8, 4, 4))
        validation_subset = Subset(kg, kg.validation_mask.nonzero(as_tuple=True)[0])
        test_subset = Subset(kg, kg.test_mask.nonzero(as_tuple=True)[0])
        tc_evaluator.evaluate(batch_size=2, knowledge_graph_subset=validation_subset)
        results = tc_evaluator.accuracy(batch_size=2, kg_to_evaluate=test_subset)
        assert isinstance(results, TripletClassificationResults)
        # All metrics in valid range
        assert 0.0 <= results.accuracy <= 1.0
        assert 0.0 <= results.precision <= 1.0
        assert 0.0 <= results.recall <= 1.0
        assert 0.0 <= results.specificity <= 1.0
        assert 0.0 <= results.f1 <= 1.0
        # Consistency: F1 = 2*P*R/(P+R) when P+R > 0
        if results.precision + results.recall > 0:
            expected_f1 = 2 * results.precision * results.recall / (results.precision + results.recall)
            assert results.f1 == pytest.approx(expected_f1)

    def test_score_triplets(self, tc_evaluator, kg):
        heads = kg.graphindices[0, :4]
        tails = kg.graphindices[1, :4]
        edges = kg.graphindices[2, :4]
        scores = tc_evaluator._score_triplets(heads, tails, edges, batch_size=2)
        assert scores.shape == (4,)
        assert scores.dtype == torch.float

    def test_cache(self, tc_evaluator, kg):
        heads = kg.graphindices[0, :2]
        tails = kg.graphindices[1, :2]
        edges = kg.graphindices[2, :2]
        s1 = tc_evaluator._score_triplets(heads, tails, edges, batch_size=1)
        cached = tc_evaluator._cached_node_embeddings
        s2 = tc_evaluator._score_triplets(heads, tails, edges, batch_size=1)
        # Cache is reused
        assert tc_evaluator._cached_node_embeddings is cached
        torch.testing.assert_close(s1, s2)
