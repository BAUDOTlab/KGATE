"""
Tests for kgate.loss.
"""

import pytest
import torch
import torch.nn.functional as F

from kgate.loss import KGE_Loss, MarginLoss, BinaryCrossEntropyLoss


class TestKGE_Loss:
    def test_forward_with_terms(self):
        margin = MarginLoss(margin=1.0, reduction="mean")
        bce = BinaryCrossEntropyLoss(reduction="mean")
        loss = KGE_Loss(margin, bce)

        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([0.0, 1.0])

        result = loss(pos, neg)
        expected = margin(pos, neg) + bce(pos, neg)
        assert torch.allclose(result, expected)

    def test_forward_single_term(self):
        margin = MarginLoss(margin=1.0, reduction="mean")
        loss = KGE_Loss(margin)
        pos = torch.tensor([2.0])
        neg = torch.tensor([0.0])
        assert torch.allclose(loss(pos, neg), margin(pos, neg))

    def test_add_term(self):
        loss = KGE_Loss()
        margin = MarginLoss(margin=1.0, reduction="mean")
        loss.add_term(margin)
        assert loss.terms == [margin]

        pos = torch.tensor([2.0])
        neg = torch.tensor([0.0])
        assert torch.allclose(loss(pos, neg), margin(pos, neg))

    def test_add_callable_term(self):
        loss = KGE_Loss()
        loss.add_term(lambda pos, neg: (pos - neg).sum())
        pos = torch.tensor([1.0, 2.0])
        neg = torch.tensor([0.5, 0.5])
        assert loss(pos, neg) == pytest.approx(2.0)

    def test_empty_loss_returns_zero(self):
        loss = KGE_Loss()
        result = loss(torch.tensor([1.0]), torch.tensor([0.0]))
        assert result == 0


class TestMarginLoss:
    def test_sum_reduction(self):
        loss = MarginLoss(margin=1.0, reduction="sum")
        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([0.0, 1.0])
        # max(0, 1 - (2-0)) + max(0, 1 - (3-1)) = 0 + 0
        assert loss(pos, neg).item() == pytest.approx(0.0)

    def test_mean_reduction(self):
        loss = MarginLoss(margin=1.0, reduction="mean")
        pos = torch.tensor([0.5, 3.0])
        neg = torch.tensor([0.0, 1.0])
        # (max(0, 1 - 0.5) + max(0, 1 - 2)) / 2 = 0.25
        assert loss(pos, neg).item() == pytest.approx(0.25)

    def test_matches_torch_margin_ranking_loss(self):
        torch.manual_seed(0)
        pos = torch.randn(8)
        neg = torch.randn(8)
        loss = MarginLoss(margin=2.0, reduction="mean")
        expected = F.margin_ranking_loss(pos, neg, torch.ones_like(pos),
                                         margin=2.0, reduction="mean")
        assert torch.allclose(loss(pos, neg), expected)


class TestBinaryCrossEntropyLoss:
    def test_matches_manual_computation(self):
        pos = torch.tensor([1.0, 2.0])
        neg = torch.tensor([-1.0, -2.0])
        loss = BinaryCrossEntropyLoss(reduction="mean")
        expected = (F.binary_cross_entropy(F.sigmoid(pos), torch.ones_like(pos),
                                           reduction="mean")
                    + F.binary_cross_entropy(F.sigmoid(neg), torch.zeros_like(neg),
                                             reduction="mean"))
        assert torch.allclose(loss(pos, neg), expected)

    def test_sum_reduction(self):
        loss = BinaryCrossEntropyLoss(reduction="sum")
        pos = torch.tensor([1.0, 2.0])
        neg = torch.tensor([-1.0, -2.0])
        expected = (F.binary_cross_entropy(F.sigmoid(pos), torch.ones_like(pos),
                                           reduction="sum")
                    + F.binary_cross_entropy(F.sigmoid(neg), torch.zeros_like(neg),
                                             reduction="sum"))
        assert torch.allclose(loss(pos, neg), expected)

    def test_high_scores_give_low_loss(self):
        loss = BinaryCrossEntropyLoss(reduction="mean")
        good = loss(torch.tensor([3.0, 3.0]), torch.tensor([-3.0, -3.0]))
        bad = loss(torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
        assert good < bad
