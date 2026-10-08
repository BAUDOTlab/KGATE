"""
Tests for kgate.samplers.

Most tests use the single-node-type path (no metadata). Remaining known
limitations of the multi-node-type paths are documented in
fixes/samplers.txt.
"""

import pytest
import torch

from kgate.samplers import (
    NegativeSampler,
    UniformNegativeSampler,
    BernoulliNegativeSampler,
    PositionalNegativeSampler,
    MixedNegativeSampler,
)


class TestInterface:
    def test_corrupt_batch_raises(self):
        sampler = NegativeSampler()
        with pytest.raises(NotImplementedError):
            sampler.corrupt_batch(torch.zeros(4, 2, dtype=torch.long))


class TestUniformNegativeSampler:
    def test_corrupt_batch_shape(self, kg):
        sampler = UniformNegativeSampler(kg, negative_triplet_count=2)
        batch = kg.graphindices[:, :4]
        corrupted = sampler.corrupt_batch(batch, negative_triplet_count=2)
        assert corrupted.shape == (4, 8)
        assert corrupted.dtype == torch.long
        # Edges are never corrupted
        torch.testing.assert_close(corrupted[2], batch[2].repeat(2))
        # Triplet type indices are preserved
        torch.testing.assert_close(corrupted[3], batch[3].repeat(2))

    def test_corrupt_batch_uses_default_count(self, kg):
        sampler = UniformNegativeSampler(kg, negative_triplet_count=3)
        batch = kg.graphindices[:, :2]
        corrupted = sampler.corrupt_batch(batch)
        assert corrupted.shape == (4, 6)

    def test_corrupt_batch_modifies_something(self, kg):
        # With a large enough batch, at least one head or tail should be replaced
        sampler = UniformNegativeSampler(kg, negative_triplet_count=1)
        batch = kg.graphindices[:, :8]
        torch.manual_seed(0)
        corrupted = sampler.corrupt_batch(batch)
        assert corrupted.shape == (4, 8)
        assert not torch.equal(corrupted[:2], batch[:2])

    def test_corrupt_batch_multi_type(self, hetero_kg):
        sampler = UniformNegativeSampler(hetero_kg, negative_triplet_count=2)
        batch = hetero_kg.graphindices[:, :2]
        corrupted = sampler.corrupt_batch(batch)
        # 2 samples x 2 corrupted triplets each -> [4, 4], edges preserved
        assert corrupted.shape == (4, 4)
        torch.testing.assert_close(corrupted[2], batch[2].repeat(2))


class TestBernoulliNegativeSampler:
    def test_evaluate_bernoulli_probabilities(self, kg):
        sampler = BernoulliNegativeSampler(kg)
        probabilities = sampler.bernoulli_probabilities
        assert probabilities.shape == (2,)
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()

    def test_corrupt_batch_shape(self, kg):
        sampler = BernoulliNegativeSampler(kg, negative_triplet_count=2)
        batch = kg.graphindices[:, :4]
        corrupted = sampler.corrupt_batch(batch, negative_triplet_count=2)
        assert corrupted.shape == (4, 8)
        # Edges are never corrupted
        torch.testing.assert_close(corrupted[2], batch[2].repeat(2))
        torch.testing.assert_close(corrupted[3], batch[3].repeat(2))

    def test_corrupt_batch_with_fixed_probabilities(self, kg):
        sampler = BernoulliNegativeSampler(kg)
        sampler.bernoulli_probabilities = torch.tensor([0.0, 0.0])
        batch = kg.graphindices[:, :4]
        corrupted = sampler.corrupt_batch(batch)
        # prob 0 -> heads kept, tails replaced
        torch.testing.assert_close(corrupted[0], batch[0])
        assert corrupted.shape == (4, 4)

    def test_corrupt_batch_with_probability_one(self, kg):
        sampler = BernoulliNegativeSampler(kg)
        sampler.bernoulli_probabilities = torch.tensor([1.0, 1.0])
        batch = kg.graphindices[:, :4]
        corrupted = sampler.corrupt_batch(batch)
        # prob 1 -> heads replaced, tails kept
        torch.testing.assert_close(corrupted[1], batch[1])
        assert corrupted.shape == (4, 4)


class TestPositionalNegativeSampler:
    def test_find_possibilities(self, kg):
        sampler = PositionalNegativeSampler(kg)
        # Every node appears once as head and once as tail for each edge
        assert sampler.possible_head_count.tolist() == [8, 8]
        assert sampler.possible_tail_count.tolist() == [8, 8]
        assert set(sampler.possible_heads[0].tolist()) == set(range(8))
        assert set(sampler.possible_tails[1].tolist()) == set(range(8))

    def test_corrupt_batch_shape(self, kg):
        sampler = PositionalNegativeSampler(kg)
        batch = kg.graphindices[:, :4]
        corrupted = sampler.corrupt_batch(batch)
        assert corrupted.shape == (4, 4)
        # Edges are never corrupted
        torch.testing.assert_close(corrupted[2], batch[2])

    def test_corrupt_heads(self, kg):
        sampler = PositionalNegativeSampler(kg)
        sampler.bernoulli_probabilities = torch.tensor([1.0, 1.0])
        batch = kg.graphindices[:, :4]
        corrupted = sampler.corrupt_batch(batch)
        # Tails and edges are preserved
        torch.testing.assert_close(corrupted[1], batch[1])
        torch.testing.assert_close(corrupted[2], batch[2])
        # Heads are replaced by nodes known to be heads of the same edge
        for i in range(4):
            edge = batch[2, i].item()
            assert corrupted[0, i].item() in sampler.possible_heads[edge].tolist()

    def test_corrupt_tails(self, kg):
        sampler = PositionalNegativeSampler(kg)
        sampler.bernoulli_probabilities = torch.tensor([0.0, 0.0])
        batch = kg.graphindices[:, :4]
        corrupted = sampler.corrupt_batch(batch)
        # Heads and edges are preserved
        torch.testing.assert_close(corrupted[0], batch[0])
        torch.testing.assert_close(corrupted[2], batch[2])
        # Tails must be replaced by nodes known to be tails of the same edge
        for i in range(4):
            edge = batch[2, i].item()
            assert corrupted[1, i].item() in sampler.possible_tails[edge].tolist()
        # And at least one tail should actually have changed
        assert not torch.equal(corrupted[1], batch[1])


class TestMixedNegativeSampler:
    def test_corrupt_batch_shape(self, kg):
        sampler = MixedNegativeSampler(kg, negative_triplet_count=1)
        batch = kg.graphindices[:, :2]
        corrupted = sampler.corrupt_batch(batch, negative_triplet_count=1)
        # uniform (1) + bernoulli (1) + positional (1) per triplet
        assert corrupted.shape == (4, 6)
        assert corrupted.dtype == torch.long
