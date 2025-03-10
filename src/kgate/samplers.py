from torch import tensor, bernoulli, randint, ones, rand, cat
import torch
from torchkge.sampling import get_possible_heads_tails, PositionalNegativeSampler
from .data_structures import KGATEGraph
from typing import Tuple

class FixedPositionalNegativeSampler(PositionalNegativeSampler):
    """Simple fix of the PositionalNegativeSampler from torchkge, to solve a CPU/GPU device incompatibiltiy."""
    def __init__(self, kg:KGATEGraph, kg_val:KGATEGraph | None=None, kg_test: KGATEGraph | None=None):
        super().__init__(kg, kg_val, kg_test)

    def corrupt_batch(self, heads: torch.LongTensor, tails: torch.LongTensor, relations: torch.LongTensor, n_neg: int = 1) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """For each true triplet, produce a corrupted one not different from
        any other golden triplet. If `heads` and `tails` are cuda objects,
        then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.a
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch. This is optional here and mainly present because of the
            interface with other NegativeSampler objects.

        Returns
        -------
        neg_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the current batch.
        neg_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the current batch.
        """
        device = heads.device
        assert (device == tails.device)
        assert (device == relations.device)

        batch_size = heads.shape[0]
        neg_heads, neg_tails = heads.clone(), tails.clone()

        self.bern_probs = self.bern_probs.to(relations.device)
        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bern_probs[relations]).double()
        n_heads_corrupted = int(mask.sum().item())

        self.n_poss_heads = self.n_poss_heads.to(relations.device)
        self.n_poss_tails = self.n_poss_tails.to(relations.device)
        # Get the number of possible entities for head and tail
        n_poss_heads = self.n_poss_heads[relations[mask == 1]]
        n_poss_tails = self.n_poss_tails[relations[mask == 0]]

        assert n_poss_heads.shape[0] == n_heads_corrupted
        assert n_poss_tails.shape[0] == batch_size - n_heads_corrupted

        # Choose a rank of an entity in the list of possible entities
        choice_heads = (n_poss_heads.float() * rand((n_heads_corrupted,), device=n_poss_heads.device)).floor().long()

        choice_tails = (n_poss_tails.float() * rand((batch_size - n_heads_corrupted,), device=n_poss_tails.device)).floor().long()

        corr = []
        rels = relations[mask == 1]
        for i in range(n_heads_corrupted):
            r = rels[i].item()
            choices = self.possible_heads[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any head
                # choose one entity at random
                corr.append(randint(low=0, high=self.n_ent, size=(1,)).item())
            else:
                corr.append(choices[choice_heads[i].item()])
        neg_heads[mask == 1] = tensor(corr, device=device).long()

        corr = []
        rels = relations[mask == 0]
        for i in range(batch_size - n_heads_corrupted):
            r = rels[i].item()
            choices = self.possible_tails[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any tail
                # choose one entity at random
                corr.append(randint(low=0, high=self.n_ent, size=(1,)).item())
            else:
                corr.append(choices[choice_tails[i].item()])
        neg_tails[mask == 0] = tensor(corr, device=device).long()

        return neg_heads.long(), neg_tails.long()