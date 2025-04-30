from torch import tensor, bernoulli, randint, ones, rand, cat
import torch
from torchkge.sampling import get_possible_heads_tails, PositionalNegativeSampler, UniformNegativeSampler, BernoulliNegativeSampler, NegativeSampler
from .knowledgegraph import KnowledgeGraph
from typing import Tuple, List, Dict, Set
from torch.types import Number

class FixedPositionalNegativeSampler(PositionalNegativeSampler):
    """Simple fix of the PositionalNegativeSampler from torchkge, to solve a CPU/GPU device incompatibiltiy."""
    def __init__(self, kg:KnowledgeGraph, kg_val:KnowledgeGraph | None=None, kg_test: KnowledgeGraph | None=None):
        super().__init__(kg, kg_val, kg_test)
        self.ix2nt = {v: k for k,v in self.kg.nt2ix.items()}
        self.rel_types = {v: k for k,v in self.kg.rel2ix.items()}

    def corrupt_batch(self, batch: torch.LongTensor, n_neg: int = 1) -> torch.LongTensor:
        """For each true triplet, produce a corrupted one not different from
        any other golden triplet. If `heads` and `tails` are cuda objects,
        then the returned tensors are on the GPU.

        Parameters
        ----------
        batch: torch.Tensor, dtype: torch.long, shape: (4, batch_size)
            Tensor containing the integer key of heads, tails, relations and triples
            of the relations in the current batch.

        Returns
        -------
        neg_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the current batch.
        neg_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the current batch.
        """
        relations = batch[2]
        device = batch.device
        node_types = self.kg.node_types
        triple_types = self.kg.triple_types

        batch_size = batch.size(1)
        neg_batch: torch.LongTensor = batch.clone().long()

        self.bern_probs = self.bern_probs.to(device)
        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bern_probs[relations]).double()
        n_heads_corrupted = int(mask.sum().item())

        self.n_poss_heads = self.n_poss_heads.to(device)
        self.n_poss_tails = self.n_poss_tails.to(device)
        # Get the number of possible entities for head and tail
        n_poss_heads = self.n_poss_heads[relations[mask == 1]]
        n_poss_tails = self.n_poss_tails[relations[mask == 0]]

        assert n_poss_heads.shape[0] == n_heads_corrupted
        assert n_poss_tails.shape[0] == batch_size - n_heads_corrupted

        # Choose a rank of an entity in the list of possible entities
        choice_heads = (n_poss_heads.float() * rand((n_heads_corrupted,), device=device)).floor().long()

        choice_tails = (n_poss_tails.float() * rand((batch_size - n_heads_corrupted,), device=device)).floor().long()

        corr = tensor([], device=device, dtype=torch.long)
        corr_head_batch = batch[:,mask == 1]
        for i in range(n_heads_corrupted):
            r = corr_head_batch[2][i].item()
            t = corr_head_batch[1][i].item()
            choices: Dict[Number,Set[Number]] = self.possible_heads[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any head
                # choose one entity at random
                corr_head = randint(low=0, high=self.n_ent, size=(1,)).item()
            else:
                corr_head = choices[choice_heads[i].item()]
            cat([
                corr,
                tensor([
                    corr_head,
                    r,
                    t,
                    triple_types.index((
                        self.ix2nt[node_types[corr_head]],
                        self.rel_types[r],
                        self.ix2nt[node_types[t]]
                    ))
                ])
            ])
        neg_batch[mask == 1] = tensor(corr, device=device).long()

        corr = tensor([], device=device, dtype=torch.long)
        corr_tail_batch = batch[:,mask == 1]
        for i in range(batch_size - n_heads_corrupted):
            r = corr_tail_batch[2][i].item()
            h = corr_tail_batch[0][i].item()
            choices: Dict[Number,Set[Number]] = self.possible_tails[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any tail
                # choose one entity at random
                corr_tail = randint(low=0, high=self.n_ent, size=(1,)).item()
            else:
                corr_tail = choices[choice_tails[i].item()]
            cat([
                corr,
                tensor([
                    h,
                    r,
                    corr_tail,
                    triple_types.index((
                        self.ix2nt[node_types[h]],
                        self.rel_types[r],
                        self.ix2nt[node_types[corr_tail]]
                    ))
                ])
            ])
        neg_batch[mask == 0] = tensor(corr, device=device).long()

        return neg_batch
    
class MixedNegativeSampler(NegativeSampler):
    """
    A custom negative sampler that combines the BernoulliNegativeSampler
    and the PositionalNegativeSampler. For each triplet, it samples `n_neg` negative samples
    using both samplers.
    
    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data_structures.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data_structures.KnowledgeGraph (optional)
        Test knowledge graph.
    n_neg: int
        Number of negative sample to create from each fact.
    """
    
    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)
        # Initialize both Bernoulli and Positional samplers
        self.uniform_sampler = UniformNegativeSampler(kg, kg_val, kg_test, n_neg)
        self.bernoulli_sampler = BernoulliNegativeSampler(kg, kg_val, kg_test, n_neg)
        self.positional_sampler = FixedPositionalNegativeSampler(kg, kg_val, kg_test)
        
    def corrupt_batch(self, heads, tails, relations, n_neg=None):
        """For each true triplet, produce `n_neg` corrupted ones from the
        Unniform sampler, the Bernoulli sampler and the Positional sampler. If `heads` and `tails` are
        cuda objects, then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch.
        n_neg: int (optional)
            Number of negative samples to create from each fact. If None, the class-level
            `n_neg` value is used.

        Returns
        -------
        combined_neg_heads: torch.Tensor, dtype: torch.long
            Tensor containing the integer key of negatively sampled heads from both samplers.
        combined_neg_tails: torch.Tensor, dtype: torch.long
            Tensor containing the integer key of negatively sampled tails from both samplers.
        """

        if heads.device != tails.device or heads.device != relations.device:
            raise ValueError(f"Tensors are on different devices: h is on {heads.device}, t is on {tails.device}, r is on {relations.device}")

        if n_neg is None:
            n_neg = self.n_neg

        # Get negative samples from Uniform sampler
        uniform_neg_heads, uniform_neg_tails = self.uniform_sampler.corrupt_batch(
            heads, tails, relations, n_neg=n_neg
        )
        
        # Get negative samples from Bernoulli sampler
        bernoulli_neg_heads, bernoulli_neg_tails = self.bernoulli_sampler.corrupt_batch(
            heads, tails, relations, n_neg=n_neg
        )
        
        # Get negative samples from Positional sampler
        positional_neg_heads, positional_neg_tails = self.positional_sampler.corrupt_batch(
            heads, tails, relations
        )
        
        # Combine results from all samplers
        combined_neg_heads = cat([uniform_neg_heads, bernoulli_neg_heads, positional_neg_heads])
        combined_neg_tails = cat([uniform_neg_tails,bernoulli_neg_tails, positional_neg_tails])
        
        return combined_neg_heads, combined_neg_tails
