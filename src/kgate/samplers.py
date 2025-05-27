from torch import tensor, bernoulli, randint, ones, rand, cat
import torch
import torchkge
from torchkge.sampling import get_possible_heads_tails, NegativeSampler
import torchkge.sampling
from .knowledgegraph import KnowledgeGraph
from typing import Tuple, List, Dict, Set
from torch.types import Number, Tensor

class PositionalNegativeSampler(torchkge.sampling.PositionalNegativeSampler):
    """Adaptation of torchKGE's PositionalNegativeSampler to KGATE's edgelist format.

    Either the head or the tail of a triplet is replaced by another entity
    chosen among entities that have already appeared at the same place in a
    triplet (involving the same relation), using bernouilli sampling.

    If the corrupted triplet is of a type that doesn't exist in the original KG,
    it is createad.

    Parameters
    ----------
    kg: kgate.data_structure.KnowledgeGraph
        Knowledge Graph from which the corrupted triples will be created.
            
    Attributes
    ----------
    possible_heads: Dict[int, List[int]]
        keys : relations
        values : list of possible heads for each relation.
    possible_tails: Dict[int, List[int]]
        keys : relations
        values : list of possible tails for each relation.
    n_poss_heads: List[int]
        List of number of possible heads for each relation.
    n_poss_tails: List[int]
        List of number of possible tails for each relation.
    ix2nt : Dict[int,str]
        keys : node index
        values : node types
    rel_types : Dict[int,str]
        keys : relation index
        values : relation name
    
    Notes
    -----
    Also fixes GPU/CPU incompatibility bug.
    See original implementation here : https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/sampling.py#L330C52-L330C53
    """
    def __init__(self, kg:KnowledgeGraph):
        super().__init__(kg)
        self.ix2nt = {v: k for k,v in self.kg.nt2ix.items()}
        self.rel_types = {v: k for k,v in self.kg.rel2ix.items()}

    def corrupt_batch(self, batch: Tensor, n_neg: int = 1) -> Tensor:
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
        neg_batch: Tensor = batch.clone().long()

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

        corr_head_batch = batch[:,mask == 1]
        corr_heads = []
        triples = [0] * n_heads_corrupted if len(self.kg.nt2ix)==1 else []
        for i in range(n_heads_corrupted):
            r = corr_head_batch[2][i].item()
            choices: Dict[Number,Set[Number]] = self.possible_heads[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any head
                # choose one entity at random
                corr_head = randint(low=0, high=self.n_ent, size=(1,)).item()
            else:
                corr_head = choices[choice_heads[i].item()]
            corr_heads.append(corr_head)
            # If we don't use metadata, there is only 1 node type
            if len(self.kg.nt2ix) > 1:
                t = corr_head_batch[1][i].item()
                # Find the corrupted triplet index
                corr_tri = (
                            self.ix2nt[node_types[corr_head].item()],
                            self.rel_types[r],
                            self.ix2nt[node_types[t].item()]
                        )
                # Add it if it doesn't already exists
                if not corr_tri in triple_types:
                    triple_types.append(corr_tri)
                    triple = len(triple_types)
                else:
                    triple = triple_types.index(corr_tri)

                triples.append(triple)
            
        if len(corr_heads) > 0:
            neg_batch[:, mask == 1] = torch.stack([tensor(corr_heads, device=device), corr_head_batch[1], corr_head_batch[2], tensor(triples, device=device)]).long().to(device)

        corr_tail_batch = batch[:,mask == 0]
        corr_tails = []
        triples = [0] * (batch_size - n_heads_corrupted) if len(self.kg.nt2ix)==1 else []
        for i in range(batch_size - n_heads_corrupted):
            r = corr_tail_batch[2][i].item()
            choices: Dict[Number,Set[Number]] = self.possible_tails[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any tail
                # choose one entity at random
                corr_tail = randint(low=0, high=self.n_ent, size=(1,)).item()
            else:
                corr_tail = choices[choice_tails[i].item()]
            # If we don't use metadata, there is only 1 node type
            if len(self.kg.nt2ix) > 1:
                h = corr_tail_batch[0][i].item()
                corr_tri = (
                            self.ix2nt[node_types[h].item()],
                            self.rel_types[r],
                            self.ix2nt[node_types[corr_tail].item()]
                        )
                if not corr_tri in triple_types:
                    triple_types.append(corr_tri)
                    triple = len(triple_types)
                else:
                    triple = triple_types.index(corr_tri)
                triples.append(triple)
        
        if len(corr_tails) > 0:
            neg_batch[:, mask == 0] = torch.stack([corr_tail_batch[1], tensor(corr_tails, device=device), corr_tail_batch[2], tensor(triples, device=device)]).long().to(device)

        return neg_batch

class UniformNegativeSampler(torchkge.sampling.UniformNegativeSampler):
    def __init__(self, kg, n_neg=1):
        super().__init__(kg, n_neg=n_neg)
        self.ix2nt = {v: k for k,v in self.kg.nt2ix.items()}
        self.rel_types = {v: k for k,v in self.kg.rel2ix.items()}
    
    def corrupt_batch(self, batch: torch.Tensor, n_neg=None) -> Tensor:
        n_neg = n_neg or self.n_neg

        device = batch.device
        batch_size = batch.size(1)
        neg_heads = batch[0].repeat(n_neg)
        neg_tails = batch[1].repeat(n_neg)
        rels = batch[2].repeat(n_neg)
        
        mask = bernoulli(ones(size=(batch_size * n_neg,),
                              device = device) / 2).double()
        n_h_cor = int(mask.sum().item())

        neg_heads[mask == 1] = randint(1, self.n_ent,
                                       (n_h_cor,),
                                       device=device)
        neg_tails[mask == 0] = randint(1, self.n_ent,
                                       (batch_size * n_neg - n_h_cor,),
                                       device=device)
        
        # If we don't use metadata, there is only 1 node type
        if len(self.kg.nt2ix) == 1:
            return torch.stack([neg_heads, neg_tails, rels, batch[3].repeat(n_neg)], dim=0).long().to(device)
        
        corrupted_triples = []
        node_types = self.kg.node_types
        triple_types = self.kg.triple_types
        for i in range(batch_size):
            h = neg_heads[i]
            t = neg_tails[i]
            r = rels[i].item()
            corr_tri = (
                        self.ix2nt[node_types[h].item()],
                        self.rel_types[r],
                        self.ix2nt[node_types[t].item()]
                    )
            if not corr_tri in triple_types:
                triple_types.append(corr_tri)
                triple = len(triple_types)
            else:
                triple = triple_types.index(corr_tri)
                
            corrupted_triples.append(tensor([
                h,
                t,
                r,
                triple
            ]))

        return torch.stack(corrupted_triples, dim=1).long().to(device)
    

    
class BernoulliNegativeSampler(torchkge.sampling.BernoulliNegativeSampler):
    def __init__(self, kg, n_neg=1):
        super().__init__(kg, n_neg=n_neg)
        self.ix2nt = {v: k for k,v in self.kg.nt2ix.items()}
        self.rel_types = {v: k for k,v in self.kg.rel2ix.items()}

    def corrupt_batch(self, batch: torch.LongTensor, n_neg=None):
        n_neg = n_neg or self.n_neg

        device = batch.device
        batch_size = batch.size(1)
        neg_heads = batch[0].repeat(n_neg)
        neg_tails = batch[1].repeat(n_neg)
        rels = batch[2]

        self.bern_probs: Tensor = self.bern_probs.to(device)
        mask = bernoulli(self.bern_probs[rels].repeat(n_neg)).double()
        n_h_cor = int(mask.sum().item())

        neg_heads[mask == 1] = randint(1, self.n_ent,
                                       (n_h_cor,),
                                       device=device)
        neg_tails[mask == 0] = randint(1, self.n_ent,
                                       (batch_size * n_neg - n_h_cor,),
                                       device=device)
        
        # If we don't use metadata, there is only 1 node type
        if len(self.kg.nt2ix) == 1:
            return torch.stack([neg_heads, neg_tails, rels.repeat(n_neg), batch[3].repeat(n_neg)], dim=0).long().to(device)
        
        corrupted_triples = []
        node_types = self.kg.node_types
        triple_types = self.kg.triple_types
        for i in range(batch_size):
            h = neg_heads[i]
            t = neg_tails[i]
            r = rels[i].item()
            corr_tri = (
                        self.ix2nt[node_types[h].item()],
                        self.rel_types[r],
                        self.ix2nt[node_types[t].item()]
                    )
            if not corr_tri in triple_types:
                triple_types.append(corr_tri)
                triple = len(triple_types)
            else:
                triple = triple_types.index(corr_tri)
                
            corrupted_triples.append(tensor([
                h,
                t,
                r,
                triple
            ]))

        return torch.stack(corrupted_triples, dim=1).long().to(device)
        
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
    
    def __init__(self, kg, n_neg=1):
        super().__init__(kg, n_neg=n_neg)
        # Initialize both Bernoulli and Positional samplers
        self.uniform_sampler = UniformNegativeSampler(kg, n_neg=n_neg)
        self.bernoulli_sampler = BernoulliNegativeSampler(kg, n_neg=n_neg)
        self.positional_sampler = PositionalNegativeSampler(kg)
        
    def corrupt_batch(self, batch: torch.LongTensor, n_neg=None):
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

        n_neg = n_neg or self.n_neg

        # Get negative samples from Uniform sampler
        uniform_neg_batch = self.uniform_sampler.corrupt_batch(
            batch, n_neg=n_neg
        )
        
        # Get negative samples from Bernoulli sampler
        bernoulli_neg_batch = self.bernoulli_sampler.corrupt_batch(
            batch, n_neg=n_neg
        )
        
        # Get negative samples from Positional sampler
        positional_neg_batch = self.positional_sampler.corrupt_batch(
            batch
        )
        
        # Combine results from all samplers
        combined_neg_batch = cat([uniform_neg_batch, bernoulli_neg_batch, positional_neg_batch], dim=1)
        
        return combined_neg_batch
