"""
Negative sampling classes, to generate negative triplets during training.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Dict, Set

import torch
from torch import tensor, bernoulli, randint, ones, rand, cat
from torch.types import Number, Tensor

import torchkge
import torchkge.sampling

from .knowledgegraph import KnowledgeGraph


class PositionalNegativeSampler(torchkge.sampling.PositionalNegativeSampler):
    """
    Adaptation of torchKGE's PositionalNegativeSampler to KGATE's graphindices format.

    Either the head or the tail of a triplet is replaced by another node
    chosen among nodes that have already appeared at the same place in a
    triplet (involving the same edge), using bernoulli sampling.

    If the corrupted triplet is of a type that doesn't exist in the original KG,
    it is createad.

    Parameters
    ----------
    kg: kgate.data_structure.KnowledgeGraph
        Knowledge Graph from which the corrupted triplets will be created.
            
    Attributes
    ----------
    possible_heads: Dict[int, List[int]]
        keys: edges
        values: list of possible heads for each edge.
    possible_tails: Dict[int, List[int]]
        keys: edges
        values: list of possible tails for each edge.
    possible_head_count: List[int]
        List of number of possible heads for each edge.
    possible_tail_count: List[int]
        List of number of possible tails for each edge.
    index_to_node_type: Dict[int, str]
        keys: node index
        values: node types
    edge_types: Dict[int, str]
        keys: edge index
        values: edge name
    
    Notes
    -----
    Also fixes GPU/CPU incompatibility bug.
    See original implementation here: https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/sampling.py#L330C52-L330C53
    
    """
    def __init__(self, kg: KnowledgeGraph):
        super().__init__(kg)
        self.index_to_node_type = {value: key for key, value in self.kg.node_type_to_index.items()}
        self.edge_types = {value: key for key,value in self.kg.edge_to_index.items()}


    def corrupt_batch(  self,
                        batch: Tensor, _: int = 0
                        ) -> Tensor:
        """
        For each true triplet, produce a corrupted one not different from
        any other golden triplet. If `heads` and `tails` are cuda objects,
        then the returned tensors are on the GPU.

        Parameters
        ----------
        batch: torch.Tensor, dtype: torch.long, shape: (4, batch_size)
            Tensor containing the integer key of heads, tails, edges and triplets
            of the edges in the current batch.

        Returns
        -------
        negative_triplets_batch: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled nodes of
            the edges in the current batch.
            
        """
        edges = batch[2]
        device = batch.device
        node_types = self.kg.node_types
        triplet_types = self.kg.triple_types

        batch_size = batch.size(1)
        negative_triplets_batch: Tensor = batch.clone().long()

        self.bernoulli_probabilitiess = self.bernoulli_probabilitiess.to(device)
        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bernoulli_probabilitiess[edges]).double()
        corrupted_head_count = int(mask.sum().item())

        self.possible_head_count = self.possible_head_count.to(device)
        self.possible_tail_count = self.possible_tail_count.to(device)
        # Get the number of possible nodes for head and tail
        possible_head_count = self.possible_head_count[edges[mask == 1]]
        possible_tail_count = self.possible_tail_count[edges[mask == 0]]

        assert possible_head_count.shape[0] == corrupted_head_count
        assert possible_tail_count.shape[0] == batch_size - corrupted_head_count

        # Choose a rank of an node in the list of possible nodes
        chosen_head = (possible_head_count.float() * rand((corrupted_head_count,), device = device)).floor().long()

        chosen_tail = (possible_tail_count.float() * rand((batch_size - corrupted_head_count,), device = device)).floor().long()

        corrupted_head_batch = batch[:,mask == 1]
        corrupted_heads = []
        triplets = [0] * corrupted_head_count if len(self.kg.node_type_to_index) == 1 else []
        for i in range(corrupted_head_count):
            edge_index = corrupted_head_batch[2][i].item()
            choices: Dict[Number, Set[Number]] = self.possible_heads[edge_index]
            if len(choices) == 0:
                # in this case the edge i has never been used with any head
                # choose one node at random
                corrupted_head_index = randint(low = 0, high = self.node_count, size = (1,)).item()
            else:
                corrupted_head_index = choices[chosen_head[i].item()]
            corrupted_heads.append(corrupted_head_index)
            # If we don't use metadata, there is only 1 node type
            if len(self.kg.node_type_to_index) > 1:
                tail_index = corrupted_head_batch[1][i].item()
                # Find the corrupted triplet index
                corrupted_triplet_index = (
                            self.index_to_node_type[node_types[corrupted_head_index].item()],
                            self.edge_types[edge_index],
                            self.index_to_node_type[node_types[tail_index].item()]
                        )
                # Add it if it doesn't already exist
                if not corrupted_triplet_index in triplet_types:
                    triplet_types.append(corrupted_triplet_index)
                    triplet = len(triplet_types)
                else:
                    triplet = triplet_types.index(corrupted_triplet_index)

                triplets.append(triplet)
            
        if len(corrupted_heads) > 0:
            negative_triplets_batch[:, mask == 1] = torch.stack([tensor(corrupted_heads, device = device),
                                                                corrupted_head_batch[1],
                                                                corrupted_head_batch[2],
                                                                tensor(triplets, device = device)]
                                                                ).long().to(device)

        corrupted_tail_batch = batch[:,mask == 0]
        corrupted_tails = []
        triplets = [0] * (batch_size - corrupted_head_count) if len(self.kg.node_type_to_index) == 1 else []
        for i in range(batch_size - corrupted_head_count):
            edge_index = corrupted_tail_batch[2][i].item()
            choices: Dict[Number, Set[Number]] = self.possible_tails[edge_index]
            if len(choices) == 0:
                # in this case the edge i has never been used with any tail
                # choose one node at random
                corrupted_tail_index = randint(low = 0, high = self.node_count, size = (1,)).item()
            else:
                corrupted_tail_index = choices[chosen_tail[i].item()]
            # If we don't use metadata, there is only 1 node type
            if len(self.kg.node_type_to_index) > 1:
                head_index = corrupted_tail_batch[0][i].item()
                corrupted_triplet_index = (
                            self.index_to_node_type[node_types[head_index].item()],
                            self.edge_types[edge_index],
                            self.index_to_node_type[node_types[corrupted_tail_index].item()]
                        )
                if not corrupted_triplet_index in triplet_types:
                    triplet_types.append(corrupted_triplet_index)
                    triplet = len(triplet_types)
                else:
                    triplet = triplet_types.index(corrupted_triplet_index)
                triplets.append(triplet)
        
        if len(corrupted_tails) > 0:
            negative_triplets_batch[:, mask == 0] = torch.stack([corrupted_tail_batch[1],
                                                                tensor(corrupted_tails, device = device),
                                                                corrupted_tail_batch[2],
                                                                tensor(triplets, device = device)]
                                                                ).long().to(device)

        return negative_triplets_batch


class UniformNegativeSampler(torchkge.sampling.UniformNegativeSampler):
    
    def __init__(self,
                kg: KnowledgeGraph,
                negative_triplet_count = 1):
        
        super().__init__(kg, n_neg = negative_triplet_count)
        self.index_to_node_type = {value: key for key,value in self.kg.node_type_to_index.items()}
        self.edge_types = {value: key for key,value in self.kg.edge_to_index.items()}
    
    
    def corrupt_batch(  self,
                        batch: torch.Tensor,
                        negative_triplet_count=None
                        ) -> Tensor:
        
        negative_triplet_count = negative_triplet_count or self.n_neg

        device = batch.device
        batch_size = batch.size(1)
        negative_triplet_heads = batch[0].repeat(negative_triplet_count)
        negative_triplet_tails = batch[1].repeat(negative_triplet_count)
        negative_triplet_edges = batch[2].repeat(negative_triplet_count)
        
        mask = bernoulli(ones(  size = (batch_size * negative_triplet_count,),
                                device = device) / 2).double()
        corrupted_head_count = int(mask.sum().item())

        negative_triplet_heads[mask == 1] = randint(1, self.n_ent,
                                                    (corrupted_head_count,),
                                                    device = device)
        negative_triplet_tails[mask == 0] = randint(1, self.n_ent,
                                                    (batch_size * negative_triplet_count - corrupted_head_count,),
                                                    device = device)
        
        # If we don't use metadata, there is only 1 node type
        if len(self.kg.node_type_to_index) == 1:
            return torch.stack([negative_triplet_heads,
                                negative_triplet_tails,
                                negative_triplet_edges,
                                batch[3].repeat(negative_triplet_count)],
                                dim=0).long().to(device)
        
        corrupted_triplets = []
        node_types = self.kg.node_types
        triplet_types = self.kg.triple_types
        for i in range(batch_size):
            head = negative_triplet_heads[i]
            tail = negative_triplet_tails[i]
            edge = negative_triplet_edges[i].item()
            corrupted_triplet = (
                        self.index_to_node_type[node_types[head].item()],
                        self.edge_types[edge],
                        self.index_to_node_type[node_types[tail].item()]
                    )
            if not corrupted_triplet in triplet_types:
                triplet_types.append(corrupted_triplet)
                triplet = len(triplet_types)
            else:
                triplet = triplet_types.index(corrupted_triplet)
                
            corrupted_triplets.append(tensor([
                head,
                tail,
                edge,
                triplet
            ]))

        return torch.stack(corrupted_triplets, dim=1).long().to(device)
    
    
class BernoulliNegativeSampler(torchkge.sampling.BernoulliNegativeSampler):
    
    def __init__(self,
                kg,
                negative_triplet_count=1):
        
        super().__init__(kg, n_neg = negative_triplet_count)
        self.index_to_node_type = {value: key for key,value in self.kg.node_type_to_index.items()}
        self.edge_types = {value: key for key,value in self.kg.edge_to_index.items()}


    def corrupt_batch(  self,
                        batch: torch.LongTensor,
                        negative_triplet_count=None):
        
        negative_triplet_count = negative_triplet_count or self.n_neg

        device = batch.device
        batch_size = batch.size(1)
        negative_triplet_heads = batch[0].repeat(negative_triplet_count)
        negative_triplet_tails = batch[1].repeat(negative_triplet_count)
        negative_triplet_edges = batch[2]

        self.bernoulli_probabilities: Tensor = self.bernoulli_probabilities.to(device)
        mask = bernoulli(self.bernoulli_probabilities[negative_triplet_edges].repeat(negative_triplet_count)).double()
        corrupted_head_count = int(mask.sum().item())

        negative_triplet_heads[mask == 1] = randint(1,
                                                    self.n_ent,
                                                    (corrupted_head_count,),
                                                    device = device)
        negative_triplet_tails[mask == 0] = randint(1,
                                                    self.n_ent,
                                                    (batch_size * negative_triplet_count - corrupted_head_count,),
                                                    device = device)
        
        # If we don't use metadata, there is only 1 node type
        if len(self.kg.node_type_to_index) == 1:
            return torch.stack(
                                [negative_triplet_heads,
                                negative_triplet_tails,
                                negative_triplet_edges.repeat(negative_triplet_count),
                                batch[3].repeat(negative_triplet_count)],
                                dim=0
                                ).long().to(device)
        
        corrupted_triplets = []
        node_types = self.kg.node_types
        triplet_types = self.kg.triple_types
        for i in range(batch_size):
            head = negative_triplet_heads[i]
            tail = negative_triplet_tails[i]
            edge = negative_triplet_edges[i].item()
            corrupted_triplet = (
                                self.index_to_node_type[node_types[head].item()],
                                self.edge_types[edge],
                                self.index_to_node_type[node_types[tail].item()]
                                )
            if not corrupted_triplet in triplet_types:
                triplet_types.append(corrupted_triplet)
                triplet = len(triplet_types)
            else:
                triplet = triplet_types.index(corrupted_triplet)
                
            corrupted_triplets.append(tensor([
                head,
                tail,
                edge,
                triplet
            ]))

        return torch.stack(corrupted_triplets, dim = 1).long().to(device)
        
        
class MixedNegativeSampler(torchkge.sampling.NegativeSampler):
    """
    A custom negative sampler that combines the BernoulliNegativeSampler
    and the PositionalNegativeSampler. For each triplet, it samples `n_neg` negative samples
    using both samplers.
    
    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    negative_triplet_count: int
        Number of negative samples to create from each fact.
        
    """
    
    def __init__(self,
                kg,
                negative_triplet_count = 1):
        super().__init__(kg, n_neg = negative_triplet_count)
        # Initialize both Bernoulli and Positional samplers
        self.uniform_sampler = UniformNegativeSampler(kg, negative_triplet_count = negative_triplet_count)
        self.bernoulli_sampler = BernoulliNegativeSampler(kg, negative_triplet_count = negative_triplet_count)
        self.positional_sampler = PositionalNegativeSampler(kg)
        
        
    def corrupt_batch(  self,
                        batch: torch.LongTensor,
                        negative_triplet_count = None):
        """
        For each true triplet, produce `negative_triplet_count` corrupted ones from the
        Unniform sampler, the Bernoulli sampler and the Positional sampler. If `heads` and `tails` are
        cuda objects, then the returned tensors are on the GPU.

        Parameters
        ----------
        negative_triplet_count: int (optional)
            Number of negative samples to create from each fact. If None, the class-level
            `n_neg` value is used.

        Returns
        -------
        combined_negative_triplets_batch: torch.Tensor, dtype: torch.long
            Tensor containing the integer key of negatively sampled heads and tailsfrom both samplers.
            
        """
        negative_triplet_count = negative_triplet_count or self.n_neg

        # Get negative samples from Uniform sampler
        uniform_negative_triplets_batch = self.uniform_sampler.corrupt_batch(
            batch, negative_triplet_count = negative_triplet_count
        )
        
        # Get negative samples from Bernoulli sampler
        bernoulli_negative_triplets_batch = self.bernoulli_sampler.corrupt_batch(
            batch, negative_triplet_count = negative_triplet_count
        )
        
        # Get negative samples from Positional sampler
        positional_negative_triplets_batch = self.positional_sampler.corrupt_batch(
            batch
        )
        
        # Combine results from all samplers
        combined_negative_triplets_batch = cat([
                                                uniform_negative_triplets_batch,
                                                bernoulli_negative_triplets_batch,
                                                positional_negative_triplets_batch
                                                ], dim=1)
        
        return combined_negative_triplets_batch
