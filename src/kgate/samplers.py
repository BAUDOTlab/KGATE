"""
Negative sampling classes, to generate negative triplets during training.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.

"""

from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

import torch
from torch import tensor, bernoulli, randint, ones, rand, cat
from torch.types import Number, Tensor

from .knowledgegraph import KnowledgeGraph
from .utils import get_bernoulli_probabilities



class NegativeSampler:
    def __init__(self):
        """
        Interface for negative samplers of KGATE.

        This interface doesn't implement anything but is a type helper: its __init__ does nothing, 
        and inheriting samplers are supposed to take care of their own initialization.

        """
        pass
    
    
    def corrupt_batch(  self,
                        batch: torch.Tensor,
                        negative_triplet_count: Optional[int] = None
                        ) -> Tensor:
        """
        For each true triplet, produce a corrupted one not different from 
        any other true triplet. If `heads` and `tails` are cuda objects, 
        then the returned tensors are on the GPU.

        Arguments
        ---------
        
        **batch** *(torch.Tensor, dtype: torch.long, shape: [4, batch_size])*
        : Tensor containing the integer key of heads, tails, edges and triplets of the edges in the current batch.
        : Here, batch_size is batch.shape[1].
        
        **negative_triplet_count** *(int, optional, default to None)*
        : Number of negative samples to create from each triplet. If None, `self.negative_triplet_count` is used.

        Raises
        ------
        
        **NotImplementedError**
        : The `corrupt_batch` method must be implemented by a negative sampler inheriting from this interface.
        
        """
        raise NotImplementedError("The `corrupt_batch` method must be implemented by the negative sampler.")



class UniformNegativeSampler(NegativeSampler):
    def __init__(self,
                knowledge_graph: KnowledgeGraph,
                negative_triplet_count = 1):
        """
        This class inherits from the NegativeSampler interface.

        For each positive sample, corrupts the head or the tail of the triplet (the corrupted element
        is chosen by a Bernoulli draw with probability 1/2) by replacing it with a random node.

        In typed knowledge graphs, the (head type, edge, tail type) combination of each corrupted triplet
        is registered in `triplet_type_to_index` if it does not exist yet.

        Arguments
        ---------

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the sampling will be done.

        **negative_triplet_count** *(int, optional, default to 1)*
        : Number of negative samples to create from each triplet.

        Attributes
        ----------

        **index_to_node_type** *(Dict[int, str])*
        : keys: node type index
        : values: node type name

        **edge_types** *(Dict[int, str])*
        : keys: edge index
        : values: edge name

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the sampling will be done.

        **negative_triplet_count** *(int)*
        : Number of negative samples to create from each triplet.

        """
        self.knowledge_graph = knowledge_graph
        self.index_to_node_type: Dict[int, str] = {value: key for key, value in self.knowledge_graph.node_type_to_index.items()}
        self.edge_types: Dict[int, str] = {value: key for key, value in self.knowledge_graph.edge_to_index.items()}
    
        self.negative_triplet_count = negative_triplet_count
    
    
    def corrupt_batch(  self,
                        batch: torch.Tensor,
                        negative_triplet_count: Optional[int] = None
                        ) -> Tensor:
        """
        For each true triplet, produce a corrupted one not different from 
        any other true triplet. If `heads` and `tails` are cuda objects, 
        then the returned tensors are on the GPU.

        Arguments
        ---------
        
        **batch** *(torch.Tensor, dtype: torch.long, shape: [4, batch_size])*
        : Tensor containing the integer key of heads, tails, edges and triplets of the edges in the current batch.
        : Here, batch_size is batch.shape[1].
        
        **negative_triplet_count** *(int, optional, default to None)*
        : Number of negative samples to create from each triplet. If None, self.negative_triplet_count is used.

        Returns
        -------
        
        **negative_triplets_batch** *(torch.Tensor, dtype: torch.long, shape: [4, negative_triplet_count * batch_size])*
        : Tensor containing the integer key of negatively sampled triplets of the edges in the current batch.
        : Here, batch_size is batch.shape[1].
            
        """
        device = batch.device
        batch_size = batch.shape[1]
        negative_triplet_count = negative_triplet_count or self.negative_triplet_count

        negative_triplet_heads = batch[0].repeat(negative_triplet_count)
        negative_triplet_tails = batch[1].repeat(negative_triplet_count)
        negative_triplet_edges = batch[2].repeat(negative_triplet_count)
        
        mask = bernoulli(ones(  size = (batch_size * negative_triplet_count,),
                                device = device) / 2).double()
        corrupted_head_count = int(mask.sum().item())

        negative_triplet_heads[mask == 1] = randint(1, self.knowledge_graph.node_count,
                                                    (corrupted_head_count,),
                                                    device = device)
        negative_triplet_tails[mask == 0] = randint(1, self.knowledge_graph.node_count,
                                                    (batch_size * negative_triplet_count - corrupted_head_count,),
                                                    device = device)
        
        # If we don't use metadata, there is only 1 node type
        if len(self.knowledge_graph.node_type_to_index) == 1:
            return torch.stack([negative_triplet_heads,
                                negative_triplet_tails,
                                negative_triplet_edges,
                                batch[3].repeat(negative_triplet_count)],
                                dim = 0).long().to(device)
        
        corrupted_triplets = []
        node_types = self.knowledge_graph.node_types
        triplet_types = self.knowledge_graph.triplet_types
        for i in range(batch_size):
            heads = negative_triplet_heads[i * negative_triplet_count: (i+1) * negative_triplet_count]
            tails = negative_triplet_tails[i * negative_triplet_count: (i+1) * negative_triplet_count]
            edges = negative_triplet_edges[i * negative_triplet_count: (i+1) * negative_triplet_count]

            head_types = node_types[heads]
            tail_types = node_types[tails]

            corrupted_triplet_type = torch.stack(
                [head_types, edges, tail_types],
                dim = 1
            )

            triplet_indices = torch.empty(
                corrupted_triplet_type.shape[0],
                dtype=torch.long,
                device=corrupted_triplet_type.device
            )

            for i, triplet in enumerate(corrupted_triplet_type.tolist()):
                triplet = tuple(triplet)

                if triplet not in self.knowledge_graph.triplet_type_to_index:
                    self.knowledge_graph.triplet_type_to_index[triplet] = len(self.knowledge_graph.triplet_type_to_index)

                triplet_indices[i] = self.knowledge_graph.triplet_type_to_index[triplet]
            
            # corrupted_triplet = (
            #             self.index_to_node_type[node_types[head].item()],
            #             edges,
            #             self.index_to_node_type[node_types[tail].item()]
            #         )
            # if not corrupted_triplet in triplet_types:
            #     triplet_types.append(corrupted_triplet)
            #     triplet = len(triplet_types)
            # else:
            #     triplet = triplet_types.index(corrupted_triplet)
                
            corrupted_triplets.append(torch.stack([
                heads,
                tails,
                edges,
                triplet_indices
            ], dim = 0))

        # Concatenate along the sample axis -> [4, negative_triplet_count * batch_size],
        # the same 2-D layout as the single-node-type branch (a per-sample
        # `torch.stack(..., dim=1)` would produce a 3-D tensor here, and
        # `torch.tensor([t1, t2, t3, t4])` on multi-element 1-D tensors is
        # rejected by recent PyTorch versions)
        return torch.cat(corrupted_triplets, dim = 1).long().to(device)



class BernoulliNegativeSampler(NegativeSampler):
    def __init__(self,
                knowledge_graph: KnowledgeGraph,
                negative_triplet_count = 1):
        """
        This class inherits from the NegativeSampler interface.

        For each positive sample, corrupts the head of the triplet with the edge-specific Bernoulli
        probability and the tail with the complementary probability, replacing the corrupted element
        with a random node.

        In typed knowledge graphs, the (head type, edge, tail type) combination of each corrupted triplet
        is appended to `triplet_types` if it does not exist yet.

        Arguments
        ---------

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the sampling will be done.

        **negative_triplet_count** *(int, optional, default to 1)*
        : Number of negative samples to create from each triplet.

        Attributes
        ----------

        **index_to_node_type** *(Dict[int, str])*
        : keys: node type index
        : values: node type name

        **edge_types** *(Dict[int, str])*
        : keys: edge index
        : values: edge name

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the sampling will be done.

        **negative_triplet_count** *(int)*
        : Number of negative samples to create from each triplet.

        **bernoulli_probabilities** *(torch.Tensor, dtype: torch.float, shape: [edge_count])*
        : Tensor containing the probabilities of corrupting the head for each edge.

        """
        self.knowledge_graph = knowledge_graph
        self.index_to_node_type: Dict[int, str] = {value: key for key,value in self.knowledge_graph.node_type_to_index.items()}
        self.edge_types: Dict[int, str] = {value: key for key,value in self.knowledge_graph.edge_to_index.items()}
    
        self.negative_triplet_count = negative_triplet_count
        self.bernoulli_probabilities = self.evaluate_bernoulli_probabilities()


    def evaluate_bernoulli_probabilities(self) -> torch.Tensor:
        """
        Evaluate the Bernoulli probabilities as in the TransH original paper.
        
        Code adapted from the TorchKGE function. The bernoullis probabilities are sampled 
        from the average number of heads per tail and tails per head, for each edge type. If 
        the probability for an edge type has not been sampled, it will be set to 0.5.
        
        Returns
        -------
        
        **bernoulli_probabilities** *(torch.Tensor, dtype: torch.float, shape: [edge_count])*
        : Tensor containing the probabilities of sampling a head for each edge.
        
        """
        bernoulli_probabilities = get_bernoulli_probabilities(self.knowledge_graph)

        final_probabilities = []
        for edge_index in range(self.knowledge_graph.edge_count):
            if edge_index in bernoulli_probabilities.keys():
                final_probabilities.append(bernoulli_probabilities[edge_index])
            else:
                final_probabilities.append(0.5)

        return torch.tensor(final_probabilities).float()

    
    def corrupt_batch(  self,
                        batch: Tensor,
                        negative_triplet_count: int | None = None):
        """
        For each true triplet, produce a corrupted one not different from
        any other true triplet. If `heads` and `tails` are cuda objects,
        then the returned tensors are on the GPU.

        Arguments
        ---------
        batch: torch.Tensor, dtype: torch.long, shape: [4, batch_size]
            Tensor containing the integer key of heads, tails, edges and triplets
            of the edges in the current batch.
            Here, batch_size is batch.shape[1].
        negative_triplet_count: int, optional, default to None
            Number of negative samples to create from each triplet.

        Returns
        -------
        negative_triplets_batch: torch.Tensor, dtype: torch.long, shape: [4, negative_triplet_count * batch_size]
            Tensor containing the integer key of negatively sampled triplets of
            the edges in the current batch.
            Here, batch_size is batch.shape[1].
            
        """
        device = batch.device
        batch_size = batch.shape[1]
        
        negative_triplet_count = negative_triplet_count or self.negative_triplet_count

        negative_triplet_heads = batch[0].repeat(negative_triplet_count)
        negative_triplet_tails = batch[1].repeat(negative_triplet_count)
        negative_triplet_edges = batch[2]

        self.bernoulli_probabilities: Tensor = self.bernoulli_probabilities.to(device)
        mask = bernoulli(self.bernoulli_probabilities[negative_triplet_edges].repeat(negative_triplet_count)).double()
        corrupted_head_count = int(mask.sum().item())

        negative_triplet_heads[mask == 1] = randint(1,
                                                    self.knowledge_graph.node_count,
                                                    (corrupted_head_count,),
                                                    device = device)
        negative_triplet_tails[mask == 0] = randint(1,
                                                    self.knowledge_graph.node_count,
                                                    (batch_size * negative_triplet_count - corrupted_head_count,),
                                                    device = device)
        
        # If we don't use metadata, there is only 1 node type
        if len(self.knowledge_graph.node_type_to_index) == 1:
            return torch.stack(
                                [negative_triplet_heads,
                                negative_triplet_tails,
                                negative_triplet_edges.repeat(negative_triplet_count),
                                batch[3].repeat(negative_triplet_count)],
                                dim = 0
                                ).long().to(device)
        
        corrupted_triplets = []
        node_types = self.knowledge_graph.node_types
        triplet_types = self.knowledge_graph.triplet_types
        
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



class PositionalNegativeSampler(BernoulliNegativeSampler):
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Adaptation of torchKGE's PositionalNegativeSampler to KGATE's graphindices format.

        This class inherits from the BernoulliNegativeSampler class. It inherites its attributes as well.
        
        Either the head or the tail of a triplet is replaced by another node 
        chosen among nodes that have already appeared at the same place in a 
        triplet (involving the same edge), using bernoulli sampling.

        If the corrupted triplet is of a type that doesn't exist in the original knowledge graph, 
        it is created.

        Arguments
        ---------

        **knowledge_graph** *(kgate.knowledgegraph.KnowledgeGraph)*
        : Knowledge graph from which the corrupted triplets will be created.

        Attributes
        ----------

        **possible_heads** *(Dict[int, torch.Tensor])*
        : keys: edge index
        : values: tensor of the possible heads (node indices) for that edge, equivalent to possible_head_count

        **possible_tails** *(Dict[int, torch.Tensor])*
        : keys: edge index
        : values: tensor of the possible tails (node indices) for that edge, equivalent to possible_tail_count

        **possible_head_count** *(torch.Tensor)*
        : Number of possible heads for each edge.
        : Equivalent of List[int], but with Tensor possibilities.

        **possible_tail_count** *(torch.Tensor)*
        : Number of possible tails for each edge.
        : Equivalent of List[int], but with Tensor possibilities.

        **index_to_node_type** *(Dict[int, str])*
        : keys: node type index
        : values: node type name

        **edge_types** *(Dict[int, str])*
        : keys: edge index
        : values: edge name

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the sampling will be done.

        **bernoulli_probabilities** *(torch.Tensor, dtype: torch.float, shape: [edge_count])*
        : Tensor containing the probabilities of corrupting the head for each edge.

        **negative_triplet_count** *(int)*
        : Number of negative samples to create from each triplet.

        Notes
        -----

        Also fixes GPU/CPU incompatibility bug.

        See original implementation here: https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/sampling.py#L330C52-L330C53

        Slower than UniformNegativeSampler, BernoulliNegativeSampler and MixedNegativeSampler, as it searches 
        in the entire knowledge graph instead of a batch.

        """
        super().__init__(knowledge_graph)

        self.possible_heads, self.possible_tails, \
            self.possible_head_count, self.possible_tail_count = self.find_possibilities()

        # Flat (CSR-style) candidate tables used by the vectorized corrupt_batch:
        # all candidates of all edges are stored in a single tensor, and the
        # offsets give the [start, start + count) slice of the candidates of
        # each edge. This replaces the per-sample Python loop of the original
        # implementation (one GPU<->CPU synchronization per sample).
        self.head_candidates, self.head_offsets = self._build_candidate_table(self.possible_heads, self.possible_head_count)
        self.tail_candidates, self.tail_offsets = self._build_candidate_table(self.possible_tails, self.possible_tail_count)

        # Pre-computed lookup table of the triplet type index (row 3 of the
        # graphindices) of corrupted triplets, for typed knowledge graphs:
        # _tt_lookup[head_type, edge, tail_type]. Pre-assigning the indices
        # here means corrupt_batch never mutates the knowledge graph at
        # runtime.
        self._tt_lookup: Tensor = self._build_triplet_type_lookup()


    @staticmethod
    def _build_candidate_table(possible: Dict[int, Tensor], counts: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Flatten the per-edge candidate tensors of ``possible`` into a single
        tensor, with a CSR-style offsets tensor (offsets[e] .. offsets[e+1] are
        the candidates of edge e).

        Returns
        -------

        **candidates** *(torch.Tensor, dtype: torch.long, shape: [total_candidates])*
        : All candidates of all edges, in edge order.

        **offsets** *(torch.Tensor, dtype: torch.long, shape: [edge_count + 1])*
        : Start index of each edge's candidates (last entry is the total count).

        """
        edge_count = counts.shape[0]
        chunks = [possible[edge_index] for edge_index in range(edge_count) if possible[edge_index].numel() > 0]
        candidates = torch.cat(chunks) if chunks else torch.empty(0, dtype = torch.long)
        offsets = torch.zeros(edge_count + 1, dtype = torch.long)
        offsets[1:] = counts.cumsum(0)

        return candidates.long(), offsets


    def _build_triplet_type_lookup(self) -> Tensor:
        """
        For typed knowledge graphs (more than one node type), the corrupted
        head or tail can produce a triplet type that does not exist in the
        original knowledge graph. The original implementation appended such
        types to ``knowledge_graph.triplet_types`` during training, in batch
        order, with a Python lookup per sample.

        This pre-computes the index of every possible
        (head_type, edge, tail_type) combination instead: existing types keep
        their current index, and the other combinations get the next free
        indices (they are appended to ``knowledge_graph.triplet_types`` here,
        as the original implementation did, just ahead of time).

        For single-type knowledge graphs this is a [1, edge_count, 1] table.

        Returns
        -------

        **lookup** *(torch.Tensor, dtype: torch.long, shape: [node_type_count, edge_count, node_type_count])*
        : lookup[head_type, edge, tail_type] is the triplet type index.

        """
        kg = self.knowledge_graph
        type_count = len(kg.node_type_to_index)
        edge_count = len(kg.edge_to_index)

        lookup = torch.full((type_count, edge_count, type_count), -1, dtype = torch.long)

        for triplet_index, (head_type, edge_type, tail_type) in enumerate(kg.triplet_types):
            lookup[
                kg.node_type_to_index[head_type],
                kg.edge_to_index[edge_type],
                kg.node_type_to_index[tail_type],
            ] = triplet_index

        # New combinations get the next free indices
        next_index = len(kg.triplet_types)

        for head_type in range(type_count):
            for edge_index in range(edge_count):
                for tail_type in range(type_count):
                    if lookup[head_type, edge_index, tail_type] == -1:
                        lookup[head_type, edge_index, tail_type] = next_index
                        kg.triplet_types.append(
                            (self.index_to_node_type[head_type],
                             self.edge_types[edge_index],
                             self.index_to_node_type[tail_type]))
                        next_index += 1

        return lookup


    def find_possibilities(self) -> Tuple[
                                Dict[int, torch.Tensor],
                                Dict[int, torch.Tensor], 
                                Tensor, 
                                Tensor]:
        """
        For each edge of the knowledge graph, find all the possible heads 
        and tails in the sense of Wang et al., e.g. all nodes that occupy 
        once this position in another triplet of the same edge.

        Returns
        -------
        
        **possible_heads** *(Dict[int, torch.Tensor])*
        : keys : edge index
        : values : tensor of possible heads
        
        **possible_tails** *(Dict[int, torch.Tensor])*
        : keys : edge index
        : values : tensor of possible tails
                
        **possible_heads_count** *(torch.Tensor, dtype: torch.long, shape: (edge_count))*
        : Number of possible heads for each edge.
        
        **possible_tails_count** *(torch.Tensor, dtype: torch.long, shape: (edge_count))*
        : Number of possible tails for each edge.
        
        """
        edge_indices = self.knowledge_graph.edge_indices 
        head_indices = self.knowledge_graph.head_indices 
        tail_indices = self.knowledge_graph.tail_indices 
        edge_count = self.knowledge_graph.edge_count
        node_count = self.knowledge_graph.node_count # used as a stride
       
        # For each edge, count distinct heads and distinct tails.
        # Encode each (edge, node) pair as a single integer, unique() deduplicates,
        # then bincount gives per-edge counts in one shot.
        head_keys = edge_indices * node_count + head_indices
        tail_keys = edge_indices * node_count + tail_indices

        unique_head_keys = head_keys.unique()
        unique_tail_keys = tail_keys.unique()

        # Recover which edge each unique pair belongs to
        unique_head_edges = unique_head_keys // node_count 
        unique_tail_edges = unique_tail_keys // node_count 

        possible_heads_count = torch.bincount(unique_head_edges, minlength=edge_count) 
        possible_tails_count = torch.bincount(unique_tail_edges, minlength=edge_count) 

        # Recover the actual node index from each unique pair.
        unique_head_nodes = unique_head_keys % node_count
        unique_tail_nodes = unique_tail_keys % node_count

        # Group by edge using the sorted order that unique() already produced.
        possible_heads = dict(zip(
            unique_head_edges.unique().tolist(),
            torch.split(unique_head_nodes, possible_heads_count[possible_heads_count > 0].tolist())
        ))
        possible_tails = dict(zip(
            unique_tail_edges.unique().tolist(),
            torch.split(unique_tail_nodes, possible_tails_count[possible_tails_count > 0].tolist())
        ))

        # Fill in edges with zero possibilities have a complete dict
        for edge_index in range(edge_count):
            possible_heads.setdefault(edge_index, torch.empty(0, dtype=torch.long))
            possible_tails.setdefault(edge_index, torch.empty(0, dtype=torch.long))

        return possible_heads, possible_tails, possible_heads_count, possible_tails_count

    def corrupt_batch(  self,
                        batch: Tensor,
                        negative_triplet_count: Optional[int] = None
                        ) -> Tensor:
        """
        For each true triplet, produce a corrupted one not different from 
        any other true triplet. If `heads` and `tails` are cuda objects, 
        then the returned tensors are on the GPU.

        Arguments
        ---------
        
        **batch** *(torch.Tensor, dtype: torch.long, shape: [4, batch_size])* 
        : Tensor containing the integer key of heads, tails, edges and triplets of the edges in the current batch.
        : Here, batch_size is batch.shape[1].

        Returns
        -------
        
        **negative_triplets_batch** *(torch.Tensor, dtype: torch.long, shape: [4, batch_size])*
        : Tensor containing the integer key of negatively sampled triplets of the edges in the current batch.
        : Here, batch_size is batch.shape[1].
        
        """
        edges = batch[2]
        device = batch.device

        batch_size = batch.shape[1]
        negative_triplets_batch: Tensor = batch.clone().long()
        single_node_type = len(self.knowledge_graph.node_type_to_index) == 1

        # For untyped knowledge graphs the triplet type index of corrupted
        # triplets is always 0 (as in the original implementation)
        if single_node_type:
            negative_triplets_batch[3].zero_()

        self.bernoulli_probabilities = self.bernoulli_probabilities.to(device)
        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bernoulli_probabilities[edges]).double()

        # Node types of the corrupted nodes, only needed for typed knowledge graphs
        node_types: Tensor | None = None
        if not single_node_type:
            node_types = self.knowledge_graph.node_types.to(device).clamp(min = 0)

        # ---- Corrupt the heads ----
        head_mask = mask == 1
        head_edges = edges[head_mask]
        new_heads = self._sample_positional(head_edges,
                                            self.possible_head_count.to(device),
                                            self.head_offsets.to(device),
                                            self.head_candidates.to(device),
                                            device)
        negative_triplets_batch[0][head_mask] = new_heads
        if node_types is not None:
            negative_triplets_batch[3][head_mask] = self._tt_lookup.to(device)[
                node_types[new_heads],
                head_edges,
                node_types[batch[1][head_mask]]]

        # ---- Corrupt the tails ----
        tail_mask = mask == 0
        tail_edges = edges[tail_mask]
        new_tails = self._sample_positional(tail_edges,
                                            self.possible_tail_count.to(device),
                                            self.tail_offsets.to(device),
                                            self.tail_candidates.to(device),
                                            device)
        negative_triplets_batch[1][tail_mask] = new_tails
        if node_types is not None:
            negative_triplets_batch[3][tail_mask] = self._tt_lookup.to(device)[
                node_types[batch[0][tail_mask]],
                tail_edges,
                node_types[new_tails]]

        return negative_triplets_batch


    def _sample_positional(self,
                           selected_edges: Tensor,
                           counts: Tensor,
                           offsets: Tensor,
                           candidates: Tensor,
                           device: torch.device) -> Tensor:
        """
        For each selected edge, sample one of the nodes that already occupy
        the same position (head or tail) in another triplet of that edge.
        If the edge has no candidate, fall back to a uniform random node,
        as the original per-sample loop did.

        Fully vectorized: a single gather instead of one Python iteration
        (and one GPU<->CPU synchronization) per sample.

        Arguments
        ---------

        **selected_edges** *(torch.Tensor, dtype: torch.long)*
        : Index of the edge of each sample to corrupt.

        **counts** *(torch.Tensor, dtype: torch.long, shape: [edge_count])*
        : Number of candidates per edge.

        **offsets** *(torch.Tensor, dtype: torch.long, shape: [edge_count + 1])*
        : Start index of each edge's candidates in the flat candidate table.

        **candidates** *(torch.Tensor, dtype: torch.long, shape: [total_candidates])*
        : Flat table of all candidates.

        **device** *(torch.device)*
        : Device on which the batch lives.

        Returns
        -------

        **sampled** *(torch.Tensor, dtype: torch.long, shape: [len(selected_edges)])*
        : One sampled node per selected edge.

        """
        edge_counts = counts[selected_edges]
        count = edge_counts.shape[0]
        node_count = self.knowledge_graph.node_count

        if candidates.numel() == 0:
            return randint(low = 0, high = node_count, size = (count,), device = device)

        # Choose a rank of a node in the list of possible nodes of its edge
        chosen_rank = (edge_counts.float() * rand((count,), device = device)).floor().long()

        # Position of the chosen node in the flat candidate table
        candidate_index = torch.clamp(offsets[selected_edges] + chosen_rank, max = candidates.numel() - 1)
        sampled = candidates[candidate_index]

        # Edges without any candidate get a uniform random node
        has_candidates = edge_counts > 0
        fallback = randint(low = 0, high = node_count, size = (count,), device = device)

        return torch.where(has_candidates, sampled, fallback)



class MixedNegativeSampler(NegativeSampler):
    def __init__(self,
                knowledge_graph: KnowledgeGraph,
                negative_triplet_count = 1):
        """
        A custom negative sampler that combines the BernoulliNegativeSampler, the UniformNegativeSampler 
        and the PositionalNegativeSampler. 
        
        This class inherits from the NegativeSampler class.
        
        For each triplet, it samples `negative_triplet_count` negative samples for each samplers except the Positional. Note 
        that the PositionalNegativeSampler always produces only one negative triplet per positive triplet.

        Arguments
        ---------

        **knowledge_graph** *(KnowledgeGraph)*
            Main knowledge graph (usually training one).

        **negative_triplet_count** *(int, optional, default to 1)*
            Number of negative samples to create from each triplet with the Uniform and Bernoulli samplers.
            Since the Positional sampler always adds one negative sample per triplet, the total number of
            negative samples per triplet is 2 * negative_triplet_count + 1.

        Attributes
        ----------

        **negative_triplet_count** *(int)*
            Number of negative samples to create from each triplet with the Uniform and Bernoulli samplers.

        **uniform_sampler** *(UniformNegativeSampler)*
            Initialization of the UniformNegativeSampler class as an attribute.

        **bernoulli_sampler** *(BernoulliNegativeSampler)*
            Initialization of the BernoulliNegativeSampler class as an attribute.

        **positional_sampler** *(PositionalNegativeSampler)*
            Initialization of the PositionalNegativeSampler class as an attribute.

        Notes
        -----

        This is an example of a custom negative sampler using other existing samplers, and may produce 
        unexpected behaviour if used as is.

        """
        
        self.knowledge_graph = knowledge_graph
        self.index_to_node_type: Dict[int, str] = {value: key for key,value in self.knowledge_graph.node_type_to_index.items()}
        self.edge_types: Dict[int, str] = {value: key for key,value in self.knowledge_graph.edge_to_index.items()}
    
        self.negative_triplet_count = negative_triplet_count

        # Initialize both Bernoulli, Uniform and Positional samplers
        self.uniform_sampler = UniformNegativeSampler(self.knowledge_graph, negative_triplet_count = negative_triplet_count)
        self.bernoulli_sampler = BernoulliNegativeSampler(self.knowledge_graph, negative_triplet_count = negative_triplet_count)
        self.positional_sampler = PositionalNegativeSampler(self.knowledge_graph)
        
        
    def corrupt_batch(  self,
                        batch: Tensor,
                        negative_triplet_count: int = 1):
        """
        For each true triplet, produce `negative_triplet_count` corrupted ones from the 
        Uniform sampler, the Bernoulli sampler and the Positional sampler. If `heads` and `tails` are 
        cuda objects, then the returned tensors are on the GPU.

        Arguments
        ---------
        
        batch: torch.Tensor, dtype: torch.long, shape: [4, batch_size]
        : Tensor containing the integer key of heads, tails, edges and triplets of the edges in the current batch.
        : Here, batch_size is batch.shape[1].
        negative_triplet_count: int, optional, default to 1
        : Number of negative samples to create from each triplet.

        Returns
        -------
        
        combined_negative_triplets_batch: torch.Tensor, dtype: torch.long, shape: [4, 2 * negative_triplet_count * batch_size + batch_size]
        : Tensor containing the integer key of negatively sampled heads and tails from both samplers.
        : Here, batch_size is batch.shape[1].
        
        """
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
                                                ], dim = 1)
        
        return combined_negative_triplets_batch
