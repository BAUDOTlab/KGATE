"""
Negative sampling classes, to generate negative triplets during training.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.

"""

from typing import Dict, Set, Tuple, List
from collections import defaultdict

import torch
from torch import tensor, bernoulli, randint, ones, rand, cat
from torch.types import Number, Tensor

from .knowledgegraph import KnowledgeGraph
from .utils import get_bernoulli_probabilities

class NegativeSampler:
    """This class is a simple interface to ease typing and use of negative samplers."""
    def corrupt_batch(  self,
                        batch: torch.Tensor,
                        negative_triplet_count = None
                        ) -> Tensor:
        raise NotImplementedError()

class UniformNegativeSampler:
    """
    TODO.What_the_class_is_about_globally
    For each edge, choose simultenously head and tail from Bernoulli random distribution.
    
    Check that no true triplet is created by accident.

    If the corrupted triplet is of a type that doesn't exist in the original knowledge graph,
    it is created.

    References
    ----------
    TODO

    Arguments
    ---------
    kg: KnowledgeGraph
        Knowledge graph on which the sampling will be done.
    negative_triplet_count: int, optional, default to 1
        Number of negative samples to create from each triplet. If None, the class-level
        `n_neg` value is used.

    Attributes
    ----------
    index_to_node_type: Dict[int, str]
        keys: node index
    edge_types: Dict[int, str]
        keys: edge index
        values: edge name
    kg: KnowledgeGraph
        Knowledge graph on which the sampling will be done.
    n_neg: int
        Number of negative samples to create from each triplet.
        Inherited attribute, equivalent to negative_triplet_count.
    n_ent: int
        Number of nodes.
        Inherited attribute equivalent to node_count.
    
    TODO.inherited_attributes
    
    """
    def __init__(self,
                knowledge_graph: KnowledgeGraph,
                negative_triplet_count = 1):
        
        self.knowledge_graph = knowledge_graph
        self.index_to_node_type: Dict[int, str] = {value: key for key,value in self.knowledge_graph.node_type_to_index.items()}
        self.edge_types: Dict[int, str] = {value: key for key,value in self.knowledge_graph.edge_to_index.items()}
    
        self.negative_triplet_count = negative_triplet_count
    
    
    def corrupt_batch(  self,
                        batch: torch.Tensor,
                        negative_triplet_count = None
                        ) -> Tensor:
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
            Number of negative samples to create from each triplet. If None, self.negative_triplet_count is used.

        Returns
        -------
        negative_triplets_batch: torch.Tensor, dtype: torch.long, shape: [4, negative_triplet_count * batch_size]
            Tensor containing the integer key of negatively sampled triplets of
            the edges in the current batch.
            Here, batch_size is batch.shape[1].
            
        """
        negative_triplet_count = negative_triplet_count or self.negative_triplet_count

        device = batch.device
        batch_size = batch.shape[1]
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



class BernoulliNegativeSampler:
    """
    TODO.What_the_class_is_about_globally
    For each edge, choose head from Bernoulli random distribution, then tail from Bernoulli random distribution.
    
    Check that no true triplet is created by accident.

    If the corrupted triplet is of a type that doesn't exist in the original knowledge graph,
    it is created.

    References
    ----------
    TODO

    Arguments
    ---------
    kg: KnowledgeGraph
        Knowledge graph on which the sampling will be done.
    negative_triplet_count: int, optional, default to 1
        Number of negative samples to create from each triplet. If None, the class-level
        `n_neg` value is used.

    Attributes
    ----------
    index_to_node_type: Dict[int, str]
        keys: node index
    edge_types: Dict[int, str]
        keys: edge index
        values: edge name
    kg: KnowledgeGraph
        Knowledge graph on which the sampling will be done.
    n_neg: int
        Number of negative samples to create from each triplet.
        Inherited attribute, equivalent to negative_triplet_count.
    n_ent: int
        Number of nodes.
        Inherited attribute equivalent to node_count.
    bernoulli_probabilities: torch.Tensor, dtype: torch.float, shape: [edge_count]
        Tensor containing the probabilities of sampling a head for each edge.
    TODO.inherited_attributes
    
    """
    def __init__(self,
                knowledge_graph: KnowledgeGraph,
                negative_triplet_count = 1):
        
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
        bernoulli_probabilities: torch.Tensor, dtype: torch.float, shape: [edge_count]
            Tensor containing the probabilities of sampling a head for each edge.
        
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
                        batch: torch.LongTensor,
                        negative_triplet_count = None):
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
        negative_triplet_count: int, optional
            Number of negative samples to create from each triplet. If None, the class-level
            `n_neg` value is used.

        Returns
        -------
        negative_triplets_batch: torch.Tensor, dtype: torch.long, shape: [4, negative_triplet_count * batch_size]
            Tensor containing the integer key of negatively sampled triplets of
            the edges in the current batch.
            Here, batch_size is batch.shape[1].
            
        """
        negative_triplet_count = negative_triplet_count or self.n_neg

        device = batch.device
        batch_size = batch.shape[1]
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
    """
    Adaptation of torchKGE's PositionalNegativeSampler to KGATE's graphindices format.

    Either the head or the tail of a triplet is replaced by another node
    chosen among nodes that have already appeared at the same place in a
    triplet (involving the same edge), using bernoulli sampling.

    If the corrupted triplet is of a type that doesn't exist in the original knowledge graph,
    it is created.

    Arguments
    ---------
    kg: kgate.data_structure.KnowledgeGraph
        Knowledge Graph from which the corrupted triplets will be created.

    Attributes
    ----------
    possible_heads: Dict[int, torch.Tensor]
        keys: edges
        values: list of number of possible heads for each edge, equivalent to possible_head_count
    possible_tails: Dict[int, torch.Tensor]
        keys: edges
        values: list of number of possible tails for each edge, equivalent to possible_tail_count
    possible_head_count: torch.Tensor
        List of number of possible heads for each edge.
        Equivalent of List[int], but with Tensor possibilities.
    possible_tail_count: torch.Tensor
        List of number of possible tails for each edge.
        Equivalent of List[int], but with Tensor possibilities.
    index_to_node_type: Dict[int, str]
        keys: node index
        values: node types
    edge_types: Dict[int, str]
        keys: edge index
        values: edge name
    kg: KnowledgeGraph
        Knowledge graph on which the sampling will be done.
    node_count: int
        Number of nodes.
    bernoulli_probabilities: torch.Tensor, dtype: torch.float, shape: [edge_count]
        Tensor containing the probabilities of sampling a head for each edge.
    TODO.inherited_attributes
    
    Notes
    -----
    Also fixes GPU/CPU incompatibility bug.
    See original implementation here: https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/sampling.py#L330C52-L330C53
    
    Slower than UniformNegativeSampler, BernoulliNegativeSampler and MixedNegativeSampler, as it searches
    in the entire knowledge graph instead of a batch.
    
    """
    def __init__(self, knowledge_graph: KnowledgeGraph):
        super.__init__(knowledge_graph)

        self.possible_heads, self.possible_tails, \
            self.possible_head_count, self.possible_tail_count = self.find_possibilities()


    def find_possibilities(self) -> Tuple[
                                Dict[int, List[int]],
                                Dict[int, List[int]], 
                                Tensor, 
                                Tensor]:
        """
        For each edge of the knowledge graph (and possibly the
        validation graph but not the test graph) find all the possible heads
        and tails in the sense of Wang et al., e.g. all nodes that occupy
        once this position in another triplet.

        Returns
        -------
        possible_heads: Dict[int, List[int]]
            keys : edge index, values : list of possible heads
        possible tails: Dict[int, List[int]]
            keys : edge index, values : list of possible tails
        possible_heads_count: torch.Tensor, dtype: torch.long, shape: (edge_count)
            Number of possible heads for each edge.
        possible_tails_count: torch.Tensor, dtype: torch.long, shape: (edge_count)
            Number of possible tails for each edge.
        
        """
        possible_heads = defaultdict(set)
        possible_tails = defaultdict(set)
        for triplet_index in range(self.knowledge_graph.triplet_count):
            possible_heads[self.knowledge_graph.edge_indices[triplet_index].item()].add(self.knowledge_graph.head_indices[triplet_index].item())
            possible_tails[self.knowledge_graph.edge_indices[triplet_index].item()].add(self.knowledge_graph.tail_indices[triplet_index].item())

        possible_heads_count = []
        possible_tails_count = []

        for edge_index in range(self.knowledge_graph.edge_count):
            if edge_index in possible_heads.keys():
                possible_heads_count.append(len(possible_heads[edge_index]))
                possible_tails_count.append(len(possible_tails[edge_index]))
                possible_heads[edge_index] = list(possible_heads[edge_index])
                possible_tails[edge_index] = list(possible_tails[edge_index])
            else:
                possible_heads_count.append(0)
                possible_tails_count.append(0)
                possible_heads[edge_index] = list(possible_heads[edge_index])
                possible_tails[edge_index] = list(possible_tails[edge_index])

        return possible_heads, possible_tails, torch.tensor(possible_heads_count), torch.tensor(possible_tails_count)


    def corrupt_batch(  self,
                        batch: Tensor,
                        _: int = 0
                        ) -> Tensor:
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

        Raises
        ------
        AssertionError #1
            The size/shape of possible_head_count must be corrupted_head_count.
        AssertionError #2
            The size/shape of possible_head_count must be (batch_size - corrupted_head_count).

        Returns
        -------
        negative_triplets_batch: torch.Tensor, dtype: torch.long, shape: [4, batch_size]
            Tensor containing the integer key of negatively sampled triplets of
            the edges in the current batch.
            Here, batch_size is batch.shape[1].
        
        """
        edges = batch[2]
        device = batch.device
        node_types = self.knowledge_graph.node_types
        triplet_types = self.knowledge_graph.triplet_types

        batch_size = batch.shape[1]
        negative_triplets_batch: Tensor = batch.clone().long()

        self.bernoulli_probabilities = self.bernoulli_probabilities.to(device)
        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bernoulli_probabilities[edges]).double()
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
        triplets = [0] * corrupted_head_count if len(self.knowledge_graph.node_type_to_index) == 1 else []
        for i in range(corrupted_head_count):
            edge_index = corrupted_head_batch[2][i].item()
            choices: Dict[Number, Set[Number]] = self.possible_heads[edge_index]
            if len(choices) == 0:
                # In this case the edge i has never been used with any head
                # Choose one node at random
                corrupted_head_index = randint(low = 0, high = self.node_count, size = (1,)).item()
            else:
                corrupted_head_index = choices[chosen_head[i].item()]
            corrupted_heads.append(corrupted_head_index)
            # If we don't use metadata, there is only 1 node type
            if len(self.knowledge_graph.node_type_to_index) > 1:
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
        triplets = [0] * (batch_size - corrupted_head_count) if len(self.knowledge_graph.node_type_to_index) == 1 else []
        for i in range(batch_size - corrupted_head_count):
            edge_index = corrupted_tail_batch[2][i].item()
            choices: Dict[Number, Set[Number]] = self.possible_tails[edge_index]
            if len(choices) == 0:
                # In this case the edge i has never been used with any tail
                # Choose one node at random
                corrupted_tail_index = randint(low = 0, high = self.node_count, size = (1,)).item()
            else:
                corrupted_tail_index = choices[chosen_tail[i].item()]
            # If we don't use metadata, there is only 1 node type
            if len(self.knowledge_graph.node_type_to_index) > 1:
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



class MixedNegativeSampler:
    """
    A custom negative sampler that combines the BernoulliNegativeSampler, the UniformNegativeSampler
    and the PositionalNegativeSampler. 
    
    For each triplet, it samples `negative_triplet_count` negative samples for each samplers except the Positional. Note
    that the PositionalNegativeSampler always produces only one negative triplet per positive triplet.
    
    Arguments
    ---------
    kg: KnowledgeGraph
        Main knowledge graph (usually training one).
    negative_triplet_count: int, optional, default to 1
        Third of the number of negative samples to create from each triplet. Since it uses 3 sampler
        methods, it generates 3 times the amount of negative_triplet_count indicated.
        If None, the class-level `n_neg` value is used.

    Attributes
    ----------
    n_neg: int
        Number of negative samples to create from each triplet.
        Inherited attribute, equivalent to negative_triplet_count.
    uniform_sampler: UniformNegativeSampler
        TODO.brief_description_of_the_class
    bernoulli_sampler: BernoulliNegativeSampler
        TODO.brief_description_of_the_class
    positional_sampler: PositionalNegativeSampler
        TODO.brief_description_of_the_class
    TODO.inherited_attributes
    
    Notes
    -----
    This is an example of a custom negative sampler using other existing samplers, and may produce
    unexpected behaviour if used as is.
    
    """
    
    def __init__(self,
                knowledge_graph: KnowledgeGraph,
                negative_triplet_count = 1):
        
        self.knowledge_graph = knowledge_graph
        self.index_to_node_type: Dict[int, str] = {value: key for key,value in self.knowledge_graph.node_type_to_index.items()}
        self.edge_types: Dict[int, str] = {value: key for key,value in self.knowledge_graph.edge_to_index.items()}
    
        self.negative_triplet_count = negative_triplet_count

        # Initialize both Bernoulli, Uniform and Positional samplers
        self.uniform_sampler = UniformNegativeSampler(self.knowledge_graph, negative_triplet_count = negative_triplet_count)
        self.bernoulli_sampler = BernoulliNegativeSampler(self.knowledge_graph, negative_triplet_count = negative_triplet_count)
        self.positional_sampler = PositionalNegativeSampler(self.knowledge_graph)
        
        
    def corrupt_batch(  self,
                        batch: torch.LongTensor,
                        negative_triplet_count = None):
        """
        For each true triplet, produce `negative_triplet_count` corrupted ones from the
        Uniform sampler, the Bernoulli sampler and the Positional sampler. If `heads` and `tails` are
        cuda objects, then the returned tensors are on the GPU.

        Arguments
        ---------
        batch: torch.Tensor, dtype: torch.long, shape: [4, batch_size]
            Tensor containing the integer key of heads, tails, edges and triplets
            of the edges in the current batch.
            Here, batch_size is batch.shape[1].
        negative_triplet_count: int, optional, default to None
            Number of negative samples to create from each triplet. If None, the class-level
            `n_neg` value is used.

        Returns
        -------
        combined_negative_triplets_batch: torch.Tensor, dtype: torch.long, shape: [4, 2 * negative_triplet_count * batch_size + batch_size]
            Tensor containing the integer key of negatively sampled heads and tails from both samplers.
            Here, batch_size is batch.shape[1].
        
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
                                                ], dim = 1)
        
        return combined_negative_triplets_batch