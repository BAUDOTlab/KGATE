"""
Evaluator classes to evaluate model performances.

Original code for the predictors from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Dict

from tqdm import tqdm

import torch
from torch import empty, zeros, cat, Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

import torchkge.evaluation as eval
from torchkge.utils import get_rank
from torchkge.data_structures import SmallKG
from torchkge.models import Model

from torch_geometric.utils import k_hop_subgraph

from .knowledgegraph import KnowledgeGraph
from .utils import filter_scores
from .samplers import PositionalNegativeSampler
from .encoders import GNN, DefaultEncoder


class LinkPredictionEvaluator(eval.LinkPredictionEvaluator):
    """Evaluate performance of given embedding using link prediction method.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
      and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    Parameters
    ----------
    full_edgelist: Tensor
        Tensor of shape [4,n_triples] containing every true triple. 

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    rank_true_heads: torch.Tensor, shape: (n_triples), dtype: `torch.int`
        For each fact, this is the rank of the true head when all entities
        are ranked as possible replacement of the head entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    rank_true_tails: torch.Tensor, shape: (n_triples), dtype: `torch.int`
        For each fact, this is the rank of the true tail when all entities
        are ranked as possible replacement of the tail entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    filt_rank_true_heads: torch.Tensor, shape: (n_triples), dtype: `torch.int`
        This is the same as the `rank_of_true_heads` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    filt_rank_true_tails: torch.Tensor, shape: (n_triples), dtype: `torch.int`
        This is the same as the `rank_of_true_tails` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    evaluated: bool
        Indicates if the method LinkPredictionEvaluator.evaluate has already
        been called.

    """

    def __init__(self, full_graphindices: Tensor):
        self.full_graphindices = full_graphindices
        self.evaluated = False

    def evaluate(self, 
                batch_size: int,
                encoder: DefaultEncoder | GNN,
                decoder: Model, 
                knowledge_graph: KnowledgeGraph, 
                node_embeddings: nn.ParameterList, 
                edge_embeddings: nn.Embedding,
                verbose: bool=True):
        """
        Run the Link Prediction evaluation.

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        encoder: DefaultEncoder or GNN
            Encoder model to embed the nodes. Deactivated with DefaultEncoder
        decoder: torchkge.Model
            Decoder model to evaluate, inheriting from the torchkge.Model class.
        knowledge_graph: kgate.KnowledgeGraph
            The test Knowledge Graph that will be used for the evaluation.
        node_embeddings: nn.ParameterList
            
        relation_embeddings: nn.Embedding
            A tensor containing one embedding by relation type.
        mappings: kgate.HeteroMappings
            An object containing mapping between the knowledge graph and 
            embeddings.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.
        """
        device = edge_embeddings.weight.device
        use_cuda = edge_embeddings.weight.is_cuda

        self.rank_true_heads = empty(size=(knowledge_graph.triplet_count,)).long().to(device)
        self.rank_true_tails = empty(size=(knowledge_graph.triplet_count,)).long().to(device)
        self.filtered_rank_true_heads = empty(size=(knowledge_graph.triplet_count,)).long().to(device)
        self.filtered_rank_true_tails = empty(size=(knowledge_graph.triplet_count,)).long().to(device)

        dataloader = DataLoader(knowledge_graph, batch_size=batch_size)
        edgelist = knowledge_graph.graphindices.to(device)
        if decoder is not None and hasattr(decoder,"embedding_spaces"):
            encoder_node_embedding_dimensions = decoder.emb_dim * decoder.embedding_spaces
        else:
            encoder_node_embedding_dimensions = decoder.emb_dim

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit="batch", disable=(not verbose),
                             desc="Link prediction evaluation"):
            batch:Tensor = batch.T.to(device)
            head_index, tail_index, edge_index = batch[0], batch[1], batch[2]

            if isinstance(encoder, GNN):
                seed_nodes = batch[:2].unique()
                hop_count = encoder.n_layers
                edge_index = knowledge_graph.edge_list

                _,_,_, edge_mask = k_hop_subgraph(
                    node_idx = seed_nodes,
                    num_hops = hop_count,
                    edge_index = edge_index
                    )
                
                input = knowledge_graph.get_encoder_input(edgelist[:, edge_mask], node_embeddings)
                encoder_output: Dict[str, Tensor] = encoder(input.x_dict, input.edge_list)
                
                embeddings: torch.Tensor = torch.zeros((knowledge_graph.node_count, encoder_node_embedding_dimensions), device=device, dtype=torch.float)

                for node_type, index in input.mapping.items():
                    embeddings[index] = encoder_output[node_type]
            else:
                embeddings = node_embeddings[0].data

            head_embeddings, tail_embeddings, edge_embeddings, candidates = decoder.inference_prepare_candidates(h_idx = head_index, 
                                                                                   t_idx = tail_index, 
                                                                                   r_idx = edge_index, 
                                                                                   node_embeddings = embeddings, 
                                                                                   relation_embeddings = edge_embeddings,
                                                                                   entities=True)

            scores = decoder.inference_scoring_function(head_embeddings, candidates, edge_embeddings)
            filtered_scores = filter_scores(
                scores = scores, 
                edgelist = self.full_graphindices.to(device),
                missing = "tail",
                first_index=head_index,
                second_index=edge_index,
                true_index=tail_index
            )
            self.rank_true_tails[i * batch_size: (i + 1) * batch_size] = get_rank(scores, tail_index).detach()
            self.filtered_rank_true_tails[i * batch_size: (i + 1) * batch_size] = get_rank(filtered_scores, tail_index).detach()

            scores = decoder.inference_scoring_function(candidates, tail_embeddings, edge_embeddings)
            filtered_scores = filter_scores(
                scores = scores, 
                edgelist = self.full_graphindices.to(device),
                missing = "head",
                first_index=tail_index,
                second_index=edge_index,
                true_index=head_index
            )
            self.rank_true_heads[i * batch_size: (i + 1) * batch_size] = get_rank(scores, head_index).detach()
            self.filtered_rank_true_heads[i * batch_size: (i + 1) * batch_size] = get_rank(filtered_scores, head_index).detach()

        self.evaluated = True

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filtered_rank_true_heads = self.filtered_rank_true_heads.cpu()
            self.filtered_rank_true_tails = self.filtered_rank_true_tails.cpu()


class TripletClassificationEvaluator(eval.TripletClassificationEvaluator):
    """Evaluate performance of given embedding using triplet classification
    method.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
      Reasoning With Neural Tensor Networks for Knowledge Base Completion.
      In Advances in Neural Information Processing Systems 26, pages 926-934.
      2013.
      https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg_val: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the testing evaluation will be done.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg_val: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    evaluated: bool
        Indicate whether the `evaluate` function has been called.
    thresholds: float
        Value of the thresholds for the scoring function to consider a
        triplet as true. It is defined by calling the `evaluate` method.
    sampler: torchkge.sampling.NegativeSampler
        Negative sampler.

    """

    def __init__(self, architect, kg_validation, kg_test):
        self.architect = architect
        self.kg_validation = kg_validation
        self.kg_test = kg_test
        self.is_cuda = self.architect.device.type == "cuda"

        self.evaluated = False
        self.thresholds = None

        self.sampler = PositionalNegativeSampler(self.kg_validation)

    def get_scores(self, heads: Tensor, tails: Tensor, edges: Tensor, batch_size: int):
        """With head, tail and relation indexes, compute the value of the
        scoring function of the model.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: n_triples
            List of heads indices.
        tails: torch.Tensor, dtype: torch.long, shape: n_triples
            List of tails indices.
        relations: torch.Tensor, dtype: torch.long, shape: n_triples
            List of relation indices.
        batch_size: int

        Returns
        -------
        scores: torch.Tensor, dtype: torch.float, shape: n_triples
            List of scores of each triplet.
        """
        scores = []
        #print(heads, heads.shape, tails, tails.shape, relations, relations.shape)

        small_kg = SmallKG(heads, tails, edges)
        if self.is_cuda:
            dataloader = DataLoader(small_kg, batch_size=batch_size,
                                    use_cuda="batch")
        else:
            dataloader = DataLoader(small_kg, batch_size=batch_size)

        for i, batch in enumerate(dataloader):
            head_index, tail_index, edge_index = batch[0].to(self.architect.device), batch[1].to(self.architect.device), batch[2].to(self.architect.device)
            scores.append(self.architect.scoring_function(head_index, tail_index, edge_index, train = False))

        return cat(scores, dim=0)

    def evaluate(self, batch_size: int, knowledge_graph: KnowledgeGraph):
        """Find relation thresholds using the validation set. As described in
        the paper by Socher et al., for a relation, the threshold is a value t
        such that if the score of a triplet is larger than t, the fact is true.
        If a relation is not present in any fact of the validation set, then
        the largest value score of all negative samples is used as threshold.

        Parameters
        ----------
        b_size: int
            Batch size.
        """
        sampler = PositionalNegativeSampler(knowledge_graph)
        edge_indices = knowledge_graph.edge_indices

        negative_heads, negative_tails = sampler.corrupt_kg(batch_size, self.is_cuda,
                                                       which="main")
        negative_scores = self.get_scores(negative_heads, negative_tails, edge_indices, batch_size)

        self.thresholds = zeros(self.kg_validation.n_rel)

        for i in range(self.kg_validation.n_rel):
            mask = (edge_indices == i).bool()
            if mask.sum() > 0:
                self.thresholds[i] = negative_scores[mask].max()
            else:
                self.thresholds[i] = negative_scores.max()

        self.evaluated = True
        self.thresholds.detach_()

    def accuracy(self, batch_size:int, kg_test: KnowledgeGraph, kg_validation: KnowledgeGraph | None = None):
        """

        Parameters
        ----------
        b_size: int
            Batch size.

        Returns
        -------
        acc: float
            Share of all triplets (true and negatively sampled ones) that where
            correctly classified using the thresholds learned from the
            validation set.

        """
        if not self.evaluated:
            kg_to_evaluate = kg_validation if kg_validation is not None else kg_test
            self.evaluate(batch_size=batch_size, knowledge_graph=kg_to_evaluate)

        sampler = PositionalNegativeSampler(kg_test)
        edge_indices = kg_test.edge_indices

        negative_heads, negative_tails = sampler.corrupt_kg(batch_size,
                                                self.is_cuda,
                                                which="main")
        scores = self.get_scores(kg_test.head_indices,
                                 kg_test.tail_indices,
                                 edge_indices,
                                 batch_size)
        negative_scores = self.get_scores(negative_heads, negative_tails, edge_indices, batch_size)

        if self.is_cuda:
            self.thresholds = self.thresholds.cuda()
            
        scores = (scores > self.thresholds[edge_indices])
        negative_scores = (negative_scores < self.thresholds[edge_indices])

        return (scores.sum().item() +
                negative_scores.sum().item()) / (2 * self.kg_test.n_triples)