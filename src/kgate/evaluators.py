"""
Evaluator classes to evaluate model performances.

Original code for the predictors from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Dict, Tuple, TYPE_CHECKING

from tqdm import tqdm

import torch
from torch import empty, zeros, cat, Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_geometric.utils import k_hop_subgraph

from torchkge.utils import get_rank
from torchkge.data_structures import SmallKG

if TYPE_CHECKING:
    from .architect import Architect
from .decoders import BilinearDecoder, ConvolutionalDecoder, TranslationalDecoder
from .encoders import GNN, DefaultEncoder
from .knowledgegraph import KnowledgeGraph
from .samplers import NegativeSampler, PositionalNegativeSampler, BernoulliNegativeSampler, UniformNegativeSampler, MixedNegativeSampler
from .utils import filter_scores



class Predictions:
    """
    Object holding the predictions output of an Evaluator.
    
    Predictions are available as a dataframe, but can also be accessed through
    builtin methods to get specific metrics.
    
    TODO: the object could be better structured altogether

    Arguments
    ---------
    true_predictions_rank: torch.Tensor
        Among the ranking of all predictions, the rank of the true result.
    filtered_true_predictions_rank: torch.Tensor
        Among the ranking of all filtered predictions, the rank of the true result.
        True triplets that are not the target of the prediction are filtered out.

    Attributes
    ----------
    true_predictions_rank: torch.Tensor
        Among the ranking of all predictions, the rank of the true result.
    filtered_true_predictions_rank: torch.Tensor
        Among the ranking of all filtered predictions, the rank of the true result.
        True triplets that are not the target of the prediction are filtered out.
    
    
    """
    def __init__(self,
                true_predictions_rank: Tensor,
                filtered_true_predictions_rank: Tensor):

        self.true_predictions_rank = true_predictions_rank
        self.filtered_true_predictions_rank = filtered_true_predictions_rank
    
    
    def __str__(self):
        k = 10
        message = f"""
        Hit@{k}: {round(self.hit_at_k(k)[0],3)} \t Filtered Hit@{k}: {round(self.hit_at_k(k)[1],3)} 

        MRR: {round(self.mrr[0],3)} \t Filtered MRR: {round(self.mrr[1],3)}

        Mean Rank: {int(self.mean_rank[0])} \t Filtered Mean Rank: {int(self.mean_rank[1])}
        """
        
        return message


    @property
    def mean_rank(self) -> Tuple[float, float]:
        """
        Mean rank metric
        
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO.reference
        
        Returns
        -------
        mean_rank_score: float
            Mean value of `true_predictions_rank` scores.
            Among the ranking of all predictions, `true_predictions_rank` is the rank of the true result.
        filtered_mean_rank_score: float
            Mean value of `filtered_true_predictions_rank` scores.
            Among the ranking of all predictions, `filtered_true_predictions_rank` is the rank of the true result.
            True triplets that are not the target of the prediction are filtered out.
        
        """
        mean_rank_score = self.true_predictions_rank.float().mean().item()

        filtered_mean_rank_score = self.filtered_true_predictions_rank.float().mean().item()

        return mean_rank_score, filtered_mean_rank_score
    
    
    def hit_at_k(self,
                k: int = 10
                ) -> Tuple[float, float]:
        """
        Return the frequence at which the true triplet is within the k first predictions.

        References
        ----------
        TODO.reference
        
        Arguments
        ---------
        k: int, default to 10
            The true triplet must be within the k first predictions.
        
        Returns
        -------
        true_prediction_hit: float
            Frequence at which the true triplet is within the k first predictions.
        filtered_true_prediction_hit: float
            Frequence at which the true triplet is within the k first predictions, when ranking among filtered triplets.
            True triplets that are not the target of the prediction are filtered out.
        
        """
        true_prediction_hit = (self.true_predictions_rank <= k).float().mean().item()
        filtered_true_prediction_hit = (self.filtered_true_predictions_rank <= k).float().mean().item()
        
        return true_prediction_hit, filtered_true_prediction_hit
    
    
    @property
    def mrr(self) -> Tuple[float, float]:
        """
        Mean reciprocal rank
        
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO.reference

        Returns
        -------
        mrr: float
            Inverse of the position of the true triplet prediction.
            If the true triplet is predicted in 100th position, then mrr = 0.01
            Perfect score is 1.
        filtered_mrr: float
            Inverse of the position of the true triplet filtered prediction.
            If the true triplet is predicted in 100th position, then mrr = 0.01
            Perfect score is 1.
            True triplets that are not the target of the prediction are filtered out.
        
        """
        mrr = (self.true_predictions_rank.float()**(-1)).mean().item()
        filtered_mrr = (self.filtered_true_predictions_rank.float()**(-1)).mean().item()

        return mrr, filtered_mrr



class LinkPredictionEvaluator:
    """
    Evaluate performance of given embedding using link prediction method.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
        and Oksana Yakhnenko.
    `Translating Embeddings for Modeling Multi-relational Data.`
    https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    In Advances in Neural Information Processing Systems 26, pages 2787–2795. 2013.

    Arguments
    ---------
    full_graphindices: torch.Tensor
        Tensor of shape [4, triplet_count] containing every true triplet.
    embedding_dimensions: int
        Dimensions of embeddings.
        
    Attributes
    ----------
    full_graphindices: torch.Tensor
        Tensor of shape [4, triplet_count] containing every true triplet.
    evaluated: bool
        Indicate whether the method LinkPredictionEvaluator.evaluate has already
        been called.
    rank_true_heads: torch.Tensor, shape: [triplet_count], dtype: `torch.int`
        For each fact, this is the rank of the true head when all nodes
        are ranked as possible replacement of the head node. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    rank_true_tails: torch.Tensor, shape: [triplet_count], dtype: `torch.int`
        For each fact, this is the rank of the true tail when all nodes
        are ranked as possible replacement of the tail node. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    filtered_rank_true_heads: torch.Tensor, shape: [triplet_count], dtype: `torch.int`
        This is the same as the `rank_of_true_heads` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    filtered_rank_true_tails: torch.Tensor, shape: [triplet_count], dtype: `torch.int`
        This is the same as the `rank_of_true_tails` when in the filtered
        case. See referenced paper by Bordes et al. for more information.

    """
    def __init__(self, full_graphindices: Tensor, embedding_dimensions: int):
        self.full_graphindices = full_graphindices
        self.embedding_dimensions = embedding_dimensions
        self.evaluated = False


    def evaluate(self,
                batch_size: int,
                encoder: DefaultEncoder | GNN,
                decoder: BilinearDecoder | ConvolutionalDecoder | TranslationalDecoder,
                knowledge_graph: KnowledgeGraph,
                node_embeddings: nn.ParameterList,
                edge_embeddings: nn.Embedding,
                verbose: bool = True
                ) -> Tuple[Predictions, Predictions]:
        """
        Run the Link Prediction evaluation.

        Arguments
        ---------
        batch_size: int
            Size of the current batch.
        encoder: DefaultEncoder or GNN
            Encoder model to embed the nodes. Deactivated with DefaultEncoder.
        decoder: BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder
            Decoder model to evaluate.
        knowledge_graph: KnowledgeGraph
            Knowledge graph on which the evaluation will be done.
        node_embeddings: nn.ParameterList, keyword-only
            A list containing all embeddings for each node type.
            keys: node type index
            values: tensors of shape (node_count, embedding_dimensions)
        edge_embeddings: nn.Embedding, keyword-only
            A tensor containing one embedding by edge type, of shape (edge_count, embedding_dimensions).
        verbose: bool
            Indicate whether a progress bar should be displayed during
            evaluation.
        
        Returns
        -------
        head_predictions: Predictions
            Predictions for heads.
        tail_predictions: Predictions
            Predictions for tails.
        
        """
        device = edge_embeddings.weight.device

        self.rank_true_heads = empty(size = (knowledge_graph.triplet_count,)).long().to(device)
        self.rank_true_tails = empty(size = (knowledge_graph.triplet_count,)).long().to(device)
        self.filtered_rank_true_heads = empty(size = (knowledge_graph.triplet_count,)).long().to(device)
        self.filtered_rank_true_tails = empty(size = (knowledge_graph.triplet_count,)).long().to(device)

        dataloader = DataLoader(knowledge_graph, batch_size = batch_size)
        graphindices = knowledge_graph.graphindices.to(device)
        if decoder is not None and hasattr(decoder,"embedding_spaces"):
            encoder_node_embedding_dimensions: int = self.embedding_dimensions * decoder.embedding_spaces
        else:
            encoder_node_embedding_dimensions: int = self.embedding_dimensions

        for i, batch in tqdm(enumerate(dataloader),
                            total = len(dataloader),
                            unit = "batch",
                            disable = (not verbose),
                            desc = "Link prediction evaluation"):
            batch: Tensor = batch.T.to(device)
            head_index, tail_index, edge_index = batch[0], batch[1], batch[2]

            if isinstance(encoder, GNN):
                seed_nodes: Tensor = batch[:2].unique()
                hop_count: int = encoder.layer_count
                edge_list: Tensor = knowledge_graph.edge_list

                _, _, _, edge_mask = k_hop_subgraph(
                    node_idx = seed_nodes,
                    num_hops = hop_count,
                    edge_index = edge_list
                    )
                
                input = knowledge_graph.get_encoder_input(graphindices[:, edge_mask], node_embeddings)
                encoder_output: Dict[str, Tensor] = encoder(input.x_dict, input.edge_index)
                
                evaluation_node_embeddings: torch.Tensor = torch.zeros((knowledge_graph.node_count,
                                                            encoder_node_embedding_dimensions),
                                                            device = device,
                                                            dtype = torch.float)

                for node_type, index in input.mapping.items():
                    evaluation_node_embeddings[index] = encoder_output[node_type]
            else:
                evaluation_node_embeddings = node_embeddings[0].data

            head_embeddings, tail_embeddings, inference_edge_embeddings, candidates = decoder.inference_prepare_candidates(head_indices = head_index, 
                                                                                                                tail_indices = tail_index, 
                                                                                                                edge_indices = edge_index, 
                                                                                                                node_embeddings = evaluation_node_embeddings, 
                                                                                                                edge_embeddings = edge_embeddings,
                                                                                                                node_inference = True)

            scores = decoder.inference_score(
                head_embeddings = head_embeddings, 
                tail_embeddings = candidates, 
                edge_embeddings = inference_edge_embeddings
                )
            filtered_scores = filter_scores(
                scores = scores, 
                graphindices = self.full_graphindices.to(device),
                missing = "tail",
                first_index = head_index,
                second_index = edge_index,
                true_index = tail_index
            )
            self.rank_true_tails[i * batch_size: (i + 1) * batch_size] = get_rank(scores, tail_index).detach()
            self.filtered_rank_true_tails[i * batch_size: (i + 1) * batch_size] = get_rank(filtered_scores, tail_index).detach()

            scores = decoder.inference_score(
                head_embeddings = candidates,
                tail_embeddings = tail_embeddings,
                edge_embeddings = inference_edge_embeddings)
            filtered_scores = filter_scores(
                scores = scores, 
                graphindices = self.full_graphindices.to(device),
                missing = "head",
                first_index = tail_index,
                second_index = edge_index,
                true_index = head_index
            )
            self.rank_true_heads[i * batch_size: (i + 1) * batch_size] = get_rank(scores, head_index).detach()
            self.filtered_rank_true_heads[i * batch_size: (i + 1) * batch_size] = get_rank(filtered_scores, head_index).detach()

        self.evaluated = True

        head_predictions = Predictions(self.rank_true_heads.cpu(), self.filtered_rank_true_heads.cpu())
        tail_predictions = Predictions(self.rank_true_tails.cpu(), self.filtered_rank_true_tails.cpu())

        return head_predictions, tail_predictions



class TripletClassificationEvaluator:
    """
    Evaluates performance of given embedding using triplet classification
    method.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
    `Reasoning With Neural Tensor Networks for Knowledge Base Completion.`
    https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    In Advances in Neural Information Processing Systems 26, pages 926-934. 2013.

    Arguments
    ---------
    architect: Architect
        Embedding model inheriting from the right interface.
    kg_validation: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the testing evaluation will be done.

    Attributes
    ----------
    architect: Architect
        Embedding model inheriting from the right interface.
    kg_validation: KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    device: str, "cuda" or "cpu", default to "cuda"
        Indicate if data should be sent to GPU or CPU.
        GPU is referenced to as Cuda.
    evaluated: bool, default to False
        Indicate whether the `evaluate` function has already been called.
    thresholds: float
        Value of the thresholds for the scoring function to consider a
        triplet as true. It is defined by calling the `evaluate` method.
    sampler: torchkge.sampling.NegativeSampler
        Negative sampler.

    """
    def __init__(self,
                architect: "Architect",
                kg_validation: KnowledgeGraph,
                kg_test: KnowledgeGraph):
        
        self.architect = architect
        self.kg_validation = kg_validation
        self.kg_test = kg_test
        self.device = self.architect.device.type == "cuda"

        self.evaluated = False
        self.thresholds = None

        # PositionalNegativeSampler specifically as done in TorchKGE
        # following the original paper: https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
        self.sampler = PositionalNegativeSampler(self.kg_validation)


    def get_scores( self,
                    heads: Tensor,
                    tails: Tensor,
                    edges: Tensor,
                    batch_size: int
                    ) -> Tensor:
        """
        With head, tail and edge indices, compute the value of the
        scoring function of the model.

        Arguments
        ---------
        heads: torch.Tensor, dtype: torch.long, shape: triplet_count
            List of head indices.
        tails: torch.Tensor, dtype: torch.long, shape: triplet_count
            List of tail indices.
        edges: torch.Tensor, dtype: torch.long, shape: triplet_count
            List of edge indices.
        batch_size: int
            Size of the current batch.

        Returns
        -------
        scores: torch.Tensor, dtype: torch.float, shape: triplet_count
            List of scores of each triplet.
        
        """
        
        scores = []

        small_kg = SmallKG(heads, tails, edges)
        if self.is_cuda:
            dataloader = DataLoader(small_kg,
                                    batch_size = batch_size)
        else:
            dataloader = DataLoader(small_kg,
                                    batch_size = batch_size)

        for _, batch in enumerate(dataloader):
            head_index, tail_index, edge_index = batch[0].to(self.architect.device), batch[1].to(self.architect.device), batch[2].to(self.architect.device)
            scores.append(self.architect.scoring_function(head_index, tail_index, edge_index, train = False))

        return cat(scores, dim = 0)


    def evaluate(self,
                batch_size: int,
                kg: KnowledgeGraph):
        """
        Find edge thresholds using the validation set. As described in
        the paper by Socher et al., for an edge, the threshold is a value t
        such that if the score of a triplet is larger than t, the triplet is correct.
        If an edge is not present in any triplet of the validation set, then
        the largest value score of all negative samples is used as threshold.

        Arguments
        ---------
        batch_size: int
            Size of the current batch.
        kg: KnowledgeGraph
            Knowledge graph on which the evaluation will be done.
            
        """
        sampler = PositionalNegativeSampler(kg)
        edge_indices = kg.edge_indices

        negative_heads, negative_tails = sampler.corrupt_kg(batch_size,
                                                            self.is_cuda,
                                                            which = "main")
        negative_scores = self.get_scores(negative_heads,
                                        negative_tails,
                                        edge_indices,
                                        batch_size)

        self.thresholds = zeros(self.kg_validation.n_rel)

        for i in range(self.kg_validation.n_rel):
            mask = (edge_indices == i).bool()
            if mask.sum() > 0:
                self.thresholds[i] = negative_scores[mask].max()
            else:
                self.thresholds[i] = negative_scores.max()

        self.evaluated = True
        self.thresholds.detach_()


    def accuracy(self,
                batch_size: int,
                kg_test: KnowledgeGraph,
                kg_validation: KnowledgeGraph | None = None
                ) -> float:
        
        """
        TODO.what_that_function_does
        
        Arguments
        ---------
        batch_size: int
            Size of the current batch.
        kg_test: KnowledgeGraph
            Test split from the knowledge graph.
        kg_validation: KnowledgeGraph, optional
            Validation split from the knowledge graph.

        Returns
        -------
        accuracy: float
            Proportion of all triplets (true and negatively sampled ones) that were
            correctly classified using the thresholds learned from the validation set.

        """
        if not self.evaluated:
            kg_to_evaluate = kg_validation if kg_validation is not None else kg_test
            self.evaluate(batch_size = batch_size, kg = kg_to_evaluate)

        sampler = PositionalNegativeSampler(kg_test)
        edge_indices = kg_test.edge_indices

        negative_heads, negative_tails = sampler.corrupt_kg(batch_size,
                                                            self.is_cuda,
                                                            which = "main")
        scores = self.get_scores(kg_test.head_indices,
                                kg_test.tail_indices,
                                edge_indices,
                                batch_size)
        negative_scores = self.get_scores(negative_heads,
                                        negative_tails,
                                        edge_indices,
                                        batch_size)

        if self.is_cuda:
            self.thresholds = self.thresholds.cuda()
            
        scores = (scores > self.thresholds[edge_indices])
        negative_scores = (negative_scores < self.thresholds[edge_indices])

        return (scores.sum().item() +
                negative_scores.sum().item()) / (2 * self.kg_test.triplet_count)