from typing import Dict, Literal

from tqdm.autonotebook import tqdm

from torch import tensor, nn, Tensor
import torch
from torch.utils.data import DataLoader, Dataset

from torch_geometric.utils import k_hop_subgraph

import torchkge.inference as torchkge_inference
from torchkge.models import Model

from .utils import filter_scores
from .encoders import DefaultEncoder, GNN
from .knowledgegraph import KnowledgeGraph



class Inference_KG(Dataset):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    first_tensor_index: torch.Tensor
        TODO.What_that_argument_is_or_does
    second_tensor_index: torch.Tensor
        TODO.What_that_argument_is_or_does

    Attributes
    ----------
    first_tensor_index: torch.Tensor
        TODO.What_that_variable_is_or_does
    second_tensor_index: torch.Tensor
        TODO.What_that_variable_is_or_does
    
    Raises
    ------
    error_name
        TODO.What_that_means_comma_causes_comma_and_fixes_if_easy

    Notes
    -----
    TODO.explain_getitem
    
    """
    def __init__(self,
                first_tensor_index: Tensor,
                second_tensor_index: Tensor):
        
        # Either both tensors are nodes, or they are node and edge
        assert first_tensor_index.size() == second_tensor_index.size(), "Both index tensors must be of the same size for inference."
        self.first_tensor_index = first_tensor_index
        self.second_tensor_index = second_tensor_index


    def __len__(self):
        return self.first_tensor_index.size(0)


    def __getitem__(self, index: int):
        return (self.first_tensor_index[index], self.second_tensor_index[index])



class EdgeInference(torchkge_inference.RelationInference):
    """
    Use trained embedding model to infer missing edges in triplets.

    Arguments
    ---------
    kg: KnowledgeGraph
        Knowledge graph.

    Attributes
    ----------
    kg: KnowledgeGraph
        Knowledge graph.

    """
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def evaluate(self, 
                head_index: Tensor,
                tail_index: Tensor,
                *,
                top_k: int,
                batch_size: int,
                encoder: DefaultEncoder | GNN,
                decoder: Model,
                node_embeddings: nn.ParameterList | nn.Embedding, 
                edge_embeddings: nn.Embedding, 
                verbose: bool = True,
                **_):
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        head_index: torch.Tensor
            TODO.What_that_argument_is_or_does
        tail_index: torch.Tensor
            TODO.What_that_argument_is_or_does
        top_k: int, keyword-only
            TODO.What_that_argument_is_or_does
        batch_size: int, keyword-only
            TODO.What_that_argument_is_or_does
        encoder: DefaultEncoder or GNN, keyword-only
            TODO.What_that_argument_is_or_does
        decoder: torchkge.Model, keyword-only
            TODO.What_that_argument_is_or_does
        node_embeddings: nn.ParameterList or nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        edge_embeddings: nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        verbose: bool, default to True, keyword-only
            TODO.What_that_argument_is_or_does

        Returns
        -------
        result_name: TODO.type
            TODO.What_that_variable_is_or_does
        result_name: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        with torch.no_grad():
            device = edge_embeddings.weight.device

            inference_kg = Inference_KG(head_index, tail_index)

            dataloader = DataLoader(inference_kg, batch_size = batch_size)

            predictions = torch.empty(size = (len(head_index), top_k), device = device).long()   
            node_embeddings = node_embeddings.weight.data

            for i, batch in tqdm(enumerate(dataloader),
                                total = len(dataloader),
                                unit = "batch",
                                disable = (not verbose),
                                desc = "Inference"):
                head_index, tail_index = batch[0], batch[1]
                
                if isinstance(encoder, GNN):
                    seed_nodes = batch.unique()
                    hop_count = encoder.n_layers
                    edge_list = self.kg.edge_list

                    _,_,_, edge_mask = k_hop_subgraph(
                        node_idx = seed_nodes,
                        num_hops = hop_count,
                        edge_index = edge_list
                        )
                    
                    input = self.kg.get_encoder_input(self.kg.graphindices[:, edge_mask], node_embeddings)
                    encoder_output: Dict[str, Tensor] = encoder(input.x_dict, input.edge_list)
            
                    for node_type, index in input.mapping.items():
                        node_embeddings[index] = encoder_output[node_type]

                head_embeddings, tail_embeddings, _, candidates = decoder.inference_prepare_candidates(head_index = head_index,
                                                                                                        tail_index = tail_index, 
                                                                                                        edge_index = tensor([]).long(),
                                                                                                        node_embeddings = node_embeddings, 
                                                                                                        edge_embeddings = edge_embeddings, 
                                                                                                        node_inference = False)
                scores = decoder.inference_scoring_function(head_embeddings, tail_embeddings, candidates)

                scores = filter_scores(scores, self.kg.graphindices, "edge", head_index, tail_index, None)

                scores, indices = scores.sort(descending = True)

                predictions[i * batch_size: (i + 1) * batch_size] = indices[:, :top_k]
                scores[i * batch_size, (i + 1) * batch_size] = scores[:, :top_k]

            return predictions.cpu(), scores.cpu()



class NodeInference(torchkge_inference.EntityInference):
    """
    Use trained embedding model to infer missing entities in triples.

    Arguments
    ---------
    kg:
    
    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    known_entities: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities.
    known_relations: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known relations.
    top_k: int
        Indicates the number of top predictions to return.
    missing: str
        String indicating if the missing entities are the heads or the tails.
    dictionary: dict, optional (default=None)
        Dictionary of possible heads or tails (depending on the value of `missing`).
        It is used to filter predictions that are known to be True in the training set
        in order to return only new facts.
    predictions: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.long`
        List of the indices of predicted entities for each test fact.
    scores: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.float`
        List of the scores of resulting triples for each test fact.

    """
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def evaluate(self,
                node_index: Tensor,
                edge_list: Tensor,
                *,
                top_k: int,
                missing_triplet_part: Literal["head", "tail"],
                batch_size: int,
                encoder: DefaultEncoder | GNN,
                decoder: Model,
                node_embeddings: nn.ParameterList, 
                edge_embeddings: nn.Embedding,
                verbose: bool = True,
                **_):
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        node_index: torch.Tensor
            TODO.What_that_argument_is_or_does
        edge_list: torch.Tensor
            TODO.What_that_argument_is_or_does
        top_k: int, keyword-only
            TODO.What_that_argument_is_or_does
        missing_triplet_part: Literal["head", "tail"], keyword-only
            TODO.What_that_argument_is_or_does
        batch_size: int, keyword-only
            TODO.What_that_argument_is_or_does
        encoder: DefaultEncoder or GNN, keyword-only
            TODO.What_that_argument_is_or_does
        decoder: torchkge.Model, keyword-only
            TODO.What_that_argument_is_or_does
        node_embeddings: nn.ParameterList, keyword-only
            TODO.What_that_argument_is_or_does
        edge_embeddings: nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        verbose: bool, default to True, keyword-only
            TODO.What_that_argument_is_or_does

        Returns
        -------
        result_name: TODO.type
            TODO.What_that_variable_is_or_does
        result_name: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        with torch.no_grad():
            device = edge_embeddings.weight.device

            inference_kg = Inference_KG(node_index, edge_list)

            dataloader = DataLoader(inference_kg, batch_size = batch_size)

            predictions = torch.empty(size = (len(node_index), top_k),
                                    device = device).long()
            scores = torch.empty(size = (len(node_index), top_k),
                                device = device).long()

            for i, batch in tqdm(enumerate(dataloader),
                                total = len(dataloader),
                                unit = "batch",
                                disable = (not verbose),
                                desc = "Inference"):

                known_nodes, known_edges = batch[0], batch[1]
                
                if isinstance(encoder, GNN):
                    seed_nodes = known_nodes.unique()
                    hop_count = encoder.n_layers
                    edge_list = self.kg.edge_list

                    _, _, _, edge_mask = k_hop_subgraph(
                        node_idx = seed_nodes,
                        num_hops = hop_count,
                        edge_index = edge_list
                        )
                    
                    node_embeddings: torch.Tensor = torch.zeros(node_embeddings[0].size(),
                                                                device = device,
                                                                dtype = torch.float)

                    input = self.kg.get_encoder_input(self.kg.graphindices[:, edge_mask], node_embeddings)
                    encoder_output: Dict[str, Tensor] = encoder(input.x_dict, input.edge_list)
            
                    for node_type, index in input.mapping.items():
                        node_embeddings[index] = encoder_output[node_type]
                
                else:
                    node_embeddings = node_embeddings[0][known_nodes]

                if missing_triplet_part == "head":
                    _, tail_embeddings, edge_embeddings, candidates = decoder.inference_prepare_candidates( head_index = tensor([], device=device).long(), 
                                                                                                            tail_index = known_nodes.to(device),
                                                                                                            edge_index = known_edges.to(device),
                                                                                                            node_embeddings = node_embeddings,
                                                                                                            edge_embeddings = edge_embeddings,
                                                                                                            node_inference = True)
                    batch_scores = decoder.inference_scoring_function(candidates, tail_embeddings, edge_embeddings)
                
                else:
                    head_embeddings, _, edge_embeddings, candidates = decoder.inference_prepare_candidates( head_index = known_nodes.to(device), 
                                                                                                            tail_index = tensor([], device=device).long(),
                                                                                                            edge_index = known_edges.to(device),
                                                                                                            node_embeddings = node_embeddings,
                                                                                                            edge_embeddings = edge_embeddings,
                                                                                                            node_inference = True)
                    batch_scores = decoder.inference_scoring_function(head_embeddings, candidates, edge_embeddings)

                batch_scores = filter_scores(batch_scores,
                                            self.kg.graphindices,
                                            missing_triplet_part,
                                            known_nodes,
                                            known_edges,
                                            None)

                batch_scores, indices = batch_scores.sort(descending = True)
                batch_size = min(batch_size, len(batch_scores))
                
            predictions[i * batch_size: (i+1) * batch_size] = indices[:, :top_k]
            scores[i * batch_size: (i+1) * batch_size] = batch_scores[:, :top_k]

            return predictions.cpu(), scores.cpu()