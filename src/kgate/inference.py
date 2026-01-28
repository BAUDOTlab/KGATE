from typing import Dict, Literal

from tqdm.autonotebook import tqdm

from torch import tensor, nn, Tensor
import torch
from torch.utils.data import DataLoader, Dataset

from torch_geometric.utils import k_hop_subgraph

import torchkge.inference as torchkge_inference
from torchkge.models import Model

from .encoders import DefaultEncoder, GNN
from .knowledgegraph import KnowledgeGraph
from .utils import filter_scores



class Inference_KG(Dataset):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    first_index_tensor: torch.Tensor
        The first tensor with indices of the edges or nodes (from the knowledge graph).
    second_index_tensor: torch.Tensor
        The second tensor with indices of the edges or nodes (from the knowledge graph).
        

    Attributes
    ----------
    first_index_tensor: torch.Tensor
        The first tensor with indices of the edges or nodes (from the knowledge graph).
    second_index_tensor: torch.Tensor
        The second tensor with indices of the edges or nodes (from the knowledge graph).
    
    Raises
    ------
    AssertionError
        Index tensors are of different sizes.

    Notes
    -----
    Either both tensors are nodes, or they are node and edge.
    TODO.explain_getitem
    
    """
    def __init__(self,
                first_index_tensor: Tensor,
                second_index_tensor: Tensor):
        
        # Either both tensors are nodes, or they are node and edge
        assert first_index_tensor.size() == second_index_tensor.size(), "Both index tensors must be of the same size for inference."
        self.first_tensor_index = first_index_tensor
        self.second_tensor_index = second_index_tensor


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
        Knowledge graph on which the inference will be done.

    Attributes
    ----------
    kg: KnowledgeGraph
        Knowledge graph on which the inference will be done.

    """
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def evaluate(self, 
                head_indices: Tensor,
                tail_indices: Tensor,
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
        head_indices: torch.Tensor
            The indices of the head nodes (from the knowledge graph).
        tail_indices: torch.Tensor
            The indices of the tail nodes (from the knowledge graph).
        top_k: int, keyword-only
            Indicate the number of top predictions to return.
        batch_size: int, keyword-only
            Size of the current batch.
        encoder: DefaultEncoder or GNN, keyword-only
            Encoder model to embed the nodes. Deactivated with DefaultEncoder.
        decoder: BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder
            Decoder model to evaluate.
        node_embeddings: nn.ParameterList, keyword-only
            A list containing all embeddings for each node type.
            keys: node type index
            values: tensors of shape (node_count, embedding_dimensions)
        edge_embeddings: nn.Embedding, keyword-only
            A tensor containing one embedding by edge type, of shape (edge_count, embedding_dimensions).
        verbose: bool, default to True, keyword-only
            Indicate whether a progress bar should be displayed during evaluation.

        Returns
        -------
        predictions: TODO.type
            TODO.What_that_variable_is_or_does
        scores: TODO.type
            Tensor of shape [batch_size, n] with -Inf values for all true node/edge index except the ones being predicted.
            
        """
        with torch.no_grad():
            device = edge_embeddings.weight.device

            inference_kg = Inference_KG(head_indices, tail_indices)

            dataloader = DataLoader(inference_kg, batch_size = batch_size)

            predictions = torch.empty(size = (len(head_indices), top_k), device = device).long()   
            node_embeddings = node_embeddings.weight.data

            for i, batch in tqdm(enumerate(dataloader),
                                total = len(dataloader),
                                unit = "batch",
                                disable = (not verbose),
                                desc = "Inference"):
                head_indices, tail_indices = batch[0], batch[1]
                
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

                head_embeddings, tail_embeddings, _, candidates = decoder.inference_prepare_candidates( head_indices = head_indices,
                                                                                                        tail_indices = tail_indices, 
                                                                                                        edge_indices = tensor([]).long(),
                                                                                                        node_embeddings = node_embeddings, 
                                                                                                        edge_embeddings = edge_embeddings, 
                                                                                                        node_inference = False)
                scores = decoder.inference_score(head_embeddings, tail_embeddings, candidates)

                scores = filter_scores(scores, self.kg.graphindices, "edge", head_indices, tail_indices, None)

                scores, indices = scores.sort(descending = True)

                predictions[i * batch_size: (i + 1) * batch_size] = indices[:, :top_k]
                scores[i * batch_size, (i + 1) * batch_size] = scores[:, :top_k]

            return predictions.cpu(), scores.cpu()



class NodeInference(torchkge_inference.EntityInference):
    """
    Use trained embedding model to infer missing entities in triples.

    Arguments
    ---------
    kg: KnowledgeGraph
        Knowledge graph on which the inference will be done.
    
    Attributes
    ----------
    kg: KnowledgeGraph
        Knowledge graph on which the inference will be done.

    """
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg


    def evaluate(self,
                node_indices: Tensor,
                edge_indices: Tensor,
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
        node_indices: torch.Tensor
            The indices of nodes (from the knowledge graph).
        edge_indices: torch.Tensor
            The indices of edges (from the knowledge graph).
        top_k: int, keyword-only
            Indicate the number of top predictions to return.
        missing_triplet_part: Literal["head", "tail"], keyword-only
            String indicating if the missing nodes are the heads or the tails.
        batch_size: int, keyword-only
            Size of the current batch.
        encoder: DefaultEncoder or GNN, keyword-only
            Encoder model to embed the nodes. Deactivated with DefaultEncoder.
        decoder: BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder, keyword-only
            Decoder model to evaluate.
        node_embeddings: nn.ParameterList, keyword-only
            A list containing all embeddings for each node type.
            keys: node type index
            values: tensors of shape (node_count, embedding_dimensions)
        edge_embeddings: nn.Embedding, keyword-only
            A tensor containing one embedding by edge type, of shape (edge_count, embedding_dimensions).
        verbose: bool, default to True, keyword-only
            Indicate whether a progress bar should be displayed during
            evaluation.

        Returns
        -------
        predictions: TODO.type
            TODO.What_that_variable_is_or_does
        scores: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        with torch.no_grad():
            device = edge_embeddings.weight.device

            inference_kg = Inference_KG(node_indices, edge_indices)

            dataloader = DataLoader(inference_kg, batch_size = batch_size)

            predictions = torch.empty(size = (len(node_indices), top_k),
                                    device = device).long()
            scores = torch.empty(size = (len(node_indices), top_k),
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
                    edge_indices = self.kg.edge_list

                    _, _, _, edge_mask = k_hop_subgraph(
                        node_idx = seed_nodes,
                        num_hops = hop_count,
                        edge_index = edge_indices
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
                    _, tail_embeddings, edge_embeddings, candidates = decoder.inference_prepare_candidates( head_indices = tensor([], device = device).long(), 
                                                                                                            tail_indices = known_nodes.to(device),
                                                                                                            edge_indices = known_edges.to(device),
                                                                                                            node_embeddings = node_embeddings,
                                                                                                            edge_embeddings = edge_embeddings,
                                                                                                            node_inference = True)
                    batch_scores = decoder.inference_score(candidates, tail_embeddings, edge_embeddings)
                
                else:
                    head_embeddings, _, edge_embeddings, candidates = decoder.inference_prepare_candidates( head_indices = known_nodes.to(device), 
                                                                                                            tail_indices = tensor([], device = device).long(),
                                                                                                            edge_indices = known_edges.to(device),
                                                                                                            node_embeddings = node_embeddings,
                                                                                                            edge_embeddings = edge_embeddings,
                                                                                                            node_inference = True)
                    batch_scores = decoder.inference_score(head_embeddings, candidates, edge_embeddings)

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