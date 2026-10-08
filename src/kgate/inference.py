from typing import Dict, Literal

from tqdm import tqdm

from torch import tensor, nn, Tensor
import torch
from torch.utils.data import DataLoader, Dataset

from .encoders import GNN
from .decoders import TranslationalDecoder, BilinearDecoder, ConvolutionalDecoder
from .knowledgegraph import KnowledgeGraph
from .utils import filter_scores


class Inference_KG(Dataset):
    def __init__(self,
                first_tensor_index: Tensor,
                second_tensor_index: Tensor):
        """
        Subset of a KG used for inference.

        This class is a subclass of the PyTorch
        [`utils.data.Dataset`](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

        Arguments
        ---------

        **first_tensor_index** *(torch.Tensor)*
        : The first tensor with indices of the edges or nodes (from the knowledge graph).

        **second_tensor_index** *(torch.Tensor)*
        : The second tensor with indices of the edges or nodes (from the knowledge graph).

        Attributes
        ----------

        **first_tensor_index** *(torch.Tensor)*
        : The first tensor with indices of the edges or nodes (from the knowledge graph).

        **second_tensor_index** *(torch.Tensor)*
        : The second tensor with indices of the edges or nodes (from the knowledge graph).

        Raises
        ------

        **AssertionError**
        : Both index tensors must be of the same size.

        Notes
        -----

        Either both tensors are nodes, or they are node and edge.   

        The `__getitem__` method allows to call an `Inference_KG` object with an index, giving back a tuple containing the corresponding values of both tensors.

        """
        
        # Either both tensors are nodes, or they are node and edge
        assert first_tensor_index.size() == second_tensor_index.size(), "Both index tensors must be of the same size for inference."
        self.first_tensor_index = first_tensor_index
        self.second_tensor_index = second_tensor_index


    def __len__(self):
        return self.first_tensor_index.size(0)


    def __getitem__(self, index: int):
        return (self.first_tensor_index[index], self.second_tensor_index[index])



def generate_inference_embeddings(knowledge_graph: KnowledgeGraph,
                       seed_nodes: Tensor,
                       encoder: GNN,
                       node_embeddings: nn.ParameterList,
                       device: torch.device) -> Tensor:
    """
    Compute the forward pass of the encoder on the given set of nodes.

    Arguments
    ---------

    **knowledge_graph** *(KnowledgeGraph)*
    : The knowledge graph structure.

    **seed_nodes** *(torch.Tensor, dtype: torch.long, shape: [seed_count])*
    : The global indices of the nodes to embed.

    **encoder** *(GNN)*
    : The GNN encoder generating the embeddings.

    **node_embeddings** *(nn.ParameterList)*
    : The node embeddings of each node type.

    **device** *(torch.device)*
    : The device of the returned tensor.

    Returns
    -------

    **node_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions])*
    : The flat node embeddings, indexed by global node index.

    """
    node_embeddings: Tensor = torch.zeros((knowledge_graph.node_count,
                                               node_embeddings[0].size(1)),
                                              device = device,
                                              dtype = torch.float)

    encoder_input = knowledge_graph.get_encoder_input(seed_nodes = seed_nodes,
                                         hop_count = encoder.layer_count)
    encoder_output: Dict[str, Tensor] = encoder(encoder_input.x_dict, encoder_input.edge_index)

    for node_type, indices in encoder_input.seed_mapping.items():
        node_type_index = knowledge_graph.node_type_to_index[node_type]
        node_type_mask = (knowledge_graph.node_types[seed_nodes] == node_type_index)
        node_embeddings[seed_nodes[node_type_mask]] = encoder_output[node_type][indices]

    return node_embeddings



class EdgeInference:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Use trained embedding model to infer missing edges in triplets.

        Arguments
        ---------

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the inference will be done.

        Attributes
        ----------

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the inference will be done.

        """
        self.knowledge_graph = knowledge_graph


    def evaluate(self, 
                head_indices: Tensor,
                tail_indices: Tensor,
                *,
                top_k: int,
                batch_size: int,
                encoder: GNN | None,
                decoder: TranslationalDecoder | BilinearDecoder | ConvolutionalDecoder,
                node_embeddings: nn.ParameterList, 
                edge_embeddings: nn.Parameter,
                sphere_embeddings: bool = False,
                verbose: bool = True,
                **_):
        """
        Use a trained embedding model to infer the missing edges of triplets:
        for each given (head, tail) pair, rank all edges and return the top_k
        best ones.

        Arguments
        ---------

        **head_indices** *(torch.Tensor)*
        : The indices of the head nodes (from the knowledge graph).
        
        **tail_indices** *(torch.Tensor)*
        : The indices of the tail nodes (from the knowledge graph).
        
        **top_k** *(int, keyword-only)*
        : Indicate the number of top predictions to return.
        
        **batch_size** *(int, keyword-only)*
        : Size of the current batch.
        
        **encoder** *( GNN, keyword-only)*
        : Encoder model to embed the nodes.     
        **decoder** *(BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder)*
        : Decoder model to evaluate.
        
        **node_embeddings** *(nn.ParameterList, keyword-only)*
        : A list containing all embeddings for each node type.
        : keys: node type index
        : values: tensors of shape (node_count, embedding_dimensions)
        
        **edge_embeddings** *(nn.Parameter, keyword-only)*
        : A tensor containing one embedding by edge type, of shape (edge_count, embedding_dimensions).
        
        **sphere_embeddings** *(bool, default to False, keyword-only)*
        : If `True`, the decoder uses sphere embeddings (SpherE, Li et al. 2024):
        : the score of each candidate edge is the overlap of the two node spheres,
        : `-max(d - r_head - r_tail, -alpha*r_head - beta*r_tail)`, the candidates
        : are ranked by that score, and known (true) candidates are not filtered
        : out (set retrieval keeps them in the predicted set).
        : The decoder must be TransR or RotatE (with `sphere_embeddings` enabled).
        
        **verbose** *(bool, default to True, keyword-only)*
        : Indicate whether a progress bar should be displayed during evaluation.

        Returns
        -------

        **predictions** *(torch.Tensor, shape: [len(head_indices), top_k], dtype: torch.long)*
        : The top_k predicted edges for each (head, tail) pair of the input,
        : ranked from best to worst score.
        
        **scores** *(torch.Tensor, shape: [len(head_indices), top_k], dtype: torch.float)*
        : The scores of the predicted edges, with known (true) edges filtered out.
        : If `sphere_embeddings` is True, these are SpherE scores and the known
        : edges are not filtered out.
        
        """
        assert top_k <= self.knowledge_graph.edge_count, f"top_k cannot be larger than the number of edges of the knowledge graph ({self.knowledge_graph.edge_count})."

        with torch.no_grad():
            device = edge_embeddings.device

            inference_kg = Inference_KG(head_indices, tail_indices)

            dataloader = DataLoader(inference_kg, batch_size = batch_size)

            predictions = torch.empty(size = (len(head_indices), top_k), device = device).long()
            scores = torch.empty(size = (len(head_indices), top_k), device = device)

            for i, batch in tqdm(enumerate(dataloader),
                                total = len(dataloader),
                                unit = "batch",
                                disable = (not verbose),
                                desc = "Inference"):
                head_indices, tail_indices = batch[0].to(device), batch[1].to(device)
                batch_len = len(head_indices)

                if encoder is not None:
                    seed_nodes = torch.cat([head_indices, tail_indices], dim = 0).unique()
                    node_embeddings_flat = generate_inference_embeddings(self.knowledge_graph, seed_nodes, encoder, node_embeddings, device)
                else:
                    # Concatenation assumes node types are in the same global order, it might be wrong.
                    node_embeddings_flat = torch.cat([embeddings.data for embeddings in node_embeddings], dim = 0).to(device)

                head_embeddings, tail_embeddings, _, candidates = decoder.inference_prepare_candidates( head_indices = head_indices,
                                                                                                        tail_indices = tail_indices, 
                                                                                                        edge_indices = tensor([], device = device).long(),
                                                                                                        node_embeddings = node_embeddings_flat, 
                                                                                                        edge_embeddings = edge_embeddings, 
                                                                                                        node_inference = False)
                batch_scores = decoder.inference_score(head_embeddings = head_embeddings,
                                                       tail_embeddings = tail_embeddings,
                                                       edge_embeddings = candidates)

                if sphere_embeddings:
                    head_radius = decoder.node_radii[head_indices, 0]
                    tail_radius = decoder.node_radii[tail_indices, 0]
                    batch_scores = -torch.max(  -batch_scores - (head_radius + tail_radius).unsqueeze(1),
                                                (-decoder.alpha * head_radius - decoder.beta * tail_radius).unsqueeze(1))
                else:
                    # Known (true) edges are filtered out, except in sphere mode
                    # (set retrieval keeps them in the predicted set).
                    batch_scores = filter_scores(batch_scores, self.knowledge_graph.graphindices.to(device), "edge", head_indices, tail_indices, None)

                batch_scores, indices = batch_scores.sort(descending = True)

                predictions[i * batch_size: i * batch_size + batch_len] = indices[:, :top_k]
                scores[i * batch_size: i * batch_size + batch_len] = batch_scores[:, :top_k]

            return predictions.cpu(), scores.cpu()



class NodeInference:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Use trained embedding model to infer missing nodes in triplets.

        Arguments
        ---------

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the inference will be done.

        Attributes
        ----------

        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph on which the inference will be done.

        """
        self.knowledge_graph = knowledge_graph


    def evaluate(self,
                node_indices: Tensor,
                edge_indices: Tensor,
                *,
                top_k: int,
                missing_triplet_part: Literal["head", "tail"],
                batch_size: int,
                encoder: GNN | None,
                decoder: TranslationalDecoder | BilinearDecoder | ConvolutionalDecoder,
                node_embeddings: nn.ParameterList, 
                edge_embeddings: nn.Parameter,
                sphere_embeddings: bool = False,
                verbose: bool = True,
                **_):
        """
        Use a trained embedding model to infer the missing node of triplets:
        for each given (node, edge) pair, rank all nodes and return the top_k
        best ones.

        Arguments
        ---------
        
        **node_indices** *(torch.Tensor)*
        : The indices of nodes (from the knowledge graph).
        
        **edge_indices** *(torch.Tensor)*
        : The indices of edges (from the knowledge graph).
        
        **top_k** *(int, keyword-only)*
        : Indicate the number of top predictions to return.
        
        **missing_triplet_part** *(Literal["head", "tail"], keyword-only)*
        : String indicating if the missing nodes are the heads or the tails.
        
        **batch_size** *(int, keyword-only)*
        : Size of the current batch.
        
        **encoder** *(GNN, keyword-only)*
        : Encoder model to embed the nodes.
             
        **decoder** *(BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder, keyword-only)*
        : Decoder model to evaluate.
        
        **node_embeddings** *(nn.ParameterList, keyword-only)*
        : A list containing all embeddings for each node type.
          : keys: node type index
          : values: tensors of shape (node_count, embedding_dimensions)
        
        **edge_embeddings** *(nn.Parameter, keyword-only)*
        : A tensor containing one embedding by edge type,
        : of shape (edge_count, embedding_dimensions).
        
        **sphere_embeddings** *(bool, default to False, keyword-only)*
        : If `True`, the decoder uses sphere embeddings (SpherE, Li et al. 2024):
        : the score of each candidate node is the overlap of the sphere of the
        : known node and the sphere of the candidate,
        : `-max(d - r_known - r_candidate, -alpha*r_known - beta*r_candidate)`, the
        : candidates are ranked by that score, and known (true) candidates are not
        : filtered out (set retrieval keeps them in the predicted set).
        : The decoder must be TransR or RotatE (with `sphere_embeddings` enabled).
        
        **verbose** *(bool, default to True, keyword-only)*
        : Indicate whether a progress bar should be displayed during evaluation.

        Returns
        -------
        
        **predictions** *(torch.Tensor, shape: [len(node_indices), top_k], dtype: torch.long)*
        : The top_k predicted nodes for each (node, edge) pair of the input,
        : ranked from best to worst score.
        
        **scores** *(torch.Tensor, shape: [len(node_indices), top_k], dtype: torch.float)*
        : The scores of the predicted nodes, with known (true) nodes filtered out
        : (set to -Inf) by `filter_scores`.
        : If `sphere_embeddings` is True, these are SpherE scores and the known
        : nodes are not filtered out.
        
        """
        assert top_k <= self.knowledge_graph.node_count, f"top_k cannot be larger than the number of nodes of the knowledge graph ({self.knowledge_graph.node_count})."

        with torch.no_grad():
            device = edge_embeddings.device

            inference_kg = Inference_KG(node_indices, edge_indices)

            dataloader = DataLoader(inference_kg, batch_size = batch_size)

            predictions = torch.empty(size = (len(node_indices), top_k),
                                    device = device).long()
            scores = torch.empty(size = (len(node_indices), top_k),
                                device = device)

            for i, batch in tqdm(enumerate(dataloader),
                                total = len(dataloader),
                                unit = "batch",
                                disable = (not verbose),
                                desc = "Inference"):

                known_nodes, known_edges = batch[0].to(device), batch[1].to(device)
                batch_len = len(known_nodes)
                
                if encoder is not None:
                    seed_nodes = known_nodes.unique()
                    node_embeddings_flat = generate_inference_embeddings(self.knowledge_graph, seed_nodes, encoder, node_embeddings, device)
                else:
                    # This concatenation assumes the node types embeddings are in the same global order.
                    # It might be wrong.
                    node_embeddings_flat = torch.cat([embeddings.data for embeddings in node_embeddings], dim = 0).to(device)

                if missing_triplet_part == "head":
                    _, tail_embeddings, edge_embeddings_inferred, candidates = decoder.inference_prepare_candidates( head_indices = known_nodes,
                                                                                                            tail_indices = known_nodes,
                                                                                                            edge_indices = known_edges,
                                                                                                            node_embeddings = node_embeddings_flat,
                                                                                                            edge_embeddings = edge_embeddings,
                                                                                                            node_inference = True)
                    batch_scores = decoder.inference_score(head_embeddings = candidates,
                                                           tail_embeddings = tail_embeddings,
                                                           edge_embeddings = edge_embeddings_inferred)
                
                else:
                    head_embeddings, _, edge_embeddings_inferred, candidates = decoder.inference_prepare_candidates( head_indices = known_nodes, 
                                                                                                            tail_indices = tensor([], device = device).long(),
                                                                                                            edge_indices = known_edges,
                                                                                                            node_embeddings = node_embeddings_flat,
                                                                                                            edge_embeddings = edge_embeddings,
                                                                                                            node_inference = True)
                    batch_scores = decoder.inference_score(head_embeddings = head_embeddings,
                                                           tail_embeddings = candidates,
                                                           edge_embeddings = edge_embeddings_inferred)

                if sphere_embeddings:
                    known_radius = decoder.node_radii[known_nodes, 0]
                    candidate_radius = decoder.node_radii[:, 0]
                    batch_scores = -torch.max(  -batch_scores - known_radius.unsqueeze(1) - candidate_radius.unsqueeze(0),
                                                -decoder.alpha * known_radius.unsqueeze(1) - decoder.beta * candidate_radius.unsqueeze(0))
                else:
                    # Known (true) nodes are filtered out, except in sphere mode
                    # (set retrieval keeps them in the predicted set).
                    batch_scores = filter_scores(batch_scores,
                                                self.knowledge_graph.graphindices.to(device),
                                                missing_triplet_part,
                                                known_nodes,
                                                known_edges,
                                                None)

                batch_scores, indices = batch_scores.sort(descending = True)

                predictions[i * batch_size: i * batch_size + batch_len] = indices[:, :top_k]
                scores[i * batch_size: i * batch_size + batch_len] = batch_scores[:, :top_k]

            return predictions.cpu(), scores.cpu()
