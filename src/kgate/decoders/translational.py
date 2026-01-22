"""
Translational decoder classes for training and inference.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Tuple, Dict

from tqdm import tqdm

from torch import nn, tensor, matmul, Tensor
from torch.cuda import empty_cache
from torch.nn.functional import normalize
from torch.nn import ParameterList, Parameter

from torchkge.models import TransEModel, TransHModel, TransRModel, TransDModel, TorusEModel


class TransE(TransEModel):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    embedding_dimensions: int
        Dimensions of embeddings.
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.
    dissimilarity_type: str
        TODO.What_that_argument_is_or_does

    Attributes
    ----------
    TODO.inherited_attributes
    
    """
    def __init__(self,
                embedding_dimensions: int,
                node_count: int,
                edge_count: int,
                dissimilarity_type: str):
        
        super().__init__(embedding_dimensions, node_count, edge_count, dissimilarity_type = dissimilarity_type)
        del self.ent_emb
        del self.rel_emb


    def score(self,
            *,
            head_embeddings: Tensor,
            tail_embeddings: Tensor,
            edge_embeddings: Tensor,
            **_) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        head_embeddings: torch.Tensor
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        
        return -self.dissimilarity(head_normalized_embeddings + edge_embeddings, tail_normalized_embeddings)
    
    
    def get_embeddings(self):
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        None
        
        """
        return None
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the TransE model.
        
        According to the original paper, the node embeddings should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type. All Parameters should be of the same size
            (n_ent, emb_dim) corresponding to (node_count, embedding_dimensions)
        edge_embeddings: torch.nn.Embedding
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        
        Returns
        -------
        node_embeddings : torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings : torch.nn.Embedding
            The normalized edge embedding object.
        
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)
            
        return node_embeddings, edge_embeddings


    def inference_prepare_candidates(self,
                                    *, 
                                    head_indices: Tensor, 
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor, 
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_scoring_function` method.

        Arguments
        ---------
        head_indices : torch.Tensor
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor
            Head node embeddings.
        tail_embeddings: torch.Tensor
            Tail node embeddings.
        edge_embeddings_inference: torch.Tensor
            TODO.What_that_variable_is_or_does
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.
            
        """
        batch_size = head_indices.shape[0]

        # Get head, tail and edge embeddings
        head_embeddings = node_embeddings[head_indices]
        tail_embeddings = node_embeddings[tail_indices]
        edge_embeddings_inference = edge_embeddings(edge_indices)

        if node_inference:
            # Prepare candidates for every node
            candidates = node_embeddings
        else:
            # Prepare candidates for every edge
            candidates = edge_embeddings.weight.data
        
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)

        return head_embeddings, tail_embeddings, edge_embeddings_inference, candidates
    
    
    
class TransH(TransHModel):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    embedding_dimensions: int
        Dimensions of embeddings.
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.

    Attributes
    ----------
    TODO.inherited_attributes
    
    """
    def __init__(self,
                embedding_dimensions: int,
                node_count: int,
                edge_count: int):
        
        super().__init__(embedding_dimensions, node_count, edge_count)


    def score(self,
            *,
            head_embeddings: Tensor,
            tail_embeddings: Tensor,
            edge_embeddings: Tensor,
            edge_indices: Tensor,
            **_) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        head_embeddings: torch.Tensor
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        edge_indices: torch.Tensor
            The indices of the edges (from KG).

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        self.evaluated_projections = False
        normalized_vector = normalize(self.norm_vect(edge_indices), p = 2, dim = 1)
        
        return - self.dissimilarity(self.project(head_normalized_embeddings, normalized_vector) + edge_embeddings,
                                    self.project(tail_normalized_embeddings, normalized_vector))
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the TransH model.
        
        According to the original paper, the node embeddings, edge embeddings
        and the normalization vector should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type. All Parameters should be of the same size
            (n_ent, emb_dim) corresponding to (node_count, embedding_dimensions)
        edge_embeddings: torch.nn.Embedding
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        
        Returns
        -------
        node_embeddings : torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings : torch.nn.Embedding
            The normalized edge embedding object.
    
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)
        edge_embeddings.weight.data = normalize(edge_embeddings.weight.data, p = 2, dim = 1)
        self.norm_vect.weight.data = normalize(self.norm_vect.weight.data, p = 2, dim = 1)
        
        return node_embeddings, edge_embeddings


    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        result_1: Dict[str, Tensor]
            TODO.What_that_variable_is_or_does
            
        """
        return {"norm_vect": self.norm_vect.weight.data}
    
    
    def inference_prepare_candidates(self,
                                    *, 
                                    head_indices: Tensor, 
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor, 
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        Arguments
        ---------
        head_indices : torch.Tensor
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        projected_heads: torch.Tensor
            TODO.What_that_variable_is_or_does
        projected_tails: torch.Tensor
            TODO.What_that_variable_is_or_does
        edge_embeddings_inference: torch.Tensor
            TODO.What_that_variable_is_or_does
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        edge_embeddings_inference = edge_embeddings(edge_indices)

        if node_inference:
            projected_heads = self.projected_entities[edge_indices, head_indices]  # shape: (batch_size, self.emb_dim)
            projected_tails = self.projected_entities[edge_indices, tail_indices]  # shape: (batch_size, self.emb_dim)
            candidates = self.projected_entities[edge_indices]  # shape: (batch_size, self.n_rel, self.emb_dim)
        else:
            projected_heads = self.projected_entities[:, head_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
            projected_tails = self.projected_entities[:, tail_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
            candidates = edge_embeddings.weight.data.unsqueeze(0).expand(batch_size, self.n_rel, self.emb_dim)  # shape: (batch_size, self.n_rel, self.emb_dim)

        return projected_heads, projected_tails, edge_embeddings_inference, candidates


    def evaluate_projections(self,
                            node_embeddings: Tensor):
        """
        Link prediction evaluation helper function. Project all nodes
        according to each edge. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        References
        ----------
        TODO

        Arguments
        ---------
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does

        Returns
        -------
        TODO
            
        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit = "nodes", desc = "Projecting nodes"):

            normalized_vector = self.norm_vect.weight.data.view(self.n_rel, self.emb_dim)
            mask = tensor([i], device = normalized_vector.device).long()

            if normalized_vector.is_cuda:
                empty_cache()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]

            normalized_components = (masked_node_embeddings.view(1, -1) * normalized_vector).sum(dim = 1)
            self.projected_entities[:, i, :] = (masked_node_embeddings.view(1, -1) - normalized_components.view(-1, 1) * normalized_vector)

            del normalized_components

        self.evaluated_projections = True



class TransR(TransRModel):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    node_embedding_dimensions: int
        Dimensions of node embeddings.
    edge_embedding_dimensions: int
        Dimensions of edge embeddings.
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.

    Attributes
    ----------
    TODO.inherited_attributes
    
    """
    def __init__(self,
                node_embedding_dimensions: int,
                edge_embedding_dimensions: int,
                node_count: int,
                edge_count: int):
        
        super().__init__(node_embedding_dimensions, edge_embedding_dimensions, node_count, edge_count)


    def score(self,
            *,
            head_embeddings: Tensor,
            tail_embeddings: Tensor,
            edge_embeddings: Tensor,
            edge_indices: Tensor,
            **_) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        head_embeddings: torch.Tensor
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        edge_indices: torch.Tensor
            The indices of the edges (from KG).

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        self.evaluated_projections = False
        batch_size = head_normalized_embeddings.shape[0]

        projected_matrix = self.proj_mat(edge_indices).view(batch_size,
                                                            self.rel_emb_dim,
                                                            self.ent_emb_dim)
        return - self.dissimilarity(self.project(head_normalized_embeddings, projected_matrix) + edge_embeddings,
                                    self.project(tail_normalized_embeddings, projected_matrix))
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the RESCAL model.
        
        According to the original paper, the node embeddings and edge embeddings
        should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type. All Parameters should be of the same size
            (n_ent, emb_dim) corresponding to (node_count, embedding_dimensions)
        edge_embeddings: torch.nn.Embedding
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        
        Returns
        -------
        node_embeddings : torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings : torch.nn.Embedding
            The normalized edge embedding object.
        
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)

        edge_embeddings.weight.data = normalize(edge_embeddings.weight.data, p = 2, dim = 1)
        
        return node_embeddings, edge_embeddings
    
    
    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        result_1: Dict[str, Tensor]
            TODO.What_that_variable_is_or_does
            
        """
        return {"proj_mat": self.proj_mat.weight.data.view(-1,
                                                        self.rel_emb_dim,
                                                        self.ent_emb_dim)}
    
    
    def inference_prepare_candidates(self,
                                    *, 
                                    head_indices: Tensor, 
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor, 
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_indices : torch.Tensor
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        projected_heads: torch.Tensor
            TODO.What_that_variable_is_or_does
        projected_tails: torch.Tensor
            TODO.What_that_variable_is_or_does
        edge_embeddings_inference: torch.Tensor
            TODO.What_that_variable_is_or_does
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        edge_embeddings_inference = edge_embeddings(edge_indices)
        if node_inference:
            projected_heads = self.projected_entities[edge_indices, head_indices]  # shape: (batch_size, self.emb_dim)
            projected_tails = self.projected_entities[edge_indices, tail_indices]  # shape: (batch_size, self.emb_dim)
            candidates = self.projected_entities[edge_indices]  # shape: (batch_size, self.n_rel, self.emb_dim)
        else:
            projected_heads = self.projected_entities[:, head_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
            projected_tails = self.projected_entities[:, tail_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
            candidates = edge_embeddings.weight.data.unsqueeze(0).expand(batch_size, self.n_rel, self.emb_dim)  # shape: (batch_size, self.n_rel, self.emb_dim)

        return projected_heads, projected_tails, edge_embeddings_inference, candidates
    
    
    def evaluate_projections(self,
                            node_embeddings: Tensor):
        """
        Link prediction evaluation helper function. Project all nodes
        according to each edge. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        References
        ----------
        TODO

        Arguments
        ---------
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does

        Returns
        -------
        TODO
            
        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit = "nodes", desc = "Projecting nodes"):
            projection_matrices = self.proj_mat.weight.data
            projection_matrices = projection_matrices.view(self.n_rel, self.rel_emb_dim, self.ent_emb_dim)

            mask = tensor([i], device = projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]
            
            projected_masked_node_embeddings = matmul(projection_matrices, masked_node_embeddings.view(self.ent_emb_dim))
            projected_masked_node_embeddings = projected_masked_node_embeddings.view(self.n_rel, self.rel_emb_dim, 1)
            self.projected_entities[:, i, :] = projected_masked_node_embeddings.view(self.n_rel, self.rel_emb_dim)
            # TODO: comment that projected_entities equivalent to projected_masked_node_embeddings

            del projected_masked_node_embeddings

        self.evaluated_projections = True



class TransD(TransDModel):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    node_embedding_dimensions: int
        Dimensions of node embeddings.
    edge_embedding_dimensions: int
        Dimensions of edge embeddings.
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.

    Attributes
    ----------
    TODO.inherited_attributes
    
    """
    def __init__(self,
                node_embedding_dimensions: int,
                edge_embedding_dimensions: int,
                node_count: int,
                edge_count: int):
    
        super().__init__(node_embedding_dimensions, edge_embedding_dimensions, node_count, edge_count)


    def score(self,
            *,
            head_embeddings: Tensor,
            tail_embeddings: Tensor,
            edge_embeddings: Tensor,
            head_indices: Tensor,
            edge_indices: Tensor,
            tail_indices: Tensor,
            **_) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        head_embeddings: torch.Tensor
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        head_indices: torch.Tensor
            The indices of the head nodes (from KG).
        edge_indices: torch.Tensor
            The indices of the edges (from KG).
        tail_indices: torch.Tensor
            The indices of the tail nodes (from KG).

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        edge_normalized_embeddings = normalize(edge_embeddings, p = 2, dim = 1)

        head_projected_vectors = normalize(self.ent_proj_vect(head_indices), p = 2, dim = 1)
        tail_projected_vectors = normalize(self.ent_proj_vect(tail_indices), p = 2, dim = 1)
        edge_projected_vectors = normalize(self.rel_proj_vect(edge_indices), p = 2, dim = 1)

        projected_heads = self.project(head_normalized_embeddings, head_projected_vectors, edge_projected_vectors)
        projected_tails = self.project(tail_normalized_embeddings, tail_projected_vectors, edge_projected_vectors)
        
        return - self.dissimilarity(projected_heads + edge_normalized_embeddings, projected_tails)
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the TransD model.
        
        According to the original paper, the node embeddings, the edge embeddings
        and both projection vectors should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type. All Parameters should be of the same size
            (n_ent, emb_dim) corresponding to (node_count, embedding_dimensions)
        edge_embeddings: torch.nn.Embedding
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        
        Returns
        -------
        node_embeddings : torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings : torch.nn.Embedding
            The normalized edge embedding object.
        
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)

        edge_embeddings.weight.data = normalize(edge_embeddings.weight.data, p = 2, dim = 1)

        self.ent_proj_vect.weight.data = normalize(self.ent_proj_vect.weight.data, p = 2, dim = 1)
        self.rel_proj_vect.weight.data = normalize(self.rel_proj_vect.weight.data, p = 2, dim = 1)

        return node_embeddings, edge_embeddings


    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        result_1: Dict[str, Tensor]
            TODO.What_that_variable_is_or_does
            
        """
        return {"ent_proj_vect": self.ent_proj_vect.weight.data,
                "rel_proj_vect": self.rel_proj_vect.weight.data}
    
    
    def inference_prepare_candidates(self,
                                    *, 
                                    head_indices: Tensor, 
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor, 
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_indices : torch.Tensor
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        projected_heads: torch.Tensor
            TODO.What_that_variable_is_or_does
        projected_tails: torch.Tensor
            TODO.What_that_variable_is_or_does
        edge_embeddings_inference: torch.Tensor
            TODO.What_that_variable_is_or_does
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        edge_embeddings_inference = edge_embeddings(edge_indices)

        if node_inference:
            projected_heads = self.projected_entities[edge_indices, head_indices]  # shape: (batch_size, self.emb_dim)
            projected_tails = self.projected_entities[edge_indices, tail_indices]  # shape: (batch_size, self.emb_dim)
            candidates = self.projected_entities[edge_indices]  # shape: (batch_size, self.n_rel, self.emb_dim)
        else:
            projected_heads = self.projected_entities[:, head_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.rel_emb_dim)
            projected_tails = self.projected_entities[:, tail_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.rel_emb_dim)
            candidates = self.rel_emb.weight.data.unsqueeze(0).expand(batch_size, self.n_rel, self.rel_emb_dim)  # shape: (batch_size, self.n_rel, self.emb_dim)

        return projected_heads, projected_tails, edge_embeddings_inference, candidates


    def evaluate_projections(self,
                            node_embeddings: Tensor):
        """
        Link prediction evaluation helper function. Project all nodes
        according to each edge. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        References
        ----------
        TODO

        Arguments
        ---------
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does

        Returns
        -------
        TODO
            
        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit = "nodes", desc = "Projecting nodes"):
            edge_projected_vectors = self.rel_proj_vect.weight.data

            mask = tensor([i], device=edge_projected_vectors.device).long()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]

            node_projected_vectors = self.ent_proj_vect.weight[i]

            # TODO PLACEHOLDER TODO PLACEHOLDER TODO
            sc_prod = (node_projected_vectors * masked_node_embeddings).sum(dim=0)
            projected_nodes = sc_prod * edge_projected_vectors + masked_node_embeddings[:self.rel_emb_dim].view(1, -1)

            self.projected_entities[:, i, :] = projected_nodes

            del projected_nodes

        self.evaluated_projections = True



class TorusE(TorusEModel):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    embedding_dimensions: int
        Dimensions of embeddings.
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.
    dissimilarity_type: str
        TODO.What_that_argument_is_or_does

    Attributes
    ----------
    TODO.inherited_attributes
    
    """
    def __init__(self,
                embedding_dimensions: int,
                node_count: int,
                edge_count: int,
                dissimilarity_type: str):
        
        super().__init__(embedding_dimensions, node_count, edge_count, dissimilarity_type)
    
    
    def score(self,
            *,
            head_embeddings: Tensor,
            tail_embeddings: Tensor,
            edge_embeddings: Tensor,
            head_indices: Tensor,
            tail_indices: Tensor,
            edge_indices: Tensor,
            **_) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        head_embeddings: torch.Tensor
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        head_indices: torch.Tensor
            Unused.
        tail_indices: torch.Tensor
            Unused.
        edge_indices: torch.Tensor
            Unused.

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        self.normalized = False

        fractionned_head_embeddings = head_embeddings.frac()
        fractionned_tail_embeddings = tail_embeddings.frac()
        fractionned_edge_embeddings = edge_embeddings.frac()

        return - self.dissimilarity(fractionned_head_embeddings + fractionned_edge_embeddings, fractionned_tail_embeddings)


    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the TorusE model.
        
        According to the original paper, only the fraction of the embeddings 
        should be kept in the normalization step.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type. All parameters should be of the same size
            (n_ent, emb_dim) corresponding to (node_count, embedding_dimensions)
        edge_embeddings: torch.nn.Embedding
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        
        Returns
        -------
        node_embeddings : torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings : torch.nn.Embedding
            The normalized edge embedding object.
            
        """
        for embedding in node_embeddings:
            embedding.data.frac_()

        edge_embeddings.weight.data = edge_embeddings.weight.data.frac()
        self.normalized = True

        return node_embeddings, edge_embeddings
    
    
    def get_embeddings(self):
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        None
        
        """
        return None
    
    
    def inference_prepare_candidates(self,
                                    *, 
                                    head_indices: Tensor, 
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor, 
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_indices : torch.Tensor
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor
            Head node embeddings.
        tail_embeddings: torch.Tensor
            Tail node embeddings.
        edge_embeddings_inference: torch.Tensor
            TODO.What_that_variable_is_or_does
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        if not self.normalized:
            # Very ugly transformation of the node embeddings into a ParameterList just for normalization
            # TODO: smoothen the cast (or avoid it)
            self.normalize_parameters(ParameterList([Parameter(node_embeddings)]), edge_embeddings)

        head_embeddings = node_embeddings[head_indices]
        tail_embeddings = node_embeddings[tail_indices]
        edge_embeddings_inference = edge_embeddings(edge_indices)

        if node_inference:
            # Prepare candidates for every node
            candidates = node_embeddings
        else:
            # Prepare candidates for every edge
            candidates = edge_embeddings.weight.data
            
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)
        
        return head_embeddings, tail_embeddings, edge_embeddings_inference, candidates
    