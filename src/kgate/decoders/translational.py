"""
Translational decoder classes for training and inference.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Tuple, Dict, Literal

from tqdm import tqdm

from torch import nn, tensor, matmul, Tensor, empty
from torch.cuda import empty_cache
from torch.nn.functional import normalize
from torch.nn import Module, Parameter, ParameterList

from torchkge.utils.dissimilarities import  l1_dissimilarity, \
                                            l2_dissimilarity, \
                                            l1_torus_dissimilarity, \
                                            l2_torus_dissimilarity, \
                                            el2_torus_dissimilarity

from ..utils import initialize_embedding



class TranslationalDecoder(Module):
    """
    Interface for translational decoders of KGATE.

    This interface is largely inspired by TorchKGE's TranslationModel, and exposes
    the methods that all translational decoders must use to be compatible with KGATE.
    The interface doesn't have an __init__ method as inheriting decoders are supposed
    to take care of their initialization, and only requires one attribute to be set.

    Attributes
    ----------
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings. Most translational vectors use either L1 or L2, but
        TorusE has a specific set of dissimilarity functions.
    """
    
    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                head_indices: Tensor,
                tail_indices: Tensor,
                edge_indices: Tensor
                ) -> Tensor:
        """
        Interface method for the decoder's score function.

        Refer to the specific decoder for details on scoring function implementation.
        While all arguments are given when called from the Architect class, most 
        decoders only use some of them. 

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the head entities for the current batch of length `batch_size`
            (or the whole graph, if it fits in memory)
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the tail entities for the current batch of length `batch_size` 
            (or the whole graph, if it fits in memory)
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the edges for the current batch of length `batch_size` 
            (or the whole graph, if it fits in memory)
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head entities for the current batch of length `batch_size` 
            (or the whole graph, if it fits in memory)
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail entities for the current batch of length `batch_size` 
            (or the whole graph, if it fits in memory)
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges for the current batch of length `batch_size` 
            (or the whole graph, if it fits in memory)

        Returns
        -------
            batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
                The score of each triplet as a tensor.
        
        """
        raise NotImplementedError("The score method must be implemented by the translational decoder.")


    def normalize_parameters(self) -> Tuple[nn.ParameterList, nn.Embedding] | None:
        """
        TODO.docstring
        
        """
        return None


    def get_embeddings(self) -> Dict[str, Tensor] | None:
        """
        Get the decoder-specific embeddings.
        
        If the decoder doesn't have dedicated embeddings, nothing is returned. In 
        this case, it is not necessary to implement this method from the interface.
        
        Returns
        -------
            embeddings: Dict[str, torch.Tensor] or None
                Decoder-specific embeddings, or None.
        
        """
        return None


    def inference_prepare_candidates(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        TODO.docstring
        
        """
        raise NotImplementedError("The inference_prepare_candidates method must be implemented by the translational decoder.")

    def inference_score(self, 
                        *,
                        projected_heads: Tensor,
                        projected_tails: Tensor,
                        edges: Tensor
                        ) -> Tensor:
        """
        TODO docstring
        
        """
        batch_size = projected_heads.shape[0]

        # When the shape of the edges is (batch_size, embedding_dimensions)
        if len(edges.shape) == 2:
            if len(projected_tails.shape) == 3:
                assert len(projected_heads.shape) == 2, "When inferring tails, the projected heads tensor should be of shape (batch_size, embedding_dimensions)"

                translated_heads = (projected_heads + edges).view(batch_size, 1, edges.size(1))
                return - self.dissimilarity(translated_heads, projected_tails)
            else:
                assert (len(projected_heads.shape) == 3) and (len(projected_tails) == 2), "When inferring heads, the projected tails tensor should be of shape (batch_size, embedding_dimensions)"

                edges_extended = edges.view(batch_size, 1, edges.size(1))
                tails_extended = projected_tails.view(batch_size, 1, edges.size(1))
                
                return - self.dissimilarity(projected_heads + edges_extended, tails_extended)
        elif len(edges.shape) == 3:
            if hasattr(self, "evaluated_projections"):
                projected_heads = projected_heads.view(batch_size, -1, edges.size(1))
                projected_tails = projected_tails.view(batch_size, -1, edges.size(1))
            else:
                projected_heads = projected_heads.view(batch_size, -1, projected_heads.size(1))
                projected_tails = projected_tails.view(batch_size, -1, projected_tails.size(1))

            return - self.dissimilarity(projected_heads + edges, projected_tails)



class TransE(TranslationalDecoder):
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
    def __init__(self, dissimilarity_type: Literal["L1","L2"] = "L2"):
        
        match dissimilarity_type:
            case "L1":
                self.dissimilarity = l1_dissimilarity
            case "L2":
                self.dissimilarity = l2_dissimilarity
            case _:
                raise ValueError(f"TransE decoder can only use L1 or L2 dissimlarity, but got \"{dissimilarity_type}\"")


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
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        
        return -self.dissimilarity(head_normalized_embeddings + edge_embeddings, tail_normalized_embeddings)
    
    
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
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding
            The edge embeddings, which are not normalized as per the paper's recommendation.
        
        Returns
        -------
        node_embeddings : torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings : torch.nn.Embedding
            The untouched edge embedding object.
        
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
        head_indices : torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, keyword-only
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True, keyword-only
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
    
    
    
class TransH(TranslationalDecoder):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Attributes
    ----------
    TODO.inherited_attributes
    
    """
    def __init__(self,
                node_count: int,
                edge_count: int,
                embedding_dimensions: int):
        self.normal_vector = initialize_embedding(edge_count, embedding_dimensions)
        self.dissimilarity = l2_dissimilarity

        self.evaluated_projections = False
        self.projected_nodes = Parameter(empty(size=(edge_count,
                                                    node_count,
                                                    embedding_dimensions)),
                                                    requires_grad = False)


    @staticmethod
    def project(nodes, normal_vector):
        """
        TODO.docstring
        
        """
        return nodes - (nodes * normal_vector).sum(dim = 1).view(-1, 1) * normal_vector


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
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        edge_indices: torch.Tensor, keyword-only
            The indices of the edges (from KG).

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        self.evaluated_projections = False
        normal_vector = normalize(self.normal_vector(edge_indices), p = 2, dim = 1)
        
        return - self.dissimilarity(self.project(head_normalized_embeddings, normal_vector) + edge_embeddings,
                                    self.project(tail_normalized_embeddings, normal_vector))
    
    
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
        self.normal_vector.weight.data = normalize(self.normal_vector.weight.data, p = 2, dim = 1)
        
        return node_embeddings, edge_embeddings


    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        result_1: Dict[str, Tensor]
            TODO.What_that_variable_is_or_does
            
        """
        return {"normal_vector": self.normal_vector.weight.data}
    
    
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
        head_indices : torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, keyword-only
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True, keyword-only
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
            projected_heads = self.projected_nodes[edge_indices, head_indices]  # shape: (batch_size, self.emb_dim)
            projected_tails = self.projected_nodes[edge_indices, tail_indices]  # shape: (batch_size, self.emb_dim)
            candidates = self.projected_nodes[edge_indices]  # shape: (batch_size, self.n_rel, self.emb_dim)
        else:
            projected_heads = self.projected_nodes[:, head_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
            projected_tails = self.projected_nodes[:, tail_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
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

            normal_vector = self.norm_vect.weight.data.view(self.n_rel, self.emb_dim)
            mask = tensor([i], device = normal_vector.device).long()

            if normal_vector.is_cuda:
                empty_cache()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]

            normalized_components = (masked_node_embeddings.view(1, -1) * normal_vector).sum(dim = 1)
            self.projected_nodes[:, i, :] = (masked_node_embeddings.view(1, -1) - normalized_components.view(-1, 1) * normal_vector)

            del normalized_components

        self.evaluated_projections = True



class TransR(TranslationalDecoder):
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
        
        self.node_count = node_count
        self.edge_count = edge_count
        self.node_embedding_dimensions = node_embedding_dimensions
        self.edge_embedding_dimensions = edge_embedding_dimensions

        self.projection_matrix = initialize_embedding(node_count, edge_embedding_dimensions * node_embedding_dimensions)

        self.dissimilarity = l2_dissimilarity

        self.evaluated_projections = False
        self.projected_nodes = Parameter(empty(size = (edge_count,
                                                        node_count,
                                                        embedding_dimensions)),
                                                        requires_grad = False)


    def score(  self,
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
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        edge_indices: torch.Tensor, keyword-only
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

        projection_matrix = self.proj_mat(edge_indices).view(batch_size,
                                                            self.edge_embedding_dimensions,
                                                            self.node_embedding_dimensions)
        
        return - self.dissimilarity(self.project(head_normalized_embeddings, projection_matrix) + edge_embeddings,
                                    self.project(tail_normalized_embeddings, projection_matrix))
    
    
    def project(self,
                nodes: Tensor,
                projection_matrix: Tensor):
        """
        Project the given nodes onto the projection matrix.
        TODO
        
        """
        projected_nodes = matmul(projection_matrix, nodes.view(-1, self.node_embedding_dimensions, 1))
        
        return projected_nodes.view(-1, self.edge_embedding_dimensions)
    
    
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
        return {"projection_matrix": self.projection_matrix.weight.data.view(-1,
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
        head_indices : torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, keyword-only
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True, keyword-only
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
            projected_heads = self.projected_nodes[edge_indices, head_indices]  # shape: (batch_size, self.emb_dim)
            projected_tails = self.projected_nodes[edge_indices, tail_indices]  # shape: (batch_size, self.emb_dim)
            candidates = self.projected_nodes[edge_indices]  # shape: (batch_size, self.n_rel, self.emb_dim)
        else:
            projected_heads = self.projected_nodes[:, head_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
            projected_tails = self.projected_nodes[:, tail_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.emb_dim)
            candidates = edge_embeddings.weight.data.unsqueeze(0).expand(batch_size, edge_embeddings.num_embeddings, edge_embeddings.embedding_dim)  # shape: (batch_size, self.n_rel, self.emb_dim)

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
            projection_matrices = self.projection_matrix.weight.data
            projection_matrices = projection_matrices.view(self.edge_count, self.edge_embedding_dimensions, self.node_embedding_dimension)

            mask = tensor([i], device = projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]
            
            projected_masked_node_embeddings = matmul(projection_matrices, masked_node_embeddings.view(self.node_embedding_dimension))
            projected_masked_node_embeddings = projected_masked_node_embeddings.view(self.n_rel, self.edge_embedding_dimensions, 1)
            self.projected_nodes[:, i, :] = projected_masked_node_embeddings.view(self.n_rel, self.edge_embedding_dimensions)
            # TODO: comment that projected_nodes equivalent to projected_masked_node_embeddings

            del projected_masked_node_embeddings

        self.evaluated_projections = True



class TransD(TranslationalDecoder):
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
        self.node_count = node_count
        self.edge_count = edge_count
        self.node_embedding_dimensions = node_embedding_dimensions
        self.edge_embedding_dimensions = edge_embedding_dimensions

        # Might be changed to have 2 embedding spaces instead (meaning it will be encoded by a GNN if present)
        self.node_projection_vector = initialize_embedding(self.node_count, self.node_embedding_dimensions)
        self.edge_projection_vector = initialize_embedding(self.edge_count, self.edge_embedding_dimensions)

        self.dissimilarity = l2_dissimilarity

        self.evaluated_projections = False
        self.projected_nodes = Parameter(empty(size = ( edge_count,
                                                        node_count,
                                                        embedding_dimensions)),
                                                        requires_grad = False)


    def score(  self,
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
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        head_indices: torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        edge_indices: torch.Tensor, keyword-only
            The indices of the edges (from KG).
        tail_indices: torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).

        Returns
        -------
        result_1: TODO.type
            TODO.What_that_variable_is_or_does
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        edge_normalized_embeddings = normalize(edge_embeddings, p = 2, dim = 1)

        head_projected_vectors = normalize(self.node_projection_vector(head_indices), p = 2, dim = 1)
        tail_projected_vectors = normalize(self.node_projection_vector(tail_indices), p = 2, dim = 1)
        edge_projected_vectors = normalize(self.edge_projection_vector(edge_indices), p = 2, dim = 1)

        projected_heads = self.project(head_normalized_embeddings, head_projected_vectors, edge_projected_vectors)
        projected_tails = self.project(tail_normalized_embeddings, tail_projected_vectors, edge_projected_vectors)
        
        return - self.dissimilarity(projected_heads + edge_normalized_embeddings, projected_tails)
    
    
    def project(self, nodes: Tensor, node_projection_vector: Tensor, edge_projection_vector: Tensor) -> Tensor:
        batch_size = nodes.shape[0]

        scalar_product = (nodes * node_projection_vector).sum(dim = 1)
        projected_nodes = (edge_projection_vector * scalar_product.view(batch_size, 1))

        return projected_nodes + nodes[:, :self.edge_embedding_dimensions]
    
    
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

        self.node_projection_vector.weight.data = normalize(self.node_projection_vector.weight.data, p = 2, dim = 1)
        self.edge_projection_vector.weight.data = normalize(self.edge_projection_vector.weight.data, p = 2, dim = 1)

        return node_embeddings, edge_embeddings


    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        result_1: Dict[str, Tensor]
            TODO.What_that_variable_is_or_does
            
        """
        return {"node_projection_vector": self.node_projection_vector.weight.data,
                "edge_projection_vector": self.edge_projection_vector.weight.data}
    
    
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
        head_indices : torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, keyword-only
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True, keyword-only
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
            projected_heads = self.projected_nodes[edge_indices, head_indices]  # shape: (batch_size, self.emb_dim)
            projected_tails = self.projected_nodes[edge_indices, tail_indices]  # shape: (batch_size, self.emb_dim)
            candidates = self.projected_nodes[edge_indices]  # shape: (batch_size, self.n_rel, self.emb_dim)
        else:
            projected_heads = self.projected_nodes[:, head_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.rel_emb_dim)
            projected_tails = self.projected_nodes[:, tail_indices].transpose(0, 1)  # shape: (batch_size, self.n_rel, self.rel_emb_dim)
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

        for i in tqdm(range(self.node_count), unit = "nodes", desc = "Projecting nodes"):
            edge_projection_vector = self.edge_projection_vector.weight.data

            mask = tensor([i], device = edge_projected_vectors.device).long()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]

            node_projection_vector = self.node_projection_vector.weight[i]

            scalar_product = (node_projection_vector * masked_node_embeddings).sum(dim = 0)
            projected_nodes = scalar_product * edge_projection_vector + masked_node_embeddings[:self.rel_emb_dim].view(1, -1)

            self.projected_nodes[:, i, :] = projected_nodes

            del projected_nodes

        self.evaluated_projections = True



class TorusE(TranslationalDecoder):
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
                dissimilarity_type: Literal["L1","torus_L1","torus_L2","torus_eL2"]):
        
        match dissimilarity_type:
            case "L1":
                self.dissimilarity = l1_dissimilarity
            case "torus_L1":
                self.dissimilarity = l1_torus_dissimilarity
            case "torus_L2":
                self.dissimilarity = l2_torus_dissimilarity
            case "torus_eL2":
                self.dissimilarity = el2_torus_dissimilarity
            case _:
                raise ValueError(f"TorusE decoder can only use L1, torus_L1, torus_L2 or torus_eL2 dissimlarity, but got \"{dissimilarity_type}\"")

        self.normalized = False
    
    
    def score(  self,
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
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            The edge embeddings, of size (n_rel, rel_emb_dim) corresponding to (edge_count, edge_embedding_dimensions)
        head_indices: torch.Tensor, keyword-only
            Unused.
        tail_indices: torch.Tensor, keyword-only
            Unused.
        edge_indices: torch.Tensor, keyword-only
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
            embedding.data.frac_() # Inplace fraction

        edge_embeddings.weight.data.frac_()
        self.normalized = True

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
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_indices : torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices : torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices : torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, keyword-only
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding, keyword-only
            TODO.What_that_argument_is_or_does
        node_inference : bool, optional, default to True, keyword-only
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