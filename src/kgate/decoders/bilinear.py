"""
Bilinear decoder classes for training and inference.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Tuple, Dict          

from torch import matmul, Tensor, nn, tensor_split
from torch.nn.functional import normalize
from torch.nn import Module

from ..utils import initialize_embedding



class BilinearDecoder(Module):
    """
    Interface for bilinear decoders of KGATE.

    This interface is largely inspired by TorchKGE's BilinearModel, and exposes
    the methods that all bilinear decoders must use to be compatible with KGATE.
    The interface doesn't have an __init__ method as inheriting decoders are supposed
    to take care of their initialization, and only requires one attribute to be set.

    Furthermore, this interface doesn't implement anything but is a type helper.
    However, functions from this class returning None can be used directly from inheriting classes.
    
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

        Refer to the specific decoder for details on this function's implementation.
        While all arguments are given when called from the Architect class, most 
        decoders only use some of them. 

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the head nodes for the current batch of length `batch_size`.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the tail nodes for the current batch of length `batch_size`.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the edges for the current batch of length `batch_size`.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes for the current batch of length `batch_size`.
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes for the current batch of length `batch_size`.
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges for the current batch of length `batch_size`.
        
        Raises
        ------
        NotImplementedError
            The score method must be implemented by a bilinear decoder
            inheriting from this interface.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score for each triplet given as a tensor by the decoder for the batch.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
        Here, [batch_size] is batch.shape[1].
        
        """
        raise NotImplementedError("The score method must be implemented by the bilinear decoder.")


    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding] | None:
        """
        Interface method for the decoder's parameters normalization function.

        Refer to the specific decoder for details on this function's implementation.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, shape: [node_count, embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, shape: [node_count, embedding_dimensions]
            The edge embedding as a ParameterList containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding
            The normalized edges embedding object.
        
        Notes
        -----
        The normalize_parameters method can be implemented by a bilinear decoder inheriting from this class
        if it has specific parameters to normalize.
        If the decoder doesn't have dedicated normalization, nothing is returned. In 
        this case, it is not necessary to implement this method from the interface.
        
        """    
        return None


    def get_embeddings(self) -> Dict[str, Tensor] | None:
        """
        Get the decoder-specific embeddings.
        
        Refer to the specific decoder for details on this function's implementation.
        
        Returns
        -------
        edge_embeddings_matrix: Dict[str, torch.Tensor] or None
            Decoder-specific embeddings, or None.
        
        Notes
        -----
        The get_embeddings method can be implemented by a bilinear decoder inheriting from this class
        if it needed.
        If the decoder doesn't have dedicated embeddings, nothing is returned. In 
        this case, it is not necessary to implement this method from the interface.
        
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
                                    ) -> Tuple[
                                                Tensor | Tuple,
                                                Tensor | Tuple,
                                                Tensor | Tuple,
                                                Tensor | Tuple
                                                ]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score_function` method.
        
        Refer to the specific decoder for details on this function's implementation.
        While all arguments are given when called from the Architect class, most 
        decoders only use some of them.
        
        Arguments
        ---------
        head_indices: torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, shape: [node_count, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, keyword-only
            Embeddings of all edges.
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.
        
        Raises
        ------
        NotImplementedError
            The inference_prepare_candidates method must be implemented by a bilinear decoder
            inheriting from this interface.
        
        Returns
        -------
        head_embeddings: torch.Tensor or Tuple
            Head node embeddings.
        tail_embeddings: torch.Tensor or Tuple
            Tail node embeddings.
        edge_embeddings_inference: torch.Tensor or Tuple
            Edge embeddings.
        candidates: torch.Tensor or Tuple
            Candidate embeddings for nodes or edges.
        
        """
        raise NotImplementedError("The inference_prepare_candidates method must be implemented by the bilinear decoder.")


    def inference_score(self, 
                        *,
                        head_embeddings: Tensor,
                        tail_embeddings: Tensor,
                        edge_embeddings: Tensor
                        ) -> Tensor:
        """
        TODO.what_that_function_does

        Refer to the specific decoder for details on this function's implementation.
        While all arguments are given when called from the Architect class, most 
        decoders only use some of them.
        
        Arguments
        ---------
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            Embeddings of the edges in the knowledge graph.
        
        Raises
        ------
        NotImplementedError
            The inference_score method must be implemented by a bilinear decoder
            inheriting from this interface.

        Returns
        -------
        score: torch.Tensor, shape: [batch_size, candidate_count]
            Tensor of score values.
            First dimension: incomplete triplets tested
            Second dimension: candidate indices
            For example, if the function is called to infer the score of tails:
            First dimension: (head_indices, edge_indices)
            Second dimension: tail_indices
        
        """
        raise NotImplementedError("Bilinear decoders must implement the inference_score function themselves.")



class RESCAL(BilinearDecoder):
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
        Number of nodes in the batch.
    edge_count: int
        Number of edges in the batch.
        
    Attributes
    ----------
    edge_embeddings_matrix: TODO.type
        TODO.What_that_variable_is_or_does
    embedding_dimensions: int
        Dimensions of embeddings.
    edge_count: int
        Number of edges in the batch.
    node_count: int
        Number of nodes in the batch.
    TODO.inherited_attributes
        
    Notes
    -----
    The batch can be the whole graph if it fits in memory.
    
    """
    def __init__(self, 
                embedding_dimensions: int, 
                node_count: int,
                edge_count: int):
        
        self.edge_count = edge_count
        self.node_count = node_count
        self.embedding_dimensions = embedding_dimensions

        self.edge_embeddings_matrix = initialize_embedding(self.edge_count, self.embedding_dimensions * self.embedding_dimensions)


    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_indices: Tensor,
                **_
                ) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the head nodes for the current batch of length `batch_size`.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the tail nodes for the current batch of length `batch_size`.
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges for the current batch of length `batch_size`.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score for each triplet given as a tensor by the decoder for the batch.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
        Here, [batch_size] is batch.shape[1].
        
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        edge_embeddings = self.edge_embeddings_matrix(edge_indices).view(-1, self.embedding_dimensions, self.embedding_dimensions)
        head_edge_embeddings = matmul(head_normalized_embeddings.view(-1, 1, self.embedding_dimensions), edge_embeddings)
        
        return (head_edge_embeddings.view(-1, self.embedding_dimensions) * tail_normalized_embeddings).sum(dim = 1)
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the RESCAL model.
        
        According to the original paper, the node embeddings should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, shape: [node_count, embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, shape: [node_count, embedding_dimensions]
            The edge embedding as a ParameterList containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding
            The unchanged edges embedding object.
        
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)
            
        return node_embeddings, edge_embeddings


    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        edge_embeddings_matrix: Dict[str, Tensor]
            TODO.What_that_variable_is_or_does
            
        """
        return {"edge_embeddings_matrix": self.edge_embeddings_matrix.weight.data.view(-1, self.embedding_dimensions, self.embedding_dimensions)}
    

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
        `inference_score_function` method.

        Arguments
        ---------
        head_indices: torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, shape: [node_count, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, keyword-only
            Unused.
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor
            Head node embeddings.
        tail_embeddings: torch.Tensor
            Tail node embeddings.
        edge_embeddings_inference: torch.Tensor
            Edge embeddings.
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        # Get head, tail and edge embeddings
        head_embeddings = node_embeddings[head_indices]
        tail_embeddings = node_embeddings[tail_indices]
        edge_embeddings_inference = self.edge_embeddings_matrix(edge_indices).view(-1, self.embedding_dimensions, self.embedding_dimensions)

        if node_inference:
            # Prepare candidates for every node
            candidates = node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Prepare candidates for every edge
            candidates = self.edge_embeddings_matrix.weight.data.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return head_embeddings, tail_embeddings, edge_embeddings_inference, candidates


    def inference_score(self,
                        *,
                        head_embeddings: Tensor,
                        tail_embeddings: Tensor,
                        edge_embeddings: Tensor
                        ) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            Embeddings of the edges in the knowledge graph.

        Raises
        ------
        AssertionError #1
            When inferring heads, the tensors tail_embeddings must have 2 dimensions
            and edge_embeddings must have 3 dimensions.
        AssertionError #2
            When inferring tails, the tensors head_embeddings must have 2 dimensions
            and edge_embeddings must have 3 dimensions.
        AssertionError #3
            When inferring edges, the tensors head_embeddings and tail_embeddings must have 2 dimensions.

        Returns
        -------
        score: torch.Tensor, shape: [batch_size, candidate_count]
            Tensor of score values.
            First dimension: incomplete triplets tested
            Second dimension: candidate indices
            For example, if the function is called to infer the score of tails:
            First dimension: (head_indices, edge_indices)
            Second dimension: tail_indices
        
        Notes
        -----
        The tensor of edges have one more dimension because of the projection matrix used.
        
        """        
        batch_size = head_embeddings.shape[0]

        if len(head_embeddings.shape) == 3:
            assert (len(tail_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 3), \
                "When inferring heads, ..."

            tail_edge_embeddings = matmul(edge_embeddings, tail_embeddings.view(batch_size, self.embedding_dimensions, 1)).view(batch_size, 1, self.embedding_dimensions)
            
            return (head_embeddings * tail_edge_embeddings).sum(dim = 2)
        
        elif len(tail_embeddings.shape) == 3:
            assert (len(head_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 3), \
                "When inferring tails, ..."
            
            head_edge_embeddings = matmul(head_embeddings.view(batch_size, 1, self.embedding_dimensions)).view(batch_size, 1, self.embedding_dimensions)
            
            return (head_edge_embeddings * tail_embeddings).sum(dim = 2)
        
        elif len(edge_embeddings.shape) == 4:
            assert (len(head_embeddings.shape) == 2) and (len(tail_embeddings.shape) == 2), \
                "When inferring edges, ..."

            head_embeddings = head_embeddings.view(batch_size, 1, 1, self.embedding_dimensions)
            tail_embeddings = tail_embeddings.view(batch_size, 1, self.embedding_dimensions)
            head_edge_embeddings = matmul(head_embeddings, edge_embeddings).view(batch_size, self.edge_count, self.embedding_dimensions)
        
            return (head_edge_embeddings * tail_embeddings).sum(dim = 2)
        
        # TODO: else with raise error
    
    
    
class DistMult(BilinearDecoder):
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
        
        self.embedding_dimensions = embedding_dimensions
        self.node_count = node_count
        self.edge_count = edge_count
    
    
    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                **_
                ) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            Embeddings of the edges in the knowledge graph.

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the head nodes for the current batch of length `batch_size`.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the tail nodes for the current batch of length `batch_size`.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the edges for the current batch of length `batch_size`.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score for each triplet given as a tensor by the decoder for the batch.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
        Here, [batch_size] is batch.shape[1].
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        
        return (head_normalized_embeddings * edge_embeddings * tail_normalized_embeddings).sum(dim = 1)
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the DistMult model.
        
        According to the original paper, the node embeddings should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, shape: [node_count, embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, shape: [node_count, embedding_dimensions]
            The edge embedding as a ParameterList containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding
            The unchanged edges embedding object.
        
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
        `inference_score_function` method.
        Arguments
        ---------
        head_indices: torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, shape: [node_count, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, keyword-only
            Embeddings of all edges.
        node_inference: bool, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor
            Head node embeddings.
        tail_embeddings: torch.Tensor
            Tail node embeddings.
        edge_embeddings_inference: torch.Tensor
            Edge embeddings.
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


    def inference_score(self,
                        *,
                        head_embeddings: Tensor,
                        tail_embeddings: Tensor,
                        edge_embeddings: Tensor
                        ) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            Embeddings of the edges in the knowledge graph.

        Raises
        ------
        AssertionError #1
            When inferring heads, the tensors tail_embeddings and edge_embeddings must have 2 dimensions.
        AssertionError #2
            When inferring tails, the tensors head_embeddings and edge_embeddings must have 2 dimensions.
        AssertionError #3
            When inferring edges, the tensors head_embeddings and tail_embeddings must have 2 dimensions.

        Returns
        -------
        score: torch.Tensor, shape: [batch_size, candidate_count]
            Tensor of score values.
            First dimension: incomplete triplets tested
            Second dimension: candidate indices
            For example, if the function is called to infer the score of tails:
            First dimension: (head_indices, edge_indices)
            Second dimension: tail_indices
        
        """
        batch_size = head_embeddings.shape[0]

        if len(head_embeddings.shape) == 3:
            assert (len(tail_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 2), \
                "When inferring heads, ..."

            tail_edge_embeddings = (edge_embeddings * tail_embeddings).view(batch_size, 1, self.embedding_dimensions)
            
            return (head_embeddings * tail_edge_embeddings).sum(dim = 2)
        
        elif len(tail_embeddings.shape) == 3:
            assert (len(head_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 2), \
                "When inferring tails, ..."
            
            head_edge_embeddings = (head_embeddings * edge_embeddings).view(batch_size, 1, self.embedding_dimensions)
            
            return (head_edge_embeddings * tail_embeddings).sum(dim = 2)
        
        elif len(edge_embeddings.shape) == 3:
            assert (len(head_embeddings.shape) == 2) and (len(tail_embeddings.shape) == 2), \
                "When inferring edges, ..."

            head_edge_embeddings = (head_embeddings.view(batch_size, 1, self.embedding_dimensions) * edge_embeddings)
            
            return (head_edge_embeddings * tail_embeddings.view(batch_size, 1, self.embedding_dimensions)).sum(dim = 2)
        
        # TODO: else with raise error



class ComplEx(BilinearDecoder):
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
                embedding_dimensions: int):
        self.embedding_dimensions = embedding_dimensions
        self.embedding_spaces = 2


    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                **_
                ) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the head nodes for the current batch of length `batch_size`.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the tail nodes for the current batch of length `batch_size`.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size], keyword-only
            The embeddings of the edges for the current batch of length `batch_size`.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score for each triplet given as a tensor by the decoder for the batch.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
        Here, [batch_size] is batch.shape[1].
        
        """        
        real_head_embedddings, imaginary_head_embeddings = tensor_split(head_embeddings, 2, dim = 1)
        real_tail_embedddings, imaginary_tail_embeddings = tensor_split(tail_embeddings, 2, dim = 1)
        real_edge_embedddings, imaginary_edge_embeddings = tensor_split(edge_embeddings, 2, dim = 1)
        
        return (real_head_embedddings * (real_edge_embedddings * real_tail_embedddings + imaginary_edge_embeddings * imaginary_tail_embeddings) + 
                imaginary_head_embeddings * (real_edge_embedddings * imaginary_tail_embeddings - imaginary_edge_embeddings * real_tail_embedddings)).sum(dim = 1)
    
    
    def inference_prepare_candidates(self,
                                    *, 
                                    head_indices: Tensor, 
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor, 
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool = True
                                    ) -> Tuple[
                                        Tuple[Tensor, Tensor], 
                                        Tuple[Tensor, Tensor],
                                        Tuple[Tensor, Tensor],
                                        Tuple[Tensor, Tensor]]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score_function` method.

        References
        ----------
        TODO

        Arguments
        ---------
        head_indices: torch.Tensor, keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, keyword-only
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor, shape: [node_count, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, keyword-only
            Embeddings of all edges.
        node_inference: bool, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        (real_head_embeddings, imaginary_head_embeddings): Tuple[Tensor, Tensor]
            Head node embeddings, both the real and the imaginary ones.
        (real_tail_embeddings, imaginary_tail_embeddings): Tuple[Tensor, Tensor]
            Tail node embeddings, both the real and the imaginary ones.
        (real_edge_embeddings, imaginary_edge_embeddings): Tuple[Tensor, Tensor]
            Edge embeddings, both the real and the imaginary ones.
        (real_candidates, imaginary_candidates): Tuple[Tensor, Tensor]
            Candidate embeddings for nodes or edges, both the real and the imaginary ones.

        """
        batch_size = head_indices.shape[0]

        real_head_embeddings, imaginary_head_embeddings = tensor_split(node_embeddings[head_indices], 2, dim = 1)
        real_tail_embeddings, imaginary_tail_embeddings = tensor_split(node_embeddings[tail_indices], 2, dim = 1)
        real_edge_embeddings, imaginary_edge_embeddings = tensor_split(edge_embeddings(edge_indices), 2, dim = 1)

        if node_inference:
            real_candidates, imaginary_candidates = tensor_split(node_embeddings, 2, dim = 1)
        else:
            real_candidates, imaginary_candidates = tensor_split(edge_embeddings, 2, dim = 1)
        
        real_candidates = real_candidates.unsqueeze(0).expand(batch_size, -1, -1)
        imaginary_candidates = imaginary_candidates.unsqueeze(0).expand(batch_size, -1, -1)

        return  (real_head_embeddings, imaginary_head_embeddings), \
                (real_tail_embeddings, imaginary_tail_embeddings), \
                (real_edge_embeddings, imaginary_edge_embeddings), \
                (real_candidates, imaginary_candidates)
    
    
    def inference_score(self,
                        *,
                        head_embeddings: Tensor,
                        tail_embeddings: Tensor,
                        edge_embeddings: Tensor
                        ) -> Tensor:
        """
        TODO.What_the_function_does_about_globally

        Arguments
        ---------
        head_embeddings: torch.Tensor, keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, keyword-only
            Embeddings of the edges in the knowledge graph.

        Raises
        ------
        AssertionError #1
            When inferring heads, the tensors tail_embeddings and edge_embeddings must have 2 dimensions.
        AssertionError #2
            When inferring tails, the tensors head_embeddings and edge_embeddings must have 2 dimensions.
        AssertionError #3
            When inferring edges, the tensors head_embeddings and tail_embeddings must have 2 dimensions.

        Returns
        -------
        score: torch.Tensor, shape: [batch_size, candidate_count]
            Tensor of score values.
            First dimension: incomplete triplets tested
            Second dimension: candidate indices
            For example, if the function is called to infer the score of tails:
            First dimension: (head_indices, edge_indices)
            Second dimension: tail_indices
        
        """
        real_head_embeddings, imaginary_head_embeddings = tensor_split(head_embeddings, 2, dim = 1)
        real_tail_embeddings, imaginary_tail_embeddings = tensor_split(tail_embeddings, 2, dim = 1)
        real_edge_embeddings, imaginary_edge_embeddings = tensor_split(edge_embeddings, 2, dim = 1)
        
        batch_size = real_head_embeddings.shape[0]

        if len(real_head_embeddings.shape) == 3:
            assert (len(real_tail_embeddings.shape) == 2) and (len(real_edge_embeddings.shape) == 2), \
                "When inferring heads, ..."
            
            return (real_head_embeddings * 
                        (real_edge_embeddings * real_tail_embeddings 
                         + imaginary_edge_embeddings * imaginary_tail_embeddings
                        ).view(batch_size, 1, self.embedding_dimensions)
                    + imaginary_head_embeddings * 
                        (real_edge_embeddings * imaginary_tail_embeddings
                        - imaginary_edge_embeddings * real_tail_embeddings
                        ).view(batch_size, 1, self.embedding_spaces)
                    ).sum(dim = 2)

        elif len(real_tail_embeddings.shape) == 3:
            assert (len(real_head_embeddings.shape) == 2) and (len(real_edge_embeddings.shape) == 2), \
                "When inferring tails, ..."
            
            return ((real_head_embeddings * real_edge_embeddings
                        - imaginary_head_embeddings * imaginary_edge_embeddings
                        ).view(batch_size, 1, self.embedding_dimensions)
                    * real_tail_embeddings
                    + (real_head_embeddings * imaginary_edge_embeddings
                        + imaginary_head_embeddings * real_tail_embeddings
                        ).view(batch_size, 1, self.embedding_dimensions)
                    * imaginary_tail_embeddings
                    )

        elif len(real_edge_embeddings.shape) == 3:
            assert (len(real_head_embeddings.shape) == 2) and (len(real_tail_embeddings.shape) == 2), \
                "When inferring edges, ..."
            
            return ((real_head_embeddings * real_tail_embeddings
                        + imaginary_head_embeddings * imaginary_tail_embeddings
                        ).view(batch_size, 1, self.embedding_dimensions)
                    * real_edge_embeddings
                    + (real_head_embeddings * imaginary_tail_embeddings
                        - imaginary_head_embeddings * real_tail_embeddings
                        ).view(batch_size, 1, self.embedding_dimensions)
                    * imaginary_edge_embeddings
                    ).sum(dim = 2)
        
        # TODO: else with raise error