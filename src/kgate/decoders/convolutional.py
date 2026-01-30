"""
Convolutional decoder classes for training and inference.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Tuple, Dict

from torch import Tensor, cat
import torch.nn as nn
from torch.nn import Module



class ConvolutionalDecoder(Module):
    """
    Interface for convolutional decoders of KGATE.

    This interface is largely inspired by TorchKGE's ConvKBModel, and exposes
    the methods that all convolutional decoders must use to be compatible with KGATE.
    The interface doesn't have an __init__ method as inheriting decoders are supposed
    to take care of their initialization, and only requires one attribute to be set.

    Furthermore, this interface doesn't implement anything but is a type helper.
    
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
            The score method must be implemented by a convolutional decoder
            inheriting from this interface.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score of each triplet as a tensor.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
        
        """
        raise NotImplementedError("The score method must be implemented by the convolutional decoder.")


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
        The normalize_parameters method can be implemented by a convolutional decoder inheriting from this class
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
        embeddings: Dict[str, torch.Tensor] or None
            Decoder-specific embeddings, or None.
        
        Notes
        -----
        The get_embeddings method can be implemented by a convolutional decoder inheriting from this class
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
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
            The inference_prepare_candidates method must be implemented by a convolutional decoder
            inheriting from this interface.
            
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
        raise NotImplementedError("The inference_prepare_candidates method must be implemented by the convolutional decoder.")


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
            The inference_score method must be implemented by a convolutional decoder
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
        raise NotImplementedError("Convolutional decoders must implement the inference_score function themselves.")



class ConvKB(ConvolutionalDecoder):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    embedding_dimensions: int
        Dimensions of embeddings.
    filter_count: int
        TODO.What_that_argument_is_or_does
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
                filter_count: int,
                node_count: int,
                edge_count: int):
        
        self.embedding_dimensions = embedding_dimensions
        self.node_count = node_count
        self.edge_cont = edge_count

        self.convolution_layer = nn.Sequential(
            nn.Conv1d(3, filter_count, 1, stride = 1),
            nn.ReLu()
        )
        self.output = nn.Sequential(
            nn.Linear(self.embedding_dimensions * filter_count, 2),
            nn.Softmax(dim = 1)
        )

        
    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                **_) -> Tensor:
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
            The score of each triplet as a tensor.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
            
        """
        batch_size = head_embeddings.shape[0]

        head_score = head_embeddings.view(batch_size, 1, -1)
        tail_score = tail_embeddings.view(batch_size, 1, -1)
        edge_score = edge_embeddings.view(batch_size, 1, -1)

        concat = cat((head_score, edge_score, tail_score), dim = 1)

        convolution = self.convolution_layer(concat).reshape(batch_size, -1)
        
        return self.output(convolution)[:, 1]    
    
    
    def inference_prepare_candidates(self, 
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
        edge_embeddings_inference = edge_embeddings(edge_indices)

        if node_inference:
            # Prepare candidates for every node
            candidates = node_embeddings
        else:
            # Prepare candidates for every edge
            candidates = edge_embeddings.weight.data
        
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)
        candidates = candidates.view(batch_size, -1, 1, self.embedding_dimensions)

        return head_embeddings, tail_embeddings, edge_embeddings_inference, candidates


    def inference_score(self,
                        *,
                        head_embeddings: Tensor,
                        tail_embeddings: Tensor,
                        edge_embeddings: Tensor
                        ) -> Tensor:
        """
        TODO.what_that_function_does
        
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

        if len(head_embeddings.shape) == 4:
            assert (len(tail_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 2), \
                "When inferring heads..."
            concatenation = cat((head_embeddings,
                            edge_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions),
                            tail_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions)), dim = 2)

        elif len(tail_embeddings.shape) == 4:
            assert (len(head_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 2), \
                "When inferring tails..."
            concatenation = cat((head_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions),
                                edge_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions),
                                tail_embeddings), dim=2)
        
        elif len(edge_embeddings.shape) == 4:
            assert (len(head_embeddings.shape) == 2) and (len(tail_embeddings.shape) == 2), \
                "When inferring edges..."
            concatenation = cat((head_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.edge_count, 1, self.embedding_dimensions),
                                edge_embeddings,
                                tail_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.edge_count, 1, self.embedding_dimensions)), dim = 2)
        
        concatenation = concatenation.reshape(-1, 3, self.embedding_dimensions)

        convolution = self.convolution_layer(concatenation).reshape(concatenation.shape[0], -1)

        scores = self.output(convolution)

        return scores[:, :, 1]