"""
Convolutional decoder classes for training and inference.

Original code for the decoders from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

import torch
from torch import Tensor, cat
import torch.nn as nn
from torch.nn import Module, functional as F



class ConvolutionalDecoder(Module):
    def __init__(self):
        """
        Interface for convolutional decoders of KGATE.

        This interface is largely inspired by TorchKGE's ConvKBModel, and exposes 
        the methods that all convolutional decoders must use to be compatible with KGATE.

        Furthermore, this interface doesn't implement anything but is a type helper.

        """
        super().__init__()
    
    
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
        
        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only)*
        : The embeddings of the head nodes for the current batch of length `batch_size`.
        
        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only)*
        : The embeddings of the tail nodes for the current batch of length `batch_size`.
        
        **edge_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only)*
        : The embeddings of the edges for the current batch of length `batch_size`.
        
        **head_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the head nodes for the current batch of length `batch_size`.
        
        **tail_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the tail nodes for the current batch of length `batch_size`.
        
        **edge_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the edges for the current batch of length `batch_size`.
        
        Raises
        ------
        
        **NotImplementedError**
        : The `score` method must be implemented by a convolutional decoder 
        inheriting from this interface.

        Returns
        -------
        
        **batch_score** *(torch.Tensor, dtype: torch.float, shape: [batch_size])*
        : The score of each triplet as a tensor.
        
        Notes
        -----
        
        The batch can be the whole graph if it fits in memory.
        
        """
        raise NotImplementedError("The score method must be implemented by the convolutional decoder.")


    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Parameter
                            ) -> tuple[nn.ParameterList, nn.Parameter] | None:
        """
        Interface method for the decoder's parameters normalization function.

        Refer to the specific decoder for details on this function's implementation.
        
        Arguments
        ---------
        
        **node_embeddings** *(torch.nn.ParameterList, dtype: torch.float)*
        : The node embedding as a ParameterList containing one Parameter per node type,
        : each of shape [node_count for the node type, embedding_dimensions],
        : or only one if there is no node type.
        
        **edge_embeddings** *(torch.nn.Parameter, dtype: torch.float, shape: [edge_count, embedding_dimensions])*
        : The edge embedding as a nn.Parameter containing one row per edge type,
        : or only one if there is no edge type.
        
        Returns
        -------
        
        **node_embeddings** *(torch.nn.ParameterList, dtype: torch.float)*
        : The normalized node embedding object, with the same structure as the input.
        
        **edge_embeddings** *(torch.nn.Parameter, dtype: torch.float)*
        : The normalized edge embedding object.
        
        Notes
        -----
        
        The `normalize_parameters` method can be implemented by a convolutional decoder inheriting from this class 
        if it has specific parameters to normalize.
        
        If the decoder doesn't have dedicated normalization, nothing is returned. In 
        this case, it is not necessary to implement this method from the interface.
        
        """    
        return None


    def get_embeddings(self) -> dict[str, Tensor] | None:
        """
        Get the decoder-specific embeddings.

        Refer to the specific decoder for details on this function's implementation.
        
        Returns
        -------
        
        **embeddings** *(Dict[str, torch.Tensor] or None)*
        : Decoder-specific embeddings, or None.
        
        Notes
        -----
        The `get_embeddings` method can be implemented by a convolutional decoder inheriting from this class 
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
                                    edge_embeddings: nn.Parameter,
                                    node_inference: bool = True
                                    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings 
        and edge embeddings. The output will be fed to the 
        `inference_score` method.

        Refer to the specific decoder for details on this function's implementation.
        
        While all arguments are given when called from the Architect class, most 
        decoders only use some of them.
        
        Arguments
        ---------
        
        **head_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the head nodes (from KG).
        
        **tail_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the tail nodes (from KG).
        
        **edge_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the edges (from KG).
        
        **node_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions], keyword-only)*
        : Embeddings of all nodes.
        
        **edge_embeddings** *(torch.nn.Parameter, dtype: torch.float, shape: [edge_count, edge_embedding_dimensions], keyword-only)*
        : Embeddings of all edges.
        
        **node_inference** *(bool, optional, default to True, keyword-only)*
        : If True, prepare candidate nodes; otherwise, prepare candidate edges.
        
        Raises
        ------
        
        **NotImplementedError**
        : The `inference_prepare_candidates` method must be implemented by a convolutional decoder 
        inheriting from this interface.
        
        Returns
        -------
        
        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions])*
            Head node embeddings.
        
        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions])*
            Tail node embeddings.
        
        **edge_embeddings_inferred** *(torch.Tensor, dtype: torch.float, shape: [edge_count, edge_embedding_dimensions])*
            Edge embeddings.
        
        **candidates** *(torch.Tensor)*
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
        Link prediction evaluation helper function. Compute the scores 
        of (head, candidate, edge) or (candidate, tail, edge) for any candidate.
        
        The arguments should match the ones of the output of `inference_prepare_candidates`.

        Refer to the specific decoder for details on this function's implementation.
        
        While all arguments are given when called from the Architect class, most 
        decoders only use some of them.
        
        Arguments
        ---------
        
        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions], keyword-only)*
        : Embeddings of the head nodes in the batch.
        
        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions], keyword-only)*
        : Embeddings of the tail nodes in the batch.
        
        **edge_embeddings** *(torch.Tensor, dtype: torch.float, shape: [edge_count, edge_embedding_dimensions], keyword-only)*
        : Embeddings of the edges in the batch.
        
        Raises
        ------
        **NotImplementedError**
        : The `inference_score` method must be implemented by a convolutional decoder 
        inheriting from this interface.

        Returns
        -------
        **score** *(torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count])*
        : Tensor of score values.
        : First dimension: incomplete triplets tested
        : Second dimension: candidate indices
        : For example, if the function is called to infer the score of tails:
        : First dimension: (head_indices, edge_indices)
        : Second dimension: tail_indices
        
        """
        raise NotImplementedError("Convolutional decoders must implement the inference_score function themselves.")



class ConvKB(ConvolutionalDecoder):
    def __init__(self,
                node_count: int,
                edge_count: int,
                embedding_dimensions: int,
                filter_count: int):
        """
        Implementation of ConvKB model detailed in the paper referenced below.

        This class inherits from the ConvolutionalDecoder interface, whose interface methods it implements.

        References
        ----------
        
        * Dai Quoc Nguyen, Tu Dinh Nguyen, Dat Quoc Nguyen, Dinh Phung
        
            `A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network`
        
            <https://arxiv.org/abs/1712.02121>
        
            In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational 
            Linguistics: Human Language Technologies (2018), vol. 2, pp. 327–333.

        Arguments
        ---------
        
        **embedding_dimensions** *(int)*
        : Dimensions of embeddings.
        
        **filter_count** *(int)*
        : Number of convolution filters to apply.
        
        **node_count** *(int)*
        : Number of nodes in the knowledge graph.
        
        **edge_count** *(int)*
        : Number of edges in the knowledge graph.

        Attributes
        ----------
        
        **node_count** *(int)*
        : Number of nodes in the knowledge graph.
        
        **edge_count** *(int)*
        : Number of edges in the knowledge graph.
        
        **embedding_dimensions** *(int)*
        : Dimensions of embeddings.
        
        **convolution_layer** *(torch.nn.Sequential)*
        : The convolution layer of the model.
        
        **output** *(torch.nn.Sequential)*
        : The reconstruction layer of the model.

        """
        super().__init__()
        
        self.node_count = node_count
        self.edge_count = edge_count
        self.embedding_dimensions = embedding_dimensions

        self.convolution_layer = nn.Sequential(
            nn.Conv1d(3, filter_count, 1, stride = 1),
            nn.ReLU()
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
        Compute the score function for the triplets given as argument.
        
        See referenced paper for more details on the score: 
        <https://arxiv.org/abs/1712.02121>

        Arguments
        ---------
        
        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only)*
        : The embeddings of the head nodes for the current batch of length `batch_size`.
        
        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only)*
        : The embeddings of the tail nodes for the current batch of length `batch_size`.
        
        **edge_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only)*
        : The embeddings of the edges for the current batch of length `batch_size`.

        Returns
        -------
        **batch_score** *(torch.Tensor, dtype: torch.float, shape: [batch_size])*
        : The score of each triplet as a tensor.
        
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
                                    edge_embeddings: nn.Parameter,
                                    node_inference: bool = True
                                    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings 
        and edge embeddings. The output will be fed to the 
        `inference_score` method.
        
        Arguments
        ---------
        
        **head_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the head nodes (from KG).
        
        **tail_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the tail nodes (from KG).
        
        **edge_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the edges (from KG).
        
        **node_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions], keyword-only)*
        : Embeddings of all nodes.
        
        **edge_embeddings** *(torch.nn.Parameter, dtype: torch.float, shape: [edge_count, edge_embedding_dimensions], keyword-only)*
        : Embeddings of all edges.
        
        **node_inference** *(bool, optional, default to True, keyword-only)*
        : If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        
        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions])*
        : Head node embeddings.
        
        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions])*
        : Tail node embeddings.
        
        **edge_embeddings_inferred** *(torch.Tensor)*
        : Edge embeddings.
        
        **candidates** *(torch.Tensor)*
        : Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        # Get head, tail and edge embeddings
        head_embeddings = node_embeddings[head_indices]
        tail_embeddings = node_embeddings[tail_indices]
        edge_embeddings_inferred = edge_embeddings[edge_indices]

        if node_inference:
            # Prepare candidates for every node
            candidates = node_embeddings
        else:
            # Prepare candidates for every edge
            candidates = edge_embeddings
        
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)
        candidates = candidates.view(batch_size, -1, 1, self.embedding_dimensions)

        return head_embeddings, tail_embeddings, edge_embeddings_inferred, candidates


    def inference_score(self,
                        *,
                        head_embeddings: Tensor,
                        tail_embeddings: Tensor,
                        edge_embeddings: Tensor
                        ) -> Tensor:
        """
        Link prediction evaluation helper function. Compute the scores 
        of (head, candidate, edge) or (candidate, tail, edge) for any candidate.
        
        The arguments should match the ones of the output of `inference_prepare_candidates`.
        
        Arguments
        ---------
        
        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions], keyword-only)*
        : Embeddings of the head nodes in the knowledge graph.
        
        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions], keyword-only)*
        : Embeddings of the tail nodes in the knowledge graph.
        
        **edge_embeddings** *(torch.Tensor, dtype: torch.float, shape: [edge_count, edge_embedding_dimensions], keyword-only)*
        : Embeddings of the edges in the knowledge graph.
        
        Raises
        ------
        
        **AssertionError #1**
        : When inferring heads, the tensors tail_embeddings and edge_embeddings must have 2 dimensions.
        
        **AssertionError #2**
        : When inferring tails, the tensors head_embeddings and edge_embeddings must have 2 dimensions.
        
        **AssertionError #3**
        : When inferring edges, the tensors head_embeddings and tail_embeddings must have 2 dimensions.

        Returns
        -------
        **score** *(torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count])*
        : Tensor of score values.
            : First dimension: incomplete triplets tested
            : Second dimension: candidate indices
        : For example, if the function is called to infer the score of tails:
            : First dimension: (head_indices, edge_indices)
            : Second dimension: tail_indices
        
        """        
        batch_size = head_embeddings.shape[0]

        if len(head_embeddings.shape) == 4:
            assert (len(tail_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 2), \
                "When inferring heads, the tensors `tail_embeddings` and `edge_embeddings` must have 2 dimensions."
            concatenation = cat((head_embeddings,
                            edge_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions),
                            tail_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions)), dim = 2)

        elif len(tail_embeddings.shape) == 4:
            assert (len(head_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 2), \
                "WWhen inferring tails, the tensors `head_embeddings` and `edge_embeddings` must have 2 dimensions."
            concatenation = cat((head_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions),
                                edge_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.node_count, 1, self.embedding_dimensions),
                                tail_embeddings), dim=2)
        
        elif len(edge_embeddings.shape) == 4:
            assert (len(head_embeddings.shape) == 2) and (len(tail_embeddings.shape) == 2), \
                "When inferring edges, the tensors `head_embeddings` and `tail_embeddings` must have 2 dimensions."
            concatenation = cat((head_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.edge_count, 1, self.embedding_dimensions),
                                edge_embeddings,
                                tail_embeddings.view(batch_size, 1, 1, self.embedding_dimensions).expand(batch_size, self.edge_count, 1, self.embedding_dimensions)), dim = 2)
        # TODO: is a ValueError within an 'else' needed here?
        concatenation = concatenation.reshape(-1, 3, self.embedding_dimensions)

        convolution = self.convolution_layer(concatenation).reshape(concatenation.shape[0], -1)

        scores = self.output(convolution)

        return scores[:, :, 1]


class ConvE(ConvolutionalDecoder):
    def __init__(self,
                node_count: int,
                edge_count: int,
                embedding_dimensions: int,
                filter_count: int = 32,
                first_embedding_shape: int = 0,
                device: torch.device | str = "cpu"):
        """
        Implementation of ConvE model detailed in the paper referenced below.

        ConvE reshapes the head and edge embeddings into 2D matrices, stacks them
        into a 2-channel input, applies a 2D convolution (3×3 kernel), flattens the
        result, projects it back to the embedding dimension, and scores the triplet
        by taking the dot product with the tail embedding.

        References
        ----------

        * Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel

            `Convolutional 2D Knowledge Graph Embeddings`

            <https://arxiv.org/abs/1707.01476>

            In Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18),
            pages 1861–1868. 2018.

        Arguments
        ---------

        **node_count** *(int)*
        : Number of nodes in the knowledge graph.

        **edge_count** *(int)*
        : Number of edges in the knowledge graph.

        **embedding_dimensions** *(int)*
        : Dimensions of embeddings. Must be factorizable as ``first_embedding_shape * second_embedding_shape``.

        **filter_count** *(int, default to 32)*
        : Number of convolution filters (output channels) in the 2D convolution layer.

        **first_embedding_shape** *(int, default to 0)*
        : First dimension of the 2D reshape of the embeddings.
        : If 0, the closest square factor of ``embedding_dimensions`` is used.
        : The second dimension is inferred as ``embedding_dimensions // first_embedding_shape``.

        **device** *(torch.device or str, default to "cpu")*
        : Device on which to create the model parameters.

        Raises
        ------

        **ValueError**
        : If ``embedding_dimensions`` cannot be evenly divided by ``first_embedding_shape``.

        Attributes
        ----------

        **node_count** *(int)*
        : Number of nodes in the knowledge graph.

        **edge_count** *(int)*
        : Number of edges in the knowledge graph.

        **embedding_dimensions** *(int)*
        : Dimensions of embeddings.

        **first_embedding_shape** *(int)*
        : First dimension of the 2D embedding reshape.

        **second_embedding_shape** *(int)*
        : Second dimension of the 2D embedding reshape.

        **filter_count** *(int)*
        : Number of convolution filters.

        **conv_layer** *(torch.nn.Conv2d)*
        : The 2D convolution layer (3×3 kernel, 1→filter_count channels).

        **projection** *(torch.nn.Linear)*
        : The projection layer from the flattened convolution output back to
        : ``embedding_dimensions``.

        """
        super().__init__()

        self.node_count = node_count
        self.edge_count = edge_count
        self.embedding_dimensions = embedding_dimensions
        self.filter_count = filter_count

        # Determine the 2D reshape dimensions
        if first_embedding_shape == 0:
            # Find the factor closest to sqrt(dimensions) for a near-square reshape
            square_dimension = int(embedding_dimensions ** 0.5)
            for candidate in range(square_dimension, 0, -1):
                if embedding_dimensions % candidate == 0:
                    embedding_shape1 = candidate
                    break
            else:
                first_embedding_shape = embedding_dimensions  # fallback: 1D

        self.first_embedding_shape = first_embedding_shape
        if embedding_dimensions % first_embedding_shape != 0:
            raise ValueError(
                f"embedding_dimensions ({embedding_dimensions}) must be evenly "
                f"divisible by first_embedding_shape ({first_embedding_shape})."
            )
        self.second_embedding_shape = embedding_dimensions // first_embedding_shape

        # Convolution layer: 2 input channels (head, edge), 3x3 kernel
        self.conv_layer = nn.Conv2d(2, self.filter_count, kernel_size=(3, 3), padding=1)

        # After 3x3 conv with padding=1: output spatial size = (shape1, shape2)
        # Flattened: filter_count * shape1 * shape2
        hidden_size = self.filter_count * self.first_embedding_shape * self.second_embedding_shape
        self.projection = nn.Linear(hidden_size, self.embedding_dimensions)

        self.to(device)


    def score(self,
              *,
              head_embeddings: Tensor,
              tail_embeddings: Tensor,
              edge_embeddings: Tensor,
              **_) -> Tensor:
        """
        Compute the score function for the triplets given as argument.

        The ConvE scoring function is:

        .. math::

            \\psi_r(e_s, e_o) = f(\\text{vec}(f([\\bar{e}_s; \\bar{r}_r] \\ast \\omega)) \\mathbf{W}) \\cdot e_o

        where :math:`\\bar{e}_s` and :math:`\\bar{r}_r` are the 2D reshaped
        head and edge embeddings, :math:`\\ast` denotes 2D convolution,
        and :math:`\\mathbf{W}` is the projection matrix.

        See referenced paper for more details on the score:
        <https://arxiv.org/abs/1707.01476>

        Arguments
        ---------

        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only)*
        : The embeddings of the head nodes for the current batch of length `batch_size`.

        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only)*
        : The embeddings of the tail nodes for the current batch of length `batch_size`.

        **edge_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only)*
        : The embeddings of the edges for the current batch of length `batch_size`.

        Returns
        -------

        **batch_score** *(torch.Tensor, dtype: torch.float, shape: [batch_size])*
        : The score of each triplet as a tensor.

        Notes
        -----
        The batch can be the whole graph if it fits in memory.

        """
        batch_size = head_embeddings.shape[0]

        # Reshape to 2D: [batch_size, 1, shape1, shape2]
        head_2d = head_embeddings.view(batch_size, 1, self.first_embedding_shape, self.second_embedding_shape)
        edge_2d = edge_embeddings.view(batch_size, 1, self.first_embedding_shape, self.second_embedding_shape)

        # Stack into 2-channel input: [batch_size, 2, shape1, shape2]
        stacked = cat((head_2d, edge_2d), dim=1)

        # 2D convolution + ReLU
        conv_out = F.relu(self.conv_layer(stacked))

        # Flatten: [batch_size, filter_count * shape1 * shape2]
        conv_flat = conv_out.view(batch_size, -1)

        # Projection back to embedding dimension: [batch_size, embedding_dimensions]
        projected = F.relu(self.projection(conv_flat))

        # Dot product with tail embeddings: [batch_size]
        batch_score = (projected * tail_embeddings).sum(dim=1)

        return batch_score


    def _score_heads_given_tail(self,
                                head_embeddings: Tensor,
                                tail_embedding: Tensor,
                                edge_embedding: Tensor) -> Tensor:
        """
        Internal helper: score a batch of candidate heads against a single tail and edge.

        Arguments
        ---------

        **head_embeddings** *(torch.Tensor, shape: [candidate_count, embedding_dimensions])*
        : Embeddings of the candidate heads.

        **tail_embedding** *(torch.Tensor, shape: [embedding_dimensions])*
        : Embedding of the fixed tail.

        **edge_embedding** *(torch.Tensor, shape: [embedding_dimensions])*
        : Embedding of the fixed edge.

        Returns
        -------

        **scores** *(torch.Tensor, shape: [candidate_count])*
        : Score for each candidate head.

        """
        candidate_count = head_embeddings.shape[0]

        head_2d = head_embeddings.view(candidate_count, 1, self.first_embedding_shape, self.second_embedding_shape)
        edge_2d = edge_embedding.view(1, 1, self.first_embedding_shape, self.second_embedding_shape).expand(candidate_count, -1, -1, -1)

        stacked = cat((head_2d, edge_2d), dim=1)

        conv_out = F.relu(self.conv_layer(stacked))
        conv_flat = conv_out.view(candidate_count, -1)
        projected = F.relu(self.projection(conv_flat))

        scores = (projected * tail_embedding).sum(dim=1)
        return scores


    def inference_prepare_candidates(self,
                                    *,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Parameter,
                                    node_inference: bool = True
                                    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score` method.

        ConvE has an asymmetric scoring function: the head and edge go through
        the 2D convolution, and the result is dotted with the tail. This means:

        - For **tail prediction** (``node_inference=True``, predicting tails):
          the convolution input is (head, edge) per batch item, and candidates
          are all tail nodes. The candidates tensor has shape
          ``[batch_size, node_count, embedding_dimensions]``.

        - For **head prediction** (predicting heads):
          the convolution input is (candidate_head, edge) for each candidate,
          and the tail is fixed per batch item. The candidates tensor has shape
          ``[batch_size, node_count, embedding_dimensions]`` (candidate heads),
          and ``tail_embeddings`` holds the fixed tail per batch item.

        Arguments
        ---------

        **head_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the head nodes (from KG).

        **tail_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the tail nodes (from KG).

        **edge_indices** *(torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only)*
        : The indices of the edges (from KG).

        **node_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, node_embedding_dimensions], keyword-only)*
        : Embeddings of all nodes.

        **edge_embeddings** *(torch.nn.Parameter, dtype: torch.float, shape: [edge_count, edge_embedding_dimensions], keyword-only)*
        : Embeddings of all edges.

        **node_inference** *(bool, optional, default to True, keyword-only)*
        : If True, prepare candidate tails; otherwise, prepare candidate heads.

        Returns
        -------

        **head_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions])* :or *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_count, node_embedding_dimensions])* if predicting heads
        : Head node embeddings (fixed when predicting tails, candidates when predicting heads).

        **tail_embeddings** *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions])* :or *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_count, node_embedding_dimensions])* if predicting tails
        : Tail node embeddings (candidates when predicting tails, fixed when predicting heads).

        **edge_embeddings_inferred** *(torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions])* :or *(torch.Tensor, dtype: torch.float, shape: [batch_size, node_count, edge_embedding_dimensions])* if predicting heads
        : Edge embeddings (fixed when predicting tails, expanded per candidate when predicting heads).

        **candidates** *(torch.Tensor)*
        : Not used by ConvE; returned for interface compatibility.

        """
        batch_size = head_indices.shape[0]

        # Get head, tail and edge embeddings
        head_embeddings = node_embeddings[head_indices]
        tail_embeddings = node_embeddings[tail_indices]
        edge_embeddings_inferred = edge_embeddings[edge_indices]

        if node_inference:
            # Predicting tails: candidates are all nodes as tails
            # head_embeddings: [batch_size, dim] (fixed)
            # tail_embeddings: [batch_size, node_count, dim] (candidates)
            # edge_embeddings: [batch_size, dim] (fixed)
            candidates = node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            tail_embeddings = candidates
            # head_embeddings and edge_embeddings stay [batch_size, dim]
        else:
            # Predicting heads: candidates are all nodes as heads
            # head_embeddings: [batch_size, node_count, dim] (candidates)
            # tail_embeddings: [batch_size, dim] (fixed)
            # edge_embeddings: [batch_size, node_count, dim] (expanded per candidate)
            candidates = node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            head_embeddings = candidates
            edge_embeddings_inferred = edge_embeddings_inferred.unsqueeze(1).expand(-1, self.node_count, -1)

        return head_embeddings, tail_embeddings, edge_embeddings_inferred, candidates


    def inference_score(self,
                        *,
                        head_embeddings: Tensor,
                        tail_embeddings: Tensor,
                        edge_embeddings: Tensor
                        ) -> Tensor:
        """
        Link prediction evaluation helper function. Compute the scores
        of (head, candidate, edge) or (candidate, tail, edge) for any candidate.

        The arguments should match the ones of the output of `inference_prepare_candidates`.

        ConvE's asymmetric scoring means:

        - **Tail prediction**: ``head_embeddings`` is ``[batch_size, dim]``,
          ``edge_embeddings`` is ``[batch_size, dim]``, and
          ``tail_embeddings`` is ``[batch_size, node_count, dim]``.
          For each batch item, the conv processes (head, edge) once, and the
          projected result is dotted with all tail candidates.

        - **Head prediction**: ``head_embeddings`` is ``[batch_size, node_count, dim]``,
          ``edge_embeddings`` is ``[batch_size, node_count, dim]``, and
          ``tail_embeddings`` is ``[batch_size, dim]``.
          For each batch item and each candidate head, the conv processes
          (candidate_head, edge) and the projected result is dotted with the
          fixed tail.

        Arguments
        ---------

        **head_embeddings** *(torch.Tensor, dtype: torch.float, keyword-only)*
        : Embeddings of the head nodes.

        **tail_embeddings** *(torch.Tensor, dtype: torch.float, keyword-only)*
        : Embeddings of the tail nodes.

        **edge_embeddings** *(torch.Tensor, dtype: torch.float, keyword-only)*
        : Embeddings of the edges.

        Returns
        -------

        **score** *(torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count])* :Tensor of score values.
        : First dimension: incomplete triplets tested
        : Second dimension: candidate indices

        """
        batch_size = head_embeddings.shape[0]

        if len(tail_embeddings.shape) == 3:
            # --- Tail prediction ---
            # head: [batch_size, dim], edge: [batch_size, dim], tail: [batch_size, node_count, dim]
            assert (len(head_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 2), \
                "When inferring tails, `head_embeddings` and `edge_embeddings` must have 2 dimensions."

            node_count = tail_embeddings.shape[1]

            # Process (head, edge) through conv: [batch_size, dim]
            head_2d = head_embeddings.view(batch_size, 1, self.first_embedding_shape, self.second_embedding_shape)
            edge_2d = edge_embeddings.view(batch_size, 1, self.first_embedding_shape, self.second_embedding_shape)
            stacked = cat((head_2d, edge_2d), dim=1)  # [batch_size, 2, shape1, shape2]

            conv_out = F.relu(self.conv_layer(stacked))  # [batch_size, filter_count, shape1, shape2]
            conv_flat = conv_out.view(batch_size, -1)  # [batch_size, hidden]
            projected = F.relu(self.projection(conv_flat))  # [batch_size, dim]

            # Dot with all tail candidates: [batch_size, node_count]
            scores = torch.einsum('bd,bcd->bc', projected, tail_embeddings)  # [batch_size, node_count]

        elif len(head_embeddings.shape) == 3:
            # --- Head prediction ---
            # head: [batch_size, node_count, dim], edge: [batch_size, node_count, dim], tail: [batch_size, dim]
            assert (len(tail_embeddings.shape) == 2) and (len(edge_embeddings.shape) == 3), \
                "When inferring heads, `tail_embeddings` must have 2 dimensions and `edge_embeddings` must have 3 dimensions."

            node_count = head_embeddings.shape[1]

            # Process (candidate_head, edge) through conv for all candidates:
            head_2d = head_embeddings.view(batch_size, node_count, 1, self.first_embedding_shape, self.second_embedding_shape)
            edge_2d = edge_embeddings.view(batch_size, node_count, 1, self.first_embedding_shape, self.second_embedding_shape)
            stacked = cat((head_2d, edge_2d), dim=2)  # [batch_size, node_count, 2, shape1, shape2]

            # Conv2d operates on 4D input, so we flatten batch and node_count
            stacked_flat = stacked.view(batch_size * node_count, 2, self.first_embedding_shape, self.second_embedding_shape)

            conv_out = F.relu(self.conv_layer(stacked_flat))  # [batch*node_count, filter_count, shape1, shape2]
            conv_flat = conv_out.view(batch_size * node_count, -1)  # [batch*node_count, hidden]
            projected = F.relu(self.projection(conv_flat))  # [batch*node_count, dim]

            # Dot with fixed tail per batch item: [batch*node_count]
            tail_expanded = tail_embeddings.unsqueeze(1).expand(-1, node_count, -1).reshape(batch_size * node_count, -1)
            scores = (projected * tail_expanded).sum(dim=1)  # [batch*node_count]

            scores = scores.view(batch_size, node_count)

        else:
            raise ValueError(
                f"Cannot determine inference direction from tensor shapes: "
                f"head_embeddings={head_embeddings.shape}, tail_embeddings={tail_embeddings.shape}, "
                f"edge_embeddings={edge_embeddings.shape}."
            )

        return scores