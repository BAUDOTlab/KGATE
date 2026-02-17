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
    
    The interface inherits from the Module PyTorch class.

    Furthermore, this interface doesn't implement anything but is a type helper.
    However, functions from this class returning None can be used directly from inheriting classes.
    Exception for the `inference_score` function, fully implemented in this class.

    Attributes
    ----------
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings. Most translational vectors use either L1 or L2, but
        TorusE has a specific set of dissimilarity functions.
        See details from torchkge here: https://torchkge.readthedocs.io/en/latest/reference/utils.html#dissimilarities
    
    """
    def __init__(self):
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
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            The embeddings of the head nodes for the current batch.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            The embeddings of the tail nodes for the current batch.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            The embeddings of the edges for the current batch.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes for the current batch.
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes for the current batch.
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges for the current batch.
        
        Raises
        ------
        NotImplementedError
            The score method must be implemented by a translational decoder
            inheriting from this interface.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count]
            The score of each triplet as a tensor.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
        
        """
        raise NotImplementedError("The `score` method must be implemented by the translational decoder.")


    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding] | None:
        """
        Interface method for the decoder's parameters normalization function.

        Refer to the specific decoder for details on this function's implementation.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The edge embedding as a nn.Embedding containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The normalized edge embedding object.
        
        Notes
        -----
        The normalize_parameters method can be implemented by a translational decoder inheriting from this class
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
        The get_embeddings method can be implemented by a translational decoder inheriting from this class
        if it needed.
        If the decoder doesn't have dedicated embeddings, nothing is returned. In 
        this case, it is not necessary to implement this method from the interface.
        
        """
        return None


    def inference_prepare_candidates(self,
                                    *,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
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
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of all edges.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.
        
        Raises
        ------
        NotImplementedError
            The inference_prepare_candidates method must be implemented by a translational decoder
            inheriting from this interface.
            
        Returns
        -------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Head node embeddings.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Tail node embeddings.
        edge_embeddings_inferred: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            Edge embeddings.
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.
        
        """
        raise NotImplementedError("The `inference_prepare_candidates` method must be implemented by the translational decoder.")


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
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of the edges in the knowledge graph.
        
        Raises
        ------
        AssertionError #1
            When inferring heads, the head_embeddings tensor should be of shape [batch_size, embedding_dimensions].
        AssertionError #2
            When inferring tails, the head_embeddings tensor should be of shape [batch_size, embedding_dimensions].

        Returns
        -------
        score: torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count]
            Tensor of score values.
            First dimension: incomplete triplets tested
            Second dimension: candidate indices
            For example, if the function is called to infer the score of tails:
            First dimension: (head_indices, edge_indices)
            Second dimension: tail_indices
        
        """
        batch_size = head_embeddings.shape[0]

        # When the shape of the edges is [batch_size, node_embedding_dimensions]
        if len(edge_embeddings.shape) == 2:
            if len(tail_embeddings.shape) == 3:
                assert len(head_embeddings.shape) == 2, "When inferring tails, the `head_embeddings` tensor should be of shape [batch_size, node_embedding_dimensions]"

                translated_heads = (head_embeddings + edge_embeddings).view(batch_size, 1, edge_embeddings.size(1))
                return - self.dissimilarity(translated_heads, tail_embeddings)
            
            else:
                assert (len(head_embeddings.shape) == 3) and (len(tail_embeddings.shape) == 2), "When inferring heads, the `head_embeddings` tensor should be of shape [batch_size, node_embedding_dimensions]"

                edges_extended = edge_embeddings.view(batch_size, 1, edge_embeddings.size(1))
                tails_extended = tail_embeddings.view(batch_size, 1, edge_embeddings.size(1))
                
                return - self.dissimilarity(head_embeddings + edges_extended, tails_extended)
        
        elif len(edge_embeddings.shape) == 3:
            if hasattr(self, "evaluated_projections"):
                head_embeddings = head_embeddings.view(batch_size, -1, edge_embeddings.size(1))
                tail_embeddings = tail_embeddings.view(batch_size, -1, edge_embeddings.size(1))
            
            else:
                head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(1))
                tail_embeddings = tail_embeddings.view(batch_size, -1, tail_embeddings.size(1))

            return - self.dissimilarity(head_embeddings + edge_embeddings, tail_embeddings)
        # TODO: would a ValueError in an 'else' be needed?



class TransE(TranslationalDecoder):
    """
    Implementation of TransE model detailed in the paper referenced below.
    
    This class inherits from the TranslationDecoder interface. It inherits its attributes as well.

    References
    ----------
    Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
    `Translating Embeddings for Modeling Multi-relational Data.`
    https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    In Advances in Neural Information Processing Systems 26, pages 2787–2795, 2013.

    Arguments
    ---------
    dissimilarity_type: Literal["L1", "L2"], default to "L2"
        The type of dissimilarity function that will be used,
        either "L1" or "L2".
    
    Raises
    ------
    ValueError
        The dissimilarity_type must be "L1" or "L2".

    Attributes
    ----------
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings.
        See details from torchkge here: https://torchkge.readthedocs.io/en/latest/reference/utils.html#dissimilarities
    
    """
    def __init__(self, dissimilarity_type: Literal["L1", "L2"] = "L2"):
        super().__init__()
        match dissimilarity_type:
            case "L1":
                self.dissimilarity = l1_dissimilarity
            case "L2":
                self.dissimilarity = l2_dissimilarity
            case _:
                raise ValueError(f"TransE decoder can only use L1 or L2 dissimilarity, but got \"{dissimilarity_type}\"")


    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                **_) -> Tensor:
        """
        Compute the score function for the triplets given as argument.
        
        See referenced paper for more details on the score:
        https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            The edge embeddings.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count]
            The score of each triplet as a tensor.
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        
        batch_score = - self.dissimilarity( head_normalized_embeddings + edge_embeddings,
                                            tail_normalized_embeddings)
    
        return batch_score
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the TransE model.
        
        According to the original paper, the node embeddings should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The edge embeddings, which are not normalized as per the paper's recommendation.
        
        Returns
        -------
        node_embeddings : torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The normalized node embedding object.
        edge_embeddings : torch.nn.Embedding, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The untouched edge embedding object.
        
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)
            
        return node_embeddings, edge_embeddings


    def inference_prepare_candidates(self,
                                    *,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score_function` method.

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            Embeddings of all edges.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.
        
        Returns
        -------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            Head node embeddings.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            Tail node embeddings.
        edge_embeddings_inferred: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            Edge embeddings.
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        # Get head, tail and edge embeddings
        head_embeddings = node_embeddings[head_indices]
        tail_embeddings = node_embeddings[tail_indices]
        edge_embeddings_inferred = edge_embeddings(edge_indices)

        if node_inference:
            # Prepare candidates for every node
            candidates = node_embeddings
        else:
            # Prepare candidates for every edge
            candidates = edge_embeddings.weight.data
        
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)

        return head_embeddings, tail_embeddings, edge_embeddings_inferred, candidates
    
    
    
class TransH(TranslationalDecoder):
    """
    Implementation of TransH model detailed in the paper referenced below.
    
    This class inherits from the TranslationDecoder interface. It inherits its attributes as well.

    References
    ----------
    Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
    `Knowledge Graph Embedding by Translating on Hyperplanes.`
    https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531
    In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.

    Arguments
    ---------
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.
    embedding_dimensions: int
        Dimensions of node and edge embeddings.

    Attributes
    ----------
    normal_vector: torch.nn.Embedding, shape: [edge_count, embedding_dimensions]
        Normal vectors associated to each edge and used to compute the edge-specific hyperplanes nodes are projected on.
        See paper for more details: https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531
        Initialized with Xavier uniform distribution and then normalized.
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings.
        See details from torchkge here: https://torchkge.readthedocs.io/en/latest/reference/utils.html#dissimilarities
    evaluated_projections: bool
        Indicates whether `projected_nodes` has been computed.
        This should be set to True every time a backward pass is done in train mode.
    projected_nodes: torch.nn.Parameter, shape: [edge_count, node_count, embedding_dimensions]
        Contains the projection of each node in each edge-specific sub-space.
    
    """
    def __init__(self,
                embedding_dimensions: int,
                node_count: int,
                edge_count: int):
        super().__init__()
        self.normal_vector = initialize_embedding(edge_count, embedding_dimensions)
        self.dissimilarity = l2_dissimilarity

        self.evaluated_projections = False
        self.projected_nodes = Parameter(empty(size = (edge_count,
                                                    node_count,
                                                    embedding_dimensions)),
                                                    requires_grad = False)


    @staticmethod
    def project(nodes: Tensor,
                normal_vector: Tensor
                ) -> Tensor:
        """
        Project the given nodes onto the normal vector.
        
        Arguments
        ---------
        nodes: torch.Tensor
            The not projected node embeddings.
        normal_vector: torch.Tensor: [edge_count, embedding_dimensions]
            Normal vectors associated to each edge and used to compute the edge-specific hyperplanes nodes are projected on.
            See paper for more details: https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531
        
        Returns
        -------
        projected_nodes_tensor: torch.Tensor, shape: [edge_count, node_count, embedding_dimensions]
            Give the value of the `self.projected_nodes` nn.Parameter object when called.
        
        """
        projected_nodes_tensor = nodes - (nodes * normal_vector).sum(dim = 1).view(-1, 1) * normal_vector
        
        return projected_nodes_tensor


    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                edge_indices: Tensor,
                **_) -> Tensor:
        """
        Compute the score function for the triplets given as argument.
        
        See referenced paper for more details on the score:
        https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape [batch_size, embedding_dimensions], keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape [batch_size, embedding_dimensions], keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape [batch_size, embedding_dimensions], keyword-only
            Embeddings of edges.
        edge_indices: torch.Tensor, dtype: torch.long, shape [batch_size], keyword-only
            The indices of the edges (from KG).

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count]
            The score of each triplet as a tensor.
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        self.evaluated_projections = False
        normal_vector = normalize(self.normal_vector(edge_indices), p = 2, dim = 1)
        
        batch_score = - self.dissimilarity( self.project(head_normalized_embeddings, normal_vector) + edge_embeddings,
                                            self.project(tail_normalized_embeddings, normal_vector))
        
        return batch_score
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the TransH model.
        
        According to the original paper, the node embeddings, edge embeddings
        and the normal vector should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The edge embedding as a nn.Embedding containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            The normalized edge embedding object.
    
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)
        edge_embeddings.weight.data = normalize(edge_embeddings.weight.data, p = 2, dim = 1)
        self.normal_vector.weight.data = normalize(self.normal_vector.weight.data, p = 2, dim = 1)
        
        return node_embeddings, edge_embeddings


    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        Return the embeddings of nodes and edges along with edge normal vectors.

        Returns
        -------
        embeddings: Dict[str, torch.Tensor]
            Key: "normal_vector"
            Value: tensors representing nodes and edges in current model
        
        """
        return {"normal_vector": self.normal_vector.weight.data}
    
    
    def inference_prepare_candidates(self,
                                    *,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_scoring_function` method.        

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            Embeddings of all edges.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            Head node embeddings.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            Tail node embeddings.
        edge_embeddings_inferred: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions]
            Edge embeddings.
        candidates: torch.Tensor, dtype: float, shape: [batch_size, edge_count, embedding_dimensions]
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        edge_embeddings_inferred = edge_embeddings(edge_indices)

        if node_inference:
            head_embeddings = self.projected_nodes[edge_indices, head_indices]  # shape: [batch_size, self.embedding_dimensions]
            tail_embeddings = self.projected_nodes[edge_indices, tail_indices]  # shape: [batch_size, self.embedding_dimensions]
            candidates = self.projected_nodes[edge_indices]  # shape: [batch_size, self.edge_count, self.embedding_dimensions]
        else:
            head_embeddings = self.projected_nodes[:, head_indices].transpose(0, 1)  # shape: [batch_size, self.edge_count, self.embedding_dimensions]
            tail_embeddings = self.projected_nodes[:, tail_indices].transpose(0, 1)  # shape: [batch_size, self.edge_count, self.embedding_dimensions]
            candidates = edge_embeddings.weight.data.unsqueeze(0).expand(batch_size, self.edge_count, self.embedding_dimensions)  # shape: [batch_size, self.edge_count, self.embedding_dimensions]

        return head_embeddings, tail_embeddings, edge_embeddings_inferred, candidates


    def evaluate_projections(self,
                            node_embeddings: Tensor):
        """
        Link prediction evaluation helper function. Project all nodes
        according to each edge. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, embedding_dimensions], keyword-only
            Embeddings of all nodes.
        
        Notes
        -----
        Assure that `self.evaluated_projections` is True.
        First check if it already is to avoid unnecessary calculations.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.node_count), unit = "nodes", desc = "Projecting nodes"):

            normal_vector = self.normal_vector.weight.data.view(self.edge_count, self.embedding_dimensions)
            mask = tensor([i], device = normal_vector.device).long()

            if normal_vector.is_cuda:
                empty_cache()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]

            normalized_components = (masked_node_embeddings.view(1, -1) * normal_vector).sum(dim = 1)
            self.projected_nodes[:, i, :] = (masked_node_embeddings.view(1, -1)
                                            - normalized_components.view(-1, 1)
                                            * normal_vector)

            del normalized_components

        self.evaluated_projections = True



class TransR(TranslationalDecoder):
    """
    Implementation of TransR model detailed in the paper referenced below.
    
    This class inherits from the TranslationDecoder interface. It inherits its attributes as well.

    References
    ----------
    Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu.
    `Learning Entity and Relation Embeddings for Knowledge Graph Completion.`
    https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523
    In Twenty-Ninth AAAI Conference on Artificial Intelligence, February 2015

    Arguments
    ---------
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.
    node_embedding_dimensions: int
        Dimensions of node embeddings.
    edge_embedding_dimensions: int
        Dimensions of edge embeddings.

    Attributes
    ----------
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.
    node_embedding_dimensions: int
        Dimensions of node embeddings.
    edge_embedding_dimensions: int
        Dimensions of edge embeddings.
    projection_matrix: torch.nn.Embedding, shape: [edge_count, edge_embedding_dimensions * node_embedding_dimensions]
        Edge-specific projection matrices. See paper for more details.
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings.
        See details from torchkge here: https://torchkge.readthedocs.io/en/latest/reference/utils.html#dissimilarities
    evaluated_projections: bool
        Indicates whether `projected_nodes` has been computed.
        This should be set to True every time a backward pass is done in train mode.
    projected_nodes: torch.nn.Parameter, shape: [edge_count, node_count, edge_embedding_dimensions]
        Contains the projection of each node in each edge-specific sub-space.
    
    """
    def __init__(self,
                node_count: int,
                edge_count: int,
                node_embedding_dimensions: int,
                edge_embedding_dimensions: int):
        super().__init__()

        self.node_count = node_count
        self.edge_count = edge_count
        self.node_embedding_dimensions = node_embedding_dimensions
        self.edge_embedding_dimensions = edge_embedding_dimensions

        self.projection_matrix = initialize_embedding(node_count, edge_embedding_dimensions * node_embedding_dimensions)

        self.dissimilarity = l2_dissimilarity

        self.evaluated_projections = False
        self.projected_nodes = Parameter(empty(size = ( edge_count,
                                                        node_count,
                                                        node_embedding_dimensions)),
                                                        requires_grad = False)


    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                edge_indices: Tensor,
                **_) -> Tensor:
        """
        Compute the score function for the triplets given as argument.
        
        See referenced paper for more details on the score:
        https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            The edge embeddings, of shape [edge_count, edge_embedding_dimensions]
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score of each triplet as a tensor.
            
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        self.evaluated_projections = False
        batch_size = head_normalized_embeddings.shape[0]

        projection_matrix = self.proj_mat(edge_indices).view(batch_size,
                                                            self.edge_embedding_dimensions,
                                                            self.node_embedding_dimensions)
        
        batch_score = - self.dissimilarity( self.project(head_normalized_embeddings, projection_matrix) + edge_embeddings,
                                            self.project(tail_normalized_embeddings, projection_matrix))
        
        return batch_score
    
    
    def project(self,
                nodes: Tensor,
                projection_matrix: Tensor
                ) -> Tensor:
        """
        Project the given nodes onto the projection matrix.
        
        Arguments
        ---------
        nodes: torch.Tensor
            TODO.what_that_variable_is_or_does
        projection_matrix: torch.Tensor
            TODO.what_that_variable_is_or_does
        
        Returns
        -------
        projected_nodes_tensor: torch.Tensor, shape: [edge_count, node_count, edge_embedding_dimensions]
            Give the value of the `self.projected_nodes` nn.Parameter object when called.
        
        """
        projected_nodes_tensor = matmul(projection_matrix, nodes.view(-1, self.node_embedding_dimensions, 1))
        
        return projected_nodes_tensor.view(-1, self.edge_embedding_dimensions)
    
    
    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding]:
        """
        Normalize parameters for the TransR model.
        
        According to the original paper, the node embeddings and edge embeddings
        should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The edge embedding as a ParameterList containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList, shape: [batch_size, node_embedding_dimensions]
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding, shape: [batch_size, edge_embedding_dimensions]
            The normalized edge embedding object.
        
        """
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)

        edge_embeddings.weight.data = normalize(edge_embeddings.weight.data, p = 2, dim = 1)
        
        return node_embeddings, edge_embeddings
    
    
    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        Return the embeddings of nodes and edges along with edge normal vectors.

        Returns
        -------
        embeddings: Dict[str, torch.Tensor]
            Key: "projection_matrix"
            Value: tensors representing nodes and edges in current model
            
        """
        return {"projection_matrix": self.projection_matrix.weight.data.view(-1,
                                                        self.edge_embedding_dimensions,
                                                        self.node_embedding_dimensions)}
    
    
    def inference_prepare_candidates(self,
                                    *,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score_function` method.

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of all edges.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Head node embeddings.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Tail node embeddings.
        edge_embeddings_inferred: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            Edge embeddings.
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        # TODO: check that, below, all 'edge_embedding_dimensions' should not be 'node_embedding_dimensions'
        edge_embeddings_inferred = edge_embeddings(edge_indices)
        if node_inference:
            head_embeddings = self.projected_nodes[edge_indices, head_indices]  # shape: [batch_size, self.edge_embedding_dimensions]
            tail_embeddings = self.projected_nodes[edge_indices, tail_indices]  # shape: [batch_size, self.edge_embedding_dimensions]
            candidates = self.projected_nodes[edge_indices]  # shape: [batch_size, self.edge_count, self.edge_embedding_dimensions]
        else:
            head_embeddings = self.projected_nodes[:, head_indices].transpose(0, 1)  # shape: [batch_size, self.edge_count, self.edge_embedding_dimensions]
            tail_embeddings = self.projected_nodes[:, tail_indices].transpose(0, 1)  # shape: [batch_size, self.edge_count, self.edge_embedding_dimensions]
            candidates = edge_embeddings.weight.data.unsqueeze(0).expand(batch_size, edge_embeddings.num_embeddings, edge_embeddings.embedding_dim)  # shape: [batch_size, self.edge_count, self.edge_embedding_dimensions]

        return head_embeddings, tail_embeddings, edge_embeddings_inferred, candidates
    
    
    def evaluate_projections(self,
                            node_embeddings: Tensor):
        """
        Link prediction evaluation helper function. Project all nodes
        according to each edge. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.

        Notes
        -----
        Assure that `self.evaluated_projections` is True.
        First check if it already is to avoid unnecessary calculations.
            
        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.node_count), unit = "nodes", desc = "Projecting nodes"):
            projection_matrices = self.projection_matrix.weight.data
            projection_matrices = projection_matrices.view(self.edge_count, self.edge_embedding_dimensions, self.node_embedding_dimension)

            mask = tensor([i], device = projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]
            
            projected_masked_node_embeddings = matmul(projection_matrices, masked_node_embeddings.view(self.node_embedding_dimension))
            projected_masked_node_embeddings = projected_masked_node_embeddings.view(self.edge_count, self.edge_embedding_dimensions, 1)
            self.projected_nodes[:, i, :] = projected_masked_node_embeddings.view(self.edge_count, self.edge_embedding_dimensions)
            # projected_nodes is an object equivalent to projected_masked_node_embeddings

            del projected_masked_node_embeddings

        self.evaluated_projections = True



class TransD(TranslationalDecoder):
    """
    Implementation of TransD model detailed in the paper referenced below.
    
    This class inherits from the TranslationDecoder interface. It inherits its attributes as well.

    References
    ----------
    Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao.
    `Knowledge Graph Embedding via Dynamic Mapping Matrix.`
    https://aclweb.org/anthology/papers/P/P15/P15-1067/
    In Proceedings of the 53rd Annual Meeting of the Association
    for Computational Linguistics and the 7th International Joint Conference
    on Natural Language Processing (Volume 1: Long Papers) pages 687–696,
    Beijing, China, July 2015. Association for Computational Linguistics.

    Arguments
    ---------
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.
    node_embedding_dimensions: int
        Dimensions of node embeddings.
    edge_embedding_dimensions: int
        Dimensions of edge embeddings.

    Attributes
    ----------
    node_count: int
        Number of nodes in the knowledge graph.
    edge_count: int
        Number of edges in the knowledge graph.
    node_embedding_dimensions: int
        Dimensions of node embeddings.
    edge_embedding_dimensions: int
        Dimensions of edge embeddings.
    node_projection_vector: torch.nn.Embedding, shape: [node_count, node_embedding_dimensions]
        Node-specific vector used to build projection matrices. See paper for more details.
        Initialized with Xavier uniform distribution and then normalized.
    edge_projection_vector: torch.nn.Embedding, shape: [edge_count, edge_embedding_dimensions]
        Edge-specific vector used to build projection matrices. See paper for more details.
        Initialized with Xavier uniform distribution and then normalized.
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings.
        See details from torchkge here: https://torchkge.readthedocs.io/en/latest/reference/utils.html#dissimilarities
    evaluated_projections: bool
        Indicates whether projected_nodes has been computed.
        This should be set to True every time a backward pass is done in train mode.
    projected_nodes: torch.nn.Parameter, shape: [edge_count, node_count, edge_embedding_dimensions]
        Contains the projection of each node in each edge-specific sub-space.
    
    """
    def __init__(self,
                node_count: int,
                edge_count: int,
                node_embedding_dimensions: int,
                edge_embedding_dimensions: int):
        super().__init__()

        self.node_count = node_count
        self.edge_count = edge_count
        self.node_embedding_dimensions = node_embedding_dimensions
        self.edge_embedding_dimensions = edge_embedding_dimensions

        # TODO: Might be changed to have 2 embedding spaces instead (meaning it will be encoded by a GNN if present)
        self.node_projection_vector = initialize_embedding(self.node_count, self.node_embedding_dimensions)
        self.edge_projection_vector = initialize_embedding(self.edge_count, self.edge_embedding_dimensions)

        self.dissimilarity = l2_dissimilarity

        self.evaluated_projections = False
        self.projected_nodes = Parameter(empty(size = ( edge_count,
                                                        node_count,
                                                        node_embedding_dimensions)),
                                                        requires_grad = False)

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
        Compute the score function for the triplets given as argument.
        
        See referenced paper for more details on the score:
        https://aclweb.org/anthology/papers/P/P15/P15-1067/

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            The edge embeddings, of shape [edge_count, edge_embedding_dimensions]
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score of each triplet as a tensor.
        
        """
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        edge_normalized_embeddings = normalize(edge_embeddings, p = 2, dim = 1)

        head_projected_vectors = normalize(self.node_projection_vector(head_indices), p = 2, dim = 1)
        tail_projected_vectors = normalize(self.node_projection_vector(tail_indices), p = 2, dim = 1)
        edge_projected_vectors = normalize(self.edge_projection_vector(edge_indices), p = 2, dim = 1)

        projected_heads = self.project(head_normalized_embeddings, head_projected_vectors, edge_projected_vectors)
        projected_tails = self.project(tail_normalized_embeddings, tail_projected_vectors, edge_projected_vectors)
        
        batch_score = - self.dissimilarity( projected_heads + edge_normalized_embeddings,
                                            projected_tails)
        
        return batch_score
    
    
    def project(self,
                nodes: Tensor,
                node_projection_vector: Tensor,
                edge_projection_vector: Tensor
                ) -> Tensor:
        """
        Project the given nodes onto the projection vector.
        
        Arguments
        ---------
        nodes: torch.Tensor
            TODO.what_that_variable_is_or_does
        node_projection_vector: torch.Tensor
            TODO.what_that_variable_is_or_does
        edge_projection_vector: torch.Tensor
            TODO.what_that_variable_is_or_does
        
        Returns
        -------
        projected_nodes_tensor: torch.Tensor, shape: [edge_count, node_count, edge_embedding_dimensions]
            Give the value of the `self.projected_nodes` nn.Parameter object when called.
        
        """
        batch_size = nodes.shape[0]

        scalar_product = (nodes * node_projection_vector).sum(dim = 1)
        projected_nodes_tensor = (edge_projection_vector * scalar_product.view(batch_size, 1))

        return projected_nodes_tensor + nodes[:, :self.edge_embedding_dimensions]
    
    
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
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The edge embedding as a ParameterList containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
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
        Return the embeddings of nodes and edges along with edge normal vectors.
        
        Returns
        -------
        embeddings: Dict[str, torch.Tensor]
            Key: "node_projection_vector", "edge_projection_vector"
            Value: tensors representing respectively nodes and edges in current model
            
        """
        return {"node_projection_vector": self.node_projection_vector.weight.data,
                "edge_projection_vector": self.edge_projection_vector.weight.data}
    
    
    def inference_prepare_candidates(self,
                                    *,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score_function` method.

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of all edges.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Head node embeddings.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Tail node embeddings.
        edge_embeddings_inferred: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            Edge embeddings.
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.

        """
        batch_size = head_indices.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        edge_embeddings_inferred = edge_embeddings(edge_indices)

        if node_inference:
            head_embeddings = self.projected_nodes[edge_indices, head_indices]  # shape: [batch_size, self.node_embedding_dimensions]
            tail_embeddings = self.projected_nodes[edge_indices, tail_indices]  # shape: [batch_size, self.node_embedding_dimensions]
            candidates = self.projected_nodes[edge_indices]  # shape: [batch_size, self.edge_count, self.node_embedding_dimensions]
        else:
            head_embeddings = self.projected_nodes[:, head_indices].transpose(0, 1)  # shape: [batch_size, self.edge_count, self.edge_embedding_dimensions]
            tail_embeddings = self.projected_nodes[:, tail_indices].transpose(0, 1)  # shape: [batch_size, self.edge_count, self.edge_embedding_dimensions]
            candidates = edge_embeddings.weight.data.unsqueeze(0).expand(batch_size, self.edge_count, self.edge_embedding_dimensions)  # shape: [batch_size, self.edge_count, self.node_embedding_dimensions]

        return head_embeddings, tail_embeddings, edge_embeddings_inferred, candidates


    def evaluate_projections(self,
                            node_embeddings: Tensor):
        """
        Link prediction evaluation helper function. Project all nodes
        according to each edge. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.

        Notes
        -----
        Assure that `self.evaluated_projections` is True.
        First check if it already is to avoid unnecessary calculations.
            
        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.node_count), unit = "nodes", desc = "Projecting nodes"):
            edge_projection_vector = self.edge_projection_vector.weight.data

            mask = tensor([i], device = edge_projection_vector.device).long()

            # TODO: find better name
            masked_node_embeddings = node_embeddings[mask]

            node_projection_vector = self.node_projection_vector.weight[i]

            scalar_product = (node_projection_vector * masked_node_embeddings).sum(dim = 0)
            projected_nodes = scalar_product * edge_projection_vector + masked_node_embeddings[:self.edge_embedding_dimensions].view(1, -1)

            self.projected_nodes[:, i, :] = projected_nodes

            del projected_nodes

        self.evaluated_projections = True



class TorusE(TranslationalDecoder):
    """
    Implementation of TorusE model detailed in the paper referenced below.
    
    This class inherits from the TranslationDecoder interface. It inherits its attributes as well.

    References
    ----------
    Takuma Ebisu and Ryutaro Ichise
    `TorusE: Knowledge Graph Embedding on a Lie Group.`
    https://arxiv.org/abs/1711.05435
    In Proceedings of the 32nd AAAI Conference on Artificial Intelligence
    (New Orleans, LA, USA, Feb. 2018),AAAI Press, pp. 1819–1826.

    Arguments
    ---------
    dissimilarity_type: Literal["L1", "torus_L1", "torus_L2", "torus_eL2"]
        The type of dissimilarity function that will be used,
        either "L1", "torus_L1", "torus_L2" or "torus_eL2".
    
    Raises
    ------
    ValueError
        The dissimilarity_type must be "L1", "torus_L1", "torus_L2" or "torus_eL2".

    Attributes
    ----------
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings.
        See details from torchkge here: https://torchkge.readthedocs.io/en/latest/reference/utils.html#dissimilarities
    normalized: bool
        True if parameters are normalized.
    
    """
    def __init__(self,
                dissimilarity_type: Literal["L1", "torus_L1", "torus_L2", "torus_eL2"]):
        super().__init__()

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
                raise ValueError(f"TorusE decoder can only use `L1`, `torus_L1`, `torus_L2` or `torus_eL2` dissimilarity, but got \"{dissimilarity_type}\"")

        self.normalized = False
    
    
    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor,
                **_) -> Tensor:
        """
        Compute the score function for the triplets given as argument.
        
        See referenced paper for more details on the score:
        https://arxiv.org/abs/1711.05435

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of edges.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size]
            The score of each triplet as a tensor.
        
        """
        self.normalized = False

        fractionned_head_embeddings = head_embeddings.frac()
        fractionned_tail_embeddings = tail_embeddings.frac()
        fractionned_edge_embeddings = edge_embeddings.frac()

        batch_score = - self.dissimilarity( fractionned_head_embeddings + fractionned_edge_embeddings,
                                            fractionned_tail_embeddings)
        
        return batch_score


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
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The edge embedding as a ParameterList containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The normalized edge embedding object.
        
        """
        for embedding in node_embeddings:
            embedding.data.frac_() # Inplace fraction

        edge_embeddings.weight.data.frac_()
        self.normalized = True

        return node_embeddings, edge_embeddings
    
    
    def inference_prepare_candidates(self,
                                    *,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score_function` method.

        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of all edges.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.

        Returns
        -------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Head node embeddings.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Tail node embeddings.
        edge_embeddings_inferred: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            Edge embeddings.
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
        edge_embeddings_inferred = edge_embeddings(edge_indices)

        if node_inference:
            # Prepare candidates for every node
            candidates = node_embeddings
        else:
            # Prepare candidates for every edge
            candidates = edge_embeddings.weight.data
            
        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)
        
        return head_embeddings, tail_embeddings, edge_embeddings_inferred, candidates



class SpherE(TranslationalDecoder):
    """
    Implementation of SpherE model detailed in the paper referenced below.
    
    This class inherits from the TranslationDecoder interface. It inherites its attributes as well.

    References
    ----------
    Zihao Li, Yuyi Ao, Jingrui He.
    `SpherE: Expressive and Interpretable Knowledge Graph Embedding for Set Retrieval.`
    https://arxiv.org/pdf/2404.19130
    TODO.where.

    Attributes
    ----------
    dissimilarity: function described in `torchkge.utils.dissimilarities`
        The dissimilarity function used to compare translated head embeddings 
        to tail embeddings. Most translational vectors use either L1 or L2, but
        TorusE has a specific set of dissimilarity functions.
        See details from torchkge here: https://torchkge.readthedocs.io/en/latest/reference/utils.html#dissimilarities
    
    """
    def __init__(self):
        super().__init__()
        self.normalized = False
        
        
    def score(  self,
                *,
                head_embeddings: Tensor,
                tail_embeddings: Tensor,
                edge_embeddings: Tensor
                ) -> Tensor:
        """
        Compute the score function for the triplets given as argument.
        
        Score function:
            - || normalize(head_embeddings) + edge_embeddings - normalize(tail_embeddings) ||^2
        
        See referenced paper for more details on the score:
        https://arxiv.org/pdf/2404.19130

        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            The embeddings of the head nodes for the current batch.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            The embeddings of the tail nodes for the current batch.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            The embeddings of the edges for the current batch.

        Returns
        -------
        batch_score: torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count]
            The score of each triplet as a tensor.
        
        Notes
        -----
        The batch can be the whole graph if it fits in memory.
        
        """
        # TODO: check that it matches the original code
        head_normalized_embeddings = normalize(head_embeddings, p = 2, dim = 1)
        tail_normalized_embeddings = normalize(tail_embeddings, p = 2, dim = 1)
        
        translated_head = head_normalized_embeddings + edge_embeddings
        
        batch_score = - ((translated_head - tail_normalized_embeddings) ** 2).sum(dim = 1)
    
        return batch_score


    def normalize_parameters(self,
                            node_embeddings: nn.ParameterList,
                            edge_embeddings: nn.Embedding
                            ) -> Tuple[nn.ParameterList, nn.Embedding] | None:
        """
        Normalize parameters for the TorusE model.
        
        According to the original paper, the node embeddings should be normalized.
        
        Arguments
        ---------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The node embedding as a ParameterList containing one Parameter by node type,
            or only one if there is no node type.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The edge embedding as a ParameterList containing one Parameter by edge type,
            or only one if there is no node type.
        
        Returns
        -------
        node_embeddings: torch.nn.ParameterList, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            The normalized node embedding object.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            The untouched edges embedding object.
        
        """
        # TODO: check that it matches the original code
        for embedding in node_embeddings:
            embedding.data = normalize(embedding.data, p = 2, dim = 1)
        
        self.normalized = True
        
        return node_embeddings, edge_embeddings


    def inference_prepare_candidates(self,
                                    *,
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    head_indices: Tensor,
                                    tail_indices: Tensor,
                                    edge_indices: Tensor,
                                    node_inference: bool = True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get node embeddings
        and edge embeddings. The output will be fed to the
        `inference_score_function` method.
        
        Arguments
        ---------
        node_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of all nodes.
        edge_embeddings: torch.nn.Embedding, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of all edges.
        head_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor, dtype: torch.long, shape: [batch_size], keyword-only
            The indices of the edges (from KG).
        node_inference: bool, optional, default to True, keyword-only
            If True, prepare candidate nodes; otherwise, prepare candidate edges.
            
        Returns
        -------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Head node embeddings.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions]
            Tail node embeddings.
        edge_embeddings_inferred: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions]
            Edge embeddings.
        candidates: torch.Tensor
            Candidate embeddings for nodes or edges.
        
        """
        # TODO: check that it matches the original code
        batch_size = head_indices.shape[0]

        head_embeddings = node_embeddings[head_indices]
        tail_embeddings = node_embeddings[tail_indices]
        edge_embeddings_inferred = edge_embeddings(edge_indices)

        if node_inference:
            candidates = node_embeddings
        else:
            candidates = edge_embeddings.weight.data

        candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1)

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
        The arguments should match the ones of `inference_prepare_candidates`.
        
        Arguments
        ---------
        head_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the head nodes in the knowledge graph.
        tail_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, node_embedding_dimensions], keyword-only
            Embeddings of the tail nodes in the knowledge graph.
        edge_embeddings: torch.Tensor, dtype: torch.float, shape: [batch_size, edge_embedding_dimensions], keyword-only
            Embeddings of the edges in the knowledge graph.
        
        Raises
        ------
        AssertionError #1
            When inferring heads, the head_embeddings tensor should be of shape [batch_size, embedding_dimensions].
        AssertionError #2
            When inferring tails, the head_embeddings tensor should be of shape [batch_size, embedding_dimensions].
        ValueError
            Embeddings do not have shapes adapted for inference.
        NotImplementedError
            Edge inference is not implemented for SpherE.

        Returns
        -------
        score: torch.Tensor, dtype: torch.float, shape: [batch_size, candidate_count]
            Tensor of score values.
            First dimension: incomplete triplets tested
            Second dimension: candidate indices
            For example, if the function is called to infer the score of tails:
            First dimension: (head_indices, edge_indices)
            Second dimension: tail_indices
        
        """
        # TODO: check that it matches the original code

        # When the shape of the edges is [batch_size, node_embedding_dimensions]
        if len(edge_embeddings.shape) == 2:
            if len(tail_embeddings.shape) == 3:
                assert len(head_embeddings.shape) == 2, "When inferring tails, the `head_embeddings` tensor should be of shape [batch_size, node_embedding_dimensions]"

                normalized_head = normalize(head_embeddings, p = 2, dim = 2)
                normalized_tail = normalize(tail_embeddings, p = 2, dim = 1).unsqueeze(1)
                
                translated_heads = normalized_head + edge_embeddings.unsqueeze(1)
                
                return - ((translated_heads - normalized_tail) ** 2).sum(dim = 2)
            
            else:
                assert (len(head_embeddings.shape) == 3) and (len(tail_embeddings.shape) == 2), "When inferring heads, the `head_embeddings` tensor should be of shape [batch_size, node_embedding_dimensions]"

                normalized_head = normalize(head_embeddings, p = 2, dim = 1).unsqueeze(1)
                normalized_tail = normalize(tail_embeddings, p = 2, dim = 2)

                translated_heads = normalized_head + edge_embeddings.unsqueeze(1)

                return - ((translated_heads - normalized_tail) ** 2).sum(dim = 2)
        
        elif len(edge_embeddings.shape) == 3:
            raise NotImplementedError("Edge inference is not implemented for SpherE.")
        
        else:
            raise ValueError("Embeddings do not have shapes adapted for inference.")
