"""
Convolutional decoder classes for training and inference.

Original code for the samplers from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from torch import Tensor, cat
import torch.nn as nn

from torchkge.models import ConvKBModel



class ConvKB(ConvKBModel):
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
                embedding_dimensions:int,
                filter_count: int,
                node_count: int,
                edge_count: int):
        
        super().__init__(embedding_dimensions, filter_count, node_count, edge_count)
        del self.ent_emb
        del self.rel_emb

        
    def score(self, *,
            head_embeddings: Tensor,
            tail_embeddings: Tensor,
            edge_embeddings: Tensor,
            **_):
        """
        TODO.What_the_function_does_about_globally

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
            
        Notes
        -----
        Additional_info_from_the_dev_to_the_user
            
        """
        batch_size = head_embeddings.size(0)

        head_score = head_embeddings.view(batch_size, 1, -1)
        tail_score = tail_embeddings.view(batch_size, 1, -1)
        edge_score = edge_embeddings.view(batch_size, 1, -1)

        concat = cat((head_score, edge_score, tail_score), dim=1)

        return self.output(self.convlayer(concat).reshape(batch_size, -1))[:, 1]
    
    
    def get_embeddings(self):
        """
        TODO.What_the_function_does_about_globally

        Returns
        -------
        None
        
        """
        return None
    
    
    def inference_prepare_candidates(self, 
                                    head_indices: Tensor,
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool = True):
        """
        TODO.What_the_class_is_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        head_indices: torch.Tensor
            The indices of the head nodes (from KG).
        tail_indices: torch.Tensor
            The indices of the tail nodes (from KG).
        edge_indices: torch.Tensor
            The indices of the edges (from KG).
        node_embeddings: torch.Tensor
            TODO.What_that_argument_is_or_does
        edge_embeddings: torch.nn.Embedding
            TODO.What_that_argument_is_or_does
        node_inference: bool, default to True
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
        candidates = candidates.view(batch_size, -1, 1, self.emb_dim)

        return head_embeddings, tail_embeddings, edge_embeddings_inference, candidates