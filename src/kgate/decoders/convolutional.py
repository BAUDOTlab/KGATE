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

# Code adapted from torchKGE's implementation
# 
class ConvKB(ConvKBModel):
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
            edge_embeddings: Tensor,
            tail_embeddings: Tensor,
            **_):
        batch_size = head_embeddings.size(0)

        head_score = head_embeddings.view(batch_size, 1, -1)
        edge_score = edge_embeddings.view(batch_size, 1, -1)
        tail_score = tail_embeddings.view(batch_size, 1, -1)

        concat = cat((head_score,edge_score,tail_score), dim=1)

        return self.output(self.convlayer(concat).reshape(batch_size, -1))[:, 1]
    
    def get_embeddings(self):
        return None
    
    def inference_prepare_candidates(self, 
                                    head_indices: Tensor,
                                    tail_indices: Tensor, 
                                    edge_indices: Tensor, 
                                    node_embeddings: Tensor,
                                    edge_embeddings: nn.Embedding,
                                    node_inference: bool=True):

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