from torchkge.models import ConvKBModel
from torch import Tensor, cat
import torch.nn as nn

# Code adapted from torchKGE's implementation
# 
class ConvKB(ConvKBModel):
    def __init__(self, emb_dim:int, n_filters: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_filters, n_entities, n_relations)
        del self.ent_emb
        del self.rel_emb
        
    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, **_):
        b_size = h_emb.size(0)

        h = h_emb.view(b_size, 1, -1)
        r = r_emb.view(b_size, 1, -1)
        t = t_emb.view(b_size, 1, -1)

        concat = cat((h,r,t), dim=1)

        return self.output(self.convlayer(concat).reshape(b_size, -1))[:, 1]
    
    def get_embeddings(self):
        return None
    
    def inference_prepare_candidates(self, 
                                     h_idx: Tensor,
                                     t_idx: Tensor, 
                                     r_idx: Tensor, 
                                     node_embeddings: Tensor,
                                     relation_embeddings: nn.Embedding,
                                     entities: bool=True):

        b_size = h_idx.shape[0]

        # Get head, tail and relation embeddings
        h = node_embeddings[h_idx]
        t = node_embeddings[t_idx]
        r = relation_embeddings(r_idx)

        if entities:
            # Prepare candidates for every entities
            candidates = node_embeddings
        else:
            # Prepare candidates for every relations
            candidates = relation_embeddings.weight.data
        
        candidates = candidates.unsqueeze(0).expand(b_size, -1, -1)
        candidates = candidates.view(b_size, -1, 1, self.emb_dim)

        return h, t, r, candidates