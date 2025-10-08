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

from torchkge.models import DistMultModel, RESCALModel, AnalogyModel, ComplExModel

from ..utils import init_embedding

class RESCAL(RESCALModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_entities, n_relations)
        del self.ent_emb
        self.rel_mat = init_embedding(self.n_rel, self.emb_dim * self.emb_dim)

    def score(self, *, h_emb: Tensor, t_emb: Tensor, r_idx: Tensor, **_) -> Tensor:
        h_norm = normalize(h_emb, p=2, dim=1)
        t_norm = normalize(t_emb, p=2, dim=1)
        r = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)
        hr = matmul(h_norm.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t_norm).sum(dim=1)
    
    def get_embeddings(self) -> Dict[str,Tensor]:
        return {"rel_mat" : self.rel_mat.weight.data.view(-1, self.emb_dim, self.emb_dim)}
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: Tensor, 
                                    relation_embeddings: nn.Embedding,
                                    entities: bool =True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        # Get head, tail and relation embeddings
        h = node_embeddings[h_idx]
        t = node_embeddings[t_idx]
        r_mat = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)

        if entities:
            # Prepare candidates for every entities
            candidates = node_embeddings.unsqueeze(0).expand(b_size, -1, -1)
        else:
            # Prepare candidates for every relations
            candidates = self.rel_mat.weight.data.unsqueeze(0).expand(b_size, -1, -1, -1)

        return h, t, r_mat, candidates

    
class DistMult(DistMultModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_entities, n_relations)
        del self.ent_emb
        del self.rel_emb
    
    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, r_idx: Tensor, **_) -> Tensor:
        h_norm = normalize(h_emb, p=2, dim=1)
        t_norm = normalize(t_emb, p=2, dim=1)
        return (h_norm * r_emb * t_norm).sum(dim=1)
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: Tensor, 
                                    relation_embeddings: nn.Embedding,
                                    entities: bool =True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method.

        Parameters
        ----------
        h_idx : torch.Tensor
            The indices of the head entities (from KG).
        t_idx : torch.Tensor
            The indices of the tail entities (from KG).
        r_idx : torch.Tensor
            The indices of the relations (from KG).
        entities : bool, optional
            If True, prepare candidate entities; otherwise, prepare candidate relations.

        Returns
        -------
        h: torch.Tensor
            Head entity embeddings.
        t: torch.Tensor
            Tail entity embeddings.
        r: torch.Tensor
            Relation embeddings.
        candidates: torch.Tensor
            Candidate embeddings for entities or relations.
        """
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
        
        return h, t, r, candidates

class ComplEx(ComplExModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_entities, n_relations)
        self.embedding_spaces = 2
        del self.re_ent_emb
        del self.re_rel_emb
        del self.im_ent_emb
        del self.im_rel_emb

    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, **_):
        re_h, im_h = tensor_split(h_emb, 2, dim=1)
        re_r, im_r = tensor_split(r_emb, 2, dim=1)
        re_t, im_t = tensor_split(t_emb, 2, dim=1)
        
        return (re_h * (re_r * re_t + im_r * im_t) + 
                im_h * (re_r * im_t - im_r * re_t)).sum(dim=1)
    
    def get_embeddings(self) -> Dict[str, Tensor]:
        return None
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: Tensor, 
                                    relation_embeddings: nn.Embedding,
                                    entities: bool =True) -> Tuple[
                                        Tuple[Tensor, Tensor], 
                                        Tuple[Tensor, Tensor],
                                        Tuple[Tensor, Tensor],
                                        Tuple[Tensor, Tensor]]:
        b_size = h_idx.shape[0]

        re_h, im_h = tensor_split(node_embeddings[h_idx], 2, dim=1)
        re_r, im_r = tensor_split(relation_embeddings(r_idx), 2, dim=1)
        re_t, im_t = tensor_split(node_embeddings[t_idx], 2, dim=1)

        if entities:
            re_candidates, im_candidates = tensor_split(node_embeddings, 2, dim=1)
        else:
            re_candidates, im_candidates = tensor_split(relation_embeddings, 2, dim=1)
        
        re_candidates = re_candidates.unsqueeze(0).expand(b_size, -1, -1)
        im_candidates = im_candidates.unsqueeze(0).expand(b_size, -1, -1)

        return (re_h, im_h), (re_t, im_t), (re_r, im_r), (re_candidates, im_candidates)
    