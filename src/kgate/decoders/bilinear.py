from torchkge.models import DistMultModel, RESCALModel, AnalogyModel, ComplExModel
from torch.nn.functional import normalize
import torch
from torch import matmul, tensor, Tensor, nn, split
from ..utils import init_embedding, HeteroMappings
from typing import Tuple

class RESCAL(RESCALModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_entities, n_relations)

        self.rel_mat = init_embedding(self.n_rel, self.emb_dim * self.emb_dim)

    def score(self, *, h_emb: Tensor, t_emb: Tensor, r_idx: Tensor, **_) -> Tensor:
        h_norm = normalize(h_emb, p=2, dim=1)
        t_norm = normalize(t_emb, p=2, dim=1)
        r = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)
        hr = matmul(h_norm.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t_norm).sum(dim=1)
    
    def get_embeddings(self) -> Tensor:
        return self.rel_mat.weight.data.view(-1, self.emb_dim, self.emb_dim)
    
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
        del self.re_ent_emb
        del self.re_rel_emb

    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, h_idx:Tensor, t_idx:Tensor, r_idx: Tensor, **_):
        im_h = self.im_ent_emb(h_idx)
        im_t = self.im_ent_emb(t_idx)
        im_r = self.im_rel_emb(r_idx)

        return (h_emb * (r_emb * t_emb + im_r * im_t) + 
                im_h * (r_emb * im_t - im_r * t_emb)).sum(dim=1)
    
    def get_embeddings(self):
        return self.im_ent_emb.weight.data, self.im_rel_emb.weight.data
    
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

        re_h, im_h = node_embeddings[h_idx], self.im_ent_emb(h_idx)
        re_t, im_t = node_embeddings[t_idx], self.im_ent_emb(t_idx)
        re_r, im_r = node_embeddings[r_idx], self.im_ent_emb(r_idx)

        if entities:
            re_candidates = node_embeddings
            im_candidates = self.im_ent_emb
        else:
            re_candidates = relation_embeddings
            im_candidates = self.im_rel_emb
        
        re_candidates = re_candidates.unsqueeze(0).expand(b_size, -1, -1)
        im_candidates = im_candidates.unsqueeze(0).expand(b_size, -1, -1)

        return (re_h, im_h), (re_t, im_t), (re_r, im_r), (re_candidates, im_candidates)
    