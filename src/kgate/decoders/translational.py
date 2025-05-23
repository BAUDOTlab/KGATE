from torchkge.models import TransEModel, TransHModel, TransRModel, TransDModel, TorusEModel
from torch.nn.functional import normalize
from torch import nn, tensor, split, matmul, Tensor
from torch.cuda import empty_cache
import torch
from tqdm import tqdm

from typing import Tuple, List
from ..utils import HeteroMappings

# Code adapted from torchKGE's implementation
# https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/models/translation.py#L18
class TransE(TransEModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int, dissimilarity_type: str):
        super().__init__(emb_dim, n_entities, n_relations, dissimilarity_type=dissimilarity_type)
        del self.ent_emb
        del self.rel_emb

    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, **_) -> Tensor:
        h_norm = normalize(h_emb, p=2, dim=1)
        t_norm = normalize(t_emb, p=2, dim=1)
        return -self.dissimilarity(h_norm + r_emb, t_norm)
    
    def get_embeddings(self):
        return None
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: Tensor, 
                                    relation_embeddings: nn.Embedding,
                                    entities: bool=True
                                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    
class TransH(TransHModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_entities, n_relations)

    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, r_idx: Tensor, **_) -> Tensor:
        h_norm = normalize(h_emb, p=2, dim=1)
        t_norm = normalize(t_emb, p=2, dim=1)
        self.evaluated_projections = False
        norm_vect = normalize(self.norm_vect(r_idx), p=2, dim=1)
        return - self.dissimilarity(self.project(h_norm, norm_vect) + r_emb,
                                    self.project(t_norm, norm_vect))
    
    def normalize_params(self, **_):
        self.norm_vect.weight.data = normalize(self.norm_vect.weight.data,
                                               p=2, dim=1)

    def get_embeddings(self) -> Tensor:
        return self.norm_vect.weight.data
    
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

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        r = relation_embeddings(r_idx)

        if entities:
            proj_h = self.projected_entities[r_idx, h_idx]  # shape: (b_size, emb_dim)
            proj_t = self.projected_entities[r_idx, t_idx]  # shape: (b_size, emb_dim)
            candidates = self.projected_entities[r_idx]  # shape: (b_size, self.n_rel, self.emb_dim)
        else:
            proj_h = self.projected_entities[:, h_idx].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            proj_t = self.projected_entities[:, t_idx].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            candidates = relation_embeddings.weight.data.unsqueeze(0).expand(b_size, self.n_rel, self.emb_dim)

        return proj_h, proj_t, r, candidates

    def evaluate_projections(self, node_embeddings: Tensor):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit="entities", desc="Projecting entities"):

            norm_vect = self.norm_vect.weight.data.view(self.n_rel, self.emb_dim)
            mask = tensor([i], device=norm_vect.device).long()

            if norm_vect.is_cuda:
                empty_cache()

            ent = node_embeddings[mask]

            norm_components = (ent.view(1, -1) * norm_vect).sum(dim=1)
            self.projected_entities[:, i, :] = (ent.view(1, -1) - norm_components.view(-1, 1) * norm_vect)

            del norm_components

        self.evaluated_projections = True

class TransR(TransRModel):
    def __init__(self, ent_emb_dim: int, rel_emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(ent_emb_dim, rel_emb_dim, n_entities, n_relations)

    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, r_idx: Tensor, **_) -> Tensor:
        h_norm = normalize(h_emb, p=2, dim=1)
        t_norm = normalize(t_emb, p=2, dim=1)
        self.evaluated_projections = False
        b_size = h_norm.shape[0]

        proj_mat = self.proj_mat(r_idx).view(b_size,
                                             self.rel_emb_dim,
                                             self.ent_emb_dim)
        return - self.dissimilarity(self.project(h_norm, proj_mat) + r_emb,
                                    self.project(t_norm, proj_mat))
    
    def normalize_params(self, rel_emb: nn.Embedding, **_) -> bool:
        rel_emb.weight.data = normalize(rel_emb.weight.data, p=2, dim=1)
        return False
    
    def get_embeddings(self) -> Tensor:
        return self.proj_mat.weight.data.view(-1,
                                              self.rel_emb_dim,
                                              self.ent_emb_dim)
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: Tensor, 
                                    relation_embeddings: nn.Embedding,
                                    entities: bool =True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        b_size = h_idx.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        r = relation_embeddings(r_idx)
        if entities:
            proj_h = self.projected_entities[r_idx, h_idx]  # shape: (b_size, emb_dim)
            proj_t = self.projected_entities[r_idx, t_idx]  # shape: (b_size, emb_dim)
            candidates = self.projected_entities[r_idx]  # shape: (b_size, self.n_rel, self.emb_dim)
        else:
            proj_h = self.projected_entities[:, h_idx].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            proj_t = self.projected_entities[:, t_idx].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            candidates = relation_embeddings.weight.data.unsqueeze(0).expand(b_size, self.n_rel, self.emb_dim)

        return proj_h, proj_t, r, candidates
    
    def evaluate_projections(self, node_embeddings: Tensor):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit="entities", desc="Projecting entities"):
            projection_matrices = self.proj_mat.weight.data
            projection_matrices = projection_matrices.view(self.n_rel, self.rel_emb_dim, self.ent_emb_dim)

            mask = tensor([i], device=projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            ent = node_embeddings[mask]
            
            proj_ent = matmul(projection_matrices, ent.view(self.ent_emb_dim))
            proj_ent = proj_ent.view(self.n_rel, self.rel_emb_dim, 1)
            self.projected_entities[:, i, :] = proj_ent.view(self.n_rel, self.rel_emb_dim)

            del proj_ent

        self.evaluated_projections = True

class TransD(TransDModel):
    def __init__(self, ent_emb_dim: int, rel_emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(ent_emb_dim, rel_emb_dim, n_entities, n_relations)

    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, h_idx: Tensor, r_idx: Tensor, t_idx: Tensor, **_) -> Tensor:
        h_norm = normalize(h_emb, p=2, dim=1)
        t_norm = normalize(t_emb, p=2, dim=1)
        r = normalize(r_emb, p=2, dim=1)

        h_proj_v = normalize(self.ent_proj_vect(h_idx), p=2, dim=1)
        t_proj_v = normalize(self.ent_proj_vect(t_idx), p=2, dim=1)
        r_proj_v = normalize(self.rel_proj_vect(r_idx), p=2, dim=1)

        proj_h = self.project(h_norm, h_proj_v, r_proj_v)
        proj_t = self.project(t_norm, t_proj_v, r_proj_v)
        return - self.dissimilarity(proj_h + r, proj_t)
    
    def normalize_params(self, rel_emb: nn.Embedding, **_):
        rel_emb.weight.data = normalize(rel_emb.weight.data, p=2, dim=1)
        self.ent_proj_vect.weight.data = normalize(self.ent_proj_vect.weight.data, p=2, dim=1)
        self.rel_proj_vect.weight.data = normalize(self.rel_proj_vect.weight.data, p=2, dim=1)

    def get_embeddings(self) -> Tuple[Tensor, Tensor]:
        return self.ent_proj_vect.weight.data, self.rel_proj_vect.weight.data
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: Tensor, 
                                    relation_embeddings: nn.Embedding,
                                    entities: bool =True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        b_size = h_idx.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections(node_embeddings)

        r = relation_embeddings(r_idx)

        if entities:
            proj_h = self.projected_entities[r_idx, h_idx]  # shape: (b_size, emb_dim)
            proj_t = self.projected_entities[r_idx, t_idx]  # shape: (b_size, emb_dim)
            candidates = self.projected_entities[r_idx]  # shape: (b_size, self.n_rel, self.emb_dim)
        else:
            proj_h = self.projected_entities[:, h_idx].transpose(0, 1)  # shape: (b_size, n_rel, rel_emb_dim)
            proj_t = self.projected_entities[:, t_idx].transpose(0, 1)  # shape: (b_size, n_rel, rel_emb_dim)
            candidates = self.rel_emb.weight.data.unsqueeze(0).expand(b_size, self.n_rel, self.rel_emb_dim)

        return proj_h, proj_t, r, candidates

    def evaluate_projections(self, node_embeddings: Tensor):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit="entities", desc="Projecting entities"):
            rel_proj_vects = self.rel_proj_vect.weight.data

            mask = tensor([i], device=rel_proj_vects.device).long()

            ent = node_embeddings[mask]

            ent_proj_vect = self.ent_proj_vect.weight[i]

            sc_prod = (ent_proj_vect * ent).sum(dim=0)
            proj_e = sc_prod * rel_proj_vects + ent[:self.rel_emb_dim].view(1, -1)

            self.projected_entities[:, i, :] = proj_e

            del proj_e

        self.evaluated_projections = True

class TorusE(TorusEModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int, dissimilarity_type: str):
        super().__init__(emb_dim, n_entities, n_relations, dissimilarity_type)
    
    def score(self, *, h_emb: Tensor, r_emb: Tensor, t_emb: Tensor, h_idx: Tensor, r_idx: Tensor, t_idx: Tensor, **_) -> Tensor:
        self.normalized = False

        h = h_emb.frac()
        t = t_emb.frac()
        r = r_emb.frac()

        return - self.dissimilarity(h + r, t)

    def normalize_parameters(self, ent_emb: Tensor, rel_emb: nn.Embedding):
        ent_emb.frac_()

        rel_emb.weight.data.frac_()
        self.normalized = True
    
    def get_embeddings(self):
        return None
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: Tensor, 
                                    relation_embeddings: nn.Embedding,
                                    entities: bool =True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        b_size = h_idx.shape[0]

        if not self.normalized:
            self.normalize_parameters(node_embeddings, relation_embeddings)

        device = h_idx.device        

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