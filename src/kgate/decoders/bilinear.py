from torchkge.models import DistMultModel, RESCALModel, AnalogyModel
from torch.nn.functional import normalize
import torch
from torch import matmul, tensor, Tensor, nn
from ..utils import init_embedding, HeteroMappings
from typing import Tuple

class RESCAL(RESCALModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_entities, n_relations)

        self.rel_mat = init_embedding(self.n_rel, self.emb_dim * self.emb_dim)

    def score(self, *, h_norm: Tensor, t_norm: Tensor, r_idx: Tensor, **_) -> Tensor:
        r = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)
        hr = matmul(h_norm.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t_norm).sum(dim=1)
    
    def get_embeddings(self) -> Tensor:
        return self.rel_mat.weight.data.view(-1, self.emb_dim, self.emb_dim)
    
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: nn.ModuleDict, 
                                    relation_embeddings: nn.Embedding, 
                                    mappings: HeteroMappings, 
                                    entities: bool =True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        # Get head, tail and relation embeddings
        h_emb_list = []
        t_emb_list = []
        
        for h_id in h_idx:
            node_type = mappings.kg_to_node_type[h_id.item()]
            h_emb  = node_embeddings[node_type].weight.data[mappings.kg_to_hetero[node_type][h_id.item()]]
            h_emb_list.append(h_emb)
        for t_id in t_idx:
            node_type = mappings.kg_to_node_type[t_id.item()]
            t_emb  = node_embeddings[node_type].weight.data[mappings.kg_to_hetero[node_type][t_id.item()]]
            t_emb_list.append(t_emb)

        
        h = torch.stack(h_emb_list, dim=0) if len(h_emb_list) != 0 else tensor([]).long()
        t = torch.stack(t_emb_list, dim=0) if len(t_emb_list) != 0 else tensor([]).long()

        r_mat = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)

            
        if entities:
            # Prepare candidates for every entities
            candidates = torch.cat([embedding.weight.data for embedding in node_embeddings.values()], dim=0)
            candidates = candidates.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
        else:
            # Prepare candidates for every relations
            candidates = self.rel_mat.weight.data.unsqueeze(0).expand(b_size, -1, -1, -1)

        return h, t, r_mat, candidates

    
class DistMult(DistMultModel):
    def __init__(self, emb_dim: int, n_entities: int, n_relations: int):
        super().__init__(emb_dim, n_entities, n_relations)
    
    def score(self, *, h_norm: Tensor, r_emb: Tensor, t_norm: Tensor, **_):
        return (h_norm * r_emb * t_norm).sum(dim=1)
    
    # TODO: if possible, factorize this
    def inference_prepare_candidates(self, *, 
                                    h_idx: Tensor, 
                                    t_idx: Tensor, 
                                    r_idx: Tensor, 
                                    node_embeddings: nn.ModuleDict, 
                                    relation_embeddings: nn.Embedding, 
                                    mappings: HeteroMappings, 
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
        h_emb_list = []
        t_emb_list = []
        
        for h_id in h_idx:
            node_type = mappings.kg_to_node_type[h_id.item()]
            h_emb  = node_embeddings[node_type].weight.data[mappings.kg_to_hetero[node_type][h_id.item()]]
            h_emb_list.append(h_emb)
        for t_id in t_idx:
            node_type = mappings.kg_to_node_type[t_id.item()]
            t_emb  = node_embeddings[node_type].weight.data[mappings.kg_to_hetero[node_type][t_id.item()]]
            t_emb_list.append(t_emb)

        
        h = torch.stack(h_emb_list, dim=0) if len(h_emb_list) != 0 else tensor([]).long()
        t = torch.stack(t_emb_list, dim=0) if len(t_emb_list) != 0 else tensor([]).long()

        r = relation_embeddings(r_idx)

        if entities:
            # Prepare candidates for every entities
            candidates = torch.cat([embedding.weight.data for embedding in node_embeddings.values()], dim=0)
            candidates = candidates.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
        else:
            # Prepare candidates for every relations
            candidates = relation_embeddings.weight.data.unsqueeze(0).expand(b_size, -1, -1)
        
        return h, t, r, candidates
