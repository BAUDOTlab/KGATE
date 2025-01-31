from torchkge.models import DistMultModel, RESCALModel
from torch.nn.functional import normalize
import torch
from torch import matmul
from ..utils import init_embedding

class RESCAL(RESCALModel):
    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

        self.rel_mat = init_embedding(self.n_rel, self.emb_dim * self.emb_dim)

    def score(self, h_norm, _, t_norm, r_idx, **__):
        r = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)
        hr = matmul(h_norm.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t_norm).sum(dim=1)
    
    def get_embeddings(self):
        return self.rel_mat.weight.data.view(-1, self.emb_dim, self.emb_dim)
    
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, node_embeddings, _, mapping=None,  entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        h = node_embeddings(h_idx)
        t = node_embeddings(t_idx)
        r_mat = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)

        if mapping is not None:
            h = torch.cat([node_embeddings[mapping[h_id.item()]].weight.data[h_id] for h_id in h_idx], dim=0)
            t = torch.cat([node_embeddings[mapping[t_id.item()]].weight.data[t_id] for t_id in t_idx], dim=0)
        else:
            h = node_embeddings(h_idx)
            t = node_embeddings(t_idx)
            
        if entities:
            # Prepare candidates for every entities
            if mapping is not None:
                candidates = torch.cat([embedding.weight.data for embedding in self.node_embeddings.values()], dim=0)
                candidates = candidates.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
            else:
                candidates = node_embeddings.weight.data.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
        else:
            # Prepare candidates for every relations
            candidates = self.rel_mat.weight.data.unsqueeze(0).expand(b_size, -1, -1, -1)

        return h, t, r_mat, candidates

    
class DistMult(DistMultModel):
    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)
    
    def score(self, h_norm, r_emb, t_norm, **_):
        return (h_norm * r_emb * t_norm).sum(dim=1)
    
    def normalize_parameters(self, rel_emb, **_):
        rel_emb.weight.data = normalize(rel_emb.weight.data, p=2, dim=1)
        return False
    
    # TODO: if possible, factorize this
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, node_embeddings, relation_embeddings, mapping=None, entities=True):
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
        if mapping is not None:
            h = torch.cat([node_embeddings[mapping[h_id.item()]].weight.data[h_id] for h_id in h_idx], dim=0)
            t = torch.cat([node_embeddings[mapping[t_id.item()]].weight.data[t_id] for t_id in t_idx], dim=0)
        else:
            h = node_embeddings(h_idx)
            t = node_embeddings(t_idx)
        r = relation_embeddings(r_idx)

        if entities:
            # Prepare candidates for every entities
            if mapping is not None:
                candidates = torch.cat([embedding.weight.data for embedding in self.node_embeddings.values()], dim=0)
                candidates = candidates.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
            else:
                candidates = node_embeddings.weight.data.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
        else:
            # Prepare candidates for every relations
            candidates = relation_embeddings.weight.data.unsqueeze(0).expand(b_size, -1, -1)
        
        return h, t, r, candidates
    
