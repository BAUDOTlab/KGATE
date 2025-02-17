from torchkge.models import TransEModel, TransHModel
from torch.nn.functional import normalize
from torch import tensor, split
from torch.cuda import empty_cache
import torch
from tqdm import tqdm

# Code adapted from torchKGE's implementation
# https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/models/translation.py#L18
class TransE(TransEModel):
    def __init__(self, emb_dim, n_entities, n_relations, dissimilarity_type):
        super().__init__(emb_dim, n_entities, n_relations, dissimilarity_type=dissimilarity_type)

    def score(self, h_norm, r_emb, t_norm, **_):
        return -self.dissimilarity(h_norm + r_emb, t_norm)
    
    def get_embeddigs(self):
        return None
    
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, node_embeddings, relation_embeddings, mappings, entities=True):
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

        h_emb_list = []
        t_emb_list = []
        # Get head, tail and relation embeddings
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
            # TODO : ensure candidates don't have index issues
            candidates = torch.cat([emb for embedding in node_embeddings.values() for emb in split(embedding.weight.data, 1)])
            candidates = candidates.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
        else:
            # Prepare candidates for every relations
            candidates = relation_embeddings.weight.data.unsqueeze(0).expand(b_size, -1, -1)
        
        return h, t, r, candidates
    
class TransH(TransHModel):
    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

    def score(self, h_norm, r_emb, t_norm, *, r_idx, **_):
        norm_vect = normalize(self.norm_vect(r_idx, p=2, dim=1))
        return - self.dissimilarity(self.project(h_norm, norm_vect) + r_emb,
                                    self.project(t_norm, norm_vect))

    @staticmethod
    def project(ent, norm_vect):
        return ent - (ent * norm_vect).sum(dim=1).view(-1, 1) * norm_vect
    
    def normalize_parameters(self, **_):
        self.norm_vect.weight.data = normalize(self.norm_vect.weight.data,
                                               p=2, dim=1)

    def get_embeddings(self):
        return self.norm_vect.weight.data
    
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, node_embeddings, relation_embeddings, mapping=None, entities=True):
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
            candidates = relation_embeddings.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

        return proj_h, proj_t, r, candidates

    def evaluate_projections(self, node_embeddings, mappings):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit='entities', desc='Projecting entities'):

            norm_vect = self.norm_vect.weight.data.view(self.n_rel, self.emb_dim)
            mask = tensor([i], device=norm_vect.device).long()

            if norm_vect.is_cuda:
                empty_cache()
            
            ent = node_embeddings[mappings.kg_to_node_type[mask.item()]].weight.data[mappings.kg_to_hetero[mask]]

            norm_components = (ent.view(1, -1) * norm_vect).sum(dim=1)
            self.projected_entities[:, i, :] = (ent.view(1, -1) - norm_components.view(-1, 1) * norm_vect)

            del norm_components

        self.evaluated_projections = True