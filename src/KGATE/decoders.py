from torchkge.models import TransEModel
import torch

# Code adapted from torchKGE's implementation
# https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/models/translation.py#L18
class TransE(TransEModel):
    def __init__(self, n_entities, n_relations, **kwargs):
        super().__init__(n_entities, n_relations, dissimilarity_type=kwargs.get("dissimilarity_type", "L2"))

    def score(self, h_norm, r_emb, t_norm):
        return -self.dissimilarity(h_norm + r_emb, t_norm)
    
    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
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
        h = torch.cat([self.node_embeddings[self.kg2nodetype[h_id.item()]].weight.data[h_id] for h_id in h_idx], dim=0)
        t = torch.cat([self.node_embeddings[self.kg2nodetype[t_id.item()]].weight.data[t_id] for t_id in t_idx], dim=0)
        r = self.rel_emb(r_idx)

        if entities:
            # Prepare candidates for every entities
            candidates = torch.cat([embedding.weight.data for embedding in self.node_embeddings.values()], dim=0)
            candidates = candidates.view(1, -1, self.emb_dim).expand(b_size, -1, -1)
        else:
            # Prepare candidates for every relations
            candidates = self.rel_emb.weight.data.unsqueeze(0).expand(b_size, -1, -1)
        
        return h, t, r, candidates
