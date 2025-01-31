from torchkge.models import TransEModel
import torch

# Code adapted from torchKGE's implementation
# https://github.com/torchkge-team/torchkge/blob/3adb9344dec974fc29d158025c014b0dcb48118c/torchkge/models/translation.py#L18
class TransE(TransEModel):
    def __init__(self, emb_dim, n_entities, n_relations, dissimilarity_type):
        super().__init__(emb_dim, n_entities, n_relations, dissimilarity_type=dissimilarity_type)

    def score(self, h_norm, r_emb, t_norm, **_):
        return -self.dissimilarity(h_norm + r_emb, t_norm)
    
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