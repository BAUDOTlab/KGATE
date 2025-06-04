import torchkge.inference as infer
from torchkge.models import Model
from tqdm.autonotebook import tqdm
from torch import tensor, nn, Tensor
import torch
from typing import Dict, Literal
from .utils import filter_scores
from .encoders import DefaultEncoder, GNN
from .knowledgegraph import KnowledgeGraph
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import k_hop_subgraph

class Inference_KG(Dataset):
    def __init__(self, idx_1:Tensor, idx_2:Tensor):
        assert idx_1.size() == idx_2.size(), "Both index tensors must be of the same size for inference."
        self.id1 = idx_1
        self.id2 = idx_2

    def __len__(self):
        return self.id1.size(0)

    def __getitem__(self, idx: int):
        return (self.id1[idx], self.id2[idx])


class RelationInference(infer.RelationInference):
    """Use trained embedding model to infer missing relations in triples.

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    entities1: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 1.
    entities2: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 2.
    top_k: int
        Indicates the number of top predictions to return.
    dictionary: dict, optional (default=None)
        Dictionary of possible relations. It is used to filter predictions
        that are known to be True in the training set in order to return
        only new facts.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    entities1: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 1.
    entities2: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 2.
    top_k: int
        Indicates the number of top predictions to return.
    dictionary: dict, optional (default=None)
        Dictionary of possible relations. It is used to filter predictions
        that are known to be True in the training set in order to return
        only new facts.
    predictions: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.long`
        List of the indices of predicted relations for each test fact.
    scores: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.float`
        List of the scores of resulting triples for each test fact.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def evaluate(self, 
                 h_idx: Tensor,
                 t_idx: Tensor,
                 *,
                 top_k: int,
                 b_size: int,
                 encoder: DefaultEncoder | GNN,
                 decoder: Model,
                 node_embeddings: nn.ParameterList | nn.Embedding, 
                 relation_embeddings: nn.Embedding, 
                 verbose:bool=True,
                 **_):
        
        with torch.no_grad():
            use_cuda = relation_embeddings.weight.is_cuda
            device = relation_embeddings.weight.device

            inference_kg = Inference_KG(h_idx, t_idx)

            dataloader = DataLoader(inference_kg, batch_size=b_size)

            predictions = torch.empty(size=(len(h_idx), top_k), device=device).long()   
            embeddings = node_embeddings.weight.data

            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                unit="batch", disable=(not verbose),
                                desc="Inference"):
                h_idx, t_idx = batch[0], batch[1]
                
                if isinstance(encoder, GNN):
                    seed_nodes = batch.unique()
                    num_hops = encoder.n_layers
                    edge_index = self.knowledge_graph.edge_index

                    _,_,_, edge_mask = k_hop_subgraph(
                        node_idx = seed_nodes,
                        num_hops = num_hops,
                        edge_index = edge_index
                        )
                    
                    input = self.knowledge_graph.get_encoder_input(self.knowledge_graph.edgelist[:, edge_mask], node_embeddings)
                    encoder_output: Dict[str, Tensor] = encoder(input.x_dict, input.edge_index)
            
                    for node_type, idx in input.mapping.items():
                        embeddings[idx] = encoder_output[node_type]



                h_emb, t_emb, _, candidates = decoder.inference_prepare_candidates(h_idx = h_idx,
                                                                                        t_idx = t_idx, 
                                                                                        r_idx = tensor([]).long(),
                                                                                        node_embeddings = embeddings, 
                                                                                        relation_embeddings = relation_embeddings, 
                                                                                        entities=False)
                scores = decoder.inference_scoring_function(h_emb, t_emb, candidates)

                scores = filter_scores(scores, self.knowledge_graph.edgelist, "rel", h_idx, t_idx, None)

                scores, indices = scores.sort(descending=True)

                predictions[i * b_size: (i + 1) * b_size] = indices[:, :top_k]
                scores[i * b_size, (i + 1) * b_size] = scores[:, :top_k]

            return predictions.cpu(), scores.cpu()

class EntityInference(infer.EntityInference):
    """Use trained embedding model to infer missing entities in triples.

        Attributes
        ----------
        model: torchkge.models.interfaces.Model
            Embedding model inheriting from the right interface.
        known_entities: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
            List of the indices of known entities.
        known_relations: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
            List of the indices of known relations.
        top_k: int
            Indicates the number of top predictions to return.
        missing: str
            String indicating if the missing entities are the heads or the tails.
        dictionary: dict, optional (default=None)
            Dictionary of possible heads or tails (depending on the value of `missing`).
            It is used to filter predictions that are known to be True in the training set
            in order to return only new facts.
        predictions: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.long`
            List of the indices of predicted entities for each test fact.
        scores: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.float`
            List of the scores of resulting triples for each test fact.

    """
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def evaluate(self,
                 ent_idx: Tensor,
                 rel_idx: Tensor,
                 *,
                 top_k: int,
                 missing: Literal["head","tail"],
                 b_size: int,
                 encoder: DefaultEncoder | GNN,
                 decoder: Model,
                 node_embeddings: nn.Embedding | nn.ParameterList, 
                 relation_embeddings: nn.Embedding,
                 verbose:bool=True,
                 **_):
        with torch.no_grad():
            device = relation_embeddings.weight.device

            inference_kg = Inference_KG(ent_idx, rel_idx)

            dataloader = DataLoader(inference_kg, batch_size=b_size)

            predictions = torch.empty(size=(len(ent_idx), top_k), device=device).long()
            scores = torch.empty(size=(len(ent_idx), top_k), device=device).long()
            embeddings = node_embeddings.weight.data

            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                unit="batch", disable=(not verbose),
                                desc="Inference"):

                known_ents, known_rels = batch[0], batch[1]
                
                if isinstance(encoder, GNN):
                    seed_nodes = known_ents.unique()
                    num_hops = encoder.n_layers
                    edge_index = self.knowledge_graph.edge_index

                    _,_,_, edge_mask = k_hop_subgraph(
                        node_idx = seed_nodes,
                        num_hops = num_hops,
                        edge_index = edge_index
                        )
                    
                    input = self.knowledge_graph.get_encoder_input(self.knowledge_graph.edgelist[:, edge_mask], node_embeddings)
                    encoder_output: Dict[str, Tensor] = encoder(input.x_dict, input.edge_index)
            
                    for node_type, idx in input.mapping.items():
                        embeddings[idx] = encoder_output[node_type]

                if missing == "head":
                    _, t_emb, rel_emb, candidates = decoder.inference_prepare_candidates(h_idx = tensor([], device=device).long(), 
                                                                                         t_idx = known_ents.to(device),
                                                                                         r_idx = known_rels.to(device),
                                                                                         node_embeddings = embeddings,
                                                                                         relation_embeddings = relation_embeddings,
                                                                                         entities=True)
                    batch_scores = decoder.inference_scoring_function(candidates, t_emb, rel_emb)
                else:
                    h_emb, _, rel_emb, candidates = decoder.inference_prepare_candidates(h_idx = known_ents.to(device), 
                                                                                         t_idx = tensor([], device=device).long(),
                                                                                         r_idx = known_rels.to(device),
                                                                                         node_embeddings = embeddings,
                                                                                         relation_embeddings = relation_embeddings,
                                                                                         entities=True)
                    batch_scores = decoder.inference_scoring_function(h_emb, candidates, rel_emb)

                batch_scores = filter_scores(batch_scores, self.knowledge_graph.edgelist, missing, known_ents, known_rels, None)

                batch_scores, indices = batch_scores.sort(descending=True)
                b_size = min(b_size, len(batch_scores))
                
            predictions[i * b_size: (i+1)*b_size] = indices[:, :top_k]
            scores[i*b_size: (i+1)*b_size] = batch_scores[:, :top_k]

            return predictions.cpu(), scores.cpu()