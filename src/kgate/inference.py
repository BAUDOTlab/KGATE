from torchkge.inference import RelationInference, EntityInference, DataLoader_
from torchkge.utils import filter_scores
from torchkge.models import Model
from tqdm.autonotebook import tqdm
from torch import tensor, nn
import torch
from typing import Dict, Tuple, List
from .utils import HeteroMappings

class KRelationInference(RelationInference):
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
    # TODO: add the possibility to infer link orientation as well.

    def __init__(self, model: Model, entities1: torch.Tensor, entities2: torch.Tensor, top_k: int=1, dictionary: Dict[Tuple[int, int], List[int]] | None = None):
        super().__init__(model, entities1, entities2, top_k, dictionary)

    def evaluate(self, b_size: int, node_embeddings: nn.ModuleDict, relation_embeddings: nn.Embedding, mapping: HeteroMappings, verbose:bool=True):
        with torch.no_grad():
            use_cuda = next(self.model.parameters()).is_cuda

            if use_cuda:
                dataloader = DataLoader_(self.entities1, self.entities2, batch_size=b_size, use_cuda='batch')
                self.predictions = self.predictions.cuda()
            else:
                dataloader = DataLoader_(self.entities1, self.entities2, batch_size=b_size)

            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                unit='batch', disable=(not verbose),
                                desc='Inference'):
                ents1_idx, ents2_idx = batch[0], batch[1]
                h_emb, t_emb, _, candidates = self.model.inference_prepare_candidates(ents1_idx, ents2_idx, tensor([]).long(),
                                                                                    node_embeddings, relation_embeddings, mapping, entities=False)
                scores = self.model.inference_scoring_function(h_emb, t_emb, candidates)

                if self.dictionary is not None:
                    scores = filter_scores(scores, self.dictionary, ents1_idx, ents2_idx, None)

                scores, indices = scores.sort(descending=True)

                self.predictions[i * b_size: (i + 1) * b_size] = indices[:, :self.topk]
                self.scores[i * b_size, (i + 1) * b_size] = scores[:, :self.topk]

            if use_cuda:
                self.predictions = self.predictions.cpu()
                self.scores = self.scores.cpu()

class KEntityInference(EntityInference):
    """Use trained embedding model to infer missing entities in triples.

        Parameters
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
    def __init__(self, model: Model, known_entities: torch.Tensor, known_relations: torch.Tensor, top_k:int=1, missing:str='tails', dictionary: Dict[Tuple[int, int], List[int]] | None         =None):
        super().__init__(model, known_entities, known_relations, top_k, missing, dictionary)

    def evaluate(self, b_size: int, node_embeddings: nn.ModuleDict, relation_embeddings: nn.Embedding, mapping: HeteroMappings, verbose:bool=True):
        with torch.no_grad():
            use_cuda = next(self.model.parameters()).is_cuda
            device = "cuda" if use_cuda else "cpu"
            
            if use_cuda:
                dataloader = DataLoader_(self.known_entities, self.known_relations, batch_size=b_size, use_cuda='batch')
                self.predictions = self.predictions.cuda()
            else:
                dataloader = DataLoader_(self.known_entities, self.known_relations, batch_size=b_size)

            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                unit='batch', disable=(not verbose),
                                desc='Inference'):
                known_ents, known_rels = batch[0], batch[1]
                if self.missing == 'heads':
                    _, t_emb, rel_emb, candidates = self.model.inference_prepare_candidates(tensor([]).long().to(device), known_ents,
                                                                                            known_rels, node_embeddings,
                                                                                            relation_embeddings, mapping,
                                                                                            entities=True)
                    scores = self.model.inference_scoring_function(candidates, t_emb, rel_emb)
                else:
                    h_emb, _, rel_emb, candidates = self.model.inference_prepare_candidates(known_ents, tensor([]).long().to(device),
                                                                                            known_rels, node_embeddings,
                                                                                            relation_embeddings, mapping,
                                                                                            entities=True)
                    scores = self.model.inference_scoring_function(h_emb, candidates, rel_emb)

                if self.dictionary is not None:
                    scores = filter_scores(scores, self.dictionary, known_ents, known_rels, None)

                scores, indices = scores.sort(descending=True)
                b_size = min(b_size, len(scores))
                
                self.predictions[i * b_size: (i+1)*b_size] = indices[:, :self.top_k]
                self.scores[i*b_size, (i+1)*b_size] = scores[:, :self.top_k]

            if use_cuda:
                self.predictions = self.predictions.cpu()
                self.scores = self.scores.cpu()