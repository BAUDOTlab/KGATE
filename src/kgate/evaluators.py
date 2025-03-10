"""
Original code for the predictors from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license."""

from torch import empty, zeros, cat, Tensor
from tqdm.autonotebook import tqdm

from torch import nn

from torchkge.evaluation import LinkPredictionEvaluator, TripletClassificationEvaluator
from torchkge.exceptions import NotYetEvaluatedError
from torchkge.utils import DataLoader, get_rank, filter_scores
from torchkge.data_structures import SmallKG
from torchkge.models import Model

from .data_structures import KGATEGraph
from .utils import HeteroMappings
from .samplers import FixedPositionalNegativeSampler

class KLinkPredictionEvaluator(LinkPredictionEvaluator):
    """Evaluate performance of given embedding using link prediction method.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
      and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    knowledge_graph: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true head when all entities
        are ranked as possible replacement of the head entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true tail when all entities
        are ranked as possible replacement of the tail entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    filt_rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_heads` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    filt_rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_tails` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    evaluated: bool
        Indicates if the method LinkPredictionEvaluator.evaluate has already
        been called.

    """

    def __init__(self):
        self.evaluated = False

    def evaluate(self, 
                b_size: int,
                decoder: Model, 
                knowledge_graph: KGATEGraph, 
                node_embeddings: nn.ModuleDict, 
                relation_embeddings: nn.Embedding, 
                mappings: HeteroMappings, 
                verbose: bool=True):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        decoder: torchkge.Model
            Decoder model to evaluate, inheriting from the torchkge.Model class.
        knowledge_graph: kgate.KGATEGraph
            The test Knowledge Graph that will be used for the evaluation.
        node_embeddings: nn.ModuleDict
            A dictionnary where keys are relation types and values the
            embedding tensor of this relation's nodes.
        relation_embeddings: nn.Embedding
            A tensor containing one embedding by relation type.
        mappings: kgate.HeteroMappings
            An object containing mapping between the knowledge graph and 
            embeddings.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.
        """
        self.rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()

        use_cuda = next(decoder.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader(knowledge_graph, batch_size=b_size, use_cuda='batch')
            self.rank_true_heads = self.rank_true_heads.cuda()
            self.rank_true_tails = self.rank_true_tails.cuda()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cuda()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cuda()
        else:
            dataloader = DataLoader(knowledge_graph, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Link prediction evaluation'):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            h_emb, t_emb, r_emb, candidates = decoder.inference_prepare_candidates(h_idx = h_idx, 
                                                                                   t_idx = t_idx, 
                                                                                   r_idx = r_idx, 
                                                                                   node_embeddings = node_embeddings, 
                                                                                   relation_embeddings = relation_embeddings, 
                                                                                   mappings = mappings, 
                                                                                   entities=True)

            scores = decoder.inference_scoring_function(h_emb, candidates, r_emb)
            filt_scores = filter_scores(scores, knowledge_graph.dict_of_tails, h_idx, r_idx, t_idx)
            self.rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(scores, t_idx).detach()
            self.filt_rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, t_idx).detach()

            scores = decoder.inference_scoring_function(candidates, t_emb, r_emb)
            filt_scores = filter_scores(scores, knowledge_graph.dict_of_heads, t_idx, r_idx, h_idx)
            self.rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(scores, h_idx).detach()
            self.filt_rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, h_idx).detach()

        self.evaluated = True

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cpu()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cpu()

class KTripletClassificationEvaluator(TripletClassificationEvaluator):
    """Evaluate performance of given embedding using triplet classification
    method.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
      Reasoning With Neural Tensor Networks for Knowledge Base Completion.
      In Advances in Neural Information Processing Systems 26, pages 926-934.
      2013.
      https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg_val: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the testing evaluation will be done.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg_val: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    evaluated: bool
        Indicate whether the `evaluate` function has been called.
    thresholds: float
        Value of the thresholds for the scoring function to consider a
        triplet as true. It is defined by calling the `evaluate` method.
    sampler: torchkge.sampling.NegativeSampler
        Negative sampler.

    """

    def __init__(self, architect, kg_val, kg_test):
        self.architect = architect
        self.kg_val = kg_val
        self.kg_test = kg_test
        self.is_cuda = self.architect.device.type == "cuda"

        self.evaluated = False
        self.thresholds = None

        self.sampler = FixedPositionalNegativeSampler(self.kg_val,
                                                 kg_test=self.kg_test)

    def get_scores(self, heads: Tensor, tails: Tensor, relations: Tensor, batch_size: int):
        """With head, tail and relation indexes, compute the value of the
        scoring function of the model.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: n_facts
            List of heads indices.
        tails: torch.Tensor, dtype: torch.long, shape: n_facts
            List of tails indices.
        relations: torch.Tensor, dtype: torch.long, shape: n_facts
            List of relation indices.
        batch_size: int

        Returns
        -------
        scores: torch.Tensor, dtype: torch.float, shape: n_facts
            List of scores of each triplet.
        """
        scores = []
        #print(heads, heads.shape, tails, tails.shape, relations, relations.shape)

        small_kg = SmallKG(heads, tails, relations)
        if self.is_cuda:
            dataloader = DataLoader(small_kg, batch_size=batch_size,
                                    use_cuda='batch')
        else:
            dataloader = DataLoader(small_kg, batch_size=batch_size)

        for i, batch in enumerate(dataloader):
            h_idx, t_idx, r_idx = batch[0].to(self.architect.device), batch[1].to(self.architect.device), batch[2].to(self.architect.device)
            scores.append(self.architect.scoring_function(h_idx, t_idx, r_idx, train = False))

        return cat(scores, dim=0)

    def evaluate(self, b_size: int, knowledge_graph: KGATEGraph):
        """Find relation thresholds using the validation set. As described in
        the paper by Socher et al., for a relation, the threshold is a value t
        such that if the score of a triplet is larger than t, the fact is true.
        If a relation is not present in any fact of the validation set, then
        the largest value score of all negative samples is used as threshold.

        Parameters
        ----------
        b_size: int
            Batch size.
        """
        sampler = FixedPositionalNegativeSampler(knowledge_graph)
        r_idx = knowledge_graph.relations

        neg_heads, neg_tails = sampler.corrupt_kg(b_size, self.is_cuda,
                                                       which='main')
        neg_scores = self.get_scores(neg_heads, neg_tails, r_idx, b_size)

        self.thresholds = zeros(self.kg_val.n_rel)

        for i in range(self.kg_val.n_rel):
            mask = (r_idx == i).bool()
            if mask.sum() > 0:
                self.thresholds[i] = neg_scores[mask].max()
            else:
                self.thresholds[i] = neg_scores.max()

        self.evaluated = True
        self.thresholds.detach_()

    def accuracy(self, b_size:int, kg_test: KGATEGraph, kg_val: KGATEGraph | None = None):
        """

        Parameters
        ----------
        b_size: int
            Batch size.

        Returns
        -------
        acc: float
            Share of all triplets (true and negatively sampled ones) that where
            correctly classified using the thresholds learned from the
            validation set.

        """
        if not self.evaluated:
            kg_to_eval = kg_val if kg_val is not None else kg_test
            self.evaluate(b_size=b_size, knowledge_graph=kg_to_eval)

        sampler = FixedPositionalNegativeSampler(kg_test)
        r_idx = kg_test.relations

        neg_heads, neg_tails = sampler.corrupt_kg(b_size,
                                                self.is_cuda,
                                                which='main')
        scores = self.get_scores(kg_test.head_idx,
                                 kg_test.tail_idx,
                                 r_idx,
                                 b_size)
        neg_scores = self.get_scores(neg_heads, neg_tails, r_idx, b_size)

        if self.is_cuda:
            self.thresholds = self.thresholds.cuda()
            
        scores = (scores > self.thresholds[r_idx])
        neg_scores = (neg_scores < self.thresholds[r_idx])

        return (scores.sum().item() +
                neg_scores.sum().item()) / (2 * self.kg_test.n_facts)