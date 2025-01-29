"""
Original code for the predictors from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license."""

from torch import empty, zeros, cat
from tqdm.autonotebook import tqdm

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.exceptions import NotYetEvaluatedError
from torchkge.sampling import PositionalNegativeSampler
from torchkge.utils import DataLoader, get_rank, filter_scores

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

    def evaluate(self, b_size, decoder, knowledge_graph, node_embeddings, relation_embeddings, mapping, verbose=True):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
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
            h_emb, t_emb, r_emb, candidates = decoder.inference_prepare_candidates(h_idx, t_idx, r_idx, node_embeddings, relation_embeddings, mapping, entities=True)

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

