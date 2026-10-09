"""
Evaluator classes to evaluate model performances.

Original code for the predictors from TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

Modifications and additional functionalities added by Benjamin Loire <benjamin.loire@univ-amu.fr>:
- 

The modifications are licensed under the BSD license according to the source license.
"""

from typing import Dict, Tuple, TYPE_CHECKING

from tqdm import tqdm

import torch
from torch import empty, zeros, cat, Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torch_geometric.utils import k_hop_subgraph

from torchkge.utils import get_rank
from torchkge.data_structures import SmallKG

if TYPE_CHECKING:
    from .architect import Architect
from .decoders import BilinearDecoder, ConvolutionalDecoder, TranslationalDecoder
from .encoders import GNN
from .knowledgegraph import KnowledgeGraph
from .samplers import NegativeSampler, PositionalNegativeSampler, BernoulliNegativeSampler, UniformNegativeSampler, MixedNegativeSampler
from .utils import filter_scores

import logging

class Predictions:
    def __init__(self,
                true_predictions_rank: Tensor = None,
                filtered_true_predictions_rank: Tensor = None,
                sphere_predictions: Tensor = None):
        """
        Object holding the predictions output of an Evaluator.

        Predictions are stored as rank tensors and can be accessed through 
        builtin methods to get specific metrics (e.g. `mrr`, `mean_rank`, `hit_at_k`,
        `median_rank`, `mean_reciprocal_rank_at_k`, `score_gap`, `relative_rank`),
        or flattened at once into a dictionary with `to_dict`.

        Arguments
        ---------
        
        **true_predictions_rank** *(torch.Tensor)*
        : Among the ranking of all predictions, the rank of the true result.
        
        **filtered_true_predictions_rank** *(torch.Tensor)*
        : Among the ranking of all filtered predictions, the rank of the true result.
        : True triplets that are not the target of the prediction are filtered out.

        **sphere_predictions** *(torch.Tensor, optional, keyword only)*
        : Global indices of the triplets predicted positive when `sphere_embeddings` is True
        : (per-triplet booleans, see `LinkPredictionEvaluator.evaluate`), instead of ranks.
        
        Attributes
        ----------
        
        **true_predictions_rank** *(torch.Tensor)*
        : Among the ranking of all predictions, the rank of the true result.
        
        **filtered_true_predictions_rank** *(torch.Tensor)*
        : Among the ranking of all filtered predictions, the rank of the true result.
        : True triplets that are not the target of the prediction are filtered out.

        **sphere_predictions** *(torch.Tensor)*
        : Global indices of the triplets predicted positive when `sphere_embeddings` is True.
        """

        self.true_predictions_rank = true_predictions_rank
        self.filtered_true_predictions_rank = filtered_true_predictions_rank
        self.sphere_predictions = sphere_predictions


    @staticmethod
    def _mean_reciprocal_rank(rank: Tensor) -> float:
        """
        Mean reciprocal rank of a 1-D rank tensor (1-indexed ranks),
        computed with a single tensor mean (no intermediate `.item()` calls).

        Arguments
        ---------

        **rank** *(torch.Tensor, dtype: torch.long or torch.float, shape: [triplet_count])*
        : Ranks of the true predictions.

        Returns
        -------

        **mrr** *(float)*
        : Mean of the reciprocals of the ranks. Perfect score is 1.

        """
        return (rank.float() ** (-1)).mean().item()
    
    
    def __str__(self):
        if self.true_predictions_rank is not None and self.filtered_true_predictions_rank is not None:
            k = 10
            message = f"""
            Hit@{k}: {round(self.hit_at_k(k)[0],3)} \t Filtered Hit@{k}: {round(self.hit_at_k(k)[1],3)} 

            MRR: {round(self.mrr[0],3)} \t Filtered MRR: {round(self.mrr[1],3)}

            Mean Rank: {int(self.mean_rank[0])} \t Filtered Mean Rank: {int(self.mean_rank[1])}
            """
            return message
    
        elif self.sphere_predictions is not None:
            message = f""" 
            Sphere embeddings hyperparameter is set at true.
            
            Hit@K, MRR and Mean Rank evaluations are impossible, as predictions are unranked.
            """
            return message
        
        else:
            raise ValueError("The Predictions object must receive either a tensor of predictions from sphere embeddings, or both tensors `true_predictions_rank` and `filtered_true_predictions_rank`.")
    


    @property
    def mean_rank(self) -> Tuple[float, float]:
        """
        Mean rank metric: mean of the `true_predictions_rank` values, both
        unfiltered and filtered.
        
        Returns
        -------
        
        **mean_rank_score** *(float)*
        : Mean value of `true_predictions_rank` scores.
        : Among the ranking of all predictions, `true_predictions_rank` is the rank of the true result.
        
        **filtered_mean_rank_score** *(float)*
        : Mean value of `filtered_true_predictions_rank` scores.
        : Among the ranking of all filtered predictions, `filtered_true_predictions_rank` is the rank of the true result.
        : True triplets that are not the target of the prediction are filtered out.
        
        """
        if self.true_predictions_rank is not None and self.filtered_true_predictions_rank is not None:
            mean_rank_score = self.true_predictions_rank.float().mean().item()

            filtered_mean_rank_score = self.filtered_true_predictions_rank.float().mean().item()

            return mean_rank_score, filtered_mean_rank_score
        
        else:
            raise ValueError("Mean Rank evaluation is impossible with predictions from sphere embeddings, as they are unranked. To disable sphere embeddings, set the `sphere_embeddings` hyperparameter as false in the config file.")

    
    
    def hit_at_k(self,
                k: int = 10
                ) -> Tuple[float, float]:
        """
        Return the frequence at which the true triplet is within the k first predictions.
        
        Arguments
        ---------        
        **k** *(int, default to 10)*
        : The true triplet must be within the k first predictions.
        
        Returns
        -------
        
        **true_prediction_hit** *(float)*
        : Frequence at which the true triplet is within the k first predictions.
        
        **filtered_true_prediction_hit** *(float)*
        : Frequence at which the true triplet is within the k first predictions, when ranking among filtered triplets.
        : True triplets that are not the target of the prediction are filtered out.
        
        """
        if self.true_predictions_rank is not None and self.filtered_true_predictions_rank is not None:
            true_prediction_hit = (self.true_predictions_rank <= k).float().mean().item()
            filtered_true_prediction_hit = (self.filtered_true_predictions_rank <= k).float().mean().item()

            return true_prediction_hit, filtered_true_prediction_hit
    
        else:
            raise ValueError("Hit@K evaluation is impossible with predictions from sphere embeddings, as they are unranked. To disable sphere embeddings, set the `sphere_embeddings` hyperparameter as false in the config file.")
    
    
    
    @property
    def mrr(self) -> Tuple[float, float]:
        """
        Mean reciprocal rank: mean of the inverse of the ranks of the true
        predictions, both unfiltered and filtered.

        Returns
        -------
        
        **mrr** *(float)*
        : Inverse of the position of the true triplet prediction.
        : If the true triplet is predicted in 100th position, then mrr = 0.01
        : Perfect score is 1.
        
        **filtered_mrr** *(float)*
        : Inverse of the position of the true triplet filtered prediction.
        : If the true triplet is predicted in 100th position, then mrr = 0.01
        : Perfect score is 1.
        : True triplets that are not the target of the prediction are filtered out.
        
        """
        if self.true_predictions_rank is not None and self.filtered_true_predictions_rank is not None:
            mrr = self._mean_reciprocal_rank(self.true_predictions_rank)
            filtered_mrr = self._mean_reciprocal_rank(self.filtered_true_predictions_rank)

            return mrr, filtered_mrr

        else:
            raise ValueError("MRR evaluation is impossible with predictions from sphere embeddings, as they are unranked. To disable sphere embeddings, set the `sphere_embeddings` hyperparameter as false in the config file.")


    @property
    def median_rank(self) -> Tuple[float, float]:
        """
        Median rank of the true predictions, both unfiltered and filtered.

        The median rank is a robust summary of the rank distribution: unlike
        the mean rank, it is not skewed by a small number of very badly ranked
        triplets.

        Returns
        -------

        **median_rank** *(float)*
        : Median rank of the predictions

        **filtered_median_rank** *(float)*
        : Median rank filtered to remove predictions of true triplets.

        """
        median_rank = self.true_predictions_rank.float().median().item()
        filtered_median_rank = self.filtered_true_predictions_rank.float().median().item()

        return median_rank, filtered_median_rank


    def mean_reciprocal_rank_at_k(self,
                                  k: int = 10
                                  ) -> Tuple[float, float]:
        """
        Mean reciprocal rank at k (MRR@k), both unfiltered and filtered.

        MRR@k is the mean of 1/rank when the true prediction lies within the
        top-k candidates, and 0 otherwise.

        Arguments
        ---------

        **k** *(int, default to 10)*
        : Maximum rank taken into account; true predictions ranked beyond k
        contribute 0 to the mean.

        Returns
        -------

        **mrr_at_k** *(float)*
        : Mean of the reciprocal ranks of the true predictions, where true
        predictions ranked beyond k contribute 0.

        **filtered_mrr_at_k** *(float)*
        : Same, when ranking among filtered triplets.
        : True triplets that are not the target of the prediction are filtered out.

        """
        def _mrr_at_k(rank: Tensor) -> float:
            reciprocal = torch.where(rank <= k, rank.float() ** (-1), torch.zeros_like(rank.float()))
            return reciprocal.mean().item()

        mrr_at_k = _mrr_at_k(self.true_predictions_rank)
        filtered_mrr_at_k = _mrr_at_k(self.filtered_true_predictions_rank)

        return mrr_at_k, filtered_mrr_at_k


    def score_gap(self,
                  true_scores: Tensor,
                  best_other_unfiltered: Tensor,
                  best_other_filtered: Tensor
                  ) -> Tuple[float, float]:
        """
        Mean score gap between the true prediction and the best-ranked
        incorrect candidate, both unfiltered and filtered.

        For each evaluated triplet, the gap is

            ``gap = score(true) - max score among all other candidates``

        (unfiltered: other candidates include true triplets that are not the
        prediction target; filtered: those are masked out before taking the
        maximum). A larger positive gap means the model is more confident in the true
        triplet relative to the best wrong one.

        Arguments
        ---------

        **true_scores** *(torch.Tensor, dtype: torch.float, shape: [triplet_count])*
        : Score of the true triplet for each evaluated triplet.

        **best_other_unfiltered** *(torch.Tensor, dtype: torch.float, shape: [triplet_count])*
        : Score of the highest-scoring *other* candidate triplet for each
        evaluated triplet (unfiltered and filtered, in that order)

        Returns
        -------

        **score_gap** *(float)*
        : Mean of the (true - best incorrect candidate) score gaps,
        : without filtering of true non-target triplets.

        **filtered_score_gap** *(float)*
        : Mean of the (true - best incorrect candidate) score gaps,
        : where the maximum is taken after filtering out true non-target triplets.

        """
        true_scores = true_scores.detach().float()
        best_other_unfiltered = best_other_unfiltered.detach().float()
        best_other_filtered = best_other_filtered.detach().float()

        if best_other_unfiltered.shape[0] != true_scores.shape[0] or best_other_filtered.shape[0] != true_scores.shape[0]:
            raise ValueError(f"`true_scores` ({true_scores.shape[0]}), `best_other_unfiltered` ({best_other_unfiltered.shape[0]}) and `best_other_filtered` ({best_other_filtered.shape[0]}) must all have the same number of triplets.")

        score_gap = (true_scores - best_other_unfiltered).mean().item()
        filtered_score_gap = (true_scores - best_other_filtered).mean().item()

        return score_gap, filtered_score_gap


    def relative_rank(self,
                      candidate_count: int
                      ) -> Tuple[float, float]:
        """
        Relative rank: ranks normalized by the number of candidates.

        The relative rank of a true prediction is

            ``rank / candidate_count``

        in ``[1/candidate_count, 1]`` (1-indexed ranks), so it is comparable
        across datasets of very different size, while raw ranks (mean rank, median
        rank) are not. It is the standard normalization used to compare
        KGE results across benchmarks, and it is also the natural companion
        of `hit_at_k` when k is expressed as a fraction of the candidate pool.

        Arguments
        ---------

        **candidate_count** *(int)*
        : Number of candidates each true prediction was ranked among
        : (the node count for head/tail link prediction).

        Raises
        ------

        **ValueError**
        : If `candidate_count` is less than 1.

        Returns
        -------

        **relative_rank** *(float)*
        : Mean of `true_predictions_rank / candidate_count`.

        **filtered_relative_rank** *(float)*
        : Mean of `filtered_true_predictions_rank / candidate_count`.

        """
        if candidate_count < 1:
            raise ValueError(f"`candidate_count` must be >= 1, got {candidate_count}.")

        relative_rank = (self.true_predictions_rank.float() / candidate_count).mean().item()
        filtered_relative_rank = (self.filtered_true_predictions_rank.float() / candidate_count).mean().item()

        return relative_rank, filtered_relative_rank


    def to_dict(self,
                k_values: Tuple[int, ...] = (1, 3, 10),
                true_scores: Tensor | None = None,
                best_other_unfiltered: Tensor | None = None,
                best_other_filtered: Tensor | None = None,
                candidate_count: int | None = None
                ) -> Dict[str, float]:
        """
        Flatten all metrics of this `Predictions` object into a single
        dictionary.

        Optional metrics (`score_gap`, `relative_rank`) are only included
        when their extra arguments are given.

        Arguments
        ---------

        **k_values** *(Tuple[int, ...], default to (1, 3, 10))*
        : The k values used for `hit_at_k` and `mean_reciprocal_rank_at_k`.

        **true_scores** *(torch.Tensor, optional)*
        : If given (with `best_other_unfiltered` and `best_other_filtered`),
        : include the `score_gap` metrics. See `score_gap` for the expected shape.

        **best_other_unfiltered** *(torch.Tensor, optional)*
        : If given (with `true_scores`), include the `score_gap` metrics.
        : See `score_gap` for the expected shape.

        **best_other_filtered** *(torch.Tensor, optional)*
        : If given (with `true_scores`), include the `score_gap` metrics.
        : See `score_gap` for the expected shape.

        **candidate_count** *(int, optional)*
        : If given, include the `relative_rank` metrics.
        : See `relative_rank` for the expected meaning.

        Returns
        -------

        **metrics** *(Dict[str, float])*
        : Dictionary of metric names to values.

        """
        metrics: Dict[str, float] = {}

        mean_rank, filtered_mean_rank = self.mean_rank
        metrics["mean_rank"] = mean_rank
        metrics["filtered_mean_rank"] = filtered_mean_rank

        median_rank, filtered_median_rank = self.median_rank
        metrics["median_rank"] = median_rank
        metrics["filtered_median_rank"] = filtered_median_rank

        mrr, filtered_mrr = self.mrr
        metrics["mrr"] = mrr
        metrics["filtered_mrr"] = filtered_mrr

        for k in k_values:
            hit, filtered_hit = self.hit_at_k(k)
            metrics[f"hit_at_{k}"] = hit
            metrics[f"filtered_hit_at_{k}"] = filtered_hit

            mrr_at_k, filtered_mrr_at_k = self.mean_reciprocal_rank_at_k(k)
            metrics[f"mrr_at_{k}"] = mrr_at_k
            metrics[f"filtered_mrr_at_{k}"] = filtered_mrr_at_k

        if (true_scores is not None and best_other_unfiltered is not None and best_other_filtered is not None):
            score_gap, filtered_score_gap = self.score_gap(true_scores, best_other_unfiltered, best_other_filtered)
            metrics["score_gap"] = score_gap
            metrics["filtered_score_gap"] = filtered_score_gap

        if candidate_count is not None:
            relative_rank, filtered_relative_rank = self.relative_rank(candidate_count)
            metrics["relative_rank"] = relative_rank
            metrics["filtered_relative_rank"] = filtered_relative_rank

        return metrics



class TripletClassificationResults:
    """
    Object holding the results of a `TripletClassificationEvaluator.accuracy` call.

    Provides individual metric accessors (each returning a float in [0, 1])
    and a `to_dict` method for reporting.

    All metrics are computed from the per-triplet boolean classification
    outcomes:
    
    - **positive_correct** *(Tensor[bool])* : true triplets correctly accepted
    - **positive_incorrect** *(Tensor[bool])* : true triplets incorrectly rejected
    - **negative_correct** *(Tensor[bool])* : negative triplets correctly rejected
    - **negative_incorrect** *(Tensor[bool])* : negative triplets incorrectly accepted

    Arguments
    ---------

    **positive_correct** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of true triplets that were correctly accepted.

    **positive_incorrect** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of true triplets that were incorrectly rejected.

    **negative_correct** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of negative triplets that were correctly rejected.

    **negative_incorrect** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of negative triplets that were incorrectly accepted.

    """

    def __init__(self,
                positive_correct: Tensor,
                positive_incorrect: Tensor,
                negative_correct: Tensor,
                negative_incorrect: Tensor):
        self.positive_correct = positive_correct
        self.positive_incorrect = positive_incorrect
        self.negative_correct = negative_correct
        self.negative_incorrect = negative_incorrect

    # ---- basic counts ----

    @property
    def positive_count(self) -> int:
        """Total number of true triplets evaluated."""
        return int(self.positive_correct.numel())

    @property
    def negative_count(self) -> int:
        """Total number of negative triplets evaluated."""
        return int(self.negative_correct.numel())

    # ---- core metrics ----

    @property
    def accuracy(self) -> float:
        """
        Overall accuracy: proportion of all triplets (true + negative)
        correctly classified.
        """
        total = self.positive_count + self.negative_count
        if total == 0:
            return 0.0
        correct = int(self.positive_correct.sum().item() + self.negative_correct.sum().item())
        return correct / total

    @property
    def precision(self) -> float:
        """
        Precision: of all triplets the model accepted (score > threshold),
        what proportion are actually true?

        Accepted = true triplets accepted (positive_correct) + negative
        triplets accepted (negative_incorrect).
        """
        accepted = int(self.positive_correct.sum().item() + self.negative_incorrect.sum().item())
        if accepted == 0:
            return 0.0
        return int(self.positive_correct.sum().item()) / accepted

    @property
    def recall(self) -> float:
        """
        Recall (sensitivity / true positive rate): of all true triplets,
        what proportion did the model accept?
        """
        if self.positive_count == 0:
            return 0.0
        return int(self.positive_correct.sum().item()) / self.positive_count

    @property
    def specificity(self) -> float:
        """
        Specificity (true negative rate): of all negative triplets,
        what proportion did the model correctly reject?
        """
        if self.negative_count == 0:
            return 0.0
        return int(self.negative_correct.sum().item()) / self.negative_count

    @property
    def f1(self) -> float:
        """
        F1 score: harmonic mean of precision and recall.
        """
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def false_positive_rate(self) -> float:
        """
        False positive rate (1 - specificity): proportion of negative
        triplets incorrectly accepted.
        """
        return 1.0 - self.specificity

    @property
    def false_negative_rate(self) -> float:
        """
        False negative rate (1 - recall): proportion of true triplets
        incorrectly rejected.
        """
        return 1.0 - self.recall

    @property
    def balanced_accuracy(self) -> float:
        """
        Balanced accuracy: mean of recall (TPR) and specificity (TNR).
        Robust to class imbalance between true and negative triplets.
        """
        return (self.recall + self.specificity) / 2

    @property
    def accuracy_positive(self) -> float:
        """
        Proportion of true triplets correctly accepted (= recall).
        Provided as a separate name for clarity in reports.
        """
        return self.recall

    @property
    def accuracy_negative(self) -> float:
        """
        Proportion of negative triplets correctly rejected (= specificity).
        Provided as a separate name for clarity in reports.
        """
        return self.specificity

    # ---- reporting ----

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.4f}  "
            f"Precision: {self.precision:.4f}  "
            f"Recall: {self.recall:.4f}  "
            f"Specificity: {self.specificity:.4f}  "
            f"F1: {self.f1:.4f}  "
            f"Balanced Acc: {self.balanced_accuracy:.4f}  "
            f"FPR: {self.false_positive_rate:.4f}  "
            f"FNR: {self.false_negative_rate:.4f}  "
            f"({self.positive_count} positive, {self.negative_count} negative)"
        )

    def to_dict(self) -> Dict[str, float]:
        """
        Flatten all metrics into a dictionary for reporting or serialization.

        Returns
        -------

        **metrics** *(Dict[str, float])*
        : Dictionary of metric names to values.
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


    @property
    def median_rank(self) -> Tuple[float, float]:
        """
        Median rank of the true predictions, both unfiltered and filtered.

        The median rank is a robust summary of the rank distribution: unlike
        the mean rank, it is not skewed by a small number of very badly ranked
        triplets.

        Returns
        -------

        **median_rank** *(float)*
        : Median rank of the predictions

        **filtered_median_rank** *(float)*
        : Median rank filtered to remove predictions of true triplets.

        """
        median_rank = self.true_predictions_rank.float().median().item()
        filtered_median_rank = self.filtered_true_predictions_rank.float().median().item()

        return median_rank, filtered_median_rank


    def mean_reciprocal_rank_at_k(self,
                                  k: int = 10
                                  ) -> Tuple[float, float]:
        """
        Mean reciprocal rank at k (MRR@k), both unfiltered and filtered.

        MRR@k is the mean of 1/rank when the true prediction lies within the
        top-k candidates, and 0 otherwise.

        Arguments
        ---------

        **k** *(int, default to 10)*
        : Maximum rank taken into account; true predictions ranked beyond k
        contribute 0 to the mean.

        Returns
        -------

        **mrr_at_k** *(float)*
        : Mean of the reciprocal ranks of the true predictions, where true
        predictions ranked beyond k contribute 0.

        **filtered_mrr_at_k** *(float)*
        : Same, when ranking among filtered triplets.
        : True triplets that are not the target of the prediction are filtered out.

        """
        def _mrr_at_k(rank: Tensor) -> float:
            reciprocal = torch.where(rank <= k, rank.float() ** (-1), torch.zeros_like(rank.float()))
            return reciprocal.mean().item()

        mrr_at_k = _mrr_at_k(self.true_predictions_rank)
        filtered_mrr_at_k = _mrr_at_k(self.filtered_true_predictions_rank)

        return mrr_at_k, filtered_mrr_at_k


    def score_gap(self,
                  true_scores: Tensor,
                  best_other_unfiltered: Tensor,
                  best_other_filtered: Tensor
                  ) -> Tuple[float, float]:
        """
        Mean score gap between the true prediction and the best-ranked
        incorrect candidate, both unfiltered and filtered.

        For each evaluated triplet, the gap is

            ``gap = score(true) - max score among all other candidates``

        (unfiltered: other candidates include true triplets that are not the
        prediction target; filtered: those are masked out before taking the
        maximum). A larger positive gap means the model is more confident in the true
        triplet relative to the best wrong one.

        Arguments
        ---------

        **true_scores** *(torch.Tensor, dtype: torch.float, shape: [triplet_count])*
        : Score of the true triplet for each evaluated triplet.

        **best_other_unfiltered** *(torch.Tensor, dtype: torch.float, shape: [triplet_count])*
        : Score of the highest-scoring *other* candidate triplet for each
        evaluated triplet (unfiltered and filtered, in that order)

        Returns
        -------

        **score_gap** *(float)*
        : Mean of the (true - best incorrect candidate) score gaps,
        : without filtering of true non-target triplets.

        **filtered_score_gap** *(float)*
        : Mean of the (true - best incorrect candidate) score gaps,
        : where the maximum is taken after filtering out true non-target triplets.

        """
        true_scores = true_scores.detach().float()
        best_other_unfiltered = best_other_unfiltered.detach().float()
        best_other_filtered = best_other_filtered.detach().float()

        if best_other_unfiltered.shape[0] != true_scores.shape[0] or best_other_filtered.shape[0] != true_scores.shape[0]:
            raise ValueError(f"`true_scores` ({true_scores.shape[0]}), `best_other_unfiltered` ({best_other_unfiltered.shape[0]}) and `best_other_filtered` ({best_other_filtered.shape[0]}) must all have the same number of triplets.")

        score_gap = (true_scores - best_other_unfiltered).mean().item()
        filtered_score_gap = (true_scores - best_other_filtered).mean().item()

        return score_gap, filtered_score_gap


    def relative_rank(self,
                      candidate_count: int
                      ) -> Tuple[float, float]:
        """
        Relative rank: ranks normalized by the number of candidates.

        The relative rank of a true prediction is

            ``rank / candidate_count``

        in ``[1/candidate_count, 1]`` (1-indexed ranks), so it is comparable
        across datasets of very different size, while raw ranks (mean rank, median
        rank) are not. It is the standard normalization used to compare
        KGE results across benchmarks, and it is also the natural companion
        of `hit_at_k` when k is expressed as a fraction of the candidate pool.

        Arguments
        ---------

        **candidate_count** *(int)*
        : Number of candidates each true prediction was ranked among
        : (the node count for head/tail link prediction).

        Raises
        ------

        **ValueError**
        : If `candidate_count` is less than 1.

        Returns
        -------

        **relative_rank** *(float)*
        : Mean of `true_predictions_rank / candidate_count`.

        **filtered_relative_rank** *(float)*
        : Mean of `filtered_true_predictions_rank / candidate_count`.

        """
        if candidate_count < 1:
            raise ValueError(f"`candidate_count` must be >= 1, got {candidate_count}.")

        relative_rank = (self.true_predictions_rank.float() / candidate_count).mean().item()
        filtered_relative_rank = (self.filtered_true_predictions_rank.float() / candidate_count).mean().item()

        return relative_rank, filtered_relative_rank


    def to_dict(self,
                k_values: Tuple[int, ...] = (1, 3, 10),
                true_scores: Tensor | None = None,
                best_other_unfiltered: Tensor | None = None,
                best_other_filtered: Tensor | None = None,
                candidate_count: int | None = None
                ) -> Dict[str, float]:
        """
        Flatten all metrics of this `Predictions` object into a single
        dictionary.

        Optional metrics (`score_gap`, `relative_rank`) are only included
        when their extra arguments are given.

        Arguments
        ---------

        **k_values** *(Tuple[int, ...], default to (1, 3, 10))*
        : The k values used for `hit_at_k` and `mean_reciprocal_rank_at_k`.

        **true_scores** *(torch.Tensor, optional)*
        : If given (with `best_other_unfiltered` and `best_other_filtered`),
        : include the `score_gap` metrics. See `score_gap` for the expected shape.

        **best_other_unfiltered** *(torch.Tensor, optional)*
        : If given (with `true_scores`), include the `score_gap` metrics.
        : See `score_gap` for the expected shape.

        **best_other_filtered** *(torch.Tensor, optional)*
        : If given (with `true_scores`), include the `score_gap` metrics.
        : See `score_gap` for the expected shape.

        **candidate_count** *(int, optional)*
        : If given, include the `relative_rank` metrics.
        : See `relative_rank` for the expected meaning.

        Returns
        -------

        **metrics** *(Dict[str, float])*
        : Dictionary of metric names to values.

        """
        metrics: Dict[str, float] = {}

        mean_rank, filtered_mean_rank = self.mean_rank
        metrics["mean_rank"] = mean_rank
        metrics["filtered_mean_rank"] = filtered_mean_rank

        median_rank, filtered_median_rank = self.median_rank
        metrics["median_rank"] = median_rank
        metrics["filtered_median_rank"] = filtered_median_rank

        mrr, filtered_mrr = self.mrr
        metrics["mrr"] = mrr
        metrics["filtered_mrr"] = filtered_mrr

        for k in k_values:
            hit, filtered_hit = self.hit_at_k(k)
            metrics[f"hit_at_{k}"] = hit
            metrics[f"filtered_hit_at_{k}"] = filtered_hit

            mrr_at_k, filtered_mrr_at_k = self.mean_reciprocal_rank_at_k(k)
            metrics[f"mrr_at_{k}"] = mrr_at_k
            metrics[f"filtered_mrr_at_{k}"] = filtered_mrr_at_k

        if (true_scores is not None and best_other_unfiltered is not None and best_other_filtered is not None):
            score_gap, filtered_score_gap = self.score_gap(true_scores, best_other_unfiltered, best_other_filtered)
            metrics["score_gap"] = score_gap
            metrics["filtered_score_gap"] = filtered_score_gap

        if candidate_count is not None:
            relative_rank, filtered_relative_rank = self.relative_rank(candidate_count)
            metrics["relative_rank"] = relative_rank
            metrics["filtered_relative_rank"] = filtered_relative_rank

        return metrics



class TripletClassificationResults:
    """
    Object holding the results of a `TripletClassificationEvaluator.accuracy` call.

    Provides individual metric accessors (each returning a float in [0, 1])
    and a `to_dict` method for reporting.

    All metrics are computed from the per-triplet boolean classification
    outcomes:
    
    - **positive_correct** *(Tensor[bool])* : true triplets correctly accepted
    - **positive_incorrect** *(Tensor[bool])* : true triplets incorrectly rejected
    - **negative_correct** *(Tensor[bool])* : negative triplets correctly rejected
    - **negative_incorrect** *(Tensor[bool])* : negative triplets incorrectly accepted

    Arguments
    ---------

    **positive_correct** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of true triplets that were correctly accepted.

    **positive_incorrect** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of true triplets that were incorrectly rejected.

    **negative_correct** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of negative triplets that were correctly rejected.

    **negative_incorrect** *(torch.Tensor, dtype: torch.bool, shape: [triplet_count])*
    : Boolean mask of negative triplets that were incorrectly accepted.

    """

    def __init__(self,
                positive_correct: Tensor,
                positive_incorrect: Tensor,
                negative_correct: Tensor,
                negative_incorrect: Tensor):
        self.positive_correct = positive_correct
        self.positive_incorrect = positive_incorrect
        self.negative_correct = negative_correct
        self.negative_incorrect = negative_incorrect

    # ---- basic counts ----

    @property
    def positive_count(self) -> int:
        """Total number of true triplets evaluated."""
        return int(self.positive_correct.numel())

    @property
    def negative_count(self) -> int:
        """Total number of negative triplets evaluated."""
        return int(self.negative_correct.numel())

    # ---- core metrics ----

    @property
    def accuracy(self) -> float:
        """
        Overall accuracy: proportion of all triplets (true + negative)
        correctly classified.
        """
        total = self.positive_count + self.negative_count
        if total == 0:
            return 0.0
        correct = int(self.positive_correct.sum().item() + self.negative_correct.sum().item())
        return correct / total

    @property
    def precision(self) -> float:
        """
        Precision: of all triplets the model accepted (score > threshold),
        what proportion are actually true?

        Accepted = true triplets accepted (positive_correct) + negative
        triplets accepted (negative_incorrect).
        """
        accepted = int(self.positive_correct.sum().item() + self.negative_incorrect.sum().item())
        if accepted == 0:
            return 0.0
        return int(self.positive_correct.sum().item()) / accepted

    @property
    def recall(self) -> float:
        """
        Recall (sensitivity / true positive rate): of all true triplets,
        what proportion did the model accept?
        """
        if self.positive_count == 0:
            return 0.0
        return int(self.positive_correct.sum().item()) / self.positive_count

    @property
    def specificity(self) -> float:
        """
        Specificity (true negative rate): of all negative triplets,
        what proportion did the model correctly reject?
        """
        if self.negative_count == 0:
            return 0.0
        return int(self.negative_correct.sum().item()) / self.negative_count

    @property
    def f1(self) -> float:
        """
        F1 score: harmonic mean of precision and recall.
        """
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def false_positive_rate(self) -> float:
        """
        False positive rate (1 - specificity): proportion of negative
        triplets incorrectly accepted.
        """
        return 1.0 - self.specificity

    @property
    def false_negative_rate(self) -> float:
        """
        False negative rate (1 - recall): proportion of true triplets
        incorrectly rejected.
        """
        return 1.0 - self.recall

    @property
    def balanced_accuracy(self) -> float:
        """
        Balanced accuracy: mean of recall (TPR) and specificity (TNR).
        Robust to class imbalance between true and negative triplets.
        """
        return (self.recall + self.specificity) / 2

    @property
    def accuracy_positive(self) -> float:
        """
        Proportion of true triplets correctly accepted (= recall).
        Provided as a separate name for clarity in reports.
        """
        return self.recall

    @property
    def accuracy_negative(self) -> float:
        """
        Proportion of negative triplets correctly rejected (= specificity).
        Provided as a separate name for clarity in reports.
        """
        return self.specificity

    # ---- reporting ----

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.4f}  "
            f"Precision: {self.precision:.4f}  "
            f"Recall: {self.recall:.4f}  "
            f"Specificity: {self.specificity:.4f}  "
            f"F1: {self.f1:.4f}  "
            f"Balanced Acc: {self.balanced_accuracy:.4f}  "
            f"FPR: {self.false_positive_rate:.4f}  "
            f"FNR: {self.false_negative_rate:.4f}  "
            f"({self.positive_count} positive, {self.negative_count} negative)"
        )

    def to_dict(self) -> Dict[str, float]:
        """
        Flatten all metrics into a dictionary for reporting or serialization.

        Returns
        -------

        **metrics** *(Dict[str, float])*
        : Dictionary of metric names to values.
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }



class LinkPredictionEvaluator:
    def __init__(self, graphindices: Tensor, embedding_dimensions: int):
        """
        Evaluate performance of given embedding using link prediction method.

        References
        ----------
        
        * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
            and Oksana Yakhnenko.
        
            `Translating Embeddings for Modeling Multi-relational Data.`
        
            <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>
        
            In Advances in Neural Information Processing Systems 26, pages 2787–2795. 2013.

        Arguments
        ---------
        
        **graphindices** *(torch.Tensor)*
        : Tensor of shape [4, triplet_count] containing every true triplet.
        
        **embedding_dimensions** *(int)*
        : Dimensions of embeddings.

        Attributes
        ----------
        
        **graphindices** *(torch.Tensor)*
        : Tensor of shape [4, triplet_count] containing every true triplet.
        
        **embedding_dimensions** *(int)*
        : Dimensions of embeddings.
        
        **generated_embeddings** *(bool)*
        : Indicate whether `generate_evaluation_embeddings` has already been called.
        
        **evaluated** *(bool)*
        : Indicate whether the method LinkPredictionEvaluator.evaluate has already
        been called.
        
        **rank_true_heads** *(torch.Tensor, shape: [triplet_count], dtype: torch.int)*
        : For each fact, this is the rank of the true head when all nodes 
        are ranked as possible replacement of the head node. They are 
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
        
        **rank_true_tails** *(torch.Tensor, shape: [triplet_count], dtype: torch.int)*
        : For each fact, this is the rank of the true tail when all nodes 
        are ranked as possible replacement of the tail node. They are 
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
        
        **filtered_rank_true_heads** *(torch.Tensor, shape: [triplet_count], dtype: torch.int)*
        : This is the same as the `rank_of_true_heads` when in the filtered 
        case. See referenced paper by Bordes et al. for more information.
        
        **filtered_rank_true_tails** *(torch.Tensor, shape: [triplet_count], dtype: torch.int)*
        : This is the same as the `rank_of_true_tails` when in the filtered 
        case. See referenced paper by Bordes et al. for more information.

        """
        self.graphindices = graphindices
        self.embedding_dimensions = embedding_dimensions
        self.evaluated = False
        self.generated_embeddings = False

    def reset(self):
        self.evaluated = False
        self.generated_embeddings = False

    def generate_evaluation_embeddings(self,
                                  batch_size: int,
                                  encoder: GNN,
                                  evaluation_subset: Subset[KnowledgeGraph],
                                  node_embedding_dimensions: int) -> None:
        with torch.no_grad():
            knowledge_graph = evaluation_subset.dataset
            while not isinstance(knowledge_graph, KnowledgeGraph):
                knowledge_graph = knowledge_graph.dataset
            device = knowledge_graph.embeddings.edge_embeddings.device

        self.evaluation_node_embeddings: torch.Tensor = torch.zeros((knowledge_graph.node_count,
                                                                    node_embedding_dimensions),
                                                                    device = device,
                                                                    dtype = torch.float)
    
        all_nodes = knowledge_graph.graphindices[:2].unique()
        try:
            input = knowledge_graph.get_encoder_input(
                seed_nodes=all_nodes,
                hop_count = encoder.layer_count
            )
            encoder_output = encoder(input.x_dict, input.edge_index)
            for node_type, indices in input.seed_mapping.items():
                node_type_index = knowledge_graph.node_type_to_index[node_type]
                node_type_mask = (knowledge_graph.node_types[all_nodes] == node_type_index)
                self.evaluation_node_embeddings[all_nodes[node_type_mask]] = encoder_output[node_type][indices]
            
        except torch.OutOfMemoryError:
            for i in range((len(all_nodes) // batch_size) + 1):
                seed_nodes: Tensor = all_nodes[i * batch_size: (i + 1) * batch_size]
                
                input = knowledge_graph.get_encoder_input(
                        seed_nodes = seed_nodes,
                        hop_count = encoder.layer_count
                        )
                        
                encoder_output: Dict[str, Tensor] = encoder(input.x_dict, input.edge_index)
                for node_type, indices in input.seed_mapping.items():
                    node_type_index = knowledge_graph.node_type_to_index[node_type]
                    node_type_mask = (knowledge_graph.node_types[seed_nodes] == node_type_index)
                    self.evaluation_node_embeddings[seed_nodes[node_type_mask]] = encoder_output[node_type][indices]
        
        self.generated_embeddings = True

    def evaluate(self,
                batch_size: int,
                encoder: GNN | None,
                decoder: BilinearDecoder | ConvolutionalDecoder | TranslationalDecoder,
                evaluated_subset: Subset[KnowledgeGraph],
                node_embeddings: nn.ParameterList,
                edge_embeddings: nn.Parameter,
                sphere_embeddings: bool = False,
                verbose: bool = True
                ) -> Tuple[Predictions, Predictions] | Tuple[Tensor, Tensor]:
        """
        Run the Link Prediction evaluation.

        Arguments
        ---------
        
        **batch_size** *(int)*
        : Size of the current batch.
        
        **encoder** *(GNN or None)*
        : Encoder model to embed the nodes.
        
        **decoder** *(BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder)*
        : Decoder model to evaluate.
        
        **evaluated_subset** *(Subset[KnowledgeGraph])*
        : Subset of the knowledge graph on which the evaluation will be done.
        
        **node_embeddings** *(nn.ParameterList, keyword-only)*
        : A list containing all embeddings for each node type.
        : position: node type index
        : values: tensors of shape (node_count, embedding_dimensions)
        
        **edge_embeddings** *(nn.Parameter, keyword-only)*
        : A tensor containing one embedding by edge type, of shape (edge_count, embedding_dimensions).
        
        **sphere_embeddings** *(bool, optional, default to False)*
        : If `True`, the decoder uses sphere embeddings (SpherE, Li et al. 2024),
        : and the returned `Predictions` are per-triplet positive predictions
        : (global indices of the triplets whose spheric score, computed by the
        : decoder's `sphere_score` on the true candidate, is >= 0), instead of
        : ranks. The decoder must be TransR or RotatE (with `sphere_embeddings`
        : enabled), as they are the only ones implementing `sphere_score`.

        **verbose** *(bool, default = True)*
        : Indicate whether a progress bar should be displayed during
        evaluation.
        
        Returns
        -------
        
        **head_predictions** *(Predictions)*
        : Predictions for heads.
        
        **tail_predictions** *(Predictions)*
        : Predictions for tails.

        Notes
        -----

        The returned `Predictions` also carry `true_scores`,
        `best_other_unfiltered` and `best_other_filtered` tensors
        (see `Predictions.score_gap`), so the score gap metric and the
        full `Predictions.to_dict` report are available out of the box.
        
        """
        with torch.no_grad():
            device = edge_embeddings.device

            knowledge_graph = evaluated_subset.dataset
            while not isinstance(knowledge_graph, KnowledgeGraph):
                knowledge_graph = knowledge_graph.dataset
                
            self.rank_true_heads = empty(size = (len(evaluated_subset),)).long().to(device)
            self.rank_true_tails = empty(size = (len(evaluated_subset),)).long().to(device)
            self.filtered_rank_true_heads = empty(size = (len(evaluated_subset),)).long().to(device)
            self.filtered_rank_true_tails = empty(size = (len(evaluated_subset),)).long().to(device)

            # Per-triplet scores of the true prediction and of the best
            # (incorrect) other candidate, for the `score_gap` metric of
            # `Predictions` (see `Predictions.score_gap` for the convention).
            self.true_score_heads = empty(size = (len(evaluated_subset),), dtype = torch.float).to(device)
            self.true_score_tails = empty(size = (len(evaluated_subset),), dtype = torch.float).to(device)
            self.best_other_score_heads_unfiltered = empty(size = (len(evaluated_subset),), dtype = torch.float).to(device)
            self.best_other_score_heads_filtered = empty(size = (len(evaluated_subset),), dtype = torch.float).to(device)
            self.best_other_score_tails_unfiltered = empty(size = (len(evaluated_subset),), dtype = torch.float).to(device)
            self.best_other_score_tails_filtered = empty(size = (len(evaluated_subset),), dtype = torch.float).to(device)

            dataloader = DataLoader(evaluated_subset, batch_size = batch_size)
            if decoder is not None and hasattr(decoder,"embedding_spaces"):
                encoder_node_embedding_dimensions: int = self.embedding_dimensions * decoder.embedding_spaces
            else:
                encoder_node_embedding_dimensions: int = self.embedding_dimensions

            if sphere_embeddings:
                # SpherE (Li et al. 2024): per-triplet positive predictions — the
                # decoder's spheric score of the true candidate is >= 0 (i.e. the
                # translated sphere of the head and the sphere of the tail
                # overlap). Global triplet indices accumulated over all batches.
                sphere_positive_heads = []
                sphere_positive_tails = []

            # Aggregate information for all nodes
            if encoder is not None and not self.generated_embeddings:
                self.generate_evaluation_embeddings(batch_size, encoder, evaluated_subset, encoder_node_embedding_dimensions)
            else:
                # Concatenate the embeddings of all node types (in node_type_to_global
                # order) so that global node indices can be used directly.
                self.evaluation_node_embeddings = torch.cat([embeddings.data for embeddings in node_embeddings], dim=0)

            for i, batch in tqdm(enumerate(dataloader),
                                total = len(dataloader),
                                unit = "batch",
                                disable = (not verbose),
                                desc = "Link prediction evaluation"):
                batch: Tensor = batch.T.to(device)
                head_index, tail_index, edge_index = batch[0], batch[1], batch[2]

                head_embeddings, tail_embeddings, inference_edge_embeddings, candidates = decoder.inference_prepare_candidates(head_indices = head_index, 
                                                                                                                    tail_indices = tail_index, 
                                                                                                                    edge_indices = edge_index, 
                                                                                                                    node_embeddings = self.evaluation_node_embeddings, 
                                                                                                                    edge_embeddings = edge_embeddings,
                                                                                                                    node_inference = True)

                scores = decoder.inference_score(
                    head_embeddings = head_embeddings, 
                    tail_embeddings = candidates, 
                    edge_embeddings = inference_edge_embeddings
                    )

                if sphere_embeddings:
                    arange = torch.arange(head_index.shape[0], device = device)
                    batch_sphere_score = decoder.sphere_score(head_indices = head_index,
                                                              tail_indices = tail_index,
                                                              dissimilarity_score = -scores[arange, tail_index])
                    sphere_positive_tails.append((arange[batch_sphere_score >= 0] + i * batch_size).cpu())

                filtered_scores = filter_scores(
                    scores = scores, 
                    graphindices = self.graphindices.to(device),
                    missing = "tail",
                    first_index = head_index,
                    second_index = edge_index,
                    true_index = tail_index
                )
                batch_start = i * batch_size
                actual_batch_size = head_index.shape[0]
                batch_end = batch_start + actual_batch_size

                self.rank_true_tails[batch_start: batch_end] = get_rank(scores, tail_index).detach()
                self.filtered_rank_true_tails[batch_start: batch_end] = get_rank(filtered_scores, tail_index).detach()

                # Score gap: true score minus best other (unfiltered) candidate
                arange = torch.arange(actual_batch_size, device = device)
                true_scores_batch = scores[arange, tail_index].detach()
                masked_scores = scores.clone()
                masked_scores[arange, tail_index] = - float('Inf')
                top_other_scores = masked_scores.max(dim = 1).values.detach()
                self.true_score_tails[batch_start: batch_end] = true_scores_batch
                self.best_other_score_tails_unfiltered[batch_start: batch_end] = top_other_scores
                self.best_other_score_tails_filtered[batch_start: batch_end] = filtered_scores.max(dim = 1).values.detach()

                scores = decoder.inference_score(
                    head_embeddings = candidates,
                    tail_embeddings = tail_embeddings,
                    edge_embeddings = inference_edge_embeddings)

                if sphere_embeddings:
                    arange = torch.arange(head_index.shape[0], device = device)
                    batch_sphere_score = decoder.sphere_score(head_indices = head_index,
                                                              tail_indices = tail_index,
                                                              dissimilarity_score = -scores[arange, head_index])
                    sphere_positive_heads.append((arange[batch_sphere_score >= 0] + i * batch_size).cpu())

                filtered_scores = filter_scores(
                    scores = scores, 
                    graphindices = self.graphindices.to(device),
                    missing = "head",
                    first_index = tail_index,
                    second_index = edge_index,
                    true_index = head_index
                )
                self.rank_true_heads[batch_start: batch_end] = get_rank(scores, head_index).detach()
                self.filtered_rank_true_heads[batch_start: batch_end] = get_rank(filtered_scores, head_index).detach()

                # Score gap: true score minus best other (unfiltered) candidate
                true_scores_batch = scores[arange, head_index].detach()
                masked_scores = scores.clone()
                masked_scores[arange, head_index] = - float('Inf')
                top_other_scores = masked_scores.max(dim = 1).values.detach()
                self.true_score_heads[batch_start: batch_end] = true_scores_batch
                self.best_other_score_heads_unfiltered[batch_start: batch_end] = top_other_scores
                self.best_other_score_heads_filtered[batch_start: batch_end] = filtered_scores.max(dim = 1).values.detach()

            self.evaluated = True

            if sphere_embeddings:
                # For sphere embeddings: predictions are per-triplet booleans, so
                # the rank-based Predictions fields stay None (see `Predictions.__str__`).
                # `head_predictions` / `tail_predictions` hold the global indices of
                # the triplets predicted positive (spheric score of the true candidate >= 0).
                tail_predictions = torch.cat(sphere_positive_tails) if sphere_positive_tails else torch.empty(0).long().to(device)
                head_predictions = torch.cat(sphere_positive_heads) if sphere_positive_heads else torch.empty(0).long().to(device)

                # Make Predictions object
                return Predictions(sphere_predictions = head_predictions), Predictions(sphere_predictions = tail_predictions)

            head_predictions = Predictions(self.rank_true_heads.cpu(), self.filtered_rank_true_heads.cpu())
            tail_predictions = Predictions(self.rank_true_tails.cpu(), self.filtered_rank_true_tails.cpu())

            # Attach the per-triplet scores so that `Predictions.score_gap`
            # and `Predictions.to_dict` can report the confidence margin.
            head_predictions.true_scores = self.true_score_heads.cpu()
            head_predictions.best_other_unfiltered = self.best_other_score_heads_unfiltered.cpu()
            head_predictions.best_other_filtered = self.best_other_score_heads_filtered.cpu()
            tail_predictions.true_scores = self.true_score_tails.cpu()
            tail_predictions.best_other_unfiltered = self.best_other_score_tails_unfiltered.cpu()
            tail_predictions.best_other_filtered = self.best_other_score_tails_filtered.cpu()

            return head_predictions, tail_predictions



class TripletClassificationEvaluator:
    def __init__(self,
                knowledge_graph: KnowledgeGraph,
                *,
                decoder: BilinearDecoder | ConvolutionalDecoder | TranslationalDecoder,
                encoder: GNN | None = None,
                normalizer: "nn.Module | None" = None):
        """
        Evaluates performance of given embedding using triplet classification 
        method.

        References
        ----------
        
        * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
            
            `Reasoning With Neural Tensor Networks for Knowledge Base Completion.`
            
            <https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf>
            
            In Advances in Neural Information Processing Systems 26, pages 926-934. 2013.

        Arguments
        ---------
        
        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph to evaluate.
        
        **encoder** *(GNN or None, optional)*
        : Encoder model to embed the nodes, if applicable.
        
        **decoder** *(BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder)*
        : Decoder model to evaluate.
        
        **normalizer** *(nn.Module or None, optional)*
        : Optional normalizer to apply to embeddings before scoring.

        Attributes
        ----------
        
        **knowledge_graph** *(KnowledgeGraph)*
        : Knowledge graph to evaluate.
        
        **encoder** *(GNN or None)*
        : Encoder model.
        
        **decoder** *(decoder instance or None)*
        : Decoder model.
        
        **normalizer** *(nn.Module or None)*
        : Normalizer module, if any.
        
        **device** *(torch.device)*
        : Device to run evaluation on.
        
        **evaluated** *(bool, default to False)*
        : Indicate whether the `evaluate` function has already been called.
        
        **thresholds** *(torch.Tensor)*
        : Value of the thresholds for the scoring function to consider a 
        triplet as true. It is defined by calling the `evaluate` method.
        
        **sampler** *(kgate.samplers.PositionalNegativeSampler)*
        : Negative sampler used to generate the negative samples.

        """
        self.knowledge_graph = knowledge_graph
        self.encoder = encoder
        self.decoder = decoder
        self.normalizer = normalizer

        # Determine device from embeddings if available
        if knowledge_graph.embeddings is not None:
            try:
                self.device = knowledge_graph.embeddings.edge_embeddings.device
            except AttributeError:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.evaluated = False
        self.thresholds = None

        # PositionalNegativeSampler specifically as done in TorchKGE
        # following the original paper: https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
        self.sampler = PositionalNegativeSampler(knowledge_graph)

        # Cache for node embeddings (computed once, reused across batches)
        self._cached_node_embeddings: Tensor | None = None

    def reset(self):
        self.evaluated = False
        self._cached_node_embeddings = None

    def _get_node_embeddings(self, batch_size: int) -> Tensor:
        """
        Get node embeddings, running the encoder if present.
        Results are cached to avoid recomputation across batches.

        Returns
        -------

        **node_embeddings** *(torch.Tensor, shape: [node_count, embedding_dimensions])*
        : Concatenated node embeddings for all node types.
        """
        if self._cached_node_embeddings is not None:
            return self._cached_node_embeddings

        kg = self.knowledge_graph
        with torch.no_grad():
            if self.encoder is not None:
                # Run the encoder on the full graph (like LinkPredictionEvaluator)
                all_nodes = kg.graphindices[:2].unique()
                input = kg.get_encoder_input(
                    seed_nodes=all_nodes,
                    hop_count=self.encoder.layer_count
                )
                encoder_output = self.encoder(input.x_dict, input.edge_index)
                node_embedding_dim = encoder_output[list(encoder_output.keys())[0]].shape[1]
                node_embeddings = torch.zeros(
                    (kg.node_count, node_embedding_dim),
                    device=self.device,
                    dtype=torch.float
                )
                for node_type, indices in input.seed_mapping.items():
                    node_type_index = kg.node_type_to_index[node_type]
                    node_type_mask = (kg.node_types[all_nodes] == node_type_index)
                    node_embeddings[all_nodes[node_type_mask]] = encoder_output[node_type][indices]
            else:
                # Concatenate the embeddings of all node types
                node_embeddings = torch.cat(
                    [embeddings.data for embeddings in kg.node_embeddings], dim=0
                )

        self._cached_node_embeddings = node_embeddings
        return node_embeddings

    def _score_triplets(self,
                        heads: Tensor,
                        tails: Tensor,
                        edges: Tensor,
                        batch_size: int) -> Tensor:
        """
        Score triplets using the decoder, with optional encoder and normalizer.

        Arguments
        ---------

        **heads** *(torch.Tensor, dtype: torch.long, shape: [triplet_count])*
        : Head indices.

        **tails** *(torch.Tensor, dtype: torch.long, shape: [triplet_count])*
        : Tail indices.

        **edges** *(torch.Tensor, dtype: torch.long, shape: [triplet_count])*
        : Edge indices.

        **batch_size** *(int)*
        : Batch size for processing.

        Returns
        -------

        **scores** *(torch.Tensor, dtype: torch.float, shape: [triplet_count])*
        : Scores for each triplet.
        """
        node_embeddings = self._get_node_embeddings(batch_size)
        edge_embeddings = self.knowledge_graph.edge_embeddings

        scores_list = []
        total = heads.shape[0]
        with torch.no_grad():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                h = heads[start:end].to(self.device)
                t = tails[start:end].to(self.device)
                e = edges[start:end].to(self.device)

                head_emb = node_embeddings[h]
                tail_emb = node_embeddings[t]
                edge_emb = edge_embeddings[e]

                if self.normalizer is not None:
                    head_emb, tail_emb, edge_emb = self.normalizer(
                        head_embeddings=head_emb,
                        tail_embeddings=tail_emb,
                        edge_embeddings=edge_emb
                    )

                batch_scores = self.decoder.score(
                    head_embeddings=head_emb,
                    tail_embeddings=tail_emb,
                    edge_embeddings=edge_emb,
                    head_indices=h,
                    tail_indices=t,
                    edge_indices=e,
                )
                scores_list.append(batch_scores)

        return cat(scores_list, dim=0)

    def evaluate(self,
                batch_size: int,
                knowledge_graph_subset: Subset[KnowledgeGraph]) -> None:
        """
        Find edge thresholds using the validation set. As described in 
        the paper by Socher et al., for an edge, the threshold is a value t 
        such that if the score of a triplet is larger than t, the triplet is correct. 
        If an edge is not present in any triplet of the validation set, then 
        the largest value score of all negative samples is used as threshold.

        Arguments
        ---------
        
        **batch_size** *(int)*
        : Size of the current batch.
        
        **knowledge_graph_subset** *(Subset[KnowledgeGraph])*
        : Knowledge graph subset on which the evaluation will be done.
        
        """
        with torch.no_grad():
            graphindices = knowledge_graph_subset[:]
            heads = graphindices[0]
            tails = graphindices[1]
            edge_indices = graphindices[2]

            # Corrupt triplets using the sampler's corrupt_batch
            batch_tensor = graphindices.to(self.device)
            negative_batch = self.sampler.corrupt_batch(batch_tensor)
            negative_heads = negative_batch[0]
            negative_tails = negative_batch[1]

            # Score both positive and negative triplets
            positive_scores = self._score_triplets(heads, tails, edge_indices, batch_size)
            negative_scores = self._score_triplets(negative_heads, negative_tails, edge_indices, batch_size)

            # Compute per-edge thresholds: max negative score for each edge
            self.thresholds = torch.full(
                (self.knowledge_graph.edge_count,),
                float('-inf'),
                device=self.device
            )

            # For each edge present in the subset, set threshold to max negative score
            unique_edges = edge_indices.unique()
            for edge_idx in unique_edges:
                edge_mask = (edge_indices == edge_idx)
                self.thresholds[edge_idx] = negative_scores[edge_mask].max()

            # For edges not present, use the global max negative score
            if len(unique_edges) < self.knowledge_graph.edge_count:
                global_max = negative_scores.max()
                missing_mask = torch.full(
                    (self.knowledge_graph.edge_count,),
                    float('-inf')
                ).to(self.device)
                missing_mask[unique_edges] = float('inf')
                self.thresholds[missing_mask == float('-inf')] = global_max

            self.evaluated = True
            self.thresholds = self.thresholds.detach()

    def accuracy(self,
                batch_size: int,
                kg_to_evaluate: Subset[KnowledgeGraph]
                ) -> TripletClassificationResults:
        """
        Triplet Classification evaluation: classifies both the true triplets
        (which should be accepted) and one positionally-sampled negative per
        true triplet (which should be rejected), using the thresholds learned
        from the validation set.

        If the evaluator has not been evaluated yet, it must be evaluated
        separately on the validation set first using the `evaluate` method.

        Arguments
        ---------
        
        **batch_size** *(int)*
        : Size of the current batch.
        
        **kg_to_evaluate** *(Subset[KnowledgeGraph])*
        : Knowledge graph subset to be evaluated.

        Returns
        -------
        
        **results** *(TripletClassificationResults)*
        : Object containing all classification metrics (accuracy, precision,
        : recall, specificity, F1, balanced accuracy, FPR, FNR).

        Raises
        ------

        **RuntimeError**
        : If `evaluate` has not been called yet (thresholds not set).

        """
        if not self.evaluated:
            raise RuntimeError(
                "The evaluator has not been evaluated yet. "
                "Call `evaluate(batch_size, validation_subset)` first to "
                "compute the thresholds from the validation set."
            )

        graphindices = kg_to_evaluate[:]
        heads = graphindices[0]
        tails = graphindices[1]
        edge_indices = graphindices[2]

        # Generate negatives
        batch_tensor = graphindices.to(self.device)
        negative_batch = self.sampler.corrupt_batch(batch_tensor)
        negative_heads = negative_batch[0]
        negative_tails = negative_batch[1]

        # Score
        scores = self._score_triplets(heads, tails, edge_indices, batch_size)
        negative_scores = self._score_triplets(negative_heads, negative_tails, edge_indices, batch_size)

        # Ensure thresholds are on the same device
        self.thresholds = self.thresholds.to(self.device)
        thresholds = self.thresholds[edge_indices]

        # Classification
        positive_correct = (scores > thresholds)
        negative_correct = (negative_scores < thresholds)
        positive_incorrect = ~positive_correct
        negative_incorrect = ~negative_correct

        return TripletClassificationResults(
            positive_correct=positive_correct,
            positive_incorrect=positive_incorrect,
            negative_correct=negative_correct,
            negative_incorrect=negative_incorrect,
        )
