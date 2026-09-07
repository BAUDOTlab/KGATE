from typing import Literal
from torch import ones_like, zeros_like, tensor
from torch.nn import Module
import torch.nn.functional as F
from typing import Callable


class Loss(Module):
    """
    Composite loss that aggregates multiple loss terms into a single loss.

    Combines an arbitrary number of losses, each of which take as input
    positive and negative triplets scores, and sums their outputs to
    produce a single loss value.

    If you want to use a single loss, you can instanciate their module directly.

    Arguments
    ---------
    ***terms** *(list[torch.nn.Module])*
    : Loss modules to be combined. Each term must 
    : implement a ``forward`` (or be a function) accepting
    : ``positive_scores`` and ``negative_scores`` and returning a value.
    """

    def __init__(self, *terms: list[Module]):
        self.terms = terms

    def forward(self, positive_scores: tensor, negative_scores: tensor):
        """
        Compute the composite loss across all registered terms.

        Arguments
        ---------
        **positive_scores** *(torch.Tensor)*
        : Scores assigned by the model to positive triplets.

        **negative_scores** *(torch.Tensor)*
        : Scores assigned by the model to negative triplets.

        Returns
        -------
        **loss** *(torch.Tensor)*
        : Final loss as the sum of all loss terms.
        """
        return sum([term(positive_scores, negative_scores) for term in self.terms])

    def add_term(self, new_term: Module | Callable):
        """
        Register a loss term.

        Arguments
        ---------
        **new_term** *(torch.nn.Module | Callable)*
        : A new loss module or function to append to ``self.terms``. It
        must accept ``positive_scores`` and ``negative_scores``.
        """
        self.terms.append(new_term)


class MarginLoss(Module):
    """
    Pairwise margin ranking loss for knowledge graph embedding.

    Encourages the score of positive triplets to exceed the score of
    negative triplets by at least ``margin``.

    For each element, the function applied is 
    ``max(0, -(positive_score - negative_score) + margin)``

    Arguments
    ---------
    **margin** *(int)*
    : The minimum margin by which positive scores should exceed
    negative scores.

    **reduction** *(Literal[, "sum", "mean"])*
    : Specifies the reduction to apply to the output of the margin
    ranking loss. `sum` will sum the output and `mean` will sum the 
    output before dividing it by the number of elements.
    """

    def __init__(self, margin: int, reduction: Literal["sum", "mean"]):
        self.margin = margin
        self.reduction = reduction

    def forward(self, positive_scores: tensor, negative_scores: tensor):
        """
        Compute the margin ranking loss between positive and negative scores.

        Arguments
        ---------
        **positive_scores** *(torch.Tensor)*
        : Scores assigned by the model to positive triplets.

        **negative_scores** *(torch.Tensor)*
        : Scores assigned by the model to negative triplets.

        Returns
        -------
        **loss** *(torch.Tensor)*
        : The margin ranking loss computed between
        ``positive_scores`` and ``negative_scores``
        """
        return F.margin_ranking_loss(positive_scores,
                                      negative_scores,
                                      ones_like(positive_scores),
                                      margin=self.margin,
                                      reduction=self.reduction)


class BinaryCrossEntropyLoss(Module):
    """
    Binary cross-entropy loss for knowledge graph embeddings.

    Treats link prediction as a binary classification problem: positive
    triples are pushed toward a target label of 1 and negative triples
    toward a target label of 0, after passing raw scores through a
    sigmoid.

    Arguments
    ---------
    **reduction** *(Literal["sum", "mean"])*
    : Specifies the reduction to apply to the output of each binary
    cross-entropy term.`sum` will sum the output and `mean` will sum 
    the output before dividing it by the number of elements.
    """

    def __init__(self, reduction: Literal["sum", "mean"]):
        self.reduction = reduction

    def forward(self, positive_scores: tensor, negative_scores: tensor):
        """
        Compute the combined binary cross-entropy loss for positive and negative scores.

        Arguments
        ---------
        **positive_scores** *(torch.Tensor)*
        : Raw scores assigned by the model to positive triplets,
        to be compared against a target label of 1 after applying a
        sigmoid.

        **negative_scores** *(torch.Tensor)*
        : Raw scores assigned by the model to negative
        triples, to be compared against a target label of 0 after
        applying a sigmoid.

        Returns
        -------
        **loss** *(torch.Tensor)*
        : Sum of the binary cross-entropy loss on positive scores
         and negative scores
        """
        return F.binary_cross_entropy(
            F.sigmoid(positive_scores),
            ones_like(positive_scores),
            reduction=self.reduction
        ) + F.binary_cross_entropy(
            F.sigmoid(negative_scores),
            zeros_like(negative_scores),
            reduction=self.reduction
        )