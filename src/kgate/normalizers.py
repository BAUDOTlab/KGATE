"""
Embedding normalization module for KGATE.

The `Normalizer` is a model component, initialized by the Architect after the
decoder (as soon as the embeddings it normalizes exist). It is given a set of
embeddings to normalize and the function to apply to them. It is applied
between the encoder and the decoder step, so it is done batchwise: the
function is applied to the (encoder output) embeddings of the current batch
before they are scored by the decoder (see `Architect.scoring_function`), as
a differentiable part of the forward pass.

When there is no encoder, the embeddings between the encoder and the decoder
are the node and edge embeddings themselves. Normalizing them batchwise would
recompute the same whole-graph normalization on every batch, so in that case
the embeddings are normalized once, over the whole graph, at the beginning of
each epoch (see the `EPOCH_STARTED` trainer hook and
`Architect.apply_normalizer`), and again before an evaluation or an export of
the embeddings (see `Architect.get_embeddings`). Note that, between two
epoch-start normalizations, the optimizer keeps updating the embeddings, so
the decoder may see embeddings whose norms are not exactly unit anymore: this
is the (small) approximation made in exchange for not recomputing the same
whole-graph normalization on every batch.

This module gathers what the decoders used to do on their own in their `score`
method (e.g. RESCAL, DistMult, TransE, TransH, TransR and TransD
L2-normalizing their head and tail embeddings): the same operations are now
available as normalizer functions, selected through the configuration
(`[model.normalizer]`) and applied by the Architect instead of being
hardcoded in each decoder.

@author: Benjamin Loire <benjamin.loire@univ-amu.fr>
"""
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import normalize
from typing import Callable, Iterable, Literal


class Normalizer:
    """
    Normalizer for KGATE.

    The normalizer runs before each gradient step on the embeddings to shape them
    in order to avoid gradient explosion. The normalizer runs before the decoder, and 
    after the encoder if there is one.

    Arguments
    ---------
    
    **training_normalization** *(Callable[[torch.Tensor], torch.Tensor], optional, keyword-only)*
    : The function to apply to each embedding.
    : It must accept a tensor and return a tensor of the same shape.

    **initial_normalization** *(Callable[[torch.Tensor], torch.Tensor], optional, keyword-only)*
    : The function to apply to each embedding before the beginning of the training.
    : It must accept a tensor and return a tensor of the same shape.
    : If no function is given, will default to the same as `training_normalization`
    
    #TODO complete
    Raises
    ------
    
    **TypeError**
    : If `func` is not callable, or `params` is not an iterable of
      `torch.Tensor`.
    
    **ValueError**
    : If neither the node nor the edge embeddings are normalized.
    
    """
    def __init__(self,
                *,
                initial_normalization: Callable[[Tensor], Tensor] = None,
                training_normalization: Callable[[Tensor], Tensor] = lambda x: x,
                initial_targets: Literal["node", "edge", "all"] = "all",
                training_targets: Literal["node", "edge", "all"] = "all"):
        if not callable(training_normalization):
            raise TypeError(f"The normalizer functions must be callable, but got {type(training_normalization)}.")
        
        self.initial_normalization = initial_normalization or training_normalization
        self.training_normalization = training_normalization

        self.initial_targets = initial_targets
        self.training_targets = training_targets

    @property
    def nodes(self) -> bool:
        """Whether the normalizer targets the nodes during the training."""
        return self.training_targets in ["node", "all"]
    
    @property
    def edges(self) -> bool:
        """Whether the normalizer targets the edges during the training."""
        return self.training_targets in ["edge", "all"]

    def __call__(self,
                 *,
                 head_embeddings: Tensor,
                 tail_embeddings: Tensor,
                 edge_embeddings: Tensor
                 ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Apply the training normalizer function to the embeddings given as arguments, and
        return the normalized ones.

        Arguments
        ---------
        
        **head_embeddings** *(torch.Tensor, shape: [batch_size, dimensions], keyword-only)*
        : The head embeddings of the batch.
        
        **tail_embeddings** *(torch.Tensor, shape: [batch_size, dimensions], keyword-only)*
        : The tail embeddings of the batch.
        
        **edge_embeddings** *(torch.Tensor, shape: [batch_size, dimensions], keyword-only)*
        : The edge embeddings of the batch.
        
        Returns
        -------
        
        **head_embeddings, tail_embeddings, edge_embeddings** *(tuple[torch.Tensor, torch.Tensor, torch.Tensor])*
        : The same embeddings, normalized where the normalizer is configured,
        : and left unchanged otherwise.
        
        """
        if self.nodes:
            head_embeddings = self.training_normalization(head_embeddings)
            tail_embeddings = self.training_normalization(tail_embeddings)
        if self.edges:
            edge_embeddings = self.training_normalization(edge_embeddings)

        return head_embeddings, tail_embeddings, edge_embeddings

    def initialize(self, node_embeddings: Parameter, edge_embeddings: Parameter) -> tuple[Tensor, Tensor]:
        """Apply the initial normalizer function to the initial targets
        
        Arguments
        ---------
        **node_embeddings** *(torch.Tensor, shape: [node_count, dimensions])*
        The graph node embeddings

        **edge_embeddings** *(torch.Tensor, shape: [edge_count, dimensions])*
        The graph edge embeddings

        **node_embeddings, edge_embeddings** *(tuple[torch.Tensor, torch.Tensor])*
        : The same embeddings, normalized where the normalizer is configured,
        : and left unchanged otherwise.
        """
        if self.initial_targets in ["node", "all"]:
            node_embeddings.data = self.initial_normalization(node_embeddings.data)
        if self.initial_targets in ["edge", "all"]:
            edge_embeddings.data = self.initial_normalization(edge_embeddings.data)

        return node_embeddings, edge_embeddings
    
    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"initial normalization: {getattr(self.initial_normalization, '__name__', repr(self.initial_normalization))}"
                f"training normalization:{getattr(self.training_normalization, '__name__', repr(self.training_normalization))}")


# ------------------------------------------------------------------------------------
# Builtin normalizer functions
# ------------------------------------------------------------------------------------
# These are the functions that can be selected by name in the configuration
# (`[model.normalizer] name`). They gather what the decoders used to do in
# their own `score` method:
# - `l2_normalize` is exactly the operation RESCAL, DistMult, TransE, TransH,
#   TransR and TransD applied to their head and tail embeddings (TransD also
#   to its edge embeddings), row-wise L2 normalization.


def l1_normalize(x: Tensor) -> Tensor:
    """
    Row-wise L1 normalization of the input tensor.
    
    Arguments
    ---------
    
    **x** *(torch.Tensor, shape: [count, dimensions])*
    : The tensor to normalize.
    
    Returns
    -------
    
    **x** *(torch.Tensor, shape: [count, dimensions])*
    : The tensor with each row of unit L1 norm.
    
    """
    return normalize(x, p = 1, dim = 1)


def l2_normalize(x: Tensor) -> Tensor:
    """
    Row-wise L2 normalization of the input tensor.
    
    This is the operation the RESCAL, DistMult, TransE, TransH, TransR and
    TransD decoders used to apply to their head and tail embeddings in their
    `score` method.
    
    Arguments
    ---------
    
    **x** *(torch.Tensor, shape: [count, dimensions])*
    : The tensor to normalize.
    
    Returns
    -------
    
    **x** *(torch.Tensor, shape: [count, dimensions])*
    : The tensor with each row of unit L2 norm.
    
    """
    return normalize(x, p = 2, dim = 1)


def normalize_embeddings(embedding: Tensor,
                         p: Literal[1, 2] = 2,
                         *,
                         squared: bool = False) -> Tensor:
    """
    Row-wise Lp normalization of the input tensor, optionally squared.

    General form of the builtin normalizer functions: `l1_normalize` and
    `l2_normalize` are the cases `p = 1` and `p = 2` with `squared = False`.
    
    Arguments
    ---------
    
    **embedding** *(torch.Tensor, shape: [count, dimensions])*
    : The tensor to normalize.
    
    **p** *(1 or 2, default to 2)*
    : The order of the norm.
    
    **squared** *(bool, default to False, keyword-only)*
    : If True, the normalized tensor is squared elementwise.
    
    Returns
    -------
    
    **embedding** *(torch.Tensor, shape: [count, dimensions])*
    : The tensor with each row of unit Lp norm, or of unit Lp norm squared
      elementwise if `squared` is True.
    
    """
    normalized = normalize(embedding, p, dim = 1)
    if squared:
        normalized = normalized**2

    return normalized


# The normalizer functions, indexed by the name used in the configuration.
NORMALIZER_FUNCTIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "L1": l1_normalize,
    "L2": l2_normalize,
}
