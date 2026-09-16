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

    A normalizer is given a set of embeddings to normalize and a function to
    apply to them. It is initialized by the Architect after the decoder
    (see `initialize_normalizer`), and applied by the Architect in two
    different ways, depending on whether there is an encoder or not:

    - With an encoder, it is applied batchwise, between the encoder and the
      decoder step (see `Architect.scoring_function`): the function is
      applied to the (encoder output) embeddings of the current batch, and
      the new embeddings are returned. The original tensors (and the
      parameters they are built from) are left untouched, so the gradients
      flow through the normalization.

    - Without an encoder, the embeddings between the encoder and the decoder
      are the node and edge embeddings themselves. Normalizing them batchwise
      would recompute the same whole-graph normalization on every batch, so
      they are normalized once, over the whole graph, at the beginning of
      each epoch (see `Architect.apply_normalizer`), and again before an
      evaluation or an export of the embeddings. This is done in place, on
      the `.data` of each parameter, so the parameter objects themselves (and
      therefore the optimizer's references to them) are preserved. This is
      the same mechanism the decoders used in their `normalize_parameters`
      method.

    Arguments
    ---------
    
    **func** *(Callable[[torch.Tensor], torch.Tensor])*
    : The function to apply to each embedding. It must accept a tensor and
      return a tensor of the same shape.
    
    **node** *(bool, default to True, keyword-only)*
    : Whether the node embeddings (head and tail) are normalized.
    
    **edge** *(bool, default to False, keyword-only)*
    : Whether the edge embeddings are normalized.
    
    **params** *(Iterable[torch.Tensor], default to empty, keyword-only)*
    : The set of embeddings (node and/or edge embedding parameters of the
      knowledge graph) to normalize in place when there is no encoder.
    
    Raises
    ------
    
    **TypeError**
    : If `func` is not callable, or `params` is not an iterable of
      `torch.Tensor`.
    
    **ValueError**
    : If neither the node nor the edge embeddings are normalized.
    
    """
    def __init__(self,
                func: Callable[[Tensor], Tensor],
                *,
                node: bool = True,
                edge: bool = False,
                params: Iterable[Tensor] = ()):
        if not callable(func):
            raise TypeError(f"The normalizer function must be callable, but got {type(func)}.")
        
        try:
            params = list(params)
        except TypeError as e:
            raise TypeError(f"The embeddings to normalize must be an iterable of torch.Tensor, but got {type(params)}.") from e

        for param in params:
            if not isinstance(param, Tensor):
                raise TypeError(f"Every element of `params` must be a torch.Tensor, but found {type(param)}.")
        
        if not node and not edge:
            raise ValueError(f"A normalizer must normalize at least one kind of embeddings: `node` and `edge` are both False.")
        
        self._func: Callable[[Tensor], Tensor] = func
        self._node: bool = node
        self._edge: bool = edge
        self._params: list[Tensor] = params

    @property
    def params(self) -> list[Tensor]:
        """
        The set of embeddings this normalizer is applied to in place
        (no-encoder case).
        """
        return self._params

    @property
    def func(self) -> Callable[[Tensor], Tensor]:
        """
        The function applied to each embedding of this normalizer.
        """
        return self._func

    @property
    def node(self) -> bool:
        """
        Whether the node embeddings (head and tail) are normalized.
        """
        return self._node

    @property
    def edge(self) -> bool:
        """
        Whether the edge embeddings are normalized.
        """
        return self._edge

    def __call__(self,
                 *,
                 head_embeddings: Tensor,
                 tail_embeddings: Tensor,
                 edge_embeddings: Tensor
                 ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Apply the normalizer function to the embeddings given as arguments, and
        return the normalized ones.

        This is the batchwise application of the normalizer, used between the
        encoder and the decoder step: the function is applied to the (encoder
        output) embeddings of the current batch, and the new embeddings are
        returned. The original tensors (and the parameters they are built
        from) are left untouched, so the gradients flow through the
        normalization.

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
        : The same embeddings, normalized where the normalizer is configured
          to (see the `node` and `edge` properties), and left unchanged
          otherwise.
        
        """
        if self._node:
            head_embeddings = self._func(head_embeddings)
            tail_embeddings = self._func(tail_embeddings)
        if self._edge:
            edge_embeddings = self._func(edge_embeddings)

        return head_embeddings, tail_embeddings, edge_embeddings

    def apply_whole_graph(self):
        """
        Apply the normalizer function to every embedding it was given, in place.

        This is the whole-graph application of the normalizer, used when there
        is no encoder: the embeddings between the encoder and the decoder are
        the node and edge embeddings themselves, so normalizing them batchwise
        would recompute the same whole-graph normalization on every batch.
        They are therefore normalized once, over the whole graph, at the
        beginning of each epoch.

        The function is applied to `param.data`, so the parameter objects are
        preserved (the optimizer keeps its references to the same parameters).
        This is the same mechanism the decoders used in their
        `normalize_parameters` method.
        """
        for param in self._params:
            if isinstance(param, Parameter):
                # Operate on `.data` (a non-leaf, non-grad tensor) so no
                # throwaway autograd graph is built, and assign back to `.data`
                # so the Parameter object itself is preserved (the optimizer
                # keeps its references to the same parameters).
                param.data = self._func(param.data)
            else:
                param.copy_(self._func(param))

    def __repr__(self):
        return (f"{self.__class__.__name__}(params: {len(self._params)} parameter(s), "
                f"node: {self._node}, edge: {self._edge}, "
                f"func: {getattr(self._func, '__name__', repr(self._func))})")


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
