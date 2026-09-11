"""
Parameter regularization module for KGATE.

The `Regularizer` is a model component, initialized by the Architect after the
decoder (as soon as the parameters it regularizes exist). It is given a set
of parameters to regularize and the function to apply to them. Calling the
regularizer (e.g. from a trainer hook at the end of every epoch, or before an
evaluation) applies that function to every parameter it was given, in place.

This module gathers what the decoders used to do on their own in their
`normalize_parameters` method (e.g. TransE, RESCAL and DistMult L2-normalizing
their node embeddings): the same operations are now available as regularizer
functions, selected through the configuration (`[model.regularizer]`) and
applied by the trainer hooks instead of being hardcoded in each decoder.

@author: Benjamin Loire <benjamin.loire@univ-amu.fr>
"""
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import normalize
from typing import Callable, Iterable


class Regularizer:
    """
    Regularizer for KGATE.

    A regularizer is given a set of parameters to regularize and a function
    to apply to them. It is initialized by the Architect after the decoder
    (see `Architect.initialize_regularizer`), and applied through the trainer
    hooks (see `Architect.apply_regularizer`), for example at the end of
    every epoch and before an evaluation.

    Applying the regularizer is done in place, on the `.data` of each
    parameter, so the parameter objects themselves (and therefore the
    optimizer's references to them) are preserved. This is the same mechanism
    the decoders used in their `normalize_parameters` method.

    Arguments
    ---------
    
    **params** *(Iterable[torch.nn.Parameter])*
    : The set of parameters to regularize.
    
    **func** *(Callable[[torch.Tensor], torch.Tensor])*
    : The function to apply to each parameter. It must accept a tensor and
      return a tensor of the same shape.

    Raises
    ------
    
    **TypeError**
    : If `func` is not callable, or `params` is not an iterable of
      `torch.nn.Parameter`.
    
    """
    def __init__(self,
                params: Iterable[Parameter],
                func: Callable[[Tensor], Tensor]):
        if not callable(func):
            raise TypeError(f"The regularizer function must be callable, but got {type(func)}.")
        
        try:
            params = list(params)
        except TypeError as e:
            raise TypeError(f"The parameters to regularize must be an iterable of torch.nn.Parameter, but got {type(params)}.") from e

        for param in params:
            if not isinstance(param, Parameter):
                raise TypeError(f"Every element of `params` must be a torch.nn.Parameter, but found {type(param)}.")
        
        self._params: list[Parameter] = params
        self._func: Callable[[Tensor], Tensor] = func

    @property
    def params(self) -> list[Parameter]:
        """
        The set of parameters this regularizer is applied to.
        """
        return self._params

    @property
    def func(self) -> Callable[[Tensor], Tensor]:
        """
        The function applied to each parameter of this regularizer.
        """
        return self._func

    def __call__(self):
        """
        Apply the regularizer function to every parameter it was given, in place.
        
        The function is applied to `param.data`, so the parameter objects are
        preserved (the optimizer keeps its references to the same parameters).
        """
        for param in self._params:
            param.data = self._func(param)

    def __repr__(self):
        return (f"{self.__class__.__name__}(params: {len(self._params)} parameter(s), "
                f"func: {getattr(self._func, '__name__', repr(self._func))})")


# ------------------------------------------------------------------------------------
# Builtin regularizer functions
# ------------------------------------------------------------------------------------
# These are the functions that can be selected by name in the configuration
# (`[model.regularizer] name`). They mirror what the decoders used to do in
# their own `normalize_parameters` method:
# - `l2_normalize` is exactly the operation TransE, RESCAL and DistMult applied
#   to their node embeddings (row-wise L2 normalization).


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
    
    This is the operation the TransE, RESCAL and DistMult decoders used to
    apply to their node embeddings in their `normalize_parameters` method.
    
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


# The regularizer functions, indexed by the name used in the configuration.
REGULARIZER_FUNCTIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "L1": l1_normalize,
    "L2": l2_normalize,
}
