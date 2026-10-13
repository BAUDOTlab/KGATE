"""
Embedding normalization module for KGATE.

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

    def normalize(self, 
                    node_embeddings: Parameter | Iterable[Parameter],
                    edge_embeddings: Parameter,
                    mode: Literal["initial","training"] = "training"
                    ) -> tuple[Parameter | Iterable[Parameter], Parameter]:
        """Apply the training normalizer function to the training targets.
        
        Arguments
        ---------
        **node_embeddings** *(torch.nn.Parameter or iterable of torch.nn.Parameter, shape: [node_count, dimensions])*
        : The graph node embeddings: either a single embedding table, or one
        : embedding table per node type.

        **edge_embeddings** *(torch.nn.Parameter, shape: [edge_count, dimensions])*
        : The graph edge embeddings

        **mode** *("initial" or "training", defaults to "training")
        The mode to consider for full-graph normalisation: either on the initial targets, or the training targets.

        Returns
        -------
        **node_embeddings, edge_embeddings** *(tuple[torch.nn.Parameter or iterable of torch.nn.Parameter, torch.nn.Parameter])*
        : The same embeddings, normalized where the normalizer is configured,
        : and left unchanged otherwise.
        """
        match mode:
            case "training":
                targets = self.training_targets
                normal_func = self.training_normalization
            case "initial":
                targets = self.initial_targets
                normal_func = self.initial_normalization
                
        if targets in ["node", "all"]:
            node_parameters = [node_embeddings] if isinstance(node_embeddings, Parameter) else list(node_embeddings)
            for node_embedding in node_parameters:
                node_embedding.data = normal_func(node_embedding.data)
        if targets in ["edge", "all"]:
            edge_embeddings.data = normal_func(edge_embeddings.data)

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
