"""
Module initialization functions for the KGATE Architect.

This script contains the initialization functions of the different modules of
the Architect (the functions currently named `initialize_xxx()` in
`kgate.architect`, except for `initialize_model`).

Unlike the Architect methods, these functions are standalone: they do not take
the Architect instance. Instead, each property of the Architect that they use
is fed to them as an argument. Their bodies are otherwise the same as the
corresponding Architect methods, so they produce the same objects given the
same inputs.

Wiring these standalone functions into the Architect is left to the user.

Note
----
`initialize_evaluator` is the only function whose behavior differs slightly:
the Architect method sets `architect.validation_metric` as a side effect,
while this function returns the corresponding metric name (`"MRR"` or
`"Accuracy"`) alongside the evaluator, so the caller can set it.

@author: Benjamin Loire <benjamin.loire@univ-amu.fr>
"""
import logging
import warnings
from pathlib import Path
from typing import Any, Literal, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .config import (
    Initializer_Configuration,
    Encoder_Configuration,
    Decoder_Configuration,
    Loss_Configuration,
    Regularizer_Configuration,
    Optimizer_Configuration,
    Sampler_Configuration,
    Learning_Rate_Scheduler_Configuration,
    Evaluation_Configuration
)

from .decoders import (
    BilinearDecoder,
    ConvKB,
    ConvolutionalDecoder,
    ComplEx,
    DistMult,
    RESCAL,
    TransD,
    TransE,
    TransH,
    TranslationalDecoder,
    TransR,
    TorusE,
)
from .encoders import GATEncoder, GCNEncoder
from .evaluators import LinkPredictionEvaluator, TripletClassificationEvaluator
from .initializers import FeatureInitializer, Initializer, Node2VecInitializer
from .knowledgegraph import KnowledgeGraph
from .loss import BinaryCrossEntropyLoss, KGE_Loss, MarginLoss
from .regularizers import REGULARIZER_FUNCTIONS, Regularizer
from .samplers import (
    BernoulliNegativeSampler,
    MixedNegativeSampler,
    NegativeSampler,
    PositionalNegativeSampler,
    UniformNegativeSampler,
)


def initialize_initializer(configuration: Initializer_Configuration,
                           knowledge_graph: KnowledgeGraph,
                           node_embedding_dimensions: int,
                           checkpoints_directory: Path,
                           device: torch.device) -> Initializer:
    """
    Set the method used to generate initial embeddings
    
    Options are random initialization, which is equivalent to just a lookup embedding,
    user-supplied features that can be learnt with a deep encoder, and Node2Vec.

    Arguments
    ---------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **knowledge_graph** *(KnowledgeGraph)*
    : The Architect's knowledge graph (used for its edge list and train mask,
      for the Node2Vec initializer).
    
    **node_embedding_dimensions** *(int)*
    : The Architect's `node_embedding_dimensions`.
    
    **checkpoints_directory** *(Path)*
    : The Architect's `checkpoints_directory` (output directory of the
      Node2Vec initializer).
    
    **device** *(torch.device)*
    : The Architect's `device`.
    
    Returns
    -------
    
    **initializer** *(Initializer)*
    : The initialized initializer.
    
    """
    match configuration.name:
        case "Random":
            initializer = Initializer()
        case "Feature":
            initializer = FeatureInitializer() #TODO
        case "Node2Vec":
            initializer = Node2VecInitializer(
                edge_indices = knowledge_graph.edge_list[:, knowledge_graph.train_mask],
                embedding_dimensions = node_embedding_dimensions,
                walk_length = configuration.walk_length,
                context_size = configuration.context_size,
                output_directory = checkpoints_directory,
                device = device
            )
        case _:
            raise NotImplementedError(f"The requested initializer {configuration.name} is not implemented.")
    
    logging.info(f"Using the {configuration.name} initializer")
    return initializer


def initialize_encoder(configuration: Encoder_Configuration,
                       knowledge_graph: KnowledgeGraph,
                       encoder_node_embedding_dimensions: int,
                       encoder_name: Literal["GCN", "GAT", "Node2Vec", ""] = "",
                       gnn_layers: int = 0
                       ) -> GCNEncoder | GATEncoder | None:
    """
    Create and initialize the encoder object according to the configuration or arguments.

    The encoder is created from PyG encoding layers. Currently, the implemented encoders 
    are a random initialization, **GCN** [1]_, **GAT** [2]_. See the encoder class for a detailed
    explanation of the encoders.

    If both configuration and arguments are given, the arguments take priority.

    References
    ----------
    .. [1] <https://arxiv.org/pdf/1609.02907>. Kipf, Thomas and Max Welling. “Semi-Supervised Classification with Graph Convolutional Networks.” ArXiv abs/1609.02907 (2016): n. pag.
    .. [2] <https://arxiv.org/pdf/2105.14491>. Brody, Shaked et al. “How Attentive are Graph Attention Networks?” ArXiv abs/2105.14491 (2021): n. pag.

    Arguments
    ---------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **knowledge_graph** *(KnowledgeGraph)*
    : The Architect's knowledge graph (used for its triplet types).
    
    **encoder_node_embedding_dimensions** *(int)*
    : The Architect's `encoder_node_embedding_dimensions`.
    
    **encoder_name** *({"Default", "GCN", "GAT", "Node2Vec"}, optional)*
    : Name of the encoder.
    
    **gnn_layers** *(int, optional, default to 0)*
    : Number of hidden layers for the encoder. Only used for deep learning encoders.

    Warns
    -----
    
    If the provided encoder name is not supported, it will default to a random initialization and warn the user.

    Returns
    -------
    **encoder** *(GCNEncoder or GATEncoder or None)*
        The encoder object, or None if there is no encoder.
    
    """
    if encoder_name == "":
        encoder_name = configuration.name
    
    if gnn_layers == 0:
        gnn_layers = configuration.gnn_layers

    edge_types = knowledge_graph.triplet_types

    match encoder_name:
        case "GCN": 
            encoder = GCNEncoder(edge_types, encoder_node_embedding_dimensions, gnn_layers)
        case "GAT":
            encoder = GATEncoder(edge_types, encoder_node_embedding_dimensions, gnn_layers)
        case _:
            encoder = None
            logging.warning(f"Unrecognized encoder {encoder_name}, will not use any.")
    
    return encoder


def initialize_decoder(configuration: Decoder_Configuration,
                       knowledge_graph: KnowledgeGraph,
                       node_embedding_dimensions: int,
                       edge_embedding_dimensions: int,
                       device: torch.device,
                       decoder_name: str = "",
                       dissimilarity: Literal["L1", "L2", "torus_L1", "torus_L2", "torus_eL2", ""] = "",
                       filter_count: int = None
                       ) -> Tuple[
                                   BilinearDecoder | ConvolutionalDecoder | TranslationalDecoder,
                                   MarginLoss | BinaryCrossEntropyLoss
                                   ]:
    """
    Create and initialize the decoder object according to the configuration or arguments.

    The decoders are adapted and inherit from torchKGE decoders to be able to handle heterogeneous data. 
    Not all torchKGE decoders are already implemented, but all of them and more will eventually be. Currently, 
    the available decoders are **TransE** [1]_, **TransH** [2]_, **TransR** [3]_, **TransD** [4]_, **TorusE** [5]_, 
    **RESCAL** [6]_, **DistMult** [7]_, **ComplEx** [8]_ and **ConvKB** [9]_. See the description of decoder classes for details about 
    their implementation, or read their original papers.

    Translational models are used with a `torchkge.MarginLoss` while bilinear models are used with a 
    `torchkge.BinaryCrossEntropyLoss`.

    If both configuration and arguments are given, the arguments take priority.

    References
    ----------
    
    .. [1] Bordes, Antoine et al. “Translating Embeddings for Modeling Multi-relational Data.” Neural Information Processing Systems (2013).
    .. [2] Wang, Zhen et al. “Knowledge Graph Embedding by Translating on Hyperplanes.” AAAI Conference on Artificial Intelligence (2014).
    .. [3] Lin, Yankai et al. “Learning Entity and Relation Embeddings for Knowledge Graph Completion.” AAAI Conference on Artificial Intelligence (2015).
    .. [4] Ji, Guoliang et al. “Knowledge Graph Embedding via Dynamic Mapping Matrix.” Annual Meeting of the Association for Computational Linguistics (2015).
    .. [5] *Missing documentation for TorusE*
    .. [6] Nickel, Maximilian et al. “A Three-Way Model for Collective Learning on Multi-Relational Data.” International Conference on Machine Learning (2011).
    .. [7] Yang, Bishan et al. “Embedding Entities and Relations for Learning and Inference in Knowledge Bases.” International Conference on Learning Representations (2014).
    .. [8] *Missing documentation for ComplEx*
    .. [9] Nguyen, Dai Quoc et al. “A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network.” North American Chapter of the Association for Computational Linguistics (2017).

    % TODO: add reference to TorusE and ComplEx
    % TODO: proper links to the references
    
    Arguments
    ----------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **knowledge_graph** *(KnowledgeGraph)*
    : The Architect's knowledge graph (used for its node and edge counts).
    
    **node_embedding_dimensions** *(int)*
    : The Architect's `node_embedding_dimensions`.
    
    **edge_embedding_dimensions** *(int)*
    : The Architect's `edge_embedding_dimensions`.
    
    **device** *(torch.device)*
    : The Architect's `device`.
    
    **decoder_name** *(str, optional)*
    : Name of the decoder.
    
    **dissimilarity** *({"L1", "L2"}, optional)*
    : Type of the dissimilarity metric.
            
    **filter_count** *(int, optional, default to 0)*
    : Number of convolution filters.

    Raises
    ------
    
    **NotImplementedError**
    : If the provided decoder name is not supported.

    Returns
    -------
    
    **decoder** *(BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder)*
    : The decoder object.
    
    **loss** *(MarginLoss or BinaryCrossEntropyLoss)*
    : The loss object.
    
    """
    if decoder_name == "":
        decoder_name = configuration.name
    if dissimilarity == "":
        dissimilarity = configuration.dissimilarity
    if filter_count == 0:
        filter_count = configuration.filter_count

    # Translational models
    match decoder_name:
        case "TransE":
            decoder = TransE(dissimilarity_type = dissimilarity)
        case "TransH":
            decoder = TransH(embedding_dimensions = node_embedding_dimensions,
                            node_count = knowledge_graph.node_count,
                            edge_count = knowledge_graph.edge_count,
                            device = device)
        case "TransR":
            decoder = TransR(node_embedding_dimensions = node_embedding_dimensions,
                            edge_embedding_dimensions = edge_embedding_dimensions, 
                            node_count = knowledge_graph.node_count, 
                            edge_count = knowledge_graph.edge_count,
                            device = device)
        case "TransD":
            decoder = TransD(node_embedding_dimensions = node_embedding_dimensions,
                            edge_embedding_dimensions = edge_embedding_dimensions, 
                            node_count = knowledge_graph.node_count, 
                            edge_count = knowledge_graph.edge_count,
                            device = device)
        case "TorusE":
            decoder = TorusE(dissimilarity_type = dissimilarity)
        case "RESCAL":
            decoder = RESCAL(embedding_dimensions = node_embedding_dimensions,
                            node_count = knowledge_graph.node_count,
                            edge_count = knowledge_graph.edge_count,
                            device = device)
        case "DistMult":
            decoder = DistMult(embedding_dimensions = node_embedding_dimensions,
                            node_count = knowledge_graph.node_count,
                            edge_count = knowledge_graph.edge_count)
        case "ComplEx":
            decoder = ComplEx(embedding_dimensions = node_embedding_dimensions)
        case "ConvKB":
            decoder = ConvKB(embedding_dimensions = node_embedding_dimensions, 
                            filter_count = filter_count, 
                            node_count = knowledge_graph.node_count, 
                            edge_count = knowledge_graph.edge_count)
        case _:
            raise NotImplementedError(f"The requested decoder {decoder_name} is not implemented.")

    return decoder


def initialize_loss(configuration: Loss_Configuration,
                    loss_name: str = "",
                    margin: int = -1,
                    reduction: str = ""
                    ) -> KGE_Loss:
    """
    Creates and initializes the Loss object.

    KGATE's base loss is a composite loss that can have multiple terms. Once it is 
    initialized, additional loss functions can be added using the loss.add_term method.

    Arguments
    ---------
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **loss_name** *(str)*
    : The name of the loss function. Currently supported losses are `Margin` and `BCE`

    **margin** *(int)*
    : Only for margin loss, the value by which the positive scores must
    : exceed the negative scores.

    **reduction** *(str)*
    : How the loss of each elements is aggregated into a single value.
    : Options are `mean` and `sum`.
    """
    if loss_name == "":
        loss_name = configuration.name
    if margin == -1:
        margin = configuration.margin
    if reduction == "":
        reduction = configuration.reduction

    loss = KGE_Loss()

    match loss_name:
        case "Margin":
            loss.add_term(MarginLoss(margin, reduction))
        case "BCE":
            loss.add_term(BinaryCrossEntropyLoss(reduction))
        case _:
            raise NotImplementedError(f"The requested loss {loss_name} is not implemented.")

    return loss


def initialize_regularizer(configuration: Regularizer_Configuration,
                           knowledge_graph: KnowledgeGraph) -> Regularizer | None:
    """
    Initialize the regularizer according to the configuration.

    The regularizer is a model component initialized after the decoder: it
    is given the set of parameters to regularize and the function to apply
    to them, and is applied through the trainer hooks (see
    `Architect.apply_regularizer`). It gathers what the decoders used to do
    in their own `normalize_parameters` method (e.g. TransE, RESCAL and
    DistMult L2-normalizing their node embeddings).

    The parameters it regularizes are the node and/or edge embeddings of
    the knowledge graph, selected by the `params` configuration key
    (`node`, `edge` or `all`). The function is selected by the `name`
    configuration key (see `kgate.regularizers.REGULARIZER_FUNCTIONS` for
    the builtin functions, or `Config.regularizer.register_name` for
    custom function names).

    Arguments
    ---------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **knowledge_graph** *(KnowledgeGraph)*
    : The Architect's knowledge graph (its node and edge embeddings are the
      parameters that can be regularized).
    
    Returns
    -------
    
    **regularizer** *(Regularizer or None)*
    : The initialized regularizer, or None if no regularizer is configured
      (`[model.regularizer] name = "None"`, the default).
    
    Raises
    ------
    
    **KeyError**
    : If the configured regularizer name is not a known function.
    
    """
    if configuration.name == "None":
        logging.info("No regularizer configured. Skipping.")
        return None

    func = REGULARIZER_FUNCTIONS[configuration.name]

    # Build the set of parameters to regularize, according to the configuration
    params: list[nn.Parameter] = []
    if configuration.params in ("node", "all"):
        params.extend(knowledge_graph.node_embeddings)
    if configuration.params in ("edge", "all"):
        params.append(knowledge_graph.edge_embeddings)

    regularizer = Regularizer(params = params, func = func)
    logging.info(f"Regularizer initialized: {regularizer}")

    return regularizer


def initialize_optimizer(configuration: Optimizer_Configuration,
                         knowledge_graph: KnowledgeGraph,
                         decoder: BilinearDecoder | ConvolutionalDecoder | TranslationalDecoder,
                         encoder: GCNEncoder | GATEncoder | None = None) -> optim.Optimizer:
    """
    Initialize the optimizer based on the configuration provided.
    
    Available optimizers are Adam, SGD and RMSprop. See torch.optim 
    documentation for optimizer parameters: <https://docs.pytorch.org/docs/stable/optim.html>

    Arguments
    ---------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **knowledge_graph** *(KnowledgeGraph)*
    : The Architect's knowledge graph (its node and edge embeddings are
      optimized).
    
    **decoder** *(BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder)*
    : The Architect's decoder (its parameters are optimized).
    
    **encoder** *(GCNEncoder or GATEncoder or None, optional, default to None)*
    : The Architect's encoder, if any (its parameters are optimized).
    
    Raises
    ------
    
    **NotImplementedError**
    : If the optimizer is not supported.

    Returns
    -------
    
    **optimizer** *(torch.optim.Optimizer)*
    : Initialized optimizer.
    
    """
    optimizer_name: str = configuration.name

    # Retrieve optimizer parameters, defaulting to an empty dictionnary if not specified
    optimizer_params: dict = configuration.parameters

    optimizer_class = getattr(optim, optimizer_name)

    parameters = [node_embedding for node_embedding in knowledge_graph.node_embeddings]
    parameters.append(knowledge_graph.edge_embeddings)
    parameters.extend(decoder.parameters())
    if encoder is not None:
        parameters.extend(encoder.parameters())

    # Initialize the optimizer with given parameters
    optimizer: optim.Optimizer = optimizer_class(parameters, **optimizer_params)

    logging.info(f"Optimizer '{optimizer_name}' initialized with parameters: {optimizer_params}")
    
    return optimizer


def initialize_negative_sampler(configuration: Sampler_Configuration,
                                knowledge_graph: KnowledgeGraph) -> NegativeSampler:
    """
    Initialize the sampler according to the configuration.
    
    Supported samplers are Positional, Uniform, Bernoulli and Mixed. 
    They are adapted from torchKGE's samplers to be compatible with the 
    graphindices format.

    Arguments
    ---------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **knowledge_graph** *(KnowledgeGraph)*
    : The Architect's knowledge graph.
    
    Raises
    ------
    
    **NotImplementedError**
    : If the name of the sampler is not supported.

    Returns
    -------
    
    **negative_sampler** *(NegativeSampler)*
    : The initialized sampler.
    
    """
    negative_sampler_name: str = configuration.name
    negative_triplet_count: int = configuration.negative_triplet_count

    match negative_sampler_name:
        case "Positional":
            negative_sampler = PositionalNegativeSampler(knowledge_graph)
        case "Uniform":
            negative_sampler = UniformNegativeSampler(knowledge_graph, negative_triplet_count)
        case "Bernoulli":
            negative_sampler = BernoulliNegativeSampler(knowledge_graph, negative_triplet_count)
        case "Mixed":
            negative_sampler = MixedNegativeSampler(knowledge_graph, negative_triplet_count)
        case _:
            raise NotImplementedError(f"Sampler type '{negative_sampler_name}' is not supported. Please check the configuration.")
    
    return negative_sampler


def initialize_learning_rate_scheduler(configuration: Learning_Rate_Scheduler_Configuration,
                                       optimizer: optim.Optimizer
                                       ) -> optim.lr_scheduler.LRScheduler | None:
    """
    Initializes the learning rate scheduler based on the provided configuration.
    
    Arguments
    ---------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **optimizer** *(torch.optim.Optimizer)*
    : The Architect's optimizer, which the scheduler schedules.
    
    Raises
    ------
    
    **ValueError**
    : If the scheduler type is unsupported or required parameters are missing.
    
    Warns
    -----
    
    If no learning rate scheduler is specified in the configuration, none will be used.
    
    Returns
    -------
    
    **learning_rate_scheduler** *(torch.optim.lr_scheduler._LRScheduler or None)*
    : Instance of the specified scheduler or None if no scheduler is configured.
    
    """
    learning_rate_scheduler_name: str = configuration.name

    if learning_rate_scheduler_name == "":
        warnings.warn("No learning rate scheduler specified in the configuration, none will be used.")
        return None
    
    learning_rate_scheduler_params: dict = configuration.parameters

    learning_rate_scheduler_class = getattr(optim.lr_scheduler, learning_rate_scheduler_name)
    
    # Initialize the scheduler based on its type
    try:
        learning_rate_scheduler: optim.lr_scheduler.LRScheduler = learning_rate_scheduler_class(optimizer, **learning_rate_scheduler_params)
    except TypeError as e:
        raise ValueError(f"Error initializing '{learning_rate_scheduler_name}': {e}")
    
    logging.info(f"Scheduler '{learning_rate_scheduler_name}' initialized with parameters: {learning_rate_scheduler_params}")
    
    return learning_rate_scheduler


def initialize_evaluator(configuration: Evaluation_Configuration,
                         knowledge_graph: KnowledgeGraph,
                         node_embedding_dimensions: int,
                         architect: Any = None
                         ) -> Tuple[LinkPredictionEvaluator | TripletClassificationEvaluator, str]:
    """
    Set the task for which the model will be evaluated on using the validation set.
    
    Options are Link Prediction or Triplet Classification. 
    Link Prediction evaluate the ability of a model to predict correctly the head or tail of a triple given the other 
    node and edge. 
    Triplet Classification evaluate the ability of a model to discriminate between existing and 
    fake triplet in a KG.
    
    Note
    ----
    
    The Architect method sets `architect.validation_metric` as a side effect.
    This function returns the corresponding metric name (`"MRR"` for Link
    Prediction, `"Accuracy"` for Triplet Classification) alongside the
    evaluator, so the caller can set it.
    
    The Triplet Classification evaluator inherently references the model it
    evaluates (it uses the Architect's `device` and `scoring_function`), so
    unlike the other functions in this module, an Architect instance must be
    provided for that objective (it is ignored for Link Prediction).
    
    Arguments
    ---------
    
    **configuration** *(Configuration)*
    : The Architect's configuration.
    
    **knowledge_graph** *(KnowledgeGraph)*
    : The Architect's knowledge graph (used for its graph indices).
    
    **node_embedding_dimensions** *(int)*
    : The Architect's `node_embedding_dimensions`.
    
    **architect** *(Architect, optional, default to None)*
    : The Architect instance, only needed for the Triplet Classification
      objective (the evaluator it creates uses the Architect's `device` and
      `scoring_function`). Ignored for Link Prediction.
    
    Raises
    ------
    
    **NotImplementedError**
    : If the name of the task is not supported.
    
    Returns
    -------
    evaluator: LinkPredictionEvaluator or TripletClassificationEvaluator
        The initialized evaluator, either LinkPredictionEvaluator or TripletClassificationEvaluator.
    
    validation_metric: str
        The corresponding validation metric name: `"MRR"` for Link Prediction,
        `"Accuracy"` for Triplet Classification.
    
    """
    match configuration.objective:
        case "Link Prediction":
            evaluator = LinkPredictionEvaluator(graphindices = knowledge_graph.graphindices, embedding_dimensions = node_embedding_dimensions)
            validation_metric = "MRR"
        case "Triplet Classification":
            if architect is None:
                raise ValueError("The Triplet Classification evaluator needs the Architect instance (it uses its device and scoring_function). Please provide it as the `architect` argument.")
            evaluator = TripletClassificationEvaluator(architect = architect, knowledge_graph = knowledge_graph)
            validation_metric = "Accuracy"
        case _:
            raise NotImplementedError(f"The requested evaluator {configuration.objective} is not implemented.")
    
    logging.info(f"Using {configuration.objective} evaluator.")
    
    return evaluator, validation_metric

