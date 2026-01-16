"""Architect class and methods to run a KGE model training, testing and inference from end to end."""

import os
import csv
import gc
from glob import glob
import shutil
from inspect import signature
from pathlib import Path
import logging
import warnings
import yaml
import platform
from typing import Tuple, Dict, List, Any, Set, Literal
from collections.abc import Callable

import pandas as pd
import numpy as np

from torchkge import KnowledgeGraph
from torchkge.models import Model
import torchkge.sampling as sampling
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss

from torch_geometric.utils import k_hop_subgraph

import torch
from torch import tensor, Tensor
from torch.nn import Parameter
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler as learning_rate_scheduler

from ignite.metrics import RunningAverage
from ignite.engine import Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint, Checkpoint, DiskSaver, ProgressBar

from .utils import parse_config, load_knowledge_graph, set_random_seeds, find_best_model, merge_kg, init_embedding, plot_learning_curves, save_config
from .preprocessing import prepare_knowledge_graph, SUPPORTED_SEPARATORS
from .encoders import *
from .decoders import *
from .knowledgegraph import KnowledgeGraph
from .samplers import PositionalNegativeSampler, BernoulliNegativeSampler, UniformNegativeSampler, MixedNegativeSampler
from .evaluators import LinkPredictionEvaluator, TripletClassificationEvaluator
from .inference import NodeInference, EdgeInference
from .data_leakage import permute_tails

# Configure logging
logging.captureWarnings(True)
logging_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=logging_level,  
    format="%(asctime)s - %(levelname)s - %(message)s" 
)


class Architect(Model):
    """Architect class for knowledge graph embedding training.
    
    The Architect class contains the kg and manages every step from the training to the inference.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file
    kg : Tuple of KnowledgeGraph or torchkge.KnowledgeGraph, optional
        Either a knowledge graph that has already been preprocessed by KGATE and split accordingly, or an unprocessed KnowledgeGraph object.
        In the first case, the knowledge graph won't be preprocessed even if `config.run_kg_preprocess` is set to True.
        In the second case, an error is thrown if the `config.run_kg_preprocess` is set to False.
    dataframe : pd.DataFrame, optional
        The knowledge graph as a pandas dataframe containing at least the columns from, to and rel
    metadata : pd.DataFrame, optional
        The metadata as a pandas dataframe, with at least the columns id and type, where id is the name of the node as it is in the
        knowledge graph. If this argument is not provided, the metadata will be read from config.metadata if it exists. If both are absent,
        all nodes will be considered to be the same node type.
    cuddn_benchmark : bool, default to True
        Benchmark different convolution algorithms to chose the optimal one. Initialization is slightly longer when it is enabled, and only if cuda is available.
    number_of_cores : int, default to 0
        Set the number of cpu cores used by KGATE. If set to 0, the maximum number of available cores is used.
    kwargs: dict
        Inline configuration parameters. The name of the arguments must match the parameters found in `config_template.toml`.
        
    Raises
    ------
    ValueError
        If the `config.metadata_csv` file exists but cannot be parsed, or if `kg` is given, but not a tuple of KnowledgeGraph and `config.run_kg_preprocess` is set to false.

    Examples
    --------
    Inline hyperparameter declaration
    >>> model_params = {"emb_dim": 100, "decoder": {"name":"DistMult"}}
    >>> sampler_params = {"n_neg":5}
    >>> run_preprocessing = True
    >>> architect = Architect("/path/to/configuration", model = model_params, sampler = sampler_params, run_kg_preprocess = run_preprocessing)

    Notes
    -----
    While it is possible to give any part of the configuration, even everything, as kwargs, it is recommended
    to use a separated configuration file to ensure reproducibility of training.
    """
    def __init__(self, config_path: str = "", kg: Tuple[KnowledgeGraph,KnowledgeGraph,KnowledgeGraph] | KnowledgeGraph | None = None, dataframe: pd.DataFrame | None = None, metadata: pd.DataFrame | None = None, cudnn_benchmark: bool = True, number_of_cores:int = 0, **kwargs):
        # kg should be of type KnowledgeGraph, if exists use it instead of the one in config
        # dataframe should have columns from, rel and to
        self.config: dict = parse_config(config_path, kwargs)

        if torch.cuda.is_available():
            # Benchmark convolution algorithms to chose the optimal one.
            # Initialization is slightly longer when it is enabled.
            torch.backends.cudnn.benchmark = cudnn_benchmark

        # If given, restrict the parallelisation to user-defined threads.
        # Otherwise, use all the cores the process has access to.
            
        if platform.system() == "Windows":
            number_of_cores: int = number_of_cores if number_of_cores > 0 else os.cpu_count()
        else:
            number_of_cores: int = number_of_cores if number_of_cores > 0 else len(os.sched_getaffinity(0))
        logging.info(f"Setting number of threads to {number_of_cores}")
        torch.set_num_threads(number_of_cores)

        output_directory: Path = Path(self.config["output_directory"])
        # Create output folder if it doesn't exist
        logging.info(f"Output folder: {output_directory}")
        output_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints_directory: Path = output_directory.joinpath("checkpoints")


        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Detected device: {self.device}")

        set_random_seeds(self.config["seed"])

        self.embedding_dimensions: int = self.config["model"]["emb_dim"]
        self.edge_embedding_dimensions: int = self.config["model"]["rel_emb_dim"]
        if self.edge_embedding_dimensions == -1:
            self.edge_embedding_dimensions = self.embedding_dimensions
        self.evaluation_batch_size: int = self.config["training"]["eval_batch_size"]

        if metadata is not None and not set(["id","type"]).issubset(metadata.keys()):
            raise pd.errors.InvalidColumnName("The columns \"id\" and \"type\" must be present in the given metadata dataframe.")
        
        self.metadata = metadata

        if metadata is None and self.config["metadata_csv"] != "" and Path(self.config["metadata_csv"]).exists():
            for separator in SUPPORTED_SEPARATORS:
                try:
                    self.metadata = pd.read_csv(self.config["metadata_csv"], sep=separator, usecols=["type","id"])
                    break
                except ValueError:
                    continue
        
            if self.metadata is None:
                raise ValueError(f"The metadata csv file uses a non supported separator. Supported separators are '{'\', \''.join(SUPPORTED_SEPARATORS)}'.")


        run_kg_preprocessing: bool = self.config["run_kg_preprocess"]

        if run_kg_preprocessing:
            logging.info(f"Preparing KG...")
            self.kg_train, self.kg_validation, self.kg_test = prepare_knowledge_graph(self.config, kg, dataframe, self.metadata)
            logging.info("KG preprocessed.")
        else:
            if kg is not None:
                logging.info("Using given KG...")
                if isinstance(kg, tuple):
                    self.kg_train, self.kg_validation, self.kg_test = kg
                else:
                    raise ValueError("Given KG needs to be a tuple of training, validation and test KG if it is preprocessed.")
            else:
                logging.info("Loading KG...")
                self.kg_train, self.kg_validation, self.kg_test = load_knowledge_graph(Path(self.config["kg_pkl"]))
                logging.info("Done")

        super().__init__(self.kg_train.node_count, self.kg_train.edge_count)
        # Initialize attributes
        self.encoder: DefaultEncoder | GNN = None
        self.decoder: Model = None
        self.decoder_loss: MarginLoss | BinaryCrossEntropyLoss = None
        self.optimizer: optim.Optimizer = None
        self.sampler: sampling.NegativeSampler = None
        self.scheduler: learning_rate_scheduler.LRScheduler | None = None
        self.evaluator: LinkPredictionEvaluator | TripletClassificationEvaluator = None
        self.node_embeddings: nn.ParameterList

    @property
    def encoder_node_embedding_dimensions(self):
        if self.decoder is not None and hasattr(self.decoder,"embedding_spaces"):
            return self.embedding_dimensions * self.decoder.embedding_spaces
        return self.embedding_dimensions

    @property
    def encoder_edge_embedding_dimensions(self):
        if self.decoder is not None and hasattr(self.decoder,"embedding_spaces"):
            return self.edge_embedding_dimensions * self.decoder.embedding_spaces
        return self.edge_embedding_dimensions


    def initialize_encoder(self, encoder_name: str = "", gnn_layers: int = 0) -> DefaultEncoder | GCNEncoder | GATEncoder:
        """Create and initialize the encoder object according to the configuration or arguments.

        The encoder is created from PyG encoding layers. Currently, the implemented encoders 
        are a random initialization, GCN [1]_ and GAT [2]_. See the encoder class for a detailed
        explanation of the encoders.

        If both configuration and arguments are given, the arguments take priority.

        References
        ----------
        .. [1] Kipf, Thomas and Max Welling. “Semi-Supervised Classification with Graph Convolutional Networks.” ArXiv abs/1609.02907 (2016): n. pag.
        .. [2] Brody, Shaked et al. “How Attentive are Graph Attention Networks?” ArXiv abs/2105.14491 (2021): n. pag.

        Parameters
        ----------
        encoder_name: {"Default", "GCN", "GAT"}, optional
            Name of the encoder
        gnn_layers: int, optional
            Number of hidden layers for the encoder. Only used for deep learning encoders.

        Warns
        -----
        If the provided encoder name is not supported, it will default to a random initialization and warn the user.

        Returns
        -------
        encoder
            The encoder object
        """
        encoder_config: dict = self.config["model"]["encoder"]
        if encoder_name == "":
            encoder_name = encoder_config["name"]
        
        if gnn_layers == 0:
            gnn_layers = encoder_config["gnn_layer_number"]

        last_triple_type = self.kg_train.triplets[-1]
        edge_types = self.kg_train.triple_types#[:last_triple_type + 1]

        match encoder_name:
            case "Default":
                encoder = DefaultEncoder()
            case "GCN": 
                encoder = GCNEncoder(edge_types, self.encoder_node_embedding_dimensions, gnn_layers)
            case "GAT":
                encoder = GATEncoder(edge_types, self.encoder_node_embedding_dimensions, gnn_layers)
            case "Node2vec":
                encoder = Node2VecEncoder(self.kg_train.edge_index, self.encoder_node_embedding_dimensions, device=self.device, **encoder_config["params"])
            case _:
                encoder = DefaultEncoder()
                logging.warning(f"Unrecognized encoder {encoder_name}. Defaulting to a random initialization.")
        return encoder

    def initialize_decoder(self, decoder_name: str = "", dissimilarity: Literal["L1","L2",""] = "", margin: int = 0, filter_count: int = 0) -> Tuple[Model, MarginLoss | BinaryCrossEntropyLoss]:
        """Create and initialize the decider object according to the configuration or arguments.

        The decoders are adapted and inherit from torchKGE decoders to be able to handle heterogeneous data.
        Not all torchKGE decoders are already implemented, but all of them and more will eventually be. Currently, 
        the available decoders are **TransE** [1]_, **TransH** [2]_, **TransR** [3]_, **TransD** [4]_,
        **RESCAL** [5]_, **DistMult** [6]_ and **ConvKB** [7]_. See the description of decoder classes for details about 
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
        .. [5] Nickel, Maximilian et al. “A Three-Way Model for Collective Learning on Multi-Relational Data.” International Conference on Machine Learning (2011).
        .. [6] Yang, Bishan et al. “Embedding Entities and Relations for Learning and Inference in Knowledge Bases.” International Conference on Learning Representations (2014).
        .. [7] Nguyen, Dai Quoc et al. “A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network.” North American Chapter of the Association for Computational Linguistics (2017).

        Parameters
        ----------
        decoder_name: str, optional
            Name of the decoder.
        dissimlarity: {"L1","L2"}, optional
            Type of the dissimilarity metric.
        margin: int, optional
            Margin to be used with MarginLoss. Unused with bilinear models.

        Raises
        -----
        NotImplementedError
            If the provided decoder name is not supported.

        Returns
        -------
        decoder
            The decoder object
        decoder_loss
            The loss object
        """
        
        decoder_config: dict = self.config["model"]["decoder"]

        if decoder_name == "":
            decoder_name = decoder_config["name"]
        if dissimilarity == "":
            dissimilarity = decoder_config["dissimilarity"]
        if margin == 0:
            margin = decoder_config["margin"]
        if filter_count == 0:
            filter_count = decoder_config["n_filters"]

        # Translational models
        match decoder_name:
            case "TransE":
                decoder = TransE(self.embedding_dimensions, self.kg_train.node_count, self.kg_train.edge_count,
                            dissimilarity_type=dissimilarity)
                decoder_loss = MarginLoss(margin)
            case "TransH":
                decoder = TransH(self.embedding_dimensions, self.kg_train.node_count, self.kg_train.edge_count)
                decoder_loss = MarginLoss(margin)
            case "TransR":
                decoder = TransR(self.embedding_dimensions, self.edge_embedding_dimensions, self.kg_train.node_count, self.kg_train.edge_count)
                decoder_loss = MarginLoss(margin)
            case "TransD":
                decoder = TransD(self.embedding_dimensions, self.edge_embedding_dimensions, self.kg_train.node_count, self.kg_train.edge_count)
                decoder_loss = MarginLoss(margin)
            case "RESCAL":
                decoder = RESCAL(self.embedding_dimensions, self.kg_train.node_count, self.kg_train.edge_count)
                decoder_loss = BinaryCrossEntropyLoss()
            case "DistMult":
                decoder = DistMult(self.embedding_dimensions, self.kg_train.node_count, self.kg_train.edge_count)
                decoder_loss = BinaryCrossEntropyLoss()
            case "ComplEx":
                decoder = ComplEx(self.embedding_dimensions, self.kg_train.node_count, self.kg_train.edge_count)
                decoder_loss = BinaryCrossEntropyLoss()
            case "ConvKB":
                decoder = ConvKB(self.embedding_dimensions, filter_count, self.kg_train.node_count, self.kg_train.edge_count)
                decoder_loss = BinaryCrossEntropyLoss()
            case _:
                raise NotImplementedError(f"The requested decoder {decoder_name} is not implemented.")

        return decoder, decoder_loss

    def initialize_optimizer(self) -> optim.Optimizer:
        """
        Initialize the optimizer based on the configuration provided.
        
        Available optimizers are Adam, SGD and RMSprop. See Pytorch.optim 
        documentation for optimizer parameters.

        Raises
        ------
        NotImplementedError
            If the optimizer is not supported.

        Returns
        -------
        optimizer
            Initialized optimizer.
        """

        optimizer_name: str = self.config["optimizer"]["name"]

        # Retrieve optimizer parameters, defaulting to an empty dict if not specified
        optimizer_params: dict = self.config["optimizer"]["params"]

        # Mapping of optimizer names to their corresponding PyTorch classes
        optimizer_mapping = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
            # Add other optimizers as needed
        }

        # Check if the specified optimizer is supported
        if optimizer_name not in optimizer_mapping:
            raise NotImplementedError(f"Optimizer type '{optimizer_name}' is not supported. Please check the configuration. Supported optimizers are :\n{'\n'.join(optimizer_mapping.keys())}")

        optimizer_class = optimizer_mapping[optimizer_name]
        
        
        # Initialize the optimizer with given parameters
        optimizer: optim.Optimizer = optimizer_class(self.parameters(), **optimizer_params)

        
        logging.info(f"Optimizer '{optimizer_name}' initialized with parameters: {optimizer_params}")
        return optimizer

    def initialize_negative_sampler(self) -> sampling.NegativeSampler:
        """Initialize the sampler according to the configuration.
        
            Supported samplers are Positional, Uniform, Bernoulli and Mixed.
            They are adapted from torchKGE's samplers to be compatible with the 
            edgelist format.

            Raises
            ------
            NotImplementedError
                If the name of the sampler is not supported.

            Returns
            -------
            sampler
                The initialized sampler"""
        
        negative_sampler_config: dict = self.config["sampler"]
        negative_sampler_name: str = negative_sampler_config["name"]
        negative_triplet_count: int = negative_sampler_config["n_neg"]

        match negative_sampler_name:
            case "Positional":
                negative_sampler = PositionalNegativeSampler(self.kg_train)
            case "Uniform":
                negative_sampler = UniformNegativeSampler(self.kg_train, negative_triplet_count)
            case "Bernoulli":
                negative_sampler = BernoulliNegativeSampler(self.kg_train, negative_triplet_count)
            case "Mixed":
                negative_sampler = MixedNegativeSampler(self.kg_train, negative_triplet_count)
            case _:
                raise NotImplementedError(f"Sampler type '{negative_sampler_name}' is not supported. Please check the configuration.")
            
        return negative_sampler
    
    def initialize_learning_rate_scheduler(self) -> learning_rate_scheduler.LRScheduler | None:
        """
        Initializes the learning rate scheduler based on the provided configuration.
                
        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: Instance of the specified scheduler or
                                                            None if no scheduler is configured.
        
        Raises:
            ValueError: If the scheduler type is unsupported or required parameters are missing.
        """
        learning_rate_scheduler_config: dict = self.config["lr_scheduler"]
        
        if learning_rate_scheduler_config["type"] == "":
            warnings.warn("No learning rate scheduler specified in the configuration, none will be used.")
            return None
    
        learning_rate_scheduler_type: str = learning_rate_scheduler_config["type"]
        learning_rate_scheduler_params: dict = learning_rate_scheduler_config["params"]
        # Mapping of scheduler names to their corresponding PyTorch classes
        learning_rate_scheduler_mapping = {
            "StepLR": learning_rate_scheduler.StepLR,
            "MultiStepLR": learning_rate_scheduler.MultiStepLR,
            "ExponentialLR": learning_rate_scheduler.ExponentialLR,
            "CosineAnnealingLR": learning_rate_scheduler.CosineAnnealingLR,
            "CosineAnnealingWarmRestarts": learning_rate_scheduler.CosineAnnealingWarmRestarts,
            "ReduceLROnPlateau": learning_rate_scheduler.ReduceLROnPlateau,
            "LambdaLR": learning_rate_scheduler.LambdaLR,
            "OneCycleLR": learning_rate_scheduler.OneCycleLR,
            "CyclicLR": learning_rate_scheduler.CyclicLR,
        }

        # Verify that the scheduler type is supported
        if learning_rate_scheduler_type not in learning_rate_scheduler_mapping:
            raise ValueError(f"Scheduler type '{learning_rate_scheduler_type}' is not supported. Please check the configuration.")
        learning_rate_scheduler_class = learning_rate_scheduler_mapping[learning_rate_scheduler_type]
        
        # Initialize the scheduler based on its type
        try:
                learning_rate_scheduler: learning_rate_scheduler.LRScheduler = learning_rate_scheduler_class(self.optimizer, **learning_rate_scheduler_params)
        except TypeError as e:
            raise ValueError(f"Error initializing '{learning_rate_scheduler_type}': {e}")

        
        logging.info(f"Scheduler '{learning_rate_scheduler_type}' initialized with parameters: {learning_rate_scheduler_params}")
        return learning_rate_scheduler

    def initialize_evaluator(self) -> LinkPredictionEvaluator | TripletClassificationEvaluator:
        """Set the task for which the model will be evaluated on using the validation set.
        
        Options are Link Prediction or Triplet Classification.
        Link Prediction evaluate the ability of a model to predict correctly the head or tail of a triple given the other 
        entity and relation. 
        Triplet Classification evaluate the ability of a model to discriminate between existing and 
        fake triplet in a KG.
        
        Raises
        ------
        NotImplementedError
            If the name of the task is not supported.
            
        Returns
        -------
        evaluator
            The initialized evaluator, either LinkPredictionEvaluator or TripletClassificationEvaluator."""
        match self.config["evaluation"]["objective"]:
            case "Link Prediction":
                full_edgelist = torch.cat([
                    self.kg_train.edgelist,
                    self.kg_train.removed_triplets,
                    self.kg_validation.edgelist,
                    self.kg_validation.removed_triplets,
                    self.kg_test.edgelist,
                    self.kg_test.removed_triplets
                ], dim=1)
                evaluator = LinkPredictionEvaluator(full_edgelist=full_edgelist)
                self.validation_metric = "MRR"
            case "Triplet Classification":
                evaluator = TripletClassificationEvaluator(architect=self, kg_validation = self.kg_validation, kg_test=self.kg_test)
                self.validation_metric = "Accuracy"
            case _:
                raise NotImplementedError(f"The requested evaluator {self.config["evaluation"]["objective"]} is not implemented.")
            
        logging.info(f"Using {self.config["evaluation"]["objective"]} evaluator.")
        return evaluator

    def initialize_model(self, attributes: Dict[str,pd.DataFrame]={}, pretrained: Path | None = None):
        """Initializes every components of the model. This is done automatically by running the train_model method.
        
        Arguments:
            attributes: dict(node_type, embedding) containing the embedding for each type of node.
            pretrained: path to the pretrained embeddings
        """
        # Cannot use short-circuit syntax with tuples
        logging.info("Initializing decoder...")
        if self.decoder is None:
            self.decoder, self.decoder_loss = self.initialize_decoder()
            self.decoder.to(self.device)

        logging.info("Initializing encoder...")
        self.encoder = self.encoder or self.initialize_encoder()

        logging.info("Initializing embeddings...")
        
        # If we have been given a pretrained embedding file (such as the output of a node2vec), we use that in priority
        if pretrained is not None and pretrained.exists():
            self.node_embeddings = torch.load(pretrained)
        # elif not isinstance(self.encoder, GNN):
        #     self.node_embeddings = init_embedding(self.kg_train.n_ent, self.emb_dim, self.device)
        else:
            assert isinstance(self.encoder, GNN) or len(self.kg_train.node_type_to_index) == 1, "When using a GNN as encoder, the node_type shouldn't be supplied."

            self.node_embeddings = nn.ParameterList()
            index_to_node_type = {value: key for key,value in self.kg_train.node_type_to_index.items()}
            for node_type in self.kg_train.node_type_to_global:
                node_count = self.kg_train.node_type_to_global[node_type].size(0)
                if node_type in attributes:
                    current_attribute: pd.DataFrame = attributes[node_type]
                    assert current_attribute.shape[0] == node_count, f"The length of the given attribute ({len(current_attribute)}) must match the number of nodes of this type ({node_count})."
                    input_features = torch.zeros((node_count,current_attribute.shape[1]), dtype=torch.float)
                    for node in current_attribute.index:
                        node_index = self.kg_train.node_to_index[node]
                        node_type_index = self.kg_train.node_types[node_index]
                        local_idx = self.kg_train.global_to_local_indices[node_index]
                        assert node_type_index == self.kg_train.node_type_to_index[node_type], f"The entity {node} is given as {node_type} but registered as {index_to_node_type[str(node_type_index)]} in the KG."

                        input_features[local_idx] = tensor(current_attribute.loc[node], dtype=torch.float)
                    
                    self.node_embeddings.append(Parameter(input_features).to(self.device))
                else:
                    embeddings = init_embedding(node_count, self.embedding_dimensions, self.device)
                    self.node_embeddings.append(embeddings.weight)
            # The input features are not supposed to change if we use an encoder
            self.node_embeddings = self.node_embeddings.requires_grad_(False)

        self.edge_embeddings = init_embedding(self.kg_train.edge_count, self.encoder_edge_embedding_dimensions, self.device)


        logging.info("Initializing optimizer...")
        self.optimizer = self.optimizer or self.initialize_optimizer()

        logging.info("Initializing sampler...")
        self.sampler = self.sampler or self.initialize_negative_sampler()

        logging.info("Initializing lr scheduler...")
        self.scheduler = self.scheduler or self.initialize_learning_rate_scheduler()

        logging.info("Initializing evaluator...")
        self.evaluator = self.evaluator or self.initialize_evaluator()

    def train_model(self, checkpoint_file: Path | None = None, attributes: Dict[str,pd.DataFrame]={}, dry_run = False):
        """Launch the training procedure of the Architect.
        
        This function runs the whole training from end to end, leaving out only the evaluation on the test set.
        It uses the `initialize_model` function to prepare the autoencoder as well as the optimizer, negative sampler,
        learning rate scheduler and evaluator.
        The training is executed through a `PyTorch Ignite` `Engine` with a collection of events and parameters:
        - `RunningAverage` to compute the running loss across the batches of the same epoch.
        - `EarlyStopping` to stop the training if the validation MRR does not progress after a number of epochs
            set in the configuration parameters.
        - `Checkpoint` save at a configured interval.
        - Evaluation on the validation set at a configured interval.
        - Metrics logging at each epoch, in the `training_metrics.csv` output file.

        Notes
        -----
        If there is already a configuration file in the output folder identical to the current configuration, KGATE will
        automatically attempt to restart the training from the most recent checkpoint in the `checkpoints/` folder. Otherwise,
        the output folder will be cleaned and the current configuration will be written as `kgate_config.toml`

        Arguments:
            checkpoint_file: The path to the checkpoint file to load and resume a previous training. If None, the training will start from scratch.
            attributes: dict(node_type, embedding) containing the embedding for each type of node.
            dry_run: Initialize every variable and the trainer, but doesn't start the training.
            """
        use_cuda = "all" if self.device.type == "cuda" else None

        train_config: dict = self.config["training"]
        self.max_epochs: int = train_config["max_epochs"]
        self.train_batch_size: int = train_config["train_batch_size"]
        self.patience: int = train_config["patience"]
        self.eval_interval: int = train_config["eval_interval"]
        self.save_interval: int = train_config["save_interval"]

        match train_config["pretrained_embeddings"]:
            case "auto":
                pretrained = Path(self.config["output_directory"]).joinpath("embeddings.pt")
            case "":
                pretrained = None
            case _:
                pretrained = Path(train_config["pretrained_embeddings"])
                if not pretrained.exists(): pretrained = None
        
        self.initialize_model(attributes=attributes, pretrained=pretrained)

        self.train_metrics_file: Path = Path(self.config["output_directory"], "training_metrics.csv")

        if checkpoint_file is None:
            with open(self.train_metrics_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Training Loss", f"Validation {self.validation_metric}", "Learning Rate"])
        
        self.train_losses: List[float] = []
        self.validation_metric_value: List[float] = []
        self.learning_rates: List[float] = []

        data_loader: DataLoader = DataLoader(self.kg_train, self.train_batch_size)
        logging.info(f"Number of training batches: {len(data_loader)}")

        trainer: Engine = Engine(self.process_batch)
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss_ra")

        progress_bar = ProgressBar()
        progress_bar.attach(trainer)

        early_stopping: EarlyStopping = EarlyStopping(
            patience = self.patience,
            score_function = self.score_function,
            trainer = trainer
        )

        # If we find an identical config we resume training from it, otherwise we clean the checkpoints directory.
        existing_config_path: Path = Path(self.config["output_directory"]).joinpath("kgate_config.toml")
        if existing_config_path.exists():
            existing_config = parse_config(str(existing_config_path), {})
            all_checkpoints = glob(f"{self.checkpoints_directory}/checkpoint_*.pt")
            if existing_config == self.config and len(all_checkpoints) > 0:
                checkpoint_file = checkpoint_file or Path(max(all_checkpoints, key=os.path.getctime))
                logging.info("Found previous run with the same configuration in the output folder...   ")
        elif self.checkpoints_directory.exists() and len(os.listdir(self.checkpoints_directory)) > 0:
            shutil.rmtree(self.checkpoints_directory)

        # trainer.add_event_handler(Events.EPOCH_STARTED, self.encoder_pass)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_metrics_to_csv)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.clean_memory)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.update_scheduler)

        trainer.add_event_handler(Events.COMPLETED, self.on_training_completed)

        to_save = {
            "relations": self.edge_embeddings,
            "entities": self.node_embeddings,
            "decoder": self.decoder,
            "optimizer": self.optimizer,
            "trainer": trainer,
        }

        if self.encoder.deep:
            to_save.update({"encoder":self.encoder})
        if self.scheduler is not None:
            to_save.update({"scheduler": self.scheduler})
        
        checkpoint_handler = Checkpoint(
            to_save,    # Dict of objects to save
            DiskSaver(dirname=self.checkpoints_directory, require_empty=False, create_dir=True), # Save manager
            n_saved=2,      # Only keep last 2 checkpoints
            global_step_transform=lambda *_: trainer.state.epoch     # Include epoch number
        )

        # Custom save function to move the model to CPU before saving and back to GPU after
        def save_checkpoint_to_cpu(engine: Engine):
            # Move models to CPU before saving
            if self.encoder.deep:
                self.encoder.to("cpu")
            self.decoder.to("cpu")
            self.edge_embeddings.to("cpu")
            self.node_embeddings.to("cpu")

            # Save the checkpoint
            checkpoint_handler(engine)

            # Move models back to GPU
            if self.encoder.deep:
                self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.edge_embeddings.to(self.device)
            self.node_embeddings.to(self.device)

        # Attach checkpoint handler to trainer and call save_checkpoint_to_cpu
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), save_checkpoint_to_cpu)
    
        checkpoint_best_handler: ModelCheckpoint = ModelCheckpoint(
            dirname=self.checkpoints_directory,
            filename_prefix="best_model",
            n_saved=1,
            score_function=self.get_val_metrics,
            score_name="val_metrics",
            require_empty=False,
            create_dir=True,
            atomic=True
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.eval_interval), self.evaluate)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.eval_interval), early_stopping)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.eval_interval),
            checkpoint_best_handler,
            to_save
        )

        save_config(self.config)

        if checkpoint_file is not None:
            if Path(checkpoint_file).is_file():
                logging.info(f"Resuming training from checkpoint: {checkpoint_file}")
                logging.info(f"rel_emb size : {self.edge_embeddings.weight.size()}")
                checkpoint = torch.load(checkpoint_file, weights_only=False)
                Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

                logging.info("Checkpoint loaded successfully.")
                with open(self.train_metrics_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["CHECKPOINT RESTART", "CHECKPOINT RESTART", "CHECKPOINT RESTART", "CHECKPOINT RESTART"])

                if trainer.state.epoch < self.max_epochs:
                    logging.info(f"Starting from epoch {trainer.state.epoch}")
                    if not dry_run:
                        trainer.run(data_loader)
                else:
                    logging.info(f"Training already completed. Last epoch is {trainer.state.epoch} and max_epochs is set to {self.max_epochs}")
            else:
                logging.info(f"Checkpoint file {checkpoint_file} does not exist. Starting training from scratch.")
                if not dry_run:
                    trainer.run(data_loader, max_epochs=self.max_epochs)
        else:
            if not dry_run:
                self.normalize_parameters()
                trainer.run(data_loader, max_epochs=self.max_epochs)
    

    def test(self):
        torch.cuda.empty_cache()
        gc.collect()

        self.load_best_model()
        self.evaluator = self.initialize_evaluator()

        self.eval()

        list_rel_1: List[str] = self.config["evaluation"]["made_directed_relations"]
        list_rel_2: List[str] = self.config["evaluation"]["target_relations"]
        thresholds: List[int] = self.config["evaluation"]["thresholds"]
        metrics_file: Path = Path(self.config["output_directory"], "evaluation_metrics.yaml")

        all_edges: Set[Any] = set(self.kg_test.edge_to_index.keys())
        remaining_edges = all_edges - set(list_rel_1) - set(list_rel_2)
        remaining_edges = list(remaining_edges)

        total_metrics_sum_list_1, fact_count_list_1, individual_metrics_list_1, group_metrics_list_1 = self.calculate_metrics_for_relations(
            self.kg_test, list_rel_1)
        total_metrics_sum_list_2, fact_count_list_2, individual_metrics_list_2, group_metrics_list_2 = self.calculate_metrics_for_relations(
            self.kg_test, list_rel_2)
        total_metrics_sum_remaining, fact_count_remaining, individual_metrics_remaining, group_metrics_remaining = self.calculate_metrics_for_relations(
            self.kg_test, remaining_edges)

        global_metrics = (total_metrics_sum_list_1 + total_metrics_sum_list_2 + total_metrics_sum_remaining) / (fact_count_list_1 + fact_count_list_2 + fact_count_remaining)

        logging.info(f"Final Test metrics with best model: {global_metrics}")

        results = {
            "Global_metrics": global_metrics,
            "made_directed_relations": {
                "Global_metrics": group_metrics_list_1,
                "Individual_metrics": individual_metrics_list_1
            },
            "target_relations": {
                "Global_metrics": group_metrics_list_2,
                "Individual_metrics": individual_metrics_list_2
            },
            "remaining_relations": {
                "Global_metrics": group_metrics_remaining,
                "Individual_metrics": individual_metrics_remaining
            },
            "target_relations_by_frequency": {}  
        }

        for i in range(len(list_rel_2)):
            edge: str = list_rel_2[i]
            threshold: int = thresholds[i]
            frequent_indices, infrequent_indices = self.categorize_test_nodes(edge, threshold)
            frequent_metrics, infrequent_metrics = self.calculate_metrics_for_categories(frequent_indices, infrequent_indices)
            logging.info(f"Metrics for frequent nodes (threshold={threshold}) in relation {edge}: {frequent_metrics}")
            logging.info(f"Metrics for infrequent nodes (threshold={threshold}) in relation {edge}: {infrequent_metrics}")

            results["target_relations_by_frequency"][edge] = {
                "Frequent_metrics": frequent_metrics,
                "Infrequent_metrics": infrequent_metrics,
                "Threshold": threshold
            }
                
        self.test_results = results
        
        with open(metrics_file, "w") as file:
            yaml.dump(results, file, default_flow_style=False, sort_keys=False)

        logging.info(f"Evaluation results stored in {metrics_file}")

        return results
        
    def infer(self, heads:List[str]=[], edges:List[str]=[], tails:List[str]=[], top_k:int=100, identity:str=""):
        """Infer missing entities or relations, depending on the given parameters.
        
        Only two of heads, rels and tails must be given, and the other one will be inferred. For example, when inferring tails,
        for each couple `heads[n]` and `rels[n]`, `top_k` tails will be predicted. The values in those list must correspond to
        the `identity` of the metadata, by default the current identity. If there is no metadata, the node ID is used.
        
        Arguments
        ---------
            heads: List[str], optional
                List of known head entities
            rels: List[str], optional
                List of known relations
            tails: List[str], optional
                List of known tail entities
            top_k: int, optional
                Number of prediction to return for each couple in the list.
            identity: str, optional
                The identity to use to predict links. Default is the current identity.
                
        Returns
        -------
            predictions: pd.DataFrame
                A DataFrame containing the prediction alongside their score."""
        if not sum([len(arr) > 0 for arr in [heads,edges,tails]]) == 2:
            raise ValueError("To infer missing elements, exactly 2 lists must be given between heads, relations or tails.")
        torch.cuda.empty_cache()
        gc.collect()

        self.load_best_model()

        do_heads_inference, do_edges_inference, do_tails_inference = len(heads) == 0, len(edges) == 0, len(tails) == 0

        full_kg = merge_kg([self.kg_train, self.kg_validation, self.kg_test], True)

        if do_tails_inference:
            first_known_triplet_part = tensor([self.kg_train.node_to_index[head] for head in heads]).long()
            second_known_triplet_part = tensor([self.kg_train.edge_to_index[rel] for rel in edges]).long()
            missing_triplet_part = "tail"
            inference = NodeInference(full_kg)
        elif do_heads_inference:
            first_known_triplet_part = tensor([self.kg_train.node_to_index[tail] for tail in tails]).long()
            second_known_triplet_part = tensor([self.kg_train.edge_to_index[rel] for rel in edges]).long()
            missing_triplet_part = "head"
            inference = NodeInference(full_kg)
        elif do_edges_inference:
            first_known_triplet_part = tensor([self.kg_train.node_to_index[head] for head in heads]).long()
            second_known_triplet_part = tensor([self.kg_train.node_to_index[tail] for tail in tails]).long()
            missing_triplet_part = "rel"
            inference = EdgeInference(full_kg)
            
        predictions, scores = inference.evaluate(
            first_known_triplet_part,
            second_known_triplet_part,
            encoder = self.encoder,
            decoder = self.decoder,
            top_k = top_k,
            missing = missing_triplet_part,
            b_size = self.evaluation_batch_size,
            node_embeddings=self.node_embeddings,   
            relation_embeddings=self.edge_embeddings
        )

        index_to_node = {value: key for key, value in self.kg_train.node_to_index.items()}
        prediction_index = predictions.reshape(-1).T
        prediction_names = np.vectorize(index_to_node.get)(prediction_index)

        scores = scores.reshape(-1).T
        
        return pd.DataFrame([prediction_names,scores], columns= ["Prediction","Score"])

    def load_checkpoint(self, path: Path, loose=False) -> dict:
        """Parse an Architect checkpoint to ensure it can properly be loaded.
        
        Arguments
        ---------
        path: pathlib.Path
            The path to the checkpoint that will be loaded
        loose: bool, default to False
            If true, will try to change the current configuration to match the checkpoint's and avoid errors.
            
        Returns
        -------
        checkpoint: dict
            The loaded checkpoint as a dictionnary."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Check entity and relation dict size
        assert len(checkpoint["relations"]["weight"]) == self.n_rel, f"Mismatch between the number of relations in the checkpoint ({len(checkpoint['relations']['weight'])}) and the current configuration ({self.n_rel})!"

        if isinstance(self.encoder, GNN):
            assert len(checkpoint["entities"]) == len(self.kg_train.node_type_to_index), f"Mismatch between the number of node types in the checkpoint ({len(checkpoint['entities'])}) and the current configuration ({len(self.kg_train.node_type_to_index)})!"
        else:
            assert len(checkpoint["entities"]["weight"]) != self.n_ent, f"Mismatch between the number of entities in the checkpoint ({len(checkpoint['entities'])}) and the current configuration ({self.n_ent})!"

        if "encoder" in checkpoint:
            assert checkpoint["encoder"].keys() == self.encoder.state_dict().keys(), "Mismatch between the checkpoint convolution layers and the current configuration's."

        return checkpoint


    def load_best_model(self):
        """Load into memory the checkpoint corresponding to the highest-performing model on the validation set."""
        _, nt_count = self.kg_train.node_types.unique(return_counts=True)
        self.decoder, _ = self.initialize_decoder()
        self.encoder = self.initialize_encoder()
        self.edge_embeddings = init_embedding(self.n_rel, self.encoder_edge_embedding_dimensions, self.device)

        logging.info("Loading best model.")
        best_model = find_best_model(self.checkpoints_directory)

        if not best_model:
            logging.error(f"No best model was found in {self.checkpoints_directory}. Make sure to run the training first and not rename checkpoint files before running evaluation.")
            return
        
        logging.info(f"Best model is {self.checkpoints_directory.joinpath(best_model)}")
        checkpoint = self.load_checkpoint(self.checkpoints_directory.joinpath(best_model))


        self.node_embeddings = nn.ParameterList()
        for node_type in checkpoint["entities"]:
            self.node_embeddings.append(checkpoint["entities"][node_type].to(self.device))
        
        self.edge_embeddings.load_state_dict(checkpoint["relations"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        if "encoder" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder"])
        
        self.node_embeddings.to(self.device)
        self.edge_embeddings.to(self.device)
        self.decoder.to(self.device)
        self.encoder.to(self.device)
        logging.info("Best model successfully loaded.")


    def forward(self, positive_triplets_batch, negative_triplets_batch) -> Tuple[Tensor,Tensor]:
        """Forward pass of the Architect"""
        positive_triplet: Tensor = self.scoring_function(positive_triplets_batch, self.kg_train)
        # The loss function requires the pos and neg tensors to be of the same size,
        # Thus we duplicate the pos tensor as needed to match the neg.
        negative_triplet_count = negative_triplets_batch.size(1) // positive_triplets_batch.size(1)
        positive_triplet = positive_triplet.repeat(negative_triplet_count)

        negative_triplet: Tensor = self.scoring_function(negative_triplets_batch, self.kg_train)

        return positive_triplet, negative_triplet

    def process_batch(self, engine: Engine, batch: Tensor) -> torch.types.Number:
        batch = batch.T.to(self.device)

        batch_count = self.sampler.corrupt_batch(batch)
        batch_count = batch_count.to(self.device)
        
        self.optimizer.zero_grad()

        # Compute loss with positive and negative triples
        pos, neg = self(batch, batch_count)
        loss = self.decoder_loss(pos, neg)
        loss.backward()

        self.optimizer.step()

        self.normalize_parameters()

        return loss.item()

    def scoring_function(self, batch: Tensor, kg:KnowledgeGraph) -> Tensor:
        """Runs the encoder and decoder pass on a batch for a given KG.
        
        If the encoder is not a GNN, directly runs and update the embeddings.
        Otherwise, samples a subgraph from the given batch nodes and runs the encoder before.
        
        Arguments
        ---------
        batch: torch.Tensor
            Batch of triples, in the format [4, batch_size]. The rows corresponds to:
            - head_idx
            - tail_idx
            - rel_idx
            - triple_idx
        kg: KnowledgeGraph
            The Knowledge Graph corresponding to the batch identifiers.
            
        Returns
        -------
        score: Tensor
            The score given by the decoder for the batch..
        """
        head_indices, tail_indices, edge_indices = batch[0], batch[1], batch[2]
        
        if isinstance(self.encoder, GNN):
            seed_nodes = batch[:2].unique()
            hop_count = self.encoder.n_layers
            edge_index = kg.edge_index
            
            _,_,_, edge_mask = k_hop_subgraph(
                node_idx = seed_nodes,
                num_hops = hop_count,
                edge_index = edge_index
                )
                
            input = kg.get_encoder_input(kg.edgelist[:, edge_mask].to(self.device), self.node_embeddings)

            encoder_output: Dict[str, Tensor] = self.encoder(input.x_dict, input.edge_index)

            # As I understand it, this tensor is larger than needs to be because it needs to account for every possible
            # idx of the embeddings. It's not a logic problem as only the indices from the batch will be selected for the decoder,
            # which corresponds to the indices that are filled here.
            # TODO: See if making it a sparse tensor can spare memory
            embeddings: torch.Tensor = torch.zeros((kg.node_count, self.encoder_node_embedding_dimensions), device=self.device, dtype=torch.float)

            for node_type, index in input.mapping.items():
                embeddings[index] = encoder_output[node_type]

        else:
            embeddings = self.node_embeddings.weight
        
        
        head_embeddings = embeddings[head_indices]
        tail_embeddings = embeddings[tail_indices]
        edge_embeddings = self.edge_embeddings(edge_indices)  # Relations are unchanged

        return self.decoder.score(h_emb = head_embeddings,
                                  r_emb = edge_embeddings, 
                                  t_emb = tail_embeddings, 
                                  h_idx = head_indices, 
                                  r_idx = edge_indices, 
                                  t_idx = tail_indices)

    def get_embeddings(self) -> Dict[str,Tensor]:
        """Returns the embeddings of entities and relations, as well as decoder-specific embeddings.
        
        If the encoder uses heteroData, a dict of {node_type : embedding} is returned for entity embeddings instead of a tensor."""
        self.normalize_parameters()
        
        if isinstance(self.node_embeddings, nn.ParameterList):
            input = self.kg_train.get_encoder_input(self.kg_train.edgelist.to(self.device), self.node_embeddings)

            encoder_output: Dict[str, Tensor] = self.encoder(input.x_dict, input.edge_index)
            node_embeddings: torch.Tensor = torch.zeros((self.n_ent, self.encoder_node_embedding_dimensions), device=self.device, dtype=torch.float)

            for node_type, idx in input.mapping.items():
                node_embeddings[idx] = encoder_output[node_type]
        else:
            node_embeddings = self.node_embeddings.weight.data

        edge_embeddings = self.edge_embeddings.weight.data

        decoder_embeddings = self.decoder.get_embeddings()

        embedding_dictionnary = {"entities": node_embeddings, "relations": edge_embeddings,}

        if decoder_embeddings is not None:
            embedding_dictionnary.update({"decoder": decoder_embeddings})

        return embedding_dictionnary

    def normalize_parameters(self):
        # Some decoders should not normalize parameters or do so in a different way.
        # In this case, they should implement the function themselves and we return it.
        normalize_function: Callable[..., Tuple[nn.ParameterList, nn.Embedding]] | None = getattr(self.decoder, "normalize_params", None)
        # If the function only accept one parameter, it is the base torchKGE one,
        # we don't want that.
        if callable(normalize_function) and len(signature(normalize_function).parameters) > 1:
            normalized_embeddings = normalize_function(ent_emb = self.node_embeddings, rel_emb = self.edge_embeddings)
            assert len(normalized_embeddings) == 2, "The decoder.normalize_params method should return exactly two elements, the entity embedding and the relation embedding."
            self.node_embeddings, self.edge_embeddings = normalized_embeddings
            
        logging.debug(f"Normalized all embeddings")

    ##### Metrics recording in CSV file
    def log_metrics_to_csv(self, engine: Engine):
        epoch = engine.state.epoch
        train_loss = engine.state.metrics["loss_ra"]
        validation_metric_value = engine.state.metrics.get("val_metrics", 0)
        learning_rate = self.optimizer.param_groups[0]["lr"]

        self.train_losses.append(train_loss)
        self.validation_metric_value.append(validation_metric_value)
        self.learning_rates.append(learning_rate)

        with open(self.train_metrics_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, validation_metric_value, learning_rate])

        logging.info(f"Epoch {epoch} - Train Loss: {train_loss}, Validation {self.validation_metric}: {validation_metric_value}, Learning Rate: {learning_rate}")

    ##### Memory cleaning
    def clean_memory(self, engine:Engine):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Memory cleaned.")

    ##### Evaluation on validation set
    def evaluate(self, engine:Engine):
        logging.info(f"Evaluating on validation set at epoch {engine.state.epoch}...")
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if isinstance(self.evaluator,LinkPredictionEvaluator):
                validation_score = self.link_pred(self.kg_validation) 
                engine.state.metrics["val_metrics"] = validation_score 
                logging.info(f"Validation MRR: {validation_score}")
            elif isinstance(self.evaluator, TripletClassificationEvaluator):
                validation_score = self.triplet_classif(self.kg_validation, self.kg_test)
                engine.state.metrics["val_metrics"] = validation_score
                logging.info(f"Validation Accuracy: {validation_score}")
        if self.scheduler and isinstance(self.scheduler, learning_rate_scheduler.ReduceLROnPlateau):
            self.scheduler.step(validation_score)
            logging.info("Stepping scheduler ReduceLROnPlateau.")

        self.train() # Set the model back to training mode

    ##### Scheduler update
    def update_scheduler(self, engine: Engine):
        if self.scheduler is not None and not isinstance(self.scheduler, learning_rate_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

    ##### Early stopping score function
    def score_function(self, engine: Engine) -> float:
        return engine.state.metrics.get("val_metrics", 0)
    
    ##### Checkpoint best metric
    def get_val_metrics(self, engine: Engine) -> float:
        return engine.state.metrics.get("val_metrics", 0)
    
    ##### Late stopping
    def on_training_completed(self, engine: Engine):
        """Plot the training loss and validation MRR curves once the training is over."""
        logging.info(f"Training completed after {engine.state.epoch} epochs.")

        plot_learning_curves(self.train_metrics_file, self.config["output_directory"], self.validation_metric)

    # TODO : create a script to isolate prediction functions. Maybe a Predictor class?
    def categorize_test_nodes(self, edge_name: str, threshold: int) -> Tuple[List[int], List[int]]:
        """
        Categorizes test triples with the specified relation in the test set 
        based on whether their entities have been seen with that relation in the training set,
        and separates them into two groups based on a threshold for occurrences.

        Parameters
        ----------
        relation_name : str
            The name of the relation to check (e.g., "indication").
        threshold : int
            The minimum number of occurrences of the relation for a node to be considered as "frequent".

        Returns
        -------
        frequent_indices : list
            Indices of triples in the test set with the specified relation where entities have been seen more than `threshold` times with that relation in the training set.
        infrequent_indices : list
            Indices of triples in the test set with the specified relation where entities have been seen fewer than or equal to `threshold` times with that relation in the training set.
        """
        # Get the index of the specified relation in the training graph
        if edge_name not in self.kg_train.edge_to_index:
            raise ValueError(f"The relation '{edge_name}' does not exist in the training knowledge graph.")
        edge_index = self.kg_train.edge_to_index[edge_name]

        # Count occurrences of nodes with the specified relation in the training set
        train_node_counts = {}
        for i in range(self.kg_train.triplet_count):
            if self.kg_train.edges[i].item() == edge_index:
                head = self.kg_train.head_indices[i].item()
                tail = self.kg_train.tail_indices[i].item()
                train_node_counts[head] = train_node_counts.get(head, 0) + 1
                train_node_counts[tail] = train_node_counts.get(tail, 0) + 1

        # Separate test triples with the specified relation based on the threshold
        frequent_indices = []
        infrequent_indices = []
        for i in range(self.kg_test.triplet_count):
            if self.kg_test.edges[i].item() == edge_index:  # Only consider triples with the specified relation
                head = self.kg_test.head_indices[i].item()
                tail = self.kg_test.tail_indices[i].item()
                head_count = train_node_counts.get(head, 0)
                tail_count = train_node_counts.get(tail, 0)

                # Categorize based on threshold
                if head_count > threshold or tail_count > threshold:
                    frequent_indices.append(i)
                else:
                    infrequent_indices.append(i)

        return frequent_indices, infrequent_indices
    
    def calculate_metrics_for_relations(self, kg: KnowledgeGraph, edge_indices: List[str]) -> Tuple[float, int, Dict[str, float], float]:
        # MRR computed by ponderating for each relation
        metrics_sum = 0.0
        fact_count = 0
        individual_metrics = {} 

        for edge_name in edge_indices:
            # Get triples associated with index
            relation_index = kg.edge_to_index.get(edge_name)
            indices_to_keep = torch.nonzero(kg.edges == relation_index, as_tuple=False).squeeze()

            if indices_to_keep.numel() == 0:
                continue  # Skip to next relation if no triples found
            
            new_kg = kg.keep_triples(indices_to_keep)

            if isinstance(self.evaluator, LinkPredictionEvaluator):
                test_metrics = self.link_pred(new_kg)
            elif isinstance(self.evaluator, TripletClassificationEvaluator):
                test_metrics = self.triplet_classif(kg_validation = self.kg_validation, kg_test = new_kg)
            
            # Save each relation's MRR
            individual_metrics[edge_name] = test_metrics
            
            metrics_sum += test_metrics * indices_to_keep.numel()
            fact_count += indices_to_keep.numel()
        
        # Compute global MRR for the relation group
        group_metrics = metrics_sum / fact_count if fact_count > 0 else 0
        
        return metrics_sum, fact_count, individual_metrics, group_metrics

    def calculate_metrics_for_categories(self, frequent_indices: List[int], infrequent_indices: List[int]) -> Tuple[float, float]:
        """
        Calculate the MRR for frequent and infrequent categories based on given indices.
        
        Parameters
        ----------
        frequent_indices : list
            Indices of test triples considered as frequent.
        infrequent_indices : list
            Indices of test triples considered as infrequent.

        Returns
        -------
        frequent_mrr : float
            MRR for the frequent category.
        infrequent_mrr : float
            MRR for the infrequent category.
        """

        # Create subgraph for frequent and infrequent categories
        kg_frequent = self.kg_test.keep_triples(frequent_indices)
        kg_infrequent = self.kg_test.keep_triples(infrequent_indices)
        
        # Compute each category's MRR
        if isinstance(self.evaluator, LinkPredictionEvaluator):
            frequent_metrics = self.link_pred(kg_frequent) if frequent_indices else 0
            infrequent_metrics = self.link_pred(kg_infrequent) if infrequent_indices else 0
        elif isinstance(self.evaluator, TripletClassificationEvaluator):
            frequent_metrics = self.triplet_classif(self.kg_validation, kg_frequent) if frequent_indices else 0
            infrequent_metrics = self.triplet_classif(self.kg_validation, kg_infrequent) if infrequent_indices else 0
        return frequent_metrics, infrequent_metrics

    def link_pred(self, kg: KnowledgeGraph) -> float:
        """Link prediction evaluation on test set."""
        # Test MRR measure
        if not isinstance(self.evaluator, LinkPredictionEvaluator):
            raise ValueError(f"Wrong evaluator called. Calling Link Prediction method for {type(self.evaluator)} evaluator.")

        self.evaluator.evaluate(b_size = self.evaluation_batch_size,
                        encoder=self.encoder,
                        decoder =self.decoder,
                        knowledge_graph=kg,
                        node_embeddings=self.node_embeddings, 
                        relation_embeddings=self.edge_embeddings,
                        verbose=True)
        
        test_mrr = self.evaluator.mrr()[1]
        return test_mrr
    
    def triplet_classif(self, kg_validation: KnowledgeGraph, kg_test: KnowledgeGraph) -> float:
        """Triplet Classification evaluation"""
        if not isinstance(self.evaluator, TripletClassificationEvaluator):
            raise ValueError(f"Wrong evaluator called. Calling Triplet Classification method for {type(self.evaluator)} evaluator.")
        
        self.evaluator.evaluate(b_size=self.evaluation_batch_size, knowledge_graph=kg_validation)
        return self.evaluator.accuracy(self.evaluation_batch_size, kg_test = kg_test)

    def run_dl(self, attributes: Dict[str, pd.DataFrame] ={}):
        logging.info("Preparing KG for DL evaluation pocedure...")
        data_leakage_config = self.config["data_leakage"]

        kg = merge_kg([self.kg_train, self.kg_validation, self.kg_test])

        for edge in data_leakage_config["permuted_relations"]:
            if edge not in self.kg_train.edge_to_index:
                raise ValueError(f"Relation name {edge} was not found in the knowledge graph.")
            logging.info(f"Permutting tails of relation {edge}")
            self.kg_train = permute_tails(self.kg_train, edge)

        self.kg_train, self.kg_validation, self.kg_test = kg.split_kg(shares=self.config["preprocessing"]["split"])

        self.train_model(attributes=attributes)