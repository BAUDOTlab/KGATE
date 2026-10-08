"""
Architect class and methods to run a KGE model training, testing and inference from end to end.
"""

import csv
import gc
import logging
import os
import platform
import shutil
import warnings
from collections.abc import Callable
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Set, Tuple

import numpy as np
import pandas as pd
import tomli_w
import torch
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
)
from torch import Tensor, optim, tensor
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader, Subset

from .config import Configuration, Normalizer_Configuration, Regularizer_Configuration
from .datasets import load_FB15k_237, load_PrimeKG, load_WN18RR
from .decoders import *
from .encoders import *
from .loss import KGE_Loss, MarginLoss, BinaryCrossEntropyLoss
from .evaluators import LinkPredictionEvaluator, TripletClassificationEvaluator, TripletClassificationResults
from .inference import EdgeInference, NodeInference
from .initializers import *
from .knowledgegraph import KnowledgeGraph
from .preprocessing import SUPPORTED_SEPARATORS, prepare_knowledge_graph
from .regularizers import REGULARIZER_FUNCTIONS, Regularizer
from .normalizers import NORMALIZER_FUNCTIONS, Normalizer
from .samplers import (
    BernoulliNegativeSampler,
    MixedNegativeSampler,
    NegativeSampler,
    PositionalNegativeSampler,
    UniformNegativeSampler,
)
from .modules import *
from .utils import (
    find_best_model,
    load_knowledge_graph,
    plot_learning_curves,
    set_random_seeds,
)

# Configure logging
logging.captureWarnings(True)
logging_level = logging.INFO
logging.basicConfig(
    level = logging_level,  
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


class _BatchedKGSubset(Subset):
    """
    A ``Subset`` over a ``KnowledgeGraph`` that supports PyTorch's vectorized
    ``__getitems__`` batch protocol.

    The ``DataLoader`` fetch of a full batch then reduces to a single tensor
    gather (``graphindices[:, indices]``) instead of one Python ``__getitem__``
    call per sample followed by a ``torch.stack`` in the collate function.
    On PyTorch versions without the ``__getitems__`` hook, the regular
    per-item ``Subset.__getitem__`` path is used as before.
    """

    def __getitems__(self, indices):
        # indices: sequence of positions within this subset (train triplets only)
        train_indices = self.indices[indices]
        # [4, batch_size] -> [batch_size, 4], matching the layout produced by
        # default_collate over the per-item samples
        return self.dataset.graphindices[:, train_indices].T


def _batch_collate(batch):
    """
    Collate function tolerant of both batch-fetch paths:
    a single [batch_size, 4] tensor (vectorized ``__getitems__``) or the
    legacy list of [4] samples (per-item ``__getitem__``).
    """
    if torch.is_tensor(batch):
        return batch
    return torch.stack(batch, dim = 0)


class Architect(Module):
    def __init__(self,
                config_path: str = "",
                knowledge_graph: KnowledgeGraph 
                        | Literal["FB15k-237", "WN18RR", "PrimeKG"]
                        | None = None,
                dataframe: pd.DataFrame
                        | None = None,
                metadata: pd.DataFrame
                        | None = None,
                cudnn_benchmark: bool = True,
                number_of_cores: int = 0,
                **kwargs):
        """
        Architect class for knowledge graph embedding training.

        The Architect class contains the kg and manages every step from the training to the inference.

        Arguments
        ---------
        
        **config_path** *(str, optional)*
        : Path to the configuration file
        
        **knowledge_graph** *(KnowledgeGraph or str, optional)*
        :  A knowledge graph that may have already been preprocessed by KGATE and split accordingly, or an unprocessed KnowledgeGraph object.
        : Can also be the name of a built-in dataset, one of "FB15k-237", "WN18RR" or "PrimeKG".
        
        **dataframe** *(pd.DataFrame, optional)*
        : The knowledge graph as a pandas dataframe containing at least the columns head, tail and edge, 
        and where each row corresponds to a triplet.
        
        **metadata** *(pd.DataFrame, optional)*
        : The metadata as a pandas dataframe, with at least the columns id and type, where id is the name of the node as it is in the 
        knowledge graph. If this argument is not provided, the metadata will be read from config.metadata if it exists. If both are absent, 
        all nodes will be considered to be the same node type.
        
        **cudnn_benchmark** *(bool, optional, default to True)*
        : Benchmark different convolution algorithms to chose the optimal one.
        : Initialization is slightly longer when it is enabled, and only if cuda is available.
        
        **number_of_cores** *(int, optional, default to 0)*
        : Set the number of cpu cores used by KGATE. If set to 0, all the cores the process has access to are used.
        
        **kwargs** *(dict)*
        : Inline configuration parameters. The name of the arguments must match the parameters found in `config_template.toml`.

        Attributes
        ----------
        
        **configuration** *(Configuration)*
        : The parsed configuration object (see `kgate.config.Configuration`).
        
        **knowledge_graph** *(KnowledgeGraph)*
        : The associated knowledge graph.
        
        **metadata** *(pd.DataFrame)*
        : The metadata dataframe to associate to the knowledge graph.
        
        **node_embedding_dimensions** *(int)*
        : Dimensions of node embeddings, or both node and edge embeddings if they are confounded.
        
        **edge_embedding_dimensions** *(int)*
        : Dimensions of edge embeddings.
        : For most decoders, node and edge embeddings must be identical.
        : If not explicitly different than `node_embedding_dimensions`, it is the same. Most models only support the same value for both hyperparameters.
        
        **node_embeddings** *(nn.ParameterList)*
        : A list containing the node embeddings of each node type, stored in `knowledge_graph.node_embeddings`.
        : Position in the list: node type index (order of `knowledge_graph.node_type_to_index`)
        : Values: tensors of shape [node_count of this type, node_embedding_dimensions]
        
        **edge_embeddings** *(nn.Parameter, shape: [edge_type_count, edge_embedding_dimensions])*
        : Embeddings for each edge type, stored in `knowledge_graph.edge_embeddings`.
        
        **initializer** *(Initializer)*
        : Initializer object to generate the initial embeddings.
        : For more details, refer to the `initialize_initializer` function.

        **encoder** *(GNN or None)*
        : Encoder model of the autoencoder.
        : For more details, refer to the `initialize_encoder` function.
        
        **decoder** *(BilinearDecoder or ConvolutionalDecoder or TranslationalDecoder)*
        : Decoder model of the autoencoder.
        : For more details, refer to the `initialize_decoder` function.
        
        **loss** *(MarginLoss or BinaryCrossEntropyLoss)*
        : The loss object associated with the proper decoder, but may be overwritten.
        : Either `MarginLoss(margin)` or `BinaryCrossEntropyLoss()`.
        
        **sampler** *(NegativeSampler)*
        : Negative sampler.
        : For more details, refer to the `initialize_sampler` function.
        
        **optimizer** *(torch.optim.Optimizer)*
        : Optimizer.
        : For more details, refer to the `initialize_optimizer` function.
        
        **scheduler** *(learning_rate_scheduler.LRScheduler or None)*
        : Learning rate scheduler of KGATE.
        : Modules that alter the learning rate throughout the training.
        : For more details, refer to the `initialize_scheduler` function.
        
        **evaluator** *(LinkPredictionEvaluator or TripletClassificationEvaluator)*
        : The evaluator, either LinkPredictionEvaluator or TripletClassificationEvaluator.
        : For more details, refer to the `initialize_evaluator` function.
        : GPU is referenced to as Cuda.
        
        **device** *(torch.device)*
        : Indicate if data should be sent to GPU ("cuda") or CPU ("cpu").
        
        **checkpoints_directory** *(Path)*
        : Path to the directory containing checkpoint files.
        
        **evaluation_batch_size** *(int)*
        : Size of an evaluation and inference batch.

        Raises
        ------
        
        **pd.errors.InvalidColumnName**
        : The metadata dataframe must have columns named "id" and "type".
        
        **ValueError**
        : The metadata csv file uses a non supported separator.
        : Supported separators are comma (,), tabulation (    ) and semi-colon (;).

        Examples
        --------
        
        Inline hyperparameter declaration (keys must match `config_template.toml`)
        >>> model_params = {"node_embedding_dimensions": 100, "decoder": {"name":"DistMult"}}
        >>> sampler_params = {"negative_triplet_count":5}
        >>> architect = Architect("/path/to/configuration", model = model_params, negative_sampler = sampler_params, preprocessing = {"run_preprocessing": True})

        Notes
        -----
        
        While it is possible to give any part of the configuration as kwargs, even everything, it is strongly recommended 
        to use a separated configuration file to ensure reproducibility of training.

        """
        super().__init__()

        # kg should be of type KnowledgeGraph, if exists use it instead of the one in config
        # dataframe should have columns head, tail and edge
        self.configuration: Configuration = Configuration(config_path = config_path, config_dict = kwargs)

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

        output_directory: Path = Path(self.configuration.output_directory)
        # Create output folder if it doesn't exist
        logging.info(f"Output folder: {output_directory}")
        output_directory.mkdir(parents = True, exist_ok = True)
        self.checkpoints_directory: Path = output_directory.joinpath("checkpoints")

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Detected device: {self.device}")

        set_random_seeds(self.configuration.seed)

        self.node_embedding_dimensions: int = self.configuration.node_embedding_dimensions
        self.edge_embedding_dimensions: int = self.configuration.edge_embedding_dimensions
        if self.edge_embedding_dimensions == -1:
            self.edge_embedding_dimensions = self.node_embedding_dimensions
        self.evaluation_batch_size: int = self.configuration.training.evaluation_batch_size

        self.metadata = None
        if metadata is None:
            metadata = self.configuration.metadata_path if self.configuration.metadata_path.is_file() else None
        self.set_metadata(metadata = metadata)
        
        if isinstance(knowledge_graph, str):
            match knowledge_graph:
                case "FB15k-237":
                    knowledge_graph = load_FB15k_237()
                case "WN18RR":
                    knowledge_graph = load_WN18RR()
                case "PrimeKG":
                    knowledge_graph = load_PrimeKG()
                case _:
                    raise ValueError(f"Unrecognized {knowledge_graph} knowledge graph specified.")
        
        if self.configuration.preprocessing.run:
            logging.info(f"Preparing KG...")
            self.knowledge_graph = prepare_knowledge_graph(self.configuration, knowledge_graph, dataframe, self.metadata)
            logging.info("KG preprocessed.")
        else:
            if knowledge_graph is None:
                logging.info("Loading KG...")
                self.knowledge_graph = load_knowledge_graph(Path(self.configuration.knowledge_graph_pickle_file))
                logging.info("Done")
            else:
                self.knowledge_graph = knowledge_graph

        # Initialize attributes
        self.initializer: Initializer = None
        self.encoder: GNN | None = None
        self.decoder: BilinearDecoder | ConvolutionalDecoder | TranslationalDecoder = None
        self.regularizer: Regularizer | None = None
        self.normalizer: Normalizer | None = None
        self.loss: KGE_Loss = None
        self.skip_normalization: bool = False
        self.optimizer: optim.Optimizer = None
        self.sampler: NegativeSampler = None
        self.scheduler: optim.lr_scheduler.LRScheduler | None = None
        self.evaluator: LinkPredictionEvaluator | TripletClassificationEvaluator = None


    @property
    def encoder_node_embedding_dimensions(self) -> int:
        """
        The embedding dimensions in the output of the encoder (or initialized if there is no encoder).

        For most decoders, it is the same as `self.node_embedding_dimensions`. But for decoders which use multiple 
        embedding spaces, the latent space has `[embedding_spaces_count] * node_embedding_dimensions` embedding dimensions.

        For example, ComplEx uses two embedding spaces: a real one and an imaginary one. Thus, its methods take as input 
        embedding vectors that have `2 * node_embedding_dimensions` embedding dimensions. They are then split and 
        handled correctly from within the encoder.

        Returns
        -------
        
        **node_embedding_dimensions** *(int)*
        : Dimensions of node embeddings in the output of the encoder.
        
        """
        if self.decoder is not None and hasattr(self.decoder, "embedding_spaces"):
            return self.node_embedding_dimensions * self.decoder.embedding_spaces
        
        return self.node_embedding_dimensions


    @property
    def encoder_edge_embedding_dimensions(self) -> int:
        """
        The embedding dimensions in the output of the encoder (or initialized if there is no encoder).

        For most decoders, it is the same as `self.edge_embedding_dimensions`. But for decoders which use multiple 
        embedding spaces, the latent space has `[embedding_spaces_count] * edge_embedding_dimensions` embedding dimensions.

        For example, ComplEx uses two embedding spaces: a real one and an imaginary one. Thus, its methods take as input 
        embedding vectors that have `2 * edge_embedding_dimensions` embedding dimensions. They are then split and 
        handled correctly from within the encoder.

        Returns
        -------
        
        **edge_embedding_dimensions** *(int)*
        : Dimensions of edge embeddings in the output of the encoder.
        
        """
        if self.decoder is not None and hasattr(self.decoder, "embedding_spaces"):
            return self.edge_embedding_dimensions * self.decoder.embedding_spaces
        
        return self.edge_embedding_dimensions

    def set_metadata(self, metadata: pd.DataFrame | os.PathLike | None):
        """
        Set the node metadata of the knowledge graph.

        This function accepts either a pandas DataFrame or the path to a CSV file as input.
        
        The dataframe must have at least columns:
            * "id" which uses the same identifiers as the knowledge graph;
            * "type" which records the type of the corresponding node.
        
        In addition, the metadata can have any number of supplementary columns that can be 
        used to set the identity of the nodes for the associated :class:`~kgate.knowledgegraph.KnowledgeGraph`.

        If there is no knowledge graph associated with the Architect, the `architect.metadata` property will be used 
        to initialize them. If there is already a knowledge graph, it will update the knowledge graph with 
        the new metadata.

        Alternatively, you can directly run the :func:`~kgate.knowledgegraph.KnowledgeGraph.add_metadata` method for a 
        more fine-grained metadata management.
        
        Arguments
        ---------
        
        **metadata** *(pd.DataFrame or os.PathLike)*
        : The metadata object, either as a pandas DataFrame or a path to a CSV file.

        Raises
        ------
        
        **pd.errors.InvalidColumnName**
        : If the columns 'id' and 'type' are not present.
        
        **ValueError**
        : If the CSV file uses an unsupported separator.
        
        **TypeError**
        : If the metadata object is not of the correct type.
        
        """
        match metadata:
            case pd.DataFrame():
                if not set(["id", "type"]).issubset(metadata.keys()):
                    raise pd.errors.InvalidColumnName("The columns \"id\" and \"type\" must be present in the given metadata dataframe.")
        
                self.metadata = metadata
            case os.PathLike():
                if Path(metadata).exists():
                    # Fuzzy identification of separator.
                    # TODO: find a cleaner way to do it
                    for separator in SUPPORTED_SEPARATORS:
                        try:
                            self.metadata = pd.read_csv(metadata, sep = separator, usecols = ["type", "id"])
                            break
                        except ValueError:
                            continue
                
                    if self.metadata is None:
                        raise ValueError(f"The metadata csv file uses a non supported separator. Supported separators are '{'\', \''.join(SUPPORTED_SEPARATORS)}'.")
            case None:
                return
            case _:
                return
            
        if self.metadata is not None and hasattr(self, "knowledge_graph"):
            # If the knowledge graph does not exist yet (e.g. during __init__),
            # it will be created with this metadata by prepare_knowledge_graph.
            self.knowledge_graph.add_metadata(self.metadata)


    def initialize_model(self,
                        attributes: Dict[str, pd.DataFrame] = {},
                        pretrained: Path | None = None):
        """
        Initialize every component of the model.
        
        This is done automatically by running the `train_model` method.
        
        The initialization is done in this order:
            * Initializer
            * Decoder
            * Loss
            * Encoder
            * Node and Edge Embeddings (either at random, or using given node features, or a pretrained file)
            * Regularizer
            * Normalizer
            * Optimizer
            * Negative Sampler
            * Scheduler
            * Evaluator
        
        For each of these elements, if something is already set (i.e. the attribute is not None), it is not re-initialized.
        
        Arguments
        ---------
        
        **attributes** *(dict[str, pd.DataFrame])*
        : dict(node_type, embedding) containing the embedding for each type of node.
        
        **pretrained** *(Path, optional)*
        : Path to the pretrained node embeddings.

        Raises
        ------
        
        **AssertionError #1**
        : When not using a GNN as encoder, the `node_type` should not be supplied.
        
        **AssertionError #2**
        : The length of the given attribute must match the number of nodes of this type.
        
        **AssertionError #3**
        : The node type of each node must correspond to the one registered in the knowledge graph.
        
        """
        # Cannot use short-circuit syntax with tuples
        logging.info("Initializing decoder...")
        if self.decoder is None:
            self.decoder = initialize_decoder(  self.configuration.decoder,
                                                self.knowledge_graph,
                                                self.node_embedding_dimensions,
                                                self.edge_embedding_dimensions,
                                                self.device)

        logging.info("Initializing loss...")
        self.loss = self.loss or initialize_loss(self.configuration.loss)

        logging.info("Initializing encoder...")
        self.encoder = self.encoder or initialize_encoder(self.configuration.encoder, self.knowledge_graph, self.encoder_edge_embedding_dimensions)

        logging.info("Initializing embeddings...")
        self.initializer = self.initializer or initialize_initializer(self.configuration.initializer)

        # If given a pretrained embedding file (such as the output of a Node2Vec), we use that in priority
        if pretrained is not None and pretrained.exists():
            self.knowledge_graph.node_embeddings = torch.load(pretrained)
        elif not (hasattr(self.knowledge_graph.embeddings, "node_embeddings")
                  and hasattr(self.knowledge_graph.embeddings, "edge_embeddings")):
            # We only initialize the embeddings if they don't already exist to
            # avoid optimizer parameter mismatch (among other pitfalls)
            self.initializer.initialize_all_embeddings(self.knowledge_graph,
                                                        node_embedding_dimensions = self.node_embedding_dimensions,
                                                        edge_embedding_dimensions = self.edge_embedding_dimensions,
                                                        device = self.device,
                                                        inplace = True)

        logging.info("Initializing regularizer...")
        self.regularizer = self.regularizer or initialize_regularizer(self.configuration.regularizer, self.knowledge_graph)

        logging.info("Initializing normalizer...")
        self.normalizer = self.normalizer or initialize_normalizer(self.configuration.normalizer, self.knowledge_graph)
        # Run the first normalization of the parameters. This operation not yet tracked by the 
        # optimizer
        self.normalizer.initialize(self.knowledge_graph.node_embeddings, self.knowledge_graph.edge_embeddings)
        logging.info(f"Normalized the initial parameters")

        logging.info("Initializing optimizer...")
        self.optimizer = self.optimizer or initialize_optimizer(self.configuration.optimizer, 
                                                                self.knowledge_graph, 
                                                                decoder = self.decoder, 
                                                                encoder = self.encoder)

        logging.info("Initializing sampler...")
        self.sampler = self.sampler or initialize_negative_sampler(self.configuration.negative_sampler, self.knowledge_graph)

        logging.info("Initializing learning rate scheduler...")
        self.scheduler = self.scheduler or initialize_learning_rate_scheduler(self.configuration.learning_rate_scheduler, self.optimizer)

        logging.info("Initializing evaluator...")
        self.evaluator = self.evaluator or initialize_evaluator(self.configuration.evaluation, self.knowledge_graph, self.node_embedding_dimensions, self)


    def train_model(self,
                    checkpoint_file: Path | None = None,
                    attributes: Dict[str, pd.DataFrame] = {},
                    dry_run: bool = False):
        """
        Launch the training procedure of the Architect.
        
        This function runs the whole training from end to end, leaving out only the evaluation on the test set. 
        It uses the `initialize_model` function to prepare the autoencoder as well as the optimizer, negative sampler,
        learning rate scheduler and evaluator.
        
        The training is executed through a `PyTorch Ignite` `Engine` with a collection of events and parameters:
            * `RunningAverage` to compute the running loss across the batches of the same epoch.
            * `EarlyStopping` to stop the training if the validation MRR does not progress after a number of epochs
                set in the configuration parameters.
            * `Checkpoint` save at a configured interval.
            * Evaluation on the validation set at a configured interval.
            * Metrics logging at each epoch, in the `training_metrics.csv` output file.
            * Application of the training normalizer at the
              beginning of each epoch.
            * Application of the regularizer at the end of each epoch
              (see `initialize_regularizer` and `apply_regularizer`; a no-op if
              no regularizer is configured).


        Arguments
        ---------
        
        **checkpoint_file** *(Path, optional)*
        : The path to the checkpoint file to load and resume a previous training. If None, the training will start from scratch.
        
        **attributes** *(Dict[str, pd.DataFrame])*
        : dict(node_type, embedding) containing the embedding for each type of node.
        
        **dry_run** *(bool, optional, default to False)*
        : Initialize every variable and the trainer, but doesn't start the training.

        Arguments
        ---------
        checkpoint_file: Path, optional
            The path to the checkpoint file to load and resume a previous training. If None, the training will start from scratch.
        attributes: Dict[str, pd.DataFrame]
            dict(node_type, embedding) containing the embedding for each type of node.
        dry_run: bool, optional, default to False
            Initialize every variable and the trainer, but doesn't start the training.

        Notes
        -----
        If there already is a configuration file in the output folder identical to the current configuration, KGATE will 
        automatically attempt to restart the training from the most recent checkpoint in the `checkpoints/` folder. Otherwise, 
        the output folder will be cleaned and the current configuration will be written as `kgate_config.toml`
        
        """
        train_configuration = self.configuration.training
        self.max_epochs: int = train_configuration.max_epochs
        self.train_batch_size: int = train_configuration.train_batch_size
        self.patience: int = train_configuration.patience
        self.evaluation_interval: int = train_configuration.evaluation_interval
        self.save_interval: int = train_configuration.save_interval

        match train_configuration.pretrained_embeddings:
            case "auto":
                pretrained = Path(self.configuration.output_directory).joinpath("embeddings.pt")
            case "":
                pretrained = None
            case _:
                pretrained = Path(train_configuration.pretrained_embeddings)
                if not pretrained.exists(): pretrained = None
        
        self.initialize_model(attributes = attributes, pretrained = pretrained)

        self.train_metrics_file: Path = Path(self.configuration.output_directory, "training_metrics.csv")
        self.validation_metric = "MRR" # rubberband, to fix
        if checkpoint_file is None:
            with open(self.train_metrics_file, mode = "w", newline = "") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Training Loss", f"Validation {self.validation_metric}", "Learning Rate"])
        
        self.train_losses: List[float] = []
        self.validation_metric_value: List[float] = []
        self.learning_rates: List[float] = []

        train_subset = _BatchedKGSubset(self.knowledge_graph, self.knowledge_graph.train_mask.nonzero(as_tuple = True)[0])
        data_loader: DataLoader = DataLoader(train_subset,
                                            self.train_batch_size, 
                                            shuffle=True, 
                                            pin_memory= (self.device.type == "cuda"),
                                            collate_fn = _batch_collate)
        logging.info(f"Number of training batches: {len(data_loader)}")

        trainer: Engine = Engine(self.process_batch)
        # Per-batch running average of the loss (same alpha = 0.98 as the previous
        # ignite RunningAverage, same "loss_running_average" metric key, same float
        # value at epoch end). It is computed on the loss's own device: the standard
        # `ignite.metrics.RunningAverage` calls `loss.detach().to("cpu", copy=True)` on
        # every batch, which forces a GPU->CPU synchronization that serializes the CPU
        # and GPU pipelines (measured cost: ~8-10 ms per batch on the FB15k-237
        # benchmark setup, i.e. ~80-100 s over 100 epochs).
        trainer.add_event_handler(Events.EPOCH_STARTED, self._reset_loss_running_average)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._update_loss_running_average)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._finalize_loss_running_average)

        progress_bar = ProgressBar()
        progress_bar.attach(trainer)

        early_stopping: EarlyStopping = EarlyStopping(
            patience = self.patience,
            score_function = self.get_validation_metric,
            trainer = trainer
        )

        # If we find an identical config we resume training from it, otherwise we clean the checkpoints directory.
        existing_config_path: Path = Path(self.configuration.output_directory).joinpath("kgate_config.toml")
        if existing_config_path.exists():
            existing_config = Configuration(config_path = str(existing_config_path), config_dict = {})
            all_checkpoints = glob(f"{self.checkpoints_directory}/checkpoint_*.pt")
            if existing_config == self.configuration and len(all_checkpoints) > 0:
                checkpoint_file = checkpoint_file or Path(max(all_checkpoints, key = os.path.getctime))
                logging.info("Found previous run with the same configuration in the output folder...")
        elif self.checkpoints_directory.exists() and len(os.listdir(self.checkpoints_directory)) > 0:
            shutil.rmtree(self.checkpoints_directory)

        # Apply the configured regularizer at the end of every epoch 
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.apply_regularizer)
        #trainer.add_event_handler(Events.EPOCH_COMPLETED, self.clean_memory)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.update_scheduler)

        trainer.add_event_handler(Events.COMPLETED, self.on_training_completed)

        checkpoints_count = self.configuration.training.keep_n_checkpoints

        to_save = {
            "embeddings": self.knowledge_graph.embeddings,
            "decoder": self.decoder,
            "optimizer": self.optimizer,
            "trainer": trainer,
        }

        if isinstance(self.encoder, GNN):
            to_save.update({"encoder": self.encoder})
        if self.scheduler is not None:
            to_save.update({"scheduler": self.scheduler})

        if checkpoints_count != 0:
            if checkpoints_count == -1: checkpoints_count = None

            checkpoint_handler = Checkpoint(
                to_save,   # Dictionnary of objects to save
                DiskSaver(dirname = self.checkpoints_directory,
                        require_empty = False,
                        create_dir = True),   # Save manager
                        n_saved = checkpoints_count,   # Only keep last [checkpoint_count] checkpoints
                        global_step_transform = lambda *_: trainer.state.epoch   # Include epoch number
            )

            # Attach checkpoint handler to trainer and call save_checkpoint_to_cpu
            trainer.add_event_handler(Events.EPOCH_COMPLETED(every = self.save_interval), checkpoint_handler)
    
        checkpoint_best_handler: ModelCheckpoint = ModelCheckpoint(
            dirname = self.checkpoints_directory,
            filename_prefix = "best_model",
            n_saved = 1,
            score_function = self.get_validation_metric,
            score_name = "validation_metric_value",
            require_empty = False,
            create_dir = True,
            atomic = True
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every = self.evaluation_interval), self.evaluate)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every = self.evaluation_interval), early_stopping)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every = self.evaluation_interval),
            checkpoint_best_handler,
            to_save
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_metrics_to_csv)

        self.configuration.save()

        if checkpoint_file is not None:
            if Path(checkpoint_file).is_file():
                logging.info(f"Resuming training from checkpoint: {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, weights_only = False)
                Checkpoint.load_objects(to_load = to_save, checkpoint = checkpoint)

                logging.info("Checkpoint loaded successfully.")
                with open(self.train_metrics_file, mode = "a", newline = "") as file:
                    writer = csv.writer(file)
                    writer.writerow(["CHECKPOINT RESTART", "CHECKPOINT RESTART", "CHECKPOINT RESTART", "CHECKPOINT RESTART"])

                if trainer.state.epoch < self.max_epochs:
                    logging.info(f"Starting from epoch {trainer.state.epoch}")
                    if not dry_run:
                        trainer.run(data_loader, max_epochs = self.max_epochs)
                else:
                    logging.info(f"Training already completed. Last epoch is {trainer.state.epoch} and max_epochs is set to {self.max_epochs}")
            else:
                logging.info(f"Checkpoint file {checkpoint_file} does not exist. Starting training from scratch.")
                if not dry_run:
                    trainer.run(data_loader, max_epochs = self.max_epochs)
        else:
            if not dry_run:
                self.normalize_parameters()
                trainer.run(data_loader, max_epochs = self.max_epochs)
    

    def test(self) -> Dict[str, float | Dict[str, float]]:
        """
        Run the test procedure, evaluate the metrics on the test set and return the dictionary of the results.

        The results are also written to `evaluation_metrics.toml` in the output directory.

        Returns
        -------
        
        **results** *(Dict[str, float | Dict[str, float]])*
        : Dictionary containing:
        : - "Global_metrics": the global metric (e.g. MRR) over the whole test set.
        : - "remaining_edges": "Global_metrics" and "Individual_metrics" (metric per edge) for the edges that are not target edges.
        : - "target_edges": (only if target edges are configured) "Global_metrics" and "Individual_metrics" for the target edges.
        : - "target_edges_by_frequency": reserved key, currently empty.
        
        Notes
        -----
        This function is user-facing.
        
        """
        torch.cuda.empty_cache()
        gc.collect()

        self.load_best_model()
        self.evaluator = initialize_evaluator(  self.configuration.evaluation,
                                                self.knowledge_graph,
                                                self.node_embedding_dimensions,
                                                self)

        self.eval()

        target_edges: List[str] = self.configuration.evaluation.target_edges
        metrics_file: Path = Path(self.configuration.output_directory, "evaluation_metrics.toml")

        target_edges_result = {}

        all_edges: Set[Any] = set(self.knowledge_graph.edge_to_index.keys())
        remaining_edges = all_edges - set(target_edges)
        remaining_edges = list(remaining_edges)
        
        triplet_count_target_edges = 0
        test_knowledge_graph = Subset(self.knowledge_graph, self.knowledge_graph.test_mask.nonzero(as_tuple = True)[0])
        
        metrics_sum_target_edges = 0.0
        triplet_count_target_edges = 0

        if len(remaining_edges) != len(all_edges):
            metrics_sum_target_edges, triplet_count_target_edges, individual_metrics_target_edges, group_metrics_target_edges = self.calculate_metrics_for_edges(test_knowledge_graph, target_edges)
            
            target_edges_result = {
                "target_edges": {
                    "Global_metrics": metrics_sum_target_edges,
                    "Individual_metrics": individual_metrics_target_edges
                },
            }

        total_metrics_sum_remaining, triplet_count_remaining, individual_metrics_remaining, group_metrics_remaining = self.calculate_metrics_for_edges(test_knowledge_graph, remaining_edges)

        global_metrics = (metrics_sum_target_edges + total_metrics_sum_remaining) / (triplet_count_target_edges + triplet_count_remaining)

        logging.info(f"Final Test metrics with best model: {global_metrics}")

        results = {
            "Global_metrics": global_metrics,
            **target_edges_result, # if there is no target edges, don't add the block
            "remaining_edges": {
                "Global_metrics": group_metrics_remaining,
                "Individual_metrics": individual_metrics_remaining
            },
            "target_edges_by_frequency": {}  
        }
                
        self.test_results = results
        
        with open(metrics_file, "wb") as file:
            tomli_w.dump(results, file)

        logging.info(f"Evaluation results stored in {metrics_file}")

        return results
    
    
    def infer(self,
            heads: List[str] = [],
            tails: List[str] = [],
            edges: List[str] = [],
            top_k: int = 100):
        """
        Infer missing nodes or edges, depending on the given parameters.
        
        Only two of heads, tails and edges must be given, and the other one will be inferred. For example, when inferring tails, 
        for each couple `heads[n]` and `edges[n]`, `top_k` tails will be predicted. The values in those lists must be the node IDs
        and edge names as they appear in the knowledge graph (the keys of `knowledge_graph.node_to_index` and `knowledge_graph.edge_to_index`).
        
        Arguments
        ---------
        
        **heads** *(List[str], optional)*
        : List of known head nodes.
        
        **tails** *(List[str], optional)*
        : List of known tail nodes.
        
        **edges** *(List[str], optional)*
        : List of known edges.
        
        **top_k** *(int, optional, Default to 100)*
        : Number of prediction to return for each couple in the list.
        
        Raises
        ------
        
        **ValueError**
        : To infer missing elements, exactly 2 lists must be given between heads, tails or edges.
        
        Returns
        -------
        
        **predictions** *(pd.DataFrame)*
        : A DataFrame containing the prediction alongside their score.
        
        Notes
        -----
        This function is user-facing.
        
        """
        if not sum([len(arr) > 0 for arr in [heads, edges, tails]]) == 2:
            raise ValueError("To infer missing elements, exactly 2 lists must be given between heads, tails or edges.")
        torch.cuda.empty_cache()
        gc.collect()

        self.load_best_model()

        do_heads_inference, do_tails_inference, do_edges_inference = len(heads) == 0, len(tails) == 0, len(edges) == 0

        if do_heads_inference:
            first_known_triplet_part = tensor([self.knowledge_graph.node_to_index[tail] for tail in tails]).long()
            second_known_triplet_part = tensor([self.knowledge_graph.edge_to_index[edge] for edge in edges]).long()
            missing_triplet_part = "head"
            inference = NodeInference(self.knowledge_graph)
        elif do_tails_inference:
            first_known_triplet_part = tensor([self.knowledge_graph.node_to_index[head] for head in heads]).long()
            second_known_triplet_part = tensor([self.knowledge_graph.edge_to_index[edge] for edge in edges]).long()
            missing_triplet_part = "tail"
            inference = NodeInference(self.knowledge_graph)
        elif do_edges_inference:
            first_known_triplet_part = tensor([self.knowledge_graph.node_to_index[head] for head in heads]).long()
            second_known_triplet_part = tensor([self.knowledge_graph.node_to_index[tail] for tail in tails]).long()
            missing_triplet_part = "edge"
            inference = EdgeInference(self.knowledge_graph)
            
        predictions, scores = inference.evaluate(
            first_known_triplet_part,
            second_known_triplet_part,
            encoder = self.encoder,
            decoder = self.decoder,
            top_k = top_k,
            missing_triplet_part = missing_triplet_part,
            batch_size = self.evaluation_batch_size,
            node_embeddings = self.knowledge_graph.node_embeddings,   
            edge_embeddings = self.knowledge_graph.edge_embeddings,
            # SpherE (Li et al. 2024): if the decoder uses sphere embeddings,
            # candidates are ranked by their SpherE score and the known
            # (true) candidates are not filtered out (set retrieval).
            sphere_embeddings = self.configuration.decoder.sphere_embeddings,
        )

        index_to_node = {value: key for key, value in self.knowledge_graph.node_to_index.items()}
        prediction_index = predictions.reshape(-1)
        prediction_names = np.vectorize(index_to_node.get)(prediction_index)

        scores = scores.reshape(-1)
        
        return pd.DataFrame({"Prediction":prediction_names,"Score":scores})


    def load_checkpoint(self, path: Path) -> dict:
        """
        Parse an Architect checkpoint to ensure it can properly be loaded.
        
        Arguments
        ---------
        
        **path** *(pathlib.Path)*
        : The path to the checkpoint that will be loaded.
        
        Raises
        ------
        
        **AssertionError #1**
        : The number of edges must be the same in the checkpoint and in the current configuration.
        
        **AssertionError #2**
        : The number of node types must be the same in the checkpoint and in the current configuration.
        
        **AssertionError #3**
        : The number of nodes must be the same in the checkpoint and in the current configuration.
        
        **AssertionError #4**
        : The convolution layers must be the same in the checkpoint and in the current configuration.
        
        Returns
        -------
        
        **checkpoint** *(dict)*
        : The loaded checkpoint as a dictionnary.
        
        """
        checkpoint = torch.load(path, map_location = self.device, weights_only = False)

        # Check node and edge dictionnary size
        assert len(checkpoint["embeddings"]["edge_embeddings"]) == self.knowledge_graph.edge_count, f"Mismatch between the number of edges in the checkpoint ({len(checkpoint["embeddings"]["edge_embeddings"])}) and the current configuration ({self.knowledge_graph.edge_count})!"

        # Check the number of node types, and the total number of nodes across all node types
        node_type_count = len(self.knowledge_graph.node_type_to_index)
        assert len(checkpoint["embeddings"]) - 1 == node_type_count, f"Mismatch between the number of node types in the checkpoint ({len(checkpoint['embeddings']) - 1}) and the current configuration ({node_type_count})!"
        checkpoint_node_count = sum(len(tensor) for key, tensor in checkpoint["embeddings"].items() if key.startswith("node_embeddings."))
        assert checkpoint_node_count == self.knowledge_graph.node_count, f"Mismatch between the number of nodes in the checkpoint ({checkpoint_node_count}) and the current configuration ({self.knowledge_graph.node_count})!"

        if "encoder" in checkpoint:
            assert checkpoint["encoder"].keys() == self.encoder.state_dict().keys(), "Mismatch between the checkpoint convolution layers and the current configuration's."

        return checkpoint


    def load_best_model(self) -> None:
        """
        Load into memory the checkpoint corresponding to the highest-performing model on the validation set.

        If no best-model checkpoint exists (for instance because training did not reach the
        first validation evaluation), the current in-memory model is kept as-is and a warning
        is logged, so that a model just trained can still be evaluated.

        """
        best_model = find_best_model(self.checkpoints_directory)

        if not best_model:
            logging.warning(f"No best model was found in {self.checkpoints_directory}. Evaluating the current in-memory model instead. Train for longer, or use a smaller evaluation/save interval, to produce a best-model checkpoint.")
            return

        self.decoder = initialize_decoder(  self.configuration.decoder, 
                                            self.knowledge_graph,
                                            self.node_embedding_dimensions,
                                            self.edge_embedding_dimensions,
                                            self.device)
        self.encoder = initialize_encoder(  self.configuration.encoder,
                                            self.knowledge_graph,
                                            self.encoder_node_embedding_dimensions)
        initializer = Initializer()
        initializer.initialize_all_embeddings(self.knowledge_graph,
                                            node_embedding_dimensions=self.node_embedding_dimensions,
                                            edge_embedding_dimensions=self.edge_embedding_dimensions,
                                            device = self.device,
                                            inplace=True)
        logging.info("Loading best model.")

        logging.info(f"Best model is {self.checkpoints_directory.joinpath(best_model)}")
        checkpoint = self.load_checkpoint(self.checkpoints_directory.joinpath(best_model))

        self.knowledge_graph.embeddings.load_state_dict(checkpoint["embeddings"])

        self.decoder.load_state_dict(checkpoint["decoder"], strict=False)
        if "encoder" in checkpoint and self.encoder is not None:
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.encoder.to(self.device)
        
        self.knowledge_graph.embeddings.to(self.device)
        self.decoder.to(self.device)
        logging.info("Best model successfully loaded.")


    def get_batch_embeddings(self, knowledge_graph: KnowledgeGraph, batch: Tensor, mask: Tensor | None = None) -> nn.Parameter:
        """
        Get the node embeddings of a given batch of graph indices.

        If there is no encoder, this is a straightforward return of the node embeddings.
        If there is an encoder, runs the forward pass on the initial embeddings and returns the aggregated embeddings.

        Arguments
        ---------

        **knowledge_graph** *(KnowledgeGraph)
        : The knowledge graph from which the batch is taken.

        **batch** *(torch.Tensor, dtype: torch.long, shape: [n_indices])*
        : The graph indices of the batch.

        **mask** *(torch.Tensor, dtype: torch.bool, shape: [n_triplets], optional)*
        : The mask corresponding to a dataset split, to ensure the encoder does not aggregate information
        from nodes it is not supposed to see.

        Returns
        -------
        **node_embeddings** *(torch.nn.Parameter)*
        : Parameter containing the embeddings of the corresponding nodes.
        """
        if self.encoder is not None:
            seed_nodes: Tensor = batch[:2].unique().cpu()
            hop_count: int = self.encoder.layer_count

            input = knowledge_graph.get_encoder_input(
                seed_nodes = seed_nodes,
                hop_count = hop_count,
                mask = mask)

            encoder_output: Dict[str, Tensor] = self.encoder(input.x_dict, input.edge_index)

            all_indices = torch.cat([
                index for index in input.node_mapping.values()
            ])

            all_embeddings = torch.cat([
                encoder_output[node_type] for node_type in input.node_mapping.keys()
            ])

            # As I understand it, this tensor is larger than needs to be because it needs to account for every possible
            # idx of the embeddings. It's not a logic problem as only the indices from the batch will be selected for the decoder,
            # which corresponds to the indices that are filled here.
            # TODO: See if making it a sparse tensor can spare memory
            node_embeddings = torch.zeros(
                (knowledge_graph.node_count, self.encoder_node_embedding_dimensions),
                device = self.device,
                dtype = torch.float
            ).index_put_(
                (all_indices,),
                all_embeddings
            )
        else:
            # Concatenate the embeddings of all node types, in the order of
            # node_type_to_global, so that global node indices can be used directly.
            # (A single node type is the common case, where this is a no-op.
            #  node_embeddings[0] alone would be wrong as soon as the KG has
            #  several node types, e.g. when node metadata is given.)
            node_embeddings = torch.cat(list(self.knowledge_graph.node_embeddings), dim=0)

        return node_embeddings

    def _reset_loss_running_average(self, engine: Engine) -> None:
        """
        Reset the per-epoch running average of the training loss.
        
        % Equivalent to the reset of the previous `ignite.metrics.RunningAverage`
        % attachment, kept on the loss's device to avoid a per-batch GPU->CPU sync.
        
        Arguments
        ---------
        
        **engine** *(Engine)*
        : Runner managing the training.
        
        """
        engine.state.metrics.pop("loss_running_average", None)
        engine.state.loss_running_average_value = None


    def _update_loss_running_average(self, engine: Engine) -> None:
        """
        Update the per-epoch running average of the training loss (EMA, alpha = 0.98).
        
        The value is stored in `engine.state.metrics["loss_running_average"]`, exactly like
        the previous `ignite.metrics.RunningAverage` attachment, but is computed on the
        loss's own device so that no GPU->CPU synchronization happens on every batch.
        
        Arguments
        ---------
        
        **engine** *(Engine)*
        : Runner managing the training.
        
        """
        alpha = 0.98
        loss = engine.state.output.detach()
        value = engine.state.loss_running_average_value
        if value is None or value.device != loss.device:
            value = loss
        else:
            value = value * alpha + (1.0 - alpha) * loss
        engine.state.loss_running_average_value = value
        engine.state.metrics["loss_running_average"] = value


    def _finalize_loss_running_average(self, engine: Engine) -> None:
        """
        Convert the end-of-epoch running average of the training loss to a plain
        float, exactly like the previous `ignite.metrics.RunningAverage` attachment
        did (see `ignite.metrics.metric.Metric.completed`), so that downstream
        consumers (e.g. the training metrics CSV) see the same type as before.
        
        Arguments
        ---------
        
        **engine** *(Engine)*
        : Runner managing the training.
        
        """
        value = engine.state.loss_running_average_value
        if isinstance(value, Tensor) and len(value.size()) == 0:
            engine.state.metrics["loss_running_average"] = value.item()


    def process_batch(self,
                    engine: Engine,
                    batch: Tensor
                    ) -> torch.types.Number:
        """
        Function called by the trainer to run the training loop on a mini-batch.

        Arguments
        ---------
        
        **batch** *(torch.Tensor, dtype: torch.long, shape: [batch_size, 4])* 
        : Tensor containing, for each triplet of the batch, the integer key of the head, tail, edge and triplet type.

        Returns
        -------
        
        **loss_value** *(torch.types.Number)*
        : Training loss value of the model for this batch.
        
        """
        batch = batch.to(self.device).T

        negative_batch = self.sampler.corrupt_batch(batch)
        negative_batch = negative_batch.to(self.device)

        full_batch_indices = torch.cat((batch, negative_batch), dim=1)
        node_embeddings = self.get_batch_embeddings(self.knowledge_graph, full_batch_indices, self.knowledge_graph.train_mask)
        
        
        self.optimizer.zero_grad()

        # Compute loss with positive and negative triplets
        positive_scores, negative_scores = self(batch, negative_batch, node_embeddings)
        loss = self.loss(positive_scores, negative_scores)
        loss.backward()

        self.optimizer.step()

        return loss


    def forward(self,
                positive_triplets_batch: torch.Tensor,
                negative_triplets_batch: torch.Tensor,
                node_embeddings: torch.Tensor
                ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Architect.

        Arguments
        ---------
        
        **positive_triplets_batch** *(torch.Tensor, dtype: torch.long, shape: [4, batch_size])*
        : Tensor containing the integer keys (head, tail, edge, triplet type) of the true triplets
        : in the current batch.
        
        **negative_triplets_batch** *(torch.Tensor, dtype: torch.long, shape: [4, batch_size * negative_triplet_count])*
        : Tensor containing the integer keys of the negatively sampled triplets of the same batch.
        
        **node_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, embedding_dimensions])*
        : Embeddings of the nodes of the knowledge graph.

        Returns
        -------
        
        **positive_score** *(torch.Tensor, dtype: torch.float, shape: [batch_size * negative_triplet_count])*
        : Tensor containing the score of each true triplet within the batch, repeated to match
        : the number of negative samples.
        
        **negative_score** *(torch.Tensor, dtype: torch.float, shape: [batch_size * negative_triplet_count])*
        : Tensor containing the score of each negative triplet within the batch.
        
        """
        positive_score: Tensor = self.scoring_function(positive_triplets_batch, node_embeddings)
        # The loss function requires the positive and negative tensors to be of the same size,
        # Thus we duplicate the positive tensor as needed to match the negative.
        negative_triplet_count = negative_triplets_batch.size(1) // positive_triplets_batch.size(1)
        positive_score = positive_score.repeat(negative_triplet_count)

        negative_score: Tensor = self.scoring_function(negative_triplets_batch, node_embeddings)

        return positive_score, negative_score


    def scoring_function(self,
                        batch: Tensor,
                        node_embeddings: Tensor
                        ) -> Tensor:
        """
        Runs the encoder and decoder pass on a batch for a given KG.
        
        If the encoder is not a GNN, directly runs and update the embeddings. 
        Otherwise, samples a subgraph from the given batch nodes and runs the encoder before.
        
        The embeddings are normalized at this step if it is required by the configuration
        If there is an encoder, the normalization is done after the encoder pass but before
        the decoder pass.
        
        Arguments
        ---------
        
        **batch** *(torch.Tensor, dtype: torch.long, shape: [4, batch_size])*
        : Batch of triplets. The rows correspond to: 
            * head_index
            * tail_index
            * edge_index
            * triplet_index
        : Here, batch_size is batch.shape[1].
        
        **node_embeddings** *(torch.Tensor, dtype: torch.float, shape: [node_count, embedding_dimensions])* 
        : Embeddings of the nodes of the knowledge graph.
        
        Returns
        -------
        
        **score** *(torch.Tensor)*
        : The score given by the decoder for the batch.
        
        """
        head_indices, tail_indices, edge_indices = batch[0], batch[1], batch[2]
        
        head_embeddings = node_embeddings[head_indices]
        edge_embeddings = self.knowledge_graph.edge_embeddings[edge_indices]  # Edges are unchanged
        tail_embeddings = node_embeddings[tail_indices]

        # Apply the configured normalizer, between the encoder and the decoder
        # step. This is the batchwise application of the normalizer: the
        # function is applied to the (encoder output) embeddings of the
        # current batch, and the new embeddings are returned (the parameters
        # are left untouched, so the gradients flow through the
        # normalization). Without an encoder, this step is skipped: the
        # embeddings between the encoder and the decoder are the node and edge
        # embeddings themselves, so they are normalized once, over the whole
        # graph, at the beginning of each epoch (see `apply_normalizer`).
        if self.normalizer is not None and self.encoder is not None:
            head_embeddings, tail_embeddings, edge_embeddings = self.normalizer(  head_embeddings = head_embeddings,
                                                                                  tail_embeddings = tail_embeddings,
                                                                                  edge_embeddings = edge_embeddings)

        return self.decoder.score(  head_embeddings = head_embeddings,
                                    tail_embeddings = tail_embeddings,
                                    edge_embeddings = edge_embeddings,
                                    head_indices = head_indices,
                                    tail_indices = tail_indices,
                                    edge_indices = edge_indices)


    def get_embeddings(self) -> Dict[str, Tensor]:
        """
        Returns the embeddings of nodes and edges, as well as decoder-specific embeddings if they exist.

        Returns
        -------
        embedding_dictionnary: Dict[str, Tensor | Dict[str, str]]
            Dictionary containing:
            - "nodes": the embeddings of all nodes (shape [node_count, embedding_dimensions]);
            - "node_mapping": mapping of node indices to node identifiers;
            - "edges": the embeddings of all edge types (shape [edge_count, edge_embedding_dimensions]);
            - "edge_mapping": mapping of edge indices to edge names;
            - "decoder": decoder-specific embeddings, if the decoder has any.

        Notes
        -----
        This function is user-facing.

        """
        self.normalize_parameters()
        
        if self.encoder is not None:
            node_embeddings: torch.Tensor = torch.zeros((self.knowledge_graph.node_count, self.encoder_node_embedding_dimensions), device="cpu", dtype=torch.float)

            with torch.no_grad():
                all_nodes = self.knowledge_graph.graphindices[:2].unique()
                # TODO: use not the whole graphindices but the unique nodes instead
                for i in range(self.knowledge_graph.graphindices.shape[1] // self.train_batch_size + 1):
                    seed_nodes = all_nodes[i * self.train_batch_size : (i + 1) * self.train_batch_size]
                    
                    input = self.knowledge_graph.get_encoder_input(
                        seed_nodes = seed_nodes,
                        hop_count = self.encoder.layer_count
                        )

                    encoder_output: Dict[str, Tensor] = self.encoder(input.x_dict, input.edge_index)

                    for node_type, indices in input.seed_mapping.items():
                        node_type_index = self.knowledge_graph.node_type_to_index[node_type]
                        node_type_mask = (self.knowledge_graph.node_types[seed_nodes] == node_type_index)
                        node_embeddings[seed_nodes[node_type_mask]] = encoder_output[node_type][indices].cpu()
        else:
            # Concatenate the embeddings of all node types (in node_type_to_global
            # order) so that global node indices can be used directly.
            node_embeddings = torch.cat([embeddings.data for embeddings in self.knowledge_graph.node_embeddings], dim=0).cpu()

        edge_embeddings = self.knowledge_graph.edge_embeddings.data.cpu()

        decoder_embeddings = self.decoder.get_embeddings()

        embedding_dictionnary = {"nodes": node_embeddings, 
                                 "node_mapping": {v: k for k,v in self.knowledge_graph.node_to_index.items()},
                                 "edges": edge_embeddings,
                                 "edge_mapping": {v: k for k,v in self.knowledge_graph.edge_to_index.items()}}

        if decoder_embeddings is not None:
            embedding_dictionnary.update({"decoder": decoder_embeddings})

        return embedding_dictionnary


    def apply_regularizer(self):
        """
        Apply the configured regularizer to the parameters it was given.

        This is the entry point of the regularizer trainer hooks: it is called
        at the end of every epoch, and before the training and test procedures.
        It is a no-op if no regularizer was configured
        (`[model.regularizer] name = "None"`, the default), in which case
        nothing is changed.

        The regularizer itself is initialized after the decoder in
        `initialize_model`, and is given the set of parameters to regularize
        and the function to apply to them (see `initialize_regularizer`).
        """
        if self.regularizer is None:
            return

        self.regularizer()

        logging.debug(f"Applied regularizer to the configured parameters.")


    def apply_normalizer(self):
        """
        Apply the configured normalizer to the whole-graph embeddings, in place.
        """
        if self.normalizer is None:
            return

        self.normalizer.initialize(self.knowledge_graph.node_embeddings, self.knowledge_graph.edge_embeddings)

        logging.debug(f"Applied normalizer to the configured parameters.")


    def normalize_parameters(self):
        """
        Normalize all parameters of the model.

        Kept for backward compatibility: the normalization routines that used
        to be implemented in each decoder's `normalize_parameters` method
        (e.g. TransE, RESCAL and DistMult L2-normalizing their node embeddings)
        are now gathered in the `Regularizer` module (see
        `kgate.regularizers`), selected through the configuration
        (`[model.regularizer]`) and applied by the trainer hooks. The
        normalization routines that used to be applied to the head and tail
        embeddings in each decoder's `score` method (e.g. RESCAL, DistMult,
        TransE, TransH, TransR and TransD L2-normalizing their head and tail
        embeddings) are now gathered in the `Normalizer` module (see
        `kgate.normalizers`), selected through the configuration
        (`[model.normalizer]`) and applied by the Architect between the
        encoder and the decoder step.

        This function therefore applies the configured regularizer (a no-op
        if no regularizer is configured, which is the default), and the
        configured normalizer if there is no encoder (a no-op if no normalizer
        is configured).
        """
        self.apply_regularizer()
        self.apply_normalizer()
        
        logging.debug(f"Normalized all embeddings.")



    def log_metrics_to_csv(self, engine: Engine):
        """
        Metrics recording in CSV file.

        Arguments
        ---------
        
        **engine** *(Engine)*
        : Runner managing the training.
        
        """
        epoch = engine.state.epoch
        train_loss = engine.state.metrics["loss_running_average"]
        validation_metric_value = engine.state.metrics.get("validation_metric_value", 0)
        learning_rate = self.optimizer.param_groups[0]["lr"]

        self.train_losses.append(train_loss)
        self.validation_metric_value.append(validation_metric_value)
        self.learning_rates.append(learning_rate)

        with open(self.train_metrics_file, mode = "a", newline = "") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, validation_metric_value, learning_rate])

        logging.info(f"Epoch {epoch} - Train Loss: {train_loss}, Validation {self.validation_metric}: {validation_metric_value}, Learning Rate: {learning_rate}")


    def clean_memory(self):
        """
        Memory cleaning.
        
        """
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Memory cleaned.")


    def evaluate(self, engine:Engine):
        """
        Evaluation on validation set.

        Arguments
        ---------
        
        **engine** *(Engine)*
        : Runner managing the training.
        
        """
        logging.info(f"Evaluating on validation set at epoch {engine.state.epoch}...")
        self.eval()  # Set the model to evaluation mode
        validation_score = 0
        with torch.no_grad():
            self.evaluator.reset()
            validation_subset = Subset(self.knowledge_graph, self.knowledge_graph.validation_mask.nonzero(as_tuple = True)[0])

            if isinstance(self.evaluator,LinkPredictionEvaluator):
                validation_score = self.link_prediction(validation_subset) 
                engine.state.metrics["validation_metric_value"] = validation_score 
                logging.info(f"Validation MRR: {validation_score}")

            elif isinstance(self.evaluator, TripletClassificationEvaluator):
                validation_score = self.triplet_classification()
                engine.state.metrics["validation_metric_value"] = validation_score
                logging.info(f"Validation Accuracy: {validation_score}")
        
        if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(validation_score)
            logging.info("Stepping scheduler ReduceLROnPlateau.")

        self.train() # Set the model back to training mode

    
    def update_scheduler(self):
        """
        Scheduler update.
            
        """
        if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

    
    def get_validation_metric(self, engine: Engine) -> float:
        """
        Early stopping score function & checkpoint best metric.

        Arguments
        ---------
        
        **engine** *(Engine)*
         : Runner managing the training.

        Returns
        -------
        
        **validation_metric_value** *(float)*
        : The validation metric value (e.g. the validation MRR) computed during the last evaluation,
        : or 0 if no evaluation has been run yet.
        
        """
        return engine.state.metrics.get("validation_metric_value", 0)
    
    
    def on_training_completed(self, engine: Engine):
        """
        Run at the end of the training, even with early stopping.
        
        Plot the training loss and validation MRR curves once the training is over.

        Arguments
        ---------
        
        **engine** *(Engine)*
        : Runner managing the training.
        
        """
        logging.info(f"Training completed after {engine.state.epoch} epochs.")

        plot_learning_curves(self.train_metrics_file, self.configuration.output_directory, self.validation_metric)


    def calculate_metrics_for_edges(self,
                                        knowledge_graph: KnowledgeGraph | Subset[KnowledgeGraph],
                                        edge_indices: List[str]
                                        ) -> Tuple[float, int, Dict[str, float], float]:
        """
        Compute the metrics for each individual edge.
        
        Arguments
        ---------
        
        **knowledge_graph** *(KnowledgeGraph or Subset[KnowledgeGraph])*
        : Knowledge graph on which the metrics will be calculated.
        Can be a subset of a full knowledge graph when run on test split.

        **edge_indices** *(List[str])*
        : Names of the edges for which the metrics are computed.

        Returns
        -------
        
        **metrics_sum** *(float)*
        : Sum of all individual metrics.
        
        **triplet_count** *(int)*
        : Number of triplets considered.
        
        **individual_metrics** *(Dict[str, float])*
        : Metrics computed for a single edge.
        
        **group_metrics** *(float)*
        : Global metrics computed for the edge group.
        
        Notes
        -----
        
        The metrics calculated here are only the Mean Reciprocal Rank (MRR).
        
        More could be implemented in the future.
        
        """
        # Mean Reciprocal Rank (MRR) computed by ponderating for each edge
        metrics_sum = 0.0
        triplet_count = 0
        individual_metrics = {}
        if isinstance(knowledge_graph, Subset):
            graphindices = knowledge_graph[:]
            edge_to_index = knowledge_graph.dataset.edge_to_index
        elif isinstance(knowledge_graph, KnowledgeGraph):
            graphindices = knowledge_graph.graphindices
            edge_to_index = knowledge_graph.edge_to_index

        for edge_name in edge_indices:
            # Get triplets associated with index
            edge_index = edge_to_index.get(edge_name)
            indices_to_keep = torch.nonzero(graphindices[2] == edge_index, as_tuple = False).view(-1)

            if indices_to_keep.numel() == 0:
                continue  # Skip to next edge if no triplet found
            
            if isinstance(self.evaluator, LinkPredictionEvaluator):
                test_metrics = self.link_prediction(Subset(knowledge_graph, indices_to_keep))
            elif isinstance(self.evaluator, TripletClassificationEvaluator):
                test_metrics = self.triplet_classification() # TODO
            
            # Save each edge's MRR
            individual_metrics[edge_name] = test_metrics
            
            metrics_sum += test_metrics * indices_to_keep.numel()
            triplet_count += indices_to_keep.numel()
        
        # Compute global MRR for the edge group
        group_metrics = metrics_sum / triplet_count if triplet_count > 0 else 0
        
        return metrics_sum, triplet_count, individual_metrics, group_metrics


    def calculate_metrics_for_categories(self,
                                        frequent_indices: List[int],
                                        infrequent_indices: List[int]
                                        ) -> Tuple[float, float]:
        """
        Calculate the MRR for frequent and infrequent categories based on given indices.
        
        Arguments
        ---------
        
        **frequent_indices** *(List[int])*
        : Indices of test triplets considered as frequent.
        
        **infrequent_indices** *(List[int])*
        : Indices of test triplets considered as infrequent.

        Returns
        -------
        
        **frequent_metrics** *(float)*
        : MRR for the frequent category.
        
        **infrequent_metrics** *(float)*
        : MRR for the infrequent category.
        
        """
        # Create subgraph for frequent and infrequent categories
        kg_frequent = self.knowledge_graph.remove_triplets_from_training(~frequent_indices)
        kg_infrequent = self.knowledge_graph.remove_triplets_from_training(~infrequent_indices)
        
        # Compute each category's MRR
        if isinstance(self.evaluator, LinkPredictionEvaluator):
            frequent_metrics = self.link_prediction(kg_frequent) if frequent_indices else 0
            infrequent_metrics = self.link_prediction(kg_infrequent) if infrequent_indices else 0
        elif isinstance(self.evaluator, TripletClassificationEvaluator):
            frequent_metrics = self.triplet_classification(self.kg_validation, kg_frequent) if frequent_indices else 0
            infrequent_metrics = self.triplet_classification(self.kg_validation, kg_infrequent) if infrequent_indices else 0
            
        return frequent_metrics, infrequent_metrics


    def link_prediction(self, knowledge_graph_subset: Subset[KnowledgeGraph]) -> float:
        """
        Link prediction evaluation on test set, validation set or inference set.

        Arguments
        ---------
        
        **knowledge_graph_subset** *(Subset[KnowledgeGraph])*
        : Subset of the knowledge graph on which the link prediction evaluation will be done.

        Raises
        -----
        
        **ValueError**
        : Wrong evaluator called.
        : The evaluator is initialized beforehand and may be incompatible with link prediction.
        : This error is raised when an evaluator incompatible with link prediction is initialized.
        Returns
        -------
        **test_mrr** (*float)*
        : The filtered MRR resulting from the evaluation on given knowledge graph.
        : If the decoder uses sphere embeddings (SpherE, Li et al. 2024), there are
        : no ranks, hence no MRR: returns instead the fraction of the evaluated
        : (true) triplets predicted positive, averaged over the head and tail
        : directions.
        
        """
        # Test MRR measure
        if not isinstance(self.evaluator, LinkPredictionEvaluator):
            raise ValueError(f"Wrong evaluator called. Calling Link Prediction method for {type(self.evaluator)} evaluator.")

        # SpherE (Li et al. 2024): the decoder may use sphere embeddings, in
        # which case the evaluation returns per-triplet positive predictions
        # instead of ranks (see `LinkPredictionEvaluator.evaluate`).
        sphere_embeddings = self.configuration.decoder.sphere_embeddings

        head_predictions, tail_predictions = self.evaluator.evaluate(batch_size = self.evaluation_batch_size,
                                encoder = self.encoder,
                                decoder = self.decoder,
                                evaluated_subset = knowledge_graph_subset,
                                node_embeddings = self.knowledge_graph.node_embeddings, 
                                edge_embeddings = self.knowledge_graph.edge_embeddings,
                                verbose = True,
                                sphere_embeddings = sphere_embeddings)
        
        if sphere_embeddings:
            # No ranks, hence no MRR: the scalar metric used by the training loop
            # is the fraction of the evaluated (true) triplets predicted positive,
            # averaged over the head and tail directions.
            triplet_count = len(knowledge_graph_subset)
            if triplet_count == 0:
                return 0.0
            head_positive = head_predictions.sphere_predictions.numel()
            tail_positive = tail_predictions.sphere_predictions.numel()
            return (head_positive + tail_positive) / (2 * triplet_count)
        
        test_mrr = (head_predictions.mrr[1] + tail_predictions.mrr[1]) / 2
        
        return test_mrr
    
    
    def triplet_classification(self) -> TripletClassificationResults:
        """
        Triplet Classification evaluation.

        Raises
        ------
        
        **ValueError**
        : Wrong evaluator called: this error is raised when an evaluator
          incompatible with triplet classification is initialized.

        Returns
        -------
        
        **results** *(TripletClassificationResults)*
        : Object containing all classification metrics (accuracy, precision,
        : recall, specificity, F1, balanced accuracy, FPR, FNR). 
        
        """
        if not isinstance(self.evaluator, TripletClassificationEvaluator):
            raise ValueError(f"Wrong evaluator called. Calling Triplet Classification method for {type(self.evaluator)} evaluator.")

        validation_subset = Subset(self.knowledge_graph, self.knowledge_graph.validation_mask.nonzero(as_tuple = True)[0])
        test_subset = Subset(self.knowledge_graph, self.knowledge_graph.test_mask.nonzero(as_tuple = True)[0])

        self.evaluator.evaluate(batch_size = self.evaluation_batch_size,
                                knowledge_graph_subset = validation_subset)
        
        return self.evaluator.accuracy( batch_size = self.evaluation_batch_size,
                                        kg_to_evaluate = test_subset)

#TODO
    # def run_data_leakage(self, attributes: Dict[str, pd.DataFrame] = {}):
    #     """
    #     Data leakage evaluation.
        
        # % TODO: detail the data leakage procedure
        
        # Arguments
        # ---------
        
        # **attributes** *(Dict[str, pd.DataFrame], optional)*
        # : dict(node_type, embedding) containing the embedding for each type of node.
        
        # Raises
        # ------
        
        # **ValueError**
        # : An edge was not found in the knowledge graph.
        
    #     """
    #     logging.info("Preparing KG for data leakage evaluation procedure...")
    #     data_leakage_config = self.config["data_leakage"]

    #     kg = merge_kg([self.kg_train, self.kg_validation, self.kg_test])

    #     for edge_type in data_leakage_config["permuted_edges"]:
    #         if edge_type not in self.kg_train.edge_to_index:
    #             raise ValueError(f"Edge type {edge_type} was not found in the knowledge graph.")
    #         logging.info(f"Permuting tails of edge type {edge_type}")
    #         self.kg_train = permute_tails(self.kg_train, edge_type)

    #     self.kg_train, self.kg_validation, self.kg_test = kg.generate_masks(split_proportions = self.config["preprocessing"]["split"])

    #     self.train_model(attributes = attributes)
