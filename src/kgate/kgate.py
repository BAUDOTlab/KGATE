import os
from glob import glob
import shutil
from inspect import signature
from pathlib import Path
from torchkge import KnowledgeGraph
from torchkge.models import Model
import torchkge.sampling as sampling
import torch
from torch import tensor, long, stack, Tensor
from torch.nn import Parameter
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from .utils import parse_config, load_knowledge_graph, set_random_seeds, find_best_model, merge_kg, init_embedding, plot_learning_curves, save_config
from .preprocessing import prepare_knowledge_graph, SUPPORTED_SEPARATORS
from .encoders import *
from .decoders import *
from .knowledgegraph import KnowledgeGraph
from .samplers import PositionalNegativeSampler, BernoulliNegativeSampler, UniformNegativeSampler, MixedNegativeSampler
from .evaluators import LinkPredictionEvaluator, TripletClassificationEvaluator
from .inference import EntityInference, RelationInference
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss
import logging
import warnings
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import csv
import gc
from ignite.metrics import RunningAverage
from ignite.engine import Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint, Checkpoint, DiskSaver, ProgressBar
import pandas as pd
import numpy as np
import yaml
import platform
from typing import Tuple, Dict, List, Any, Set, Literal

# Configure logging
logging.captureWarnings(True)
log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
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
    df : pd.DataFrame, optional
        The knowledge graph as a pandas dataframe containing at least the columns from, to and rel
    metadata : pd.DataFrame, optional
        The metadata as a pandas dataframe, with at least the columns id and type, where id is the name of the node as it is in the
        knowledge graph. If this argument is not provided, the metadata will be read from config.metadata if it exists. If both are absent,
        all nodes will be considered to be the same node type.
    cuddn_benchmark : bool, default to True
        Benchmark different convolution algorithms to chose the optimal one. Initialization is slightly longer when it is enabled, and only if cuda is available.
    num_cores : int, default to 0
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
    def __init__(self, config_path: str = "", kg: Tuple[KnowledgeGraph,KnowledgeGraph,KnowledgeGraph] | KnowledgeGraph | None = None, df: pd.DataFrame | None = None, metadata: pd.DataFrame | None = None, cudnn_benchmark: bool = True, num_cores:int = 0, **kwargs):
        # kg should be of type KnowledgeGraph, if exists use it instead of the one in config
        # df should have columns from, rel and to
        self.config: dict = parse_config(config_path, kwargs)

        if torch.cuda.is_available():
            # Benchmark convolution algorithms to chose the optimal one.
            # Initialization is slightly longer when it is enabled.
            torch.backends.cudnn.benchmark = cudnn_benchmark

        # If given, restrict the parallelisation to user-defined threads.
        # Otherwise, use all the cores the process has access to.
            
        if platform.system() == "Windows":
            num_cores: int = num_cores if num_cores > 0 else os.cpu_count()
        else:
            num_cores: int = num_cores if num_cores > 0 else len(os.sched_getaffinity(0))
        logging.info(f"Setting number of threads to {num_cores}")
        torch.set_num_threads(num_cores)

        outdir: Path = Path(self.config["output_directory"])
        # Create output folder if it doesn't exist
        logging.info(f"Output folder: {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir: Path = outdir.joinpath("checkpoints")


        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Detected device: {self.device}")

        set_random_seeds(self.config["seed"])

        self.emb_dim: int = self.config["model"]["emb_dim"]
        self.rel_emb_dim: int = self.config["model"]["rel_emb_dim"]
        if self.rel_emb_dim == -1:
            self.rel_emb_dim = self.emb_dim
        self.eval_batch_size: int = self.config["training"]["eval_batch_size"]

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


        run_kg_prep: bool = self.config["run_kg_preprocess"]

        if run_kg_prep or df is not None:
            logging.info(f"Preparing KG...")
            self.kg_train, self.kg_val, self.kg_test = prepare_knowledge_graph(self.config, kg, df, self.metadata)
            logging.info("KG preprocessed.")
        else:
            if kg is not None:
                logging.info("Using given KG...")
                if isinstance(kg, tuple):
                    self.kg_train, self.kg_val, self.kg_test = kg
                else:
                    raise ValueError("Given KG needs to be a tuple of training, validation and test KG if it is preprocessed.")
            else:
                logging.info("Loading KG...")
                self.kg_train, self.kg_val, self.kg_test = load_knowledge_graph(Path(self.config["kg_pkl"]))
                logging.info("Done")

        super().__init__(self.kg_train.n_ent, self.kg_train.n_rel)
        # Initialize attributes
        self.encoder: DefaultEncoder | GNN = None
        self.decoder: Model = None
        self.criterion: MarginLoss | BinaryCrossEntropyLoss = None
        self.optimizer: optim.Optimizer = None
        self.sampler: sampling.NegativeSampler = None
        self.scheduler: lr_scheduler.LRScheduler | None = None
        self.evaluator: LinkPredictionEvaluator | TripletClassificationEvaluator = None
        self.node_embeddings: nn.ParameterList | nn.Embedding


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

        match encoder_name:
            case "Default":
                encoder = DefaultEncoder()
            case "GCN": 
                encoder = GCNEncoder(self.kg_train.triple_types, self.emb_dim, gnn_layers)
            case "GAT":
                encoder = GATEncoder(self.kg_train.triple_types, self.emb_dim, gnn_layers)
            case "Node2vec":
                encoder = Node2VecEncoder(self.kg_train.edge_index, self.emb_dim, device=self.device, **encoder_config["params"])
            case _:
                encoder = DefaultEncoder()
                logging.warning(f"Unrecognized encoder {encoder_name}. Defaulting to a random initialization.")
        return encoder

    def initialize_decoder(self, decoder_name: str = "", dissimilarity: Literal["L1","L2",""] = "", margin: int = 0, n_filters: int = 0) -> Tuple[Model, MarginLoss | BinaryCrossEntropyLoss]:
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
        criterion
            The loss object
        """
        
        decoder_config: dict = self.config["model"]["decoder"]

        if decoder_name == "":
            decoder_name = decoder_config["name"]
        if dissimilarity == "":
            dissimilarity = decoder_config["dissimilarity"]
        if margin == 0:
            margin = decoder_config["margin"]
        if n_filters == 0:
            n_filters = decoder_config["n_filters"]

        # Translational models
        match decoder_name:
            case "TransE":
                decoder = TransE(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel,
                            dissimilarity_type=dissimilarity)
                criterion = MarginLoss(margin)
            case "TransH":
                decoder = TransH(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = MarginLoss(margin)
            case "TransR":
                decoder = TransR(self.emb_dim, self.rel_emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = MarginLoss(margin)
            case "TransD":
                decoder = TransD(self.emb_dim, self.rel_emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = MarginLoss(margin)
            case "RESCAL":
                decoder = RESCAL(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = BinaryCrossEntropyLoss()
            case "DistMult":
                decoder = DistMult(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = BinaryCrossEntropyLoss()
            case "ComplEx":
                decoder = ComplEx(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = BinaryCrossEntropyLoss()
            case "ConvKB":
                decoder = ConvKB(self.emb_dim, n_filters, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = BinaryCrossEntropyLoss()
            case _:
                raise NotImplementedError(f"The requested decoder {decoder_name} is not implemented.")

        return decoder, criterion

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

    def initialize_sampler(self) -> sampling.NegativeSampler:
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
        
        sampler_config: dict = self.config["sampler"]
        sampler_name: str = sampler_config["name"]
        n_neg: int = sampler_config["n_neg"]

        match sampler_name:
            case "Positional":
                sampler = PositionalNegativeSampler(self.kg_train)
            case "Uniform":
                sampler = UniformNegativeSampler(self.kg_train, n_neg)
            case "Bernoulli":
                sampler = BernoulliNegativeSampler(self.kg_train, n_neg)
            case "Mixed":
                sampler = MixedNegativeSampler(self.kg_train, n_neg)
            case _:
                raise NotImplementedError(f"Sampler type '{sampler_name}' is not supported. Please check the configuration.")
            
        return sampler
    
    def initialize_scheduler(self) -> lr_scheduler.LRScheduler | None:
        """
        Initializes the learning rate scheduler based on the provided configuration.
                
        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: Instance of the specified scheduler or
                                                            None if no scheduler is configured.
        
        Raises:
            ValueError: If the scheduler type is unsupported or required parameters are missing.
        """
        scheduler_config: dict = self.config["lr_scheduler"]
        
        if scheduler_config["type"] == "":
            warnings.warn("No learning rate scheduler specified in the configuration, none will be used.")
            return None
    
        scheduler_type: str = scheduler_config["type"]
        scheduler_params: dict = scheduler_config["params"]
        # Mapping of scheduler names to their corresponding PyTorch classes
        scheduler_mapping = {
            "StepLR": lr_scheduler.StepLR,
            "MultiStepLR": lr_scheduler.MultiStepLR,
            "ExponentialLR": lr_scheduler.ExponentialLR,
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
            "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
            "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
            "LambdaLR": lr_scheduler.LambdaLR,
            "OneCycleLR": lr_scheduler.OneCycleLR,
            "CyclicLR": lr_scheduler.CyclicLR,
        }

        # Verify that the scheduler type is supported
        if scheduler_type not in scheduler_mapping:
            raise ValueError(f"Scheduler type '{scheduler_type}' is not supported. Please check the configuration.")
        scheduler_class = scheduler_mapping[scheduler_type]
        
        # Initialize the scheduler based on its type
        try:
                scheduler: lr_scheduler.LRScheduler = scheduler_class(self.optimizer, **scheduler_params)
        except TypeError as e:
            raise ValueError(f"Error initializing '{scheduler_type}': {e}")

        
        logging.info(f"Scheduler '{scheduler_type}' initialized with parameters: {scheduler_params}")
        return scheduler

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
                    self.kg_train.removed_triples,
                    self.kg_val.edgelist,
                    self.kg_val.removed_triples,
                    self.kg_test.edgelist,
                    self.kg_test.removed_triples
                ], dim=1)
                evaluator = LinkPredictionEvaluator(full_edgelist=full_edgelist)
                self.validation_metric = "MRR"
            case "Triplet Classification":
                evaluator = TripletClassificationEvaluator(architect=self, kg_val = self.kg_val, kg_test=self.kg_test)
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
        logging.info("Initializing encoder...")
        self.encoder = self.encoder or self.initialize_encoder()

        logging.info("Initializing embeddings...")
        if pretrained is not None and pretrained.exists():
            self.node_embeddings = torch.load(pretrained)
        elif not isinstance(self.encoder, GNN):
            self.node_embeddings = init_embedding(self.kg_train.n_ent, self.emb_dim, self.device)
        else:
            self.node_embeddings = nn.ParameterList()
            ix2nt = {v: k for k,v in self.kg_train.nt2ix.items()}
            for node_type in self.kg_train.nt2glob:
                num_nodes = self.kg_train.nt2glob[node_type].size(0)
                if node_type in attributes:
                    current_attribute: pd.DataFrame = attributes[node_type]
                    assert current_attribute.shape[0] == num_nodes, f"The length of the given attribute ({len(current_attribute)}) must match the number of nodes of this type ({num_nodes})."
                    input_features = torch.zeros((num_nodes,current_attribute.shape[1]), dtype=torch.float)
                    for node in current_attribute.index:
                        node_idx = self.kg_train.ent2ix[node]
                        nt_idx = self.kg_train.node_types[node_idx]
                        local_idx = self.kg_train.glob2loc[node_idx]
                        assert nt_idx == self.kg_train.nt2ix[node_type], f"The entity {node} is given as {node_type} but registered as {ix2nt[str(nt_idx)]} in the KG."

                        input_features[local_idx] = tensor(current_attribute.loc[node], dtype=torch.float)
                    
                    self.node_embeddings.append(Parameter(input_features).to(self.device))
                else:
                    emb = init_embedding(num_nodes, self.emb_dim, self.device)
                    self.node_embeddings.append(emb.weight)
            # The input features are not supposed to change if we use an encoder
            self.node_embeddings = self.node_embeddings.requires_grad_(False)

        self.rel_emb = init_embedding(self.kg_train.n_rel, self.rel_emb_dim, self.device)


        # Cannot use short-circuit syntax with tuples
        if self.decoder is None:
            logging.info("Initializing decoder...")
            self.decoder, self.criterion = self.initialize_decoder()
            self.decoder.to(self.device)

        logging.info("Initializing optimizer...")
        self.optimizer = self.optimizer or self.initialize_optimizer()

        logging.info("Initializing sampler...")
        self.sampler = self.sampler or self.initialize_sampler()

        logging.info("Initializing lr scheduler...")
        self.scheduler = self.scheduler or self.initialize_scheduler()

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

        training_config: dict = self.config["training"]
        self.max_epochs: int = training_config["max_epochs"]
        self.train_batch_size: int = training_config["train_batch_size"]
        self.patience: int = training_config["patience"]
        self.eval_interval: int = training_config["eval_interval"]
        self.save_interval: int = training_config["save_interval"]

        match training_config["pretrained_embeddings"]:
            case "auto":
                pretrained = Path(self.config["output_directory"]).joinpath("embeddings.pt")
            case "":
                pretrained = None
            case _:
                pretrained = Path(training_config["pretrained_embeddings"])
                if not pretrained.exists(): pretrained = None
        
        self.initialize_model(attributes=attributes, pretrained=pretrained)

        self.training_metrics_file: Path = Path(self.config["output_directory"], "training_metrics.csv")

        if checkpoint_file is None:
            with open(self.training_metrics_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Training Loss", f"Validation {self.validation_metric}", "Learning Rate"])
        
        self.train_losses: List[float] = []
        self.val_metrics: List[float] = []
        self.learning_rates: List[float] = []

        iterator: DataLoader = DataLoader(self.kg_train, self.train_batch_size)
        logging.info(f"Number of training batches: {len(iterator)}")

        trainer: Engine = Engine(self.process_batch)
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss_ra")

        pbar = ProgressBar()
        pbar.attach(trainer)

        early_stopping: EarlyStopping = EarlyStopping(
            patience = self.patience,
            score_function = self.score_function,
            trainer = trainer
        )

        # If we find an identical config we resume training from it, otherwise we clean the checkpoints directory.
        existing_config_path: Path = Path(self.config["output_directory"]).joinpath("kgate_config.toml")
        if existing_config_path.exists():
            existing_config = parse_config(str(existing_config_path), {})
            all_checkpoints = glob(f"{self.checkpoints_dir}/checkpoint_*.pt")
            if existing_config == self.config and len(all_checkpoints) > 0:
                checkpoint_file = checkpoint_file or Path(max(all_checkpoints, key=os.path.getctime))
                logging.info("Found previous run with the same configuration in the output folder...   ")
        elif self.checkpoints_dir.exists() and len(os.listdir(self.checkpoints_dir)) > 0:
            shutil.rmtree(self.checkpoints_dir)

        # trainer.add_event_handler(Events.EPOCH_STARTED, self.encoder_pass)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_metrics_to_csv)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.clean_memory)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.update_scheduler)

        trainer.add_event_handler(Events.COMPLETED, self.on_training_completed)

        to_save = {
            "relations": self.rel_emb,
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
            DiskSaver(dirname=self.checkpoints_dir, require_empty=False, create_dir=True), # Save manager
            n_saved=2,      # Only keep last 2 checkpoints
            global_step_transform=lambda *_: trainer.state.epoch     # Include epoch number
        )

        # Custom save function to move the model to CPU before saving and back to GPU after
        def save_checkpoint_to_cpu(engine: Engine):
            # Move models to CPU before saving
            if self.encoder.deep:
                self.encoder.to("cpu")
            self.decoder.to("cpu")
            self.rel_emb.to("cpu")
            self.node_embeddings.to("cpu")

            # Save the checkpoint
            checkpoint_handler(engine)

            # Move models back to GPU
            if self.encoder.deep:
                self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.rel_emb.to(self.device)
            self.node_embeddings.to(self.device)

        # Attach checkpoint handler to trainer and call save_checkpoint_to_cpu
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), save_checkpoint_to_cpu)
    
        checkpoint_best_handler: ModelCheckpoint = ModelCheckpoint(
            dirname=self.checkpoints_dir,
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
                checkpoint = torch.load(checkpoint_file, weights_only=False)
                Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

                logging.info("Checkpoint loaded successfully.")
                with open(self.training_metrics_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["CHECKPOINT RESTART", "CHECKPOINT RESTART", "CHECKPOINT RESTART", "CHECKPOINT RESTART"])

                if trainer.state.epoch < self.max_epochs:
                    logging.info(f"Starting from epoch {trainer.state.epoch}")
                    if not dry_run:
                        trainer.run(iterator)
                else:
                    logging.info(f"Training already completed. Last epoch is {trainer.state.epoch} and max_epochs is set to {self.max_epochs}")
            else:
                logging.info(f"Checkpoint file {checkpoint_file} does not exist. Starting training from scratch.")
                if not dry_run:
                    trainer.run(iterator, max_epochs=self.max_epochs)
        else:
            if not dry_run:
                self.normalize_parameters()
                trainer.run(iterator, max_epochs=self.max_epochs)
    

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

        all_relations: Set[Any] = set(self.kg_test.rel2ix.keys())
        remaining_relations = all_relations - set(list_rel_1) - set(list_rel_2)
        remaining_relations = list(remaining_relations)

        total_metrics_sum_list_1, fact_count_list_1, individual_metrics_list_1, group_metrics_list_1 = self.calculate_metrics_for_relations(
            self.kg_test, list_rel_1)
        total_metrics_sum_list_2, fact_count_list_2, individual_metrics_list_2, group_metrics_list_2 = self.calculate_metrics_for_relations(
            self.kg_test, list_rel_2)
        total_metrics_sum_remaining, fact_count_remaining, individual_metrics_remaining, group_metrics_remaining = self.calculate_metrics_for_relations(
            self.kg_test, remaining_relations)

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
            relation: str = list_rel_2[i]
            threshold: int = thresholds[i]
            frequent_indices, infrequent_indices = self.categorize_test_nodes(relation, threshold)
            frequent_metrics, infrequent_metrics = self.calculate_metrics_for_categories(frequent_indices, infrequent_indices)
            logging.info(f"Metrics for frequent nodes (threshold={threshold}) in relation {relation}: {frequent_metrics}")
            logging.info(f"Metrics for infrequent nodes (threshold={threshold}) in relation {relation}: {infrequent_metrics}")

            results["target_relations_by_frequency"][relation] = {
                "Frequent_metrics": frequent_metrics,
                "Infrequent_metrics": infrequent_metrics,
                "Threshold": threshold
            }
                
        self.test_results = results
        
        with open(metrics_file, "w") as file:
            yaml.dump(results, file, default_flow_style=False, sort_keys=False)

        logging.info(f"Evaluation results stored in {metrics_file}")

        return results
        
    def infer(self, heads:List[str]=[], rels:List[str]=[], tails:List[str]=[], top_k:int=100):
        """Infer missing entities or relations, depending on the given parameters"""
        if not sum([len(arr) > 0 for arr in [heads,rels,tails]]) == 2:
            raise ValueError("To infer missing elements, exactly 2 lists must be given between heads, relations or tails.")
        torch.cuda.empty_cache()
        gc.collect()

        self.load_best_model()

        infer_heads, infer_rels, infer_tails = len(heads) == 0, len(rels) == 0, len(tails) == 0

        full_kg = merge_kg([self.kg_train, self.kg_val, self.kg_test], True)

        if infer_tails:
            known_1 = tensor([self.kg_train.ent2ix[head] for head in heads]).long()
            known_2 = tensor([self.kg_train.rel2ix[rel] for rel in rels]).long()
            missing = "tail"
            inference = EntityInference(full_kg)
        elif infer_heads:
            known_1 = tensor([self.kg_train.ent2ix[tail] for tail in tails]).long()
            known_2 = tensor([self.kg_train.rel2ix[rel] for rel in rels]).long()
            missing = "head"
            inference = EntityInference(full_kg)
        elif infer_rels:
            known_1 = tensor([self.kg_train.ent2ix[head] for head in heads]).long()
            known_2 = tensor([self.kg_train.ent2ix[tail] for tail in tails]).long()
            missing = "rel"
            inference = RelationInference(full_kg)

        predictions, scores = inference.evaluate(
            known_1,
            known_2,
            encoder = self.encoder,
            decoder = self.decoder,
            top_k = top_k,
            missing = missing,
            b_size = self.eval_batch_size,
            node_embeddings=self.node_embeddings,   
            relation_embeddings=self.rel_emb
        )

        ix2ent = {v: k for k, v in self.kg_train.ent2ix.items()}
        pred_idx = predictions.reshape(-1).T
        pred_names = np.vectorize(ix2ent.get)(pred_idx)

        scores = scores.reshape(-1).T
        print(pred_idx, scores)
        return pd.DataFrame([pred_names,scores], columns= ["Prediction","Score"])

    def load_best_model(self):
        """Load into memory the checkpoint corresponding to the highest-performing model on the validation set."""
        _, nt_count = self.kg_train.node_types.unique(return_counts=True)
        self.rel_emb = init_embedding(self.n_rel, self.rel_emb_dim, self.device)
        self.decoder, _ = self.initialize_decoder()
        self.encoder = self.initialize_encoder()
        if isinstance(self.encoder, GNN):
            self.node_embeddings = nn.ParameterList([init_embedding(count, self.emb_dim, self.device).weight for count in nt_count])
        else:
            self.node_embeddings = init_embedding(self.n_ent, self.emb_dim, self.device)

        logging.info("Loading best model.")
        best_model = find_best_model(self.checkpoints_dir)

        if not best_model:
            logging.error(f"No best model was found in {self.checkpoints_dir}. Make sure to run the training first and not rename checkpoint files before running evaluation.")
            return
        
        logging.info(f"Best model is {self.checkpoints_dir.joinpath(best_model)}")
        checkpoint = torch.load(self.checkpoints_dir.joinpath(best_model), map_location=self.device, weights_only=False)
        self.node_embeddings.load_state_dict(checkpoint["entities"])
        self.rel_emb.load_state_dict(checkpoint["relations"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        if "encoder" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder"])
        
        self.node_embeddings.to(self.device)
        self.rel_emb.to(self.device)
        self.decoder.to(self.device)
        self.encoder.to(self.device)
        logging.info("Best model successfully loaded.")


    def forward(self, pos_batch, neg_batch) -> Tuple[Tensor,Tensor]:
        """Forward pass of the Architect"""
        pos: Tensor = self.scoring_function(pos_batch, self.kg_train)
        # The loss function requires the pos and neg tensors to be of the same size,
        # Thus we duplicate the pos tensor as needed to match the neg.
        n_neg = neg_batch.size(1) // pos_batch.size(1)
        pos = pos.repeat(n_neg)

        neg: Tensor = self.scoring_function(neg_batch, self.kg_train)

        return pos, neg

    def process_batch(self, engine: Engine, batch: Tensor) -> torch.types.Number:
        batch = batch.T.to(self.device)

        n_batch = self.sampler.corrupt_batch(batch)
        n_batch = n_batch.to(self.device)
        
        self.optimizer.zero_grad()

        # Compute loss with positive and negative triples
        pos, neg = self(batch, n_batch)
        loss = self.criterion(pos, neg)
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
        h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
        
        if isinstance(self.encoder, GNN):
            seed_nodes = batch[:2].unique()
            num_hops = self.encoder.n_layers
            edge_index = kg.edge_index
            
            _,_,_, edge_mask = k_hop_subgraph(
                node_idx = seed_nodes,
                num_hops = num_hops,
                edge_index = edge_index
                )
                
            input = kg.get_encoder_input(kg.edgelist[:, edge_mask].to(self.device), self.node_embeddings)

            encoder_output: Dict[str, Tensor] = self.encoder(input.x_dict, input.edge_index)

            embeddings: torch.Tensor = torch.zeros((kg.n_ent, self.emb_dim), device=self.device, dtype=torch.float)

            for node_type, idx in input.mapping.items():
                embeddings[idx] = encoder_output[node_type]

            h_embeddings = embeddings[h_idx]
            t_embeddings = embeddings[t_idx]
        else:
            h_embeddings = self.node_embeddings(h_idx)
            t_embeddings = self.node_embeddings(t_idx)
        r_embeddings = self.rel_emb(r_idx)  # Relations are unchanged

        return self.decoder.score(h_emb = h_embeddings,
                                  r_emb = r_embeddings, 
                                  t_emb = t_embeddings, 
                                  h_idx = h_idx, 
                                  r_idx = r_idx, 
                                  t_idx = t_idx)

    def get_embeddings(self) -> Dict[str,Tensor | None]:
        """Returns the embeddings of entities and relations, as well as decoder-specific embeddings.
        
        If the encoder uses heteroData, a dict of {node_type : embedding} is returned for entity embeddings instead of a tensor."""
        self.normalize_parameters()
        
        if isinstance(self.node_embeddings, nn.ParameterList):
            input = self.kg_train.get_encoder_input(self.kg_train.edgelist.to(self.device), self.node_embeddings)

            encoder_output: Dict[str, Tensor] = self.encoder(input.x_dict, input.edge_index)
            ent_emb: torch.Tensor = torch.zeros((self.n_ent, self.emb_dim), device=self.device, dtype=torch.float)

            for node_type, idx in input.mapping.items():
                ent_emb[idx] = encoder_output[node_type]
        else:
            ent_emb = self.node_embeddings.weight.data

        rel_emb = self.rel_emb.weight.data

        decoder_emb = self.decoder.get_embeddings()

        return {"entities": ent_emb, 
                "relations": rel_emb,
                "decoder": decoder_emb}

    def normalize_parameters(self):
        # Some decoders should not normalize parameters or do so in a different way.
        # In this case, they should implement the function themselves and we return it.
        normalize_func = getattr(self.decoder, "normalize_params", None)
        # If the function only accept one parameter, it is the base torchKGE one,
        # we don't want that.
        if callable(normalize_func) and len(signature(normalize_func).parameters) > 1:
            stop_norm = normalize_func(rel_emb = self.rel_emb, ent_emb = self.node_embeddings)
            if stop_norm: return
        
        
        if not isinstance(self.encoder, GNN):
            self.node_embeddings.weight.data = normalize(self.node_embeddings.weight.data, p=2, dim=1)
            
        logging.debug(f"Normalized all embeddings")

    ##### Metrics recording in CSV file
    def log_metrics_to_csv(self, engine: Engine):
        epoch = engine.state.epoch
        train_loss = engine.state.metrics["loss_ra"]
        val_metrics = engine.state.metrics.get("val_metrics", 0)
        lr = self.optimizer.param_groups[0]["lr"]

        self.train_losses.append(train_loss)
        self.val_metrics.append(val_metrics)
        self.learning_rates.append(lr)

        with open(self.training_metrics_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_metrics, lr])

        logging.info(f"Epoch {epoch} - Train Loss: {train_loss}, Validation {self.validation_metric}: {val_metrics}, Learning Rate: {lr}")

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
                metric = self.link_pred(self.kg_val) 
                engine.state.metrics["val_metrics"] = metric 
                logging.info(f"Validation MRR: {metric}")
            elif isinstance(self.evaluator, TripletClassificationEvaluator):
                metric = self.triplet_classif(self.kg_val, self.kg_test)
                engine.state.metrics["val_metrics"] = metric
                logging.info(f"Validation Accuracy: {metric}")
        if self.scheduler and isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
            logging.info("Stepping scheduler ReduceLROnPlateau.")

        self.train() # Set the model back to training mode

    ##### Scheduler update
    def update_scheduler(self, engine: Engine):
        if self.scheduler is not None and not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
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

        plot_learning_curves(self.training_metrics_file, self.config["output_directory"], self.validation_metric)

    # TODO : create a script to isolate prediction functions. Maybe a Predictor class?
    def categorize_test_nodes(self, relation_name: str, threshold: int) -> Tuple[List[int], List[int]]:
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
        if relation_name not in self.kg_train.rel2ix:
            raise ValueError(f"The relation '{relation_name}' does not exist in the training knowledge graph.")
        relation_idx = self.kg_train.rel2ix[relation_name]

        # Count occurrences of nodes with the specified relation in the training set
        train_node_counts = {}
        for i in range(self.kg_train.n_triples):
            if self.kg_train.relations[i].item() == relation_idx:
                head = self.kg_train.head_idx[i].item()
                tail = self.kg_train.tail_idx[i].item()
                train_node_counts[head] = train_node_counts.get(head, 0) + 1
                train_node_counts[tail] = train_node_counts.get(tail, 0) + 1

        # Separate test triples with the specified relation based on the threshold
        frequent_indices = []
        infrequent_indices = []
        for i in range(self.kg_test.n_triples):
            if self.kg_test.relations[i].item() == relation_idx:  # Only consider triples with the specified relation
                head = self.kg_test.head_idx[i].item()
                tail = self.kg_test.tail_idx[i].item()
                head_count = train_node_counts.get(head, 0)
                tail_count = train_node_counts.get(tail, 0)

                # Categorize based on threshold
                if head_count > threshold or tail_count > threshold:
                    frequent_indices.append(i)
                else:
                    infrequent_indices.append(i)

        return frequent_indices, infrequent_indices
    
    def calculate_metrics_for_relations(self, kg: KnowledgeGraph, relations: List[str]) -> Tuple[float, int, Dict[str, float], float]:
        # MRR computed by ponderating for each relation
        metrics_sum = 0.0
        fact_count = 0
        individual_metrics = {} 

        for relation_name in relations:
            # Get triples associated with index
            relation_index = kg.rel2ix.get(relation_name)
            indices_to_keep = torch.nonzero(kg.relations == relation_index, as_tuple=False).squeeze()

            if indices_to_keep.numel() == 0:
                continue  # Skip to next relation if no triples found
            
            new_kg = kg.keep_triples(indices_to_keep)

            if isinstance(self.evaluator, LinkPredictionEvaluator):
                test_metrics = self.link_pred(new_kg)
            elif isinstance(self.evaluator, TripletClassificationEvaluator):
                test_metrics = self.triplet_classif(kg_val = self.kg_val, kg_test = new_kg)
            
            # Save each relation's MRR
            individual_metrics[relation_name] = test_metrics
            
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
            frequent_metrics = self.triplet_classif(self.kg_val, kg_frequent) if frequent_indices else 0
            infrequent_metrics = self.triplet_classif(self.kg_val, kg_infrequent) if infrequent_indices else 0
        return frequent_metrics, infrequent_metrics

    def link_pred(self, kg: KnowledgeGraph) -> float:
        """Link prediction evaluation on test set."""
        # Test MRR measure
        if not isinstance(self.evaluator, LinkPredictionEvaluator):
            raise ValueError(f"Wrong evaluator called. Calling Link Prediction method for {type(self.evaluator)} evaluator.")

        self.evaluator.evaluate(b_size = self.eval_batch_size,
                        encoder=self.encoder,
                        decoder =self.decoder,
                        knowledge_graph=kg,
                        node_embeddings=self.node_embeddings, 
                        relation_embeddings=self.rel_emb,
                        verbose=True)
        
        test_mrr = self.evaluator.mrr()[1]
        return test_mrr
    
    def triplet_classif(self, kg_val: KnowledgeGraph, kg_test: KnowledgeGraph) -> float:
        """Triplet Classification evaluation"""
        if not isinstance(self.evaluator, TripletClassificationEvaluator):
            raise ValueError(f"Wrong evaluator called. Calling Triplet Classification method for {type(self.evaluator)} evaluator.")
        
        self.evaluator.evaluate(b_size=self.eval_batch_size, knowledge_graph=kg_val)
        return self.evaluator.accuracy(self.eval_batch_size, kg_test = kg_test)
