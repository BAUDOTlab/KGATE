import os
from typing import Sequence, List, Literal
import logging

from pathlib import Path
import tomllib
from importlib.resources import open_binary

from .constants import SUPPORTED_ENCODERS, SUPPORTED_DECODERS

logging_level = logging.INFO
logging.basicConfig(
    level = logging_level,  
    format = "%(asctime)s - %(levelname)s - %(message)s" 
)


class Config:
    def __init__(self, *, config_path: os.PathLike = "", config_dict: dict  = {}):
        self._configuration = Config.parse(config_path, config_dict)

    @staticmethod
    def parse(config_path: os.PathLike, config_dictionnary: dict):
        """
        Parse the configuration file and integrates it with the default and inline configurations.
        
        For each parameter, the final parsed configuration will include in priority order: inline 
        configuration, configuration file, default configuration. If the configuration file or inline 
        configuration contain parameters not existing in the default configuration, they will be included
        in the final configuration but not validated.

        Some elements of the model may use additional parameters not included in the KGATE configuration.
        In that case, you can use the Config.[element]_kwargs dictionary that will be fed to the initialization
        of the element.
        
        Arguments
        ---------
        config_path: str
            The complete path to the configuration file. If one already exists, it will be overwritten.
        config_dictionnary: dict, optional
            The parsed configuration as a python dictionnary.
            
        Raises
        ------
        FileNotFoundError
            The configuration file is not found at the indicated path.
            Check that you gave the correct path, and that it is a str.
            If you give a relative path, it must be relative to the run script path.
        
        Returns
        -------
        config: dict
            The final parsed configuration as a python dictionnary.
            Using priority orders: inline configuration, configuration file, default configuration
        """
        if config_path != "" and not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        with open_binary("kgate", "config_template.toml") as f:
            default_config = tomllib.load(f)

        self.config = {}

        if config_path != "":
            logging.info(f"Loading parameters from {config_path}")
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

        # Make the final configuration, using priority orders:
        # 1. Inline configuration (config_dictionnary)
        # 2. Configuration file (config)
        # 3. Default configuration (default_config)
        # If a default value is None, consider it required and not defaultable
        config = {  key: set_config_key(key, default_config, config, config_dictionnary)
                    for key
                    in default_config}

        return config

    @staticmethod
    def set_config_key( key: str,
                        default: dict,
                        config: dict | None = None,
                        inline: dict | None = None
                        ) -> str | int | list | dict:
        """
        For a specific parameter, a 'key', compare default, inline and user-made configurations to give
        the key value with priority order: inline configuration, configuration file, default configuration.

        Arguments
        ---------
        default: dict
            The default parsed configuration as a python dictionnary.
        config: dict, optional
            The configuration parsed from the config file.
        inline: dict, optional
            The inline parsed configuration as a python dictionnary.

        Raises
        ------
        ValueError
            A parameter without a default value is required but not set.

        Returns
        -------
        Return one of the following values:
            inline_value: str or int or float or List or dict or None
                Value of the key given by the user in command line.
                Can only be of types dict and List within the recursive call.
            config_value: str or int or float or List or dict or None
                Value of the key from the configuration file.
                Can only be of types dict and List within the recursive call.
            default[key]: str or int or float or List or dict or None
                Value of the key from the default configuration file.
                Can only be of types dict and List within the recursive call.
        
        """
        if inline is not None and key in inline:
            inline_value = inline[key]
        else:
            inline_value = None

        if config is not None and key in config:
            config_value = config[key]
        else: 
            config_value = None

        # If the value is a dict, recursively call this function on each of its keys
        if key in default and isinstance(default[key], dict):
            new_value = {}
            # The keys are taken from default
            keys = list(default[key].keys())
            if config_value is not None:
                # If they exist, keys are taken from the config file
                keys += (list(config_value.keys()))
            if inline_value is not None:
                # If they exist, keys are taken from inline inputs
                keys += (list(inline_value.keys()))
            for child_key in set(keys):
                new_value.update({child_key: set_config_key(child_key, default[key], config_value, inline_value)})
            return new_value
        
        # Return the key value in priority from: inline, config, default
        if inline_value is not None:
            return inline_value
        elif config_value is not None:
            return config_value
        elif default[key] is not None:
            logging.info(f"No value set for parameter {key}. Defaulting to {default[key]}")
            return default[key]
        else:
            raise ValueError(f"Parameter {key} is required but not set without a default value.")

    def save(self,
            filename: Path | None = None):
        """
        Saves the configuration as a TOML file.
        
        If no filename is given, it will be created as config.output_directory/kgate_config.toml.
        
        Arguments
        ---------
        filename: Path, optional
            The complete path to the configuration file. If one already exists, it will be overwritten.
        """
        config_path = filename or Path(self.output_directory).joinpath("kgate_config.toml")

        with open(config_path, "wb") as f:
            tomli_w.dump(self._config, f)

    @property
    def seed(self) -> int:
        """
        Seed used for random number generation, used for reproducibility.

        To ensure that all part of the library have a common seed, it is set for the following libraries:
        - The standard library `random`
        - `numpy`
        - `torch`
        - `torch.cuda`

        Default to 42.
        """
        return self._configuration["seed"]
    
    @seed.setter
    def set_seed(self, new_seed: int):
        if new_seed != self.seed:
            logging.warn("The seed has changed in the configuration! All random generators reset according to the new seed.")
            set_random_seeds(new_seed)

        self._configuration["seed"] = new_seed


    @property
    def knowledge_graph_csv_file(self) -> Path:
        """
        Path to the lnowledge graph in CSV format with at least 3 columns named 
        "heads", "tails" and "edges". Additionnal columns are not regarded.
        """
        return Path(self._configuration["kg_csv"])

    @knowledge_graph_csv_file.setter
    def set_knowledge_graph_csv_file(self, path: os.PathLike):
        if path != "" and not Path(path).exists():
            raise FileNotFoundError(f"The knowledge graph file has not been found at path {path}.")

        self._configuration["kg_csv"] = path

    @property
    def knowledge_graph_pickle_file(self) -> Path:
        """
        Path to the knowledge graph in pickle format.
        """
        return Path(self._configuration["kg_pkl"])

    @knowledge_graph_pickle_file.setter
    def set_knowledge_graph_pickle_file(self, path: os.PathLike):
        if path != "" and not Path(path).exists():
            raise FileNotFoundError(f"The knowledge graph file has not been found at path {path}.")

        self._configuration["kg_pkl"] = path

    @property
    def metadata_path(self) -> Path:
        """
        Path to the CSV mapping nodes to their metadata.

        The CSV file must have at least the columns 'id' (must be the same 
        as the KG node identifier) and 'type', representing the node type.
        Additionnal columns may contain more metadata that may be used to identify nodes.
        """
        return Path(self._configuration["metadata_csv"])
    
    @metadata_path.setter
    def set_metadata_path(self, path: os.PathLike):
        if path != "" and not Path(path).exists():
            raise FileNotFoundError(f"The metadata file has not been found at path {path}.")

        self._configuration["metadata_csv"] = path

    @property
    def output_directory(self) -> Path:
        """
        Path to the output directory for all KGATE files.
        """
        return Path(self._configuration["output_directory"])

    @output_directory.setter
    def set_output_directory(self, path: os.PathLike):
        self._configuration["output_directory"] = path

    @property
    def verbose(self) -> bool:
        """
        Whether to get the full information output from the library.

        Default to True.
        """
        return self._configuration["verbose"]
    
    @verbose.setter
    def set_verbose(self, verbose: bool):
        self._configuration["verbose"] = verbose

    @property
    def node_embedding_dimensions(self) -> int:
        """
        The dimension of the embedding vectors for nodes.

        This is the dimension of the latent space, which means the input
        features can have a different dimension if there is an encoder able
        to work with different dimensions. With only a decoder, the input features
        must match this parameter.

        Default to 256.
        """
        return self._configuration["model"]["node_embedding_dimensions"]

    @node_embedding_dimensions.setter
    def set_node_embedding_dimensions(self, dimensions: int):
        assert dimensions > 0 and isintance(dimensions, int), f"The node embedding dimension must be a positive integer, but got {dimensions}"

        self._configuration["model"]["node_embedding_dimensions"] = dimensions

    @property
    def edge_embedding_dimensions(self) -> int:
        """
        The dimension of the embedding vectors for edges.

        Some models need to have the same dimensions for node and edge vectors.
        To ensure that this is the case, a value of -1 for this property will
        make the edge embedding vectors to have the same dimension as the node embedding
        vector.

        Default to -1.
        """
        return self._configuration["model"]["edge_embedding_dimensions"]

    @edge_embedding_dimensions.setter
    def set_edge_embedding_dimensions(self, dimensions: int):
        assert (dimensions > 0 or dimensions == -1) and isintance(dimensions, int), f"The edge embedding dimension must be a positive integer or -1, but got {dimensions}"

        self._configuration["model"]["edge_embedding_dimensions"] = dimensions

class Preprocessing_Config:
    """
    Preprocessing part of the main configuration.

    This class is not meant to be used as a standalone, but to make access to 
    configuration parameter easier.

    Arguments
    ---------
    preprocessing_configuration: dict
        Dictionary containing only the preprocessing configuration.
    """
    def __init__(self, preprocessing_configuration: dict):
        self._configuration = preprocessing_configuration 

    def __str__(self):
        return f"""
        run: {self.run} \n
        remove_duplicate_triplets: {self.remove_duplicate_triplets} \n
        make_directed: {self.make_directed} \n
        flag_near_duplicate_edges: {self.flag_near_duplicate_edges} \n
        theta_first_edge_type: {self.theta_first_edge_type} \n
        theta_second_edge_type: {self.theta_second_edge_type}
        clean_train_set: {self.clean_train_set} \n
        split_proportions: {self.split_proportions} \n
        """

    @property
    def run(self) -> bool:
        """
        Whether or not to run the preprocessing procedure on the knowledge graph.
        
        If set to False, all preprocessing parameters are ignored.

        Defaults to True.
        """
        return self._configuration["run_kg_preprocessing"]

    @run.setter
    def set_run(self, run_preprocessing: bool):
        self._configuration["run_kg_preprocessing"] = run_preprocessing
    
    @property
    def remove_duplicate_triplets(self) -> bool:
        """
        Whether or not duplicate triplets should be removed in the knowledge graph.
        
        Defaults to True.
        """
        return self._configuration["remove_duplicate_triplets"]

    @remove_duplicate_triplets.setter
    def set_remove_duplicate_triplets(self, remove_duplicate_triplets: bool):
        self._configuration["remove_duplicate_triplets"] = remove_duplicate_triplets

    @property
    def make_directed(self) -> List[str] | Literal["all"]:
        """
        List of undirected edges to make directed.

        In the case of an undirected graph, this parameter can be set to "all" instead of
        a list to make all edges directed.

        For example, if we have an undirected edge (A)-[edge]-(B), to make it directed we will
        make it direct (A)-[edge]->(B) and create its reverse (A)<-[edge_reverse]-(B).
        """
        return self._configuration["make_directed_edges"]

    @make_directed.setter
    def set_make_directed(self, edges_to_direct: List[str] | Literal["all"]):
        if isinstance(edges_to_direct, str):
            assert edges_to_direct == "all", "The edges to make directed must be either a list of edges or \"all\" to make the whole graph directed."

        self._configuration["make_directed_edges"] = edges_to_direct

    @property
    def flag_near_duplicate_edges(self) -> bool:
        """
        Whether or not to flag edges that are almost duplicates of each others.

        This helps identify edges with different names but nearly the same semantic
        meaning, which may introduce data leakage should they be spread out in different sets.

        The identification of near duplicates is done through the procedure described in
        Akrami et al. (2020) [1]_ and further developed in Brière et al. (2025) [2]_.

        Defaults to True.

        References
        ----------
        .. [1] Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
        `Realistic Re-evaluation of Knowledge Graph Completion Methods:
        An Experimental Study.`
        <https://arxiv.org/pdf/2003.08001.pdf>
        SIGMOD’20, June 14–19, 2020, Portland, OR, USA
        .. [2] Brière, Galadriel, Thomas Stosskopf, Benjamin Loire, and Anaïs Baudot. 
        “Benchmarking Data Leakage on Link Prediction in Biomedical Knowledge Graph Embeddings.” 
        <https://doi.org/10.1101/2025.01.23.634511>
        Preprint, bioRxiv, January 26, 2025.
        """
        return self._configuration["flag_near_duplicate_edges"]

    @flag_near_duplicate_edges.setter
    def set_flag_near_duplicate_edges(self, flag: bool):
        self._configuration["flag_near_duplicate_edges"] = flag

    @property
    def theta_first_edge_type(self) -> float:
        """
        The threshold value for the first edge type when scanning for duplicates.

        When considering a pair of edge types, it sets the threshold over which 
        the first edge is considered a duplicate of the second.

        For example, with a theta value of 0.8, if more than 80% of the edges of the first type
        are duplicates of the second edge type, the whole edge type is considered to be a duplicate.

        See the paper by Akrami et al. (2020) [1]_ for full reference.

        Defaults to 0.8.

        References
        ----------
        .. [1] Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
        `Realistic Re-evaluation of Knowledge Graph Completion Methods:
        An Experimental Study.`
        <https://arxiv.org/pdf/2003.08001.pdf>
        SIGMOD’20, June 14–19, 2020, Portland, OR, USA
        """
        return self._configuration["theta_first_edge_type"]

    @theta_first_edge_type.setter
    def set_theta_first_edge_type(self, theta: float):
        assert theta >= 0 and theta <= 1, f"Theta value must be between 0 and 1, got {theta}"

        self._configuration["theta_first_edge_type"] = theta

    @property
    def theta_second_edge_type(self) -> float:
        """
        The threshold value for the second edge type when scanning for duplicates.

        When considering a pair of edge types, it sets the threshold over which 
        the second edge is considered a duplicate of the first.

        For example, with a theta value of 0.8, if more than 80% of the edges of the second type
        are duplicates of the first edge type, the whole edge type is considered to be a duplicate.

        See the paper by Akrami et al. (2020) [1]_ for full reference.

        Defaults to 0.8.

        References
        ----------
        .. [1] Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
        `Realistic Re-evaluation of Knowledge Graph Completion Methods:
        An Experimental Study.`
        <https://arxiv.org/pdf/2003.08001.pdf>
        SIGMOD’20, June 14–19, 2020, Portland, OR, USA
        """
        return self._configuration["theta_second_edge_type"]

    @theta_second_edge_type.setter
    def set_theta_second_edge_type(self, theta: float):
        assert theta >= 0 and theta <= 1, f"Theta value must be between 0 and 1, got {theta}"

        self._configuration["theta_second_edge_type"] = theta
        
    @property
    def clean_train_set(self) -> bool:
        """
        Whether or not to remove all flagged edges from the train set.

        These edges are completely removed from the dataset (not moved to the validation
        or test set), but still retained as truth for the evaluation.

        Defaults to True.
        """
        return self._configuration["clean_train_set"]
    
    @clean_train_set.setter
    def set_clean_train_set(self, clean: bool):
        self._configuration["clean_train_set"] = clean

    @property
    def split_proportions(self) -> Tuple[int, int, int]:
        """
        How the knowledge graph should be split between 'train', 
        'validation' and 'test'.
        
        First value is train set proportion, second is validation set,
        and third value is test set. The three values must sum to 1.
        """
        return self._configuration["split"]
    
    @split_proportions.setter
    def set_split_proportions(self, proportions: Sequence[int, int, int]):
        assert sum(proportions) == 1, f"The sum of all proportions must be 1 but got {sum(proportions)}."

        self._configuration["split"] = proportions

class Encoder_Config:
    """
    Encoder part of the main configuration.

    This class is not meant to be used as a standalone, but to make access to 
    configuration parameter easier.

    Arguments
    ---------
    encoder_configuration: dict
        Dictionary containing only the encoder configuration.
    """

    def __init__(self, encoder_config):
        self._configuration = encoder_config
        self.supported_encoders = SUPPORTED_ENCODERS

        # If we load a configuration with an unsupported encoder name, 
        # assume it is correct but warn the user.
        if encoder_config["name"] not in SUPPORTED_ENCODERS:
            logging.warn(f"Encoder name {encoder_config["name"]} is not a builtin KGATE encoder. It will be considered a custom encoder.")
            self.register_name(encoder_config["name"])

    @property
    def name(self) -> str:
        """
        The name of the encoder.

        When using builtin KGATE encoders, possible values are:
        - `Default`: not a proper encoder but randomly initialized vectors.
        - `GCN`: GNN encoder from :class:`~torch_geometric.nn.SAGEConv`.
        - `GAT`: GNN encoder from :class:`~torch_geometric.nn.GATv2Con`.

        It is also possible to add your own custom encoder to the configuration, in
        which case you should call :func:`~Config.encoder.register_name` to make sure
        it is acknowledged as a valid encoder name.

        Defaults to Default.
        """
        return self._configuration["name"]

    @name.setter
    def set_name(self, name: str):
        assert name in self.supported_encoders, f"Unsupported encoder given. KGATE supports {', '.join(supported_encoders)} but got {name}. If you want to register a custom encoder name, use Config.encoder.register_name()"

        self._configuration["name"] = name

    def register_name(self, name: str):
        """
        Register this name as a valid encoder.

        Adds the given name to the list of supported encoders and set it
        as the current encoder name in the configuration.

        KGATE has a limited set of builtin encoders and validates inputs
        against this list. To make sure your custom encoder pass the
        sanitization checks, it needs to be registered as valid.

        Arguments
        ---------
            name: str
                The name of the encoder to register.
        """
        self.supported_encoders.append(name)

        self.name = name

    @property
    def gnn_layers(self) -> int:
        """
        The number of GNN layers to build.

        With the builtin encoders, the same encoder is used in all layers,
        but it is possible to build a custom encoder with multiple encoders for each layer.

        Default to 1.
        """
        return self._configuration["gnn_layer_number"]

    @gnn_layers.setter
    def set_gnn_layers(self, gnn_layers: int):
        assert gnn_layers >= 0 and isinstance(gnn_layers, int), f"GNN layers must be 0 or a positive integer, but got {gnn_layers}"

        self._configuration["gnn_layer_number"] = gnn_layers

class Decoder_Config:
    """
    Decoder part of the main configuration.

    This class is not meant to be used as a standalone, but to make access to 
    configuration parameter easier.

    Arguments
    ---------
    decoder_configuration: dict
        Dictionary containing only the decoder configuration.
    """

    def __init__(self, decoder_config):
        self._configuration = decoder_config
        self.supported_decoders = SUPPORTED_DECODERS

        # If we load a configuration with an unsupported decoder name, 
        # assume it is correct but warn the user.
        if decoder_config["name"] not in SUPPORTED_DECODERS:
            logging.warn(f"decoder name {decoder_config["name"]} is not a builtin KGATE decoder. It will be considered a custom decoder.")
            self.register_name(decoder_config["name"])
            
    @property
    def name(self) -> str:
        """
        The name of the decoder.

        When using builtin KGATE decoders, possible values are:
        - :class:`~kgate.decoder.TransE`: Translational model proposed by Bordes et al. 2013
        - :class:`~kgate.decoder.TransH`: Translational model proposed by Wang et al. 2014
        - :class:`~kgate.decoder.TransR`: Translational model proposed by Lin et al. 2015
        - :class:`~kgate.decoder.TransD`: Translational model proposed by Ji et al. 2015
        - :class:`~kgate.decoder.TorusE`: Translational model proposed by Ebisu and Ichise 2017
        - :class:`~kgate.decoder.RESCAL`: Bilinear model proposed by Nickel et al. 2011
        - :class:`~kgate.decoder.DistMult`: Bilinear model proposed by Yang et al. 2014
        - :class:`~kgate.decoder.ComplEx`: Bilinear model proposed by Trouillon et al. 2016
        - :class:`~kgate.decoder.ConvKB`: Convolutional model proposed by Nguyen et al. 2018

        It is also possible to add your own custom decoder to the configuration, in
        which case you should call :func:`~Config.decoder.register_name` to make sure
        it is acknowledged as a valid decoder name.

        Defaults to TransE.
        """
        return self._configuration["name"]

    @name.setter
    def set_name(self, name: str):
        assert name in self.supported_decoders, f"Unsupported decoder given. KGATE supports {', '.join(supported_decoders)} but got {name}. If you want to register a custom decoder name, use Config.decoder.register_name()"

        self._configuration["name"] = name

    def register_name(self, name: str):
        """
        Register this name as a valid decoder.

        Adds the given name to the list of supported decoders and set it
        as the current decoder name in the configuration.

        KGATE has a limited set of builtin decoders and validates inputs
        against this list. To make sure your custom decoder pass the
        sanitization checks, it needs to be registered as valid.

        Arguments
        ---------
            name: str
                The name of the decoder to register.
        """
         
        self.supported_decoders.append(name)

        self.name = name

    @property
    def margin(self) -> int:
        """
        Margin value when using a Margin Loss.

        The margin loss is only used with translational decoders.
        This is ignored for non-translational decoders.

        Default is 1.
        """
        return self._configuration["margin"]

    @margin.setter
    def set_margin(self, margin: int) -> int:
        self._configuration["margin"] = margin

