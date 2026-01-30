"""
Utility functions for Knowledge Graph manipulation, data visualisation or file handling.
"""

import os
import tomllib
import random
import logging 
import pickle
from pathlib import Path
from importlib.resources import open_binary
from typing import List, Tuple, Literal, Dict

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tomli_w

import torch
import torch.nn as nn
from torch import cat, Tensor

from .knowledgegraph import KnowledgeGraph


logging_level = logging.INFO
logging.basicConfig(
    level = logging_level,  
    format = "%(asctime)s - %(levelname)s - %(message)s" 
)


def parse_config(config_path: str,
                config_dictionnary: dict
                ) -> dict:
    """
    TODO.What_the_function_does_about_globally
    
    References
    ----------
    TODO
    
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

    config = {}

    if config_path != "":
        logging.info(f"Loading parameters from {config_path}")
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    
    # Make the final configuration, using priority orders:
    # 1. Inline configuration (config_dictionnary)
    # 2. Configuration file (config)
    # 3. Default configuration (default_config)
    # If a default value is None, consider it required and not defaultable
    config = {key: set_config_key(key, default_config, config, config_dictionnary) for key in default_config}

    return config


def set_config_key( key: str,
                    default: dict,
                    config: dict | None = None,
                    inline: dict | None = None
                    ) -> str | int | list | dict:
    """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        default: dict
            The default parsed configuration as a python dictionnary.
        config: dict, optional
            TODO.What_that_argument_is_or_does
        inline: dict, optional
            The inline parsed configuration as a python dictionnary.

        Raises
        ------
        ValueError
            A parameter without a default value is required but not set.

        Returns
        -------
        inline_value: TODO.type
            TODO.What_that_variable_is_or_does
        config_value: TODO.type
            TODO.What_that_variable_is_or_does
        default[key]: TODO.type
            TODO.What_that_variable_is_or_does
            
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
            new_value.update({child_key: set_config_key(child_key, default[key], config_value,  inline_value)})
        return new_value
    
    # Return the key value in priority from: inline, config, default
    # TODO: invert conditions, starting with 'if inline_value is not None:', for lisibility
    if inline_value is None:
        if config_value is None:
            if default[key] is None:
                raise ValueError(f"Parameter {key} is required but not set without a default value.")
            else:
                logging.info(f"No value set for parameter {key}. Defaulting to {default[key]}")
                return default[key]
        else:
            return config_value
    else:
        return inline_value


def save_config(config: dict,
                filename: Path | None = None):
    """
    Saves the Architect configuration as a TOML file.
    
    If no filename is given, it will be created as config.output_directory/kgate_config.toml.
    
    Arguments
    ---------
    config: dict
        The parsed config as a python dictionnary.
    filename: Path, optional
        The complete path to the configuration file. If one already exists, it will be overwritten.
        
    """
    config_path = filename or Path(config["output_directory"]).joinpath("kgate_config.toml")

    with open(config_path, "wb") as f:
        tomli_w.dump(config,f)


def load_knowledge_graph(pickle_filename: Path
                        ) -> Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    """
    Load the knowledge graph from pickle files.
    
    Arguments
    ---------
    pickle_filename: Path
        The complete path to the pickle file (.pkl).

    Returns
    -------
    kg_train: KnowledgeGraph
        Train split from the knowledge graph, directly loaded from the pickle file.
    kg_validation: KnowledgeGraph
        Validation split from the knowledge graph, directly loaded from the pickle file.
    kg_test: KnowledgeGraph
        Test split from the knowledge graph, directly loaded from the pickle file.
    
    """
    logging.info(f"Will not run the preparation step. Using knowledge graph stored in: {pickle_filename}")
    with open(pickle_filename, "rb") as file:
        kg_train = pickle.load(file)
        kg_validation = pickle.load(file)
        kg_test = pickle.load(file)
        
    return kg_train, kg_validation, kg_test


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Arguments
    ---------
    seed: int
        TODO.What_that_argument_is_or_does
    
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_node_type(node_name: str):
    """
    Extracts the node type from the node name, based on the string before the first underscore.
    
    Arguments
    ---------
    node_name: str
        TODO.What_that_argument_is_or_does

    Returns
    -------
    node_type: str
        TODO.What_that_variable_is_or_does
    
    """
    return node_name.split("_")[0]


def compute_triplet_proportions(kg_train: KnowledgeGraph,
                                kg_test: KnowledgeGraph,
                                kg_validation: KnowledgeGraph
                                ) -> dict:
    """
    Computes the proportion of triplets for each edge in each of the KnowledgeGraphs
    (train, test, validation) relative to the total number of triplets for that edge.

    Arguments
    ---------
    kg_train: KnowledgeGraph
        Train split from the knowledge graph.
    kg_test: KnowledgeGraph
        Test split from the knowledge graph.
    kg_validation: KnowledgeGraph
        Validation split from the knowledge graph.

    Returns
    -------
    proportions: dict
        A dictionary where keys are edge identifiers and values are sub-dictionaries
        with the respective proportions of each edge in kg_train, kg_test, and kg_validation.
        
    """
    # Concatenate edges from all knowledge graphs
    all_edges = torch.cat(( kg_train.triplets,
                            kg_test.triplets,
                            kg_validation.triplets))

    # Compute the number of triplets for all edges
    total_counts = torch.bincount(all_edges)

    # Compute occurences of each edge
    train_count = torch.bincount(kg_train.triplets,
                                minlength = len(total_counts))
    test_count = torch.bincount(kg_test.triplets,
                                minlength = len(total_counts))
    validation_count = torch.bincount(kg_validation.triplets,
                                    minlength = len(total_counts))

    # Compute proportions for each knowledge graph
    proportions = {}
    for edge_index in range(len(total_counts)):
        if total_counts[edge_index] > 0:
            proportions[edge_index] = {
                "train": train_count[edge_index].item() / total_counts[edge_index].item(),
                "test": test_count[edge_index].item() / total_counts[edge_index].item(),
                "validation": validation_count[edge_index].item() / total_counts[edge_index].item()
            }

    return proportions


def concat_kgs( kg_train: KnowledgeGraph,
                kg_validation: KnowledgeGraph,
                kg_test: KnowledgeGraph
                ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Merge the 3 splits of a knowledge graph into the original knowledge graph.

    Arguments
    ---------
    kg_train: KnowledgeGraph
        Train split from the knowledge graph.
    kg_test: KnowledgeGraph
        Test split from the knowledge graph.
    kg_validation: KnowledgeGraph
        Validation split from the knowledge graph.

    Returns
    -------
    head: torch.Tensor, shape: [merged_kg.node_count]
        List of head indices.
    tail: torch.Tensor, shape: [merged_kg.node_count]
        List of tail indices.
    edge: torch.Tensor, shape: [merged_kg.node_count]
        List of edge indices.
    
    Notes
    -----
    (merged_kg.node_count) is the number of nodes of the newly merged knowledge graph.
    
    """
    head = cat((kg_train.head_indices,
                kg_validation.head_indices,
                kg_test.head_indices))
    
    tail = cat((kg_train.tail_indices,
                kg_validation.tail_indices,
                kg_test.tail_indices))
    
    edge = cat((kg_train.edge_indices,
                kg_validation.edge_indices,
                kg_test.edge_indices))
    
    return head, tail, edge


def count_triplets( kg1: KnowledgeGraph,
                    kg2: KnowledgeGraph,
                    duplicates: List[Tuple[int, int]],
                    reverse_duplicates: List[Tuple[int, int]]
                    ) -> Tuple[int, int]:
    """
    TODO.What_the_function_does_about_globally
    
    Arguments
    ---------
    kg1: KnowledgeGraph
        First knowledge graph.
    kg2: KnowledgeGraph
        Second knowledge graph.
    duplicates: List[Tuple[int, int]]
        List returned by torchkge.utils.data_redundancy.duplicates.
    reverse_duplicates: List[Tuple[int, int]]
        List returned by torchkge.utils.data_redundancy.duplicates.

    Returns
    -------
    duplicate_count: int
        Number of triplets in kg2 that have their duplicate triplet
        in kg1
    reverse_duplicate_count: int
        Number of triplets in kg2 that have their reverse duplicate
        triplet in kg1.
        
    """
    duplicate_count = 0
    for first_edge_type, second_edge_type in duplicates:
        head_tail_train = kg1.get_pairs(second_edge_type, type = "head_tail")
        head_tail_test = kg2.get_pairs(first_edge_type, type = "head_tail")

        duplicate_count += len(head_tail_test.intersection(head_tail_train))

        head_tail_train = kg1.get_pairs(first_edge_type, type = "head_tail")
        head_tail_test = kg2.get_pairs(second_edge_type, type = "head_tail")

        duplicate_count += len(head_tail_test.intersection(head_tail_train))

    reverse_duplicate_count = 0
    for first_edge_type, second_edge_type in reverse_duplicates:
        tail_head_train = kg1.get_pairs(second_edge_type, type = "tail_head")
        head_tail_test = kg2.get_pairs(first_edge_type, type = "head_tail")

        reverse_duplicate_count += len(head_tail_test.intersection(tail_head_train))

        tail_head_train = kg1.get_pairs(first_edge_type, type = "tail_head")
        head_tail_test = kg2.get_pairs(second_edge_type, type = "head_tail")

        reverse_duplicate_count += len(head_tail_test.intersection(tail_head_train))

    return duplicate_count, reverse_duplicate_count


def find_best_model(dir: Path):
    """
    TODO.What_the_function_does_about_globally
    
    Arguments
    ---------
    dir: Path
        TODO.What_that_argument_is_or_does
        
    Returns
    -------
    TODO.result_name: TODO.type
        TODO.What_that_variable_is_or_does

    """
    return max(
        (filename
        for filename
        in os.listdir(dir)
        if filename.startswith("best_model_checkpoint_validation_metrics=")
        and filename.endswith(".pt")),
        
        key = lambda filename: float(filename.split("validation_metrics=")[1].rstrip(".pt")),
        
        default = None
    )
    
    
def initialize_embedding(embedding_count: int,
                        embedding_dimensions: int,
                        device: str = "cpu"
                        ) -> nn.Embedding:
    """
    TODO.What_the_function_does_about_globally
    
    Arguments
    ---------
    embedding_count: int
        TODO.What_that_argument_is_or_does
    embedding_dimensions: int
        TODO.What_that_argument_is_or_does
    device: str, default to "cpu"
        TODO.What_that_argument_is_or_does
        
    Returns
    -------
    embedding: nn.Embedding
        TODO.What_that_variable_is_or_does
    
    """
    embedding = nn.Embedding(embedding_count, embedding_dimensions, device = device)
    nn.init.xavier_uniform_(embedding.weight.data)
    
    return embedding


def read_train_metrics(train_metrics_file: Path
                        ) -> pd.DataFrame:
    """
    TODO.What_the_function_does_about_globally
    
    Arguments
    ---------
    train_metrics_file: Path
        TODO.What_that_argument_is_or_does
        
    Returns
    -------
    dataframe: pd.DataFrame
        TODO.What_that_variable_is_or_does
    
    """
    dataframe = pd.read_csv(train_metrics_file)

    dataframe = dataframe[~dataframe["Epoch"].astype(str).str.contains("CHECKPOINT RESTART")]

    dataframe["Epoch"] = dataframe["Epoch"].astype(int)
    dataframe = dataframe.sort_values(by = "Epoch")

    dataframe = dataframe.drop_duplicates(subset = ["Epoch"], keep = "last")

    return dataframe


def plot_learning_curves(train_metrics_file: Path,
                        output_directory: Path,
                        validation_metric_value: str):
    """
    TODO.What_the_function_does_about_globally
    
    Arguments
    ---------
    train_metrics_file: Path
        TODO.What_that_argument_is_or_does
    output_directory: Path
        TODO.What_that_argument_is_or_does
    validation_metric_value: str
        TODO.What_that_argument_is_or_does
    
    """    
    output_directory = Path(output_directory)
    dataframe = read_train_metrics(train_metrics_file)
    dataframe["Training Loss"] = pd.to_numeric(dataframe["Training Loss"], errors = "coerce")
    dataframe[f"Validation {validation_metric_value}"] = pd.to_numeric(dataframe[f"Validation {validation_metric_value}"], errors = "coerce")
    
    plt.figure(figsize = (12, 5))

    # Plot for training loss
    plt.subplot(1, 2, 1)
    plt.plot(dataframe["Epoch"], dataframe["Training Loss"], label = "Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig(output_directory.joinpath("training_loss_curve.png"))

    # Plot for validation MRR
    plt.subplot(1, 2, 2)
    plt.plot(dataframe["Epoch"], dataframe[f"Validation {validation_metric_value}"], label = f"Validation {validation_metric_value}")
    plt.xlabel("Epoch")
    plt.ylabel(f"Validation {validation_metric_value}")
    plt.title("Validation Metric over Epochs")
    plt.legend()
    plt.savefig(output_directory.joinpath("validation_metric_curve.png"))


def filter_scores(  scores: Tensor,
                    graphindices: Tensor,
                    missing: Literal["head", "tail", "edge"],
                    first_index: Tensor,
                    second_index: Tensor,
                    true_index: Tensor | None
                    ) -> Tensor:
    """
    Filter a score tensor to ignore the score attributed to true node or edge except the ones that are being predicted.
    
    Arguments
    ---------
    scores: torch.Tensor
        Tensor of shape [batch_size, n] where n is the number of nodes or edges, depending on what is filtered.
    graphindices: torch.Tensor
        Tensor of shape [4, triplet_count] containing every true triplet in the KG.
    missing: "head", "tail" or "edge"
        The part of the triplet that is currently being predicted.
    first_index: torch.Tensor
        Tensor containing the index of the heads (if missing is "edge" or "tails")
        or tails (if missing is "head") that are part of the triplet being predicted.
    second_index: torch.Tensor
        Tensor containing the index of the tails (if missing is "edge")
        or the edges (if missing is "head" or "tails") that are part of the triplet being predicted.
    true_index: torch.Tensor, optional
        Tensor containing the index of the nodes or edges currently being predicted.
        If omitted, every true index will be filtered out.

    Returns
    -------
    filtered_scores: torch.Tensor
        Tensor of shape [batch_size, n] with -Inf values for all true node/edge index except the ones being predicted.
        
    """
    batch_size = scores.shape[0]
    filtered_scores = scores.clone()

    if missing == "edge":
        first_mask = torch.isin(graphindices[0], first_index)
        second_mask = torch.isin(graphindices[1], second_index)
        missing_index = 2
    else:
        node_index = 0 if missing == "tail" else 1
        missing_index = 1 - node_index
        first_mask = torch.isin(graphindices[node_index], first_index)
        second_mask = torch.isin(graphindices[2], second_index)

    for i in range(batch_size):
        if true_index is None:
            true_mask = torch.zeros(graphindices.size(1), dtype = torch.bool)
        else:
            true_mask = torch.isin(graphindices[missing_index], true_index[i])

        true_targets = graphindices[missing_index, 
                                    first_mask & 
                                    second_mask & 
                                    ~true_mask
                                ]
        filtered_scores[i, true_targets] = - float('Inf')

    return filtered_scores


def merge_kg(kg_list: List[KnowledgeGraph],
            complete_graphindices: bool = False
            ) -> KnowledgeGraph:
    """
    Merge multiple KnowledgeGraph objects into a unique one.
    
    Arguments
    ---------
    kg_list: List[KnowledgeGraph]
        The list of all knowledge graphs to be merged.
    complete_graphindices: bool, default to False
        Whether or not the removed_triplets tensor should be integrated into the final KG's graphindices.

    Raises
    ------
    AssertionError #1
        Knowledge graphs in kg_list must have the same node_to_index (ent2ix).
    AssertionError #2
        Knowledge graphs in kg_list must have the same edge_to_index (rel2ix).
    AssertionError #3
        Knowledge graphs in kg_list must have the same node_type_to_index (nt2ix).
    AssertionError #4
        Knowledge graphs in kg_list must have the same triplet_types.
    
    Returns
    -------
    KnowledgeGraph
        The merged KnowledgeGraph object.
        
    """
    first_kg = kg_list[0]
    for kg in kg_list:
        kg.clean()
    assert all(first_kg.node_to_index == kg.node_to_index for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different node_to_index (ent2ix)."
    assert all(first_kg.edge_to_index == kg.edge_to_index for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different edge_to_index (rel2ix)."
    assert all(first_kg.node_type_to_index == kg.node_type_to_index for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different node_type_to_index (nt2ix)."
    assert all(first_kg.triplet_types == kg.triplet_types for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different triplet_types."

    new_graphindices = cat([kg.graphindices for kg in kg_list], dim = 1)
    if complete_graphindices:
        removed_graphindices = cat([kg.removed_triplets for kg in kg_list], dim = 1)
        new_graphindices = cat([new_graphindices, removed_graphindices], dim = 1)
    
    return first_kg.__class__(
        graphindices = new_graphindices,
        node_to_index = first_kg.node_to_index,
        edge_to_index = first_kg.edge_to_index,
        node_type_to_index = first_kg.node_type_to_index,
        triplet_types = first_kg.triplet_types
    )

def get_dictionary_mapping(dataframe: pd.DataFrame, nodes = True) -> Dict[str, int]:
    """
    Build the dictionary used to map either the node or edge identifiers to their index in the graph.

    Arguments
    ---------
    dataframe: pd.DataFrame
        Pandas dataframe containing at least three columns : "head", "edge" and "tail".
        Other columns are ignored.
    nodes: bool, optional, default to True
        If True will build the dictionary for the nodes mapping, otherwise will build
        the dictionary for the edge mapping.
    
    Returns
    -------
    node_to_index or edge_to_index: Dict[str, int]
        Mapping dictionary for nodes or edges.

    Notes
    -----
    This function is adapted from the torchkge.utils.operations.get_dictionaries() from the TorchKGE package.
    """
    if nodes:
        unique_nodes = list(set(dataframe["head"].unique()).union(set(dataframe["tail"]).unique()))
        return {node: index for index, node in enumerate(sorted(unique_nodes))}
    else:
        unique_edges = list(dataframe["edge"].unique())
        return {edge: index for index, edge in enumerate(sorted(unique_edges))}

def get_average_heads_per_tail(graphindices: Tensor) -> Dict[float, float]:
    """
    Get the average number of heads per tail across each edges.

    Arguments
    ---------
    graphindices: torch.Tensor, dtype: torch.long, shape: [4, triplet_count]
        The knowledge graph representation as a tensor with four rows, respectively
        the head, tail, edge and triplet indices.
    
    Returns
    -------
    average_heads_per_tail: Dict[float,float]
        Keys: relation indices; Values: average number of heads per tail
    """
    dataframe = pd.DataFrame(graphindices.T.cpu().numpy(), columns=["head","tail","edge","triplet"])
    dataframe = dataframe.groupby(["edge", "tail"]).count().groupby("edge").mean()
    dataframe.reset_index(inplace=True)
    return {dataframe.loc[i].values[0]: dataframe.loc[i].values[1] for i in dataframe.index}

def get_average_tails_per_head(graphindices: Tensor) -> Dict[float, float]:
    """
    Get the average number of tails per head across each edges.

    Arguments
    ---------
    graphindices: torch.Tensor, dtype: torch.long, shape: [4, triplet_count]
        The knowledge graph representation as a tensor with four rows, respectively
        the head, tail, edge and triplet indices.
    
    Returns
    -------
    average_tails_per_head: Dict[float,float]
        Keys: relation indices; Values: average number of tails per head
    """
    dataframe = pd.DataFrame(graphindices.T.cpu().numpy(), columns=["head","tail","edge","triplet"])
    dataframe = dataframe.groupby(["head", "edge"]).count().groupby("edge").mean()
    dataframe.reset_index(inplace=True)
    return {dataframe.loc[i].values[0]: dataframe.loc[i].values[1] for i in dataframe.index}

def get_bernoulli_probabilities(knowledge_graph: KnowledgeGraph) -> Dict[float, float]:
    """
    Evaluate the Bernoulli probabilities for negative sampling as in the
    TransH original paper by Wang et al. (2014).

    Arguments
    ---------
    knowledge_graph: kgate.KnowledgeGraph
        The knowledge graph to sample bernoulli probabilities from.
    
    Returns
    -------
    bernoulli_probabilities: Dict[int, float]
        Sampled probabilities of tails for each head. Keys: edge indices; Values: probabilities.
    """
    heads_per_tail = get_average_heads_per_tail(knowledge_graph.graphindices)
    tails_per_head = get_average_tails_per_head(knowledge_graph.graphindices)

    assert heads_per_tail.keys() == tails_per_head.keys(), "The edges between heads_per_tail and tails_per_edge sets do not correspond."

    for edge in tails_per_head.keys():
        tails_per_head[edge] = tails_per_head[edge] / (tails_per_head[edge] + heads_per_tail[edge])
    
    return tails_per_head