"""Knowledge Graph preprocessing functions to run before any training procedure."""

import logging
import pickle
from pathlib import Path
from typing import Tuple, List, Set

import pandas as pd

import torch
from torch import cat

import torchkge

from .knowledgegraph import KnowledgeGraph
from .utils import set_random_seeds, compute_triplet_proportions


SUPPORTED_SEPARATORS = [",","\t",";"]


def prepare_knowledge_graph(config: dict, 
                            kg: KnowledgeGraph | None = None, 
                            dataframe: pd.DataFrame | None = None,
                            metadata: pd.DataFrame | None = None
                            ) -> Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    """
    Prepare and clean the knowledge graph.
    
    This function takes an input knowledge graph either as a csv file (from the configuration), an object of type
    `torchkge.KnowledgeGraph` or a pandas `DataFrame`. It is preprocessed by the `clean_knowledge_graph` function
    and saved as a pickle file with the `save_knowledge_graph` function.

    Arguments
    ---------
    config: dict
        The full configuration, usually parsed from the KGATE configuration file.
    kg: KnowledgeGraph, optional
        The knowledge graph as a single object of class KnowledgeGraph or inheriting the class (KnowledgeGraph inherits the class)
    dataframe: pd.DataFrame, optional
        The knowledge graph as a pandas DataFrame.
    metadata: pd.DataFrame, optional
        The metadata dataframe to associate to the knowledge graph.

    Raises
    ------
    ValueError
        Knowledge graph csv file not found or using a non supported separator.
        Supported separators are "," (comma), "\t" (tabulation), ";" (semicolon).
    NotImplementedError
        Knowledge graph type not supported.
        Supported knowledge graph types are KGATE's and TorchKGE's.
        
    Returns
    -------
    kg_train: KnowledgeGraph
        Subpart of the preprocessed and split knowledge graph, for training.
    kg_validation: KnowledgeGraph
        Subpart of the preprocessed and split knowledge graph, for validation.
    kg_test: KnowledgeGraph
        Subpart of the preprocessed and split knowledge graph, for test.
    
    Notes
    -----
    The CSV file can have any number of columns but at least three named head, tail and edge.
    
    """
    # Load knowledge graph
    if kg is None and dataframe is None:
        input_file = config["kg_csv"]
        kg_dataframe: pd.DataFrame = None

        for separator in SUPPORTED_SEPARATORS:
            try:
                kg_dataframe = pd.read_csv(input_file,
                                            sep = separator,
                                            usecols = ["head", "tail", "edge"])
                break
            except ValueError:
                continue
        
        if kg_dataframe is None:
            raise ValueError(f"The knowledge graph csv file was not found or uses a non supported separator. Supported separators are '{'\', \''.join(SUPPORTED_SEPARATORS)}'.")

        kg = KnowledgeGraph(dataframe = kg_dataframe, metadata = metadata)

    else:
        if kg is not None:
            if isinstance(kg, torchkge.KnowledgeGraph):
                kg_dataframe = kg.get_dataframe()
                kg = KnowledgeGraph(dataframe = kg_dataframe, metadata = metadata)
            elif isinstance(kg, KnowledgeGraph):
                kg = kg
            else:
                raise NotImplementedError(f"Knowledge graph type {type(kg)} is not supported. Supported knowledge graph types are KGATE's and TorchKGE's.")
        elif dataframe is not None:
            kg = KnowledgeGraph(dataframe = dataframe, metadata = metadata)
                
    # Clean and process knowledge graph
    kg_train, kg_validation, kg_test = clean_knowledge_graph(kg, config)

    # Save results
    save_knowledge_graph(config, kg_train, kg_validation, kg_test)

    return kg_train, kg_validation, kg_test


def save_knowledge_graph(config: dict,
                        kg_train: KnowledgeGraph,
                        kg_validation: KnowledgeGraph,
                        kg_test: KnowledgeGraph):
    """
    Save the knowledge graph to a pickle file.
    
    If the name of a pickle file is specified in the configuration, it will be used. Otherwise, the 
    file will be created in `config["output_directory"]/kg.pkl`.
    
    Arguments
    ---------
    config: dict
        The full configuration, usually parsed from the KGATE configuration file.
    kg_train: KnowledgeGraph
        The training subpart of the knowledge graph.
    kg_validation: KnowledgeGraph
        The validation subpart of the knowledge graph.
    kg_test: KnowledgeGraph
        The testing subpart of the knowledge graph.
        
    """
    if config["kg_pkl"] == "":
        pickle_filename = Path(config["output_directory"], "kg.pkl")
    else:
        pickle_filename = config["kg_pkl"]

    with open(pickle_filename, "wb") as file:
        pickle.dump(kg_train, file)
        pickle.dump(kg_validation, file)
        pickle.dump(kg_test, file)


def load_knowledge_graph(pickle_filename: Path):
    """
    Load the knowledge graph from a pickle file.
        
    Arguments
    ---------
    pickle_filename: Path
        Path to the pickle file.
        
    Returns
    -------
    kg_train, kg_validation, kg_test: KnowledgeGraph
        A tuple containing the preprocessed and split knowledge graph.
    
    """
    with open(pickle_filename, "rb") as file:
        kg_train = pickle.load(file)
        kg_validation = pickle.load(file)
        kg_test = pickle.load(file)
        
    return kg_train, kg_validation, kg_test


def clean_knowledge_graph(  kg: KnowledgeGraph,
                            config: dict
                            ) -> Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    """
    Clean and prepare the knowledge graph according to the configuration.
        
    Arguments
    ---------
    kg: KnowledgeGraph
        Knowledge graph on which the cleaning will be done.
    config: dict
        The full configuration, usually parsed from the KGATE configuration file.
    
    Raises
    ------
    ValueError
        One or more nodes are not covered in the training set after ensuring node coverage.
        
    Returns
    -------
    new_kg_train: KnowledgeGraph
        Cleaned training knowledge graph.
    new_kg_validation: KnowledgeGraph
        Cleaned validation knowledge graph.
    new_kg_test: KnowledgeGraph
        Cleaned test knowledge graph.
    
    """
    set_random_seeds(config["seed"])

    index_to_edge_name = {value: key for key, value in kg.edge_to_index.items()}

    if config["preprocessing"]["remove_duplicate_triplets"]:
        logging.info("Removing duplicated triplets...")
        kg = kg.remove_duplicate_triplets()

    duplicated_edges_list = []

    if config["preprocessing"]["flag_near_duplicate_edges"]:
        logging.info("Checking for near duplicates edges...")
        theta_first_edge_type = config["preprocessing"]["params"]["theta_first_edge_type"]
        theta_second_edge_type = config["preprocessing"]["params"]["theta_second_edge_type"]
        duplicate_edges, reverse_duplicate_edges = kg.duplicates(theta_first_edge_type = theta_first_edge_type,
                                                                theta_second_edge_type = theta_second_edge_type)
        if duplicate_edges:
            logging.info(f"Adding {len(duplicate_edges)} synonymous edges ({[index_to_edge_name[edge] for edge in duplicate_edges]}) to the list of known duplicated edges.")
            duplicated_edges_list.extend(duplicate_edges)
        if reverse_duplicate_edges:
            logging.info(f"Adding {len(reverse_duplicate_edges)} anti-synonymous edges ({[index_to_edge_name[edge] for edge in reverse_duplicate_edges]}) to the list of known duplicated edges.")
            duplicated_edges_list.extend(reverse_duplicate_edges)
    
    if config["preprocessing"]["make_directed"]:
        undirected_edges_names = config["preprocessing"]["make_directed_edges"]
        if len(undirected_edges_names) == 0:
            undirected_edges_names = list(kg.edge_to_index.keys())
        logging.info(f"Adding reverse triplets for edges {undirected_edges_names}...")
        edges_to_process = [kg.edge_to_index[edge_name] for edge_name in undirected_edges_names]
        kg, undirected_edges_list = kg.add_reverse_edges(edges_to_process)
            
        if config["preprocessing"]["flag_near_duplicate_edges"]:
            logging.info(f"Adding created reverses {[(edge_name, edge_name + "_inv") for edge_name in undirected_edges_names]} to the list of known duplicated edges.")
            duplicated_edges_list.extend(undirected_edges_list)

    # Split the knowledge graph into 3 datasets: train, validation, set
    logging.info("Splitting the dataset into train, validation and test sets...")
    kg_train, kg_validation, kg_test = kg.split_kg(split_proportions = config["preprocessing"]["split"])

    # Verify the node coverage
    kg_train_ok, _ = verify_node_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Node coverage verification failed...")  
    else:
        logging.info("Node coverage verified successfully.")

    # Clean the dataset if set as TRUE in the config file
    if config["preprocessing"]["clean_train_set"]:
        logging.info("Cleaning the train set to avoid data leakage...")
        logging.info("Step 1: with respect to validation set.")
        kg_train = clean_datasets(kg_train, kg_validation, known_reverses = duplicated_edges_list)
        logging.info("Step 2: with respect to test set.")
        kg_train = clean_datasets(kg_train, kg_test, known_reverses = duplicated_edges_list)

    kg_train_ok, _ = verify_node_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Node coverage verification failed...")
    else:
        logging.info("Node coverage verified successfully.")

    new_kg_train, new_kg_validation, new_kg_test = ensure_node_coverage(kg_train, kg_validation, kg_test)

    kg_train_ok, missing_nodes = verify_node_coverage(new_kg_train, kg)
    if not kg_train_ok:
        logging.info(f"Node coverage verification failed. {len(missing_nodes)} nodes are missing.")
        logging.info(f"Missing nodes: {missing_nodes}")
        raise ValueError("One or more nodes are not covered in the training set after ensuring node coverage...")
    else:
        logging.info("Node coverage verified successfully.")

    logging.info("Computing triplet proportions...")
    logging.info(compute_triplet_proportions(kg_train, kg_test, kg_validation))

    return new_kg_train, new_kg_validation, new_kg_test


def verify_node_coverage(kg_train: KnowledgeGraph,
                        kg_full: KnowledgeGraph
                        ) -> Tuple[bool, List[str]]:
    """
    Verify that all nodes in the full knowledge graph are represented in the training set.

    Arguments
    ---------
    kg_train: KnowledgeGraph
        The training knowledge graph subset.
    kg_full: KnowledgeGraph
        The full knowledge graph.

    Returns
    -------
    missing_nodes: Tuple[bool, List[str]]
        A tuple where the first element is True if all nodes in the full knowledge graph are present in the training 
        knowledge graph, and the second element is a list of missing nodes (names) if any are missing.
    
    """
    # Get node identifiers for the train graph and full graph
    node_indices_train = set(cat((kg_train.head_indices, kg_train.tail_indices)).tolist())
    node_indices_full = set(cat((kg_full.head_indices, kg_full.tail_indices)).tolist())
    
    # Missing nodes in the train graph
    missing_node_indices = node_indices_full - node_indices_train
    
    if missing_node_indices:
        # Invert node_to_index dictionnary to get index_to_node
        index_to_node = {value: key for key, value in kg_full.node_to_index.items()}
        
        # Get missing node names from their indices
        missing_node_names = [index_to_node[index] for index in missing_node_indices if index in index_to_node]
        return False, missing_node_names
    
    else:
        return True, []
    

def ensure_node_coverage(kg_train: KnowledgeGraph,
                        kg_validation: KnowledgeGraph,
                        kg_test: KnowledgeGraph
                        ) -> Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    """
    Ensure that all nodes in kg_train.node_to_index are present in kg_train as head or tail.
    If a node is missing, move a triplet involving that node from kg_validation or kg_test to kg_train.

    Arguments
    ---------
    kg_train: KnowledgeGraph
        The training knowledge graph subset to ensure node coverage of.
    kg_validation: KnowledgeGraph
        The validation knowledge graph subset from which to move triplets if needed.
    kg_test: KnowledgeGraph
        The test knowledge graph subset from which to move triplets if needed.

    Returns
    -------
    kg_train: KnowledgeGraph
        The updated training knowledge graph with all nodes covered.
    kg_validation: KnowledgeGraph
        The updated validation knowledge graph.
    kg_test: KnowledgeGraph
        The updated test knowledge graph.
    
    """
    # Get the indices of all nodes in kg_train 
    train_nodes = set(kg_train.node_to_index.values())

    # Get he indices of all nodes in kg_train as heads or tails
    present_heads = set(kg_train.head_indices.tolist())
    present_tails = set(kg_train.tail_indices.tolist())
    present_nodes = present_heads.union(present_tails)

    # Identify nodes missing from kg_train
    missing_nodes = train_nodes - present_nodes

    logging.info(f"Total nodes in full kg: {len(train_nodes)}")
    logging.info(f"Nodes present in kg_train: {len(present_nodes)}")
    logging.info(f"Missing nodes in kg_train: {len(missing_nodes)}")


    def find_and_move_triplets( kg_source: KnowledgeGraph,
                                nodes: Set[int]):
        
        nonlocal kg_train, kg_validation, kg_test

        # Convert `nodes` set to a `Tensor` for compatibility with `torch.isin`
        nodes_tensor = torch.tensor(list(nodes), dtype = kg_source.head_indices.dtype)

        # Create masks for all triplets where the missing node is present
        mask_heads = torch.isin(kg_source.head_indices, nodes_tensor)
        mask_tails = torch.isin(kg_source.tail_indices, nodes_tensor)
        mask = mask_heads | mask_tails

        if mask.any():
            # Extract the indices and corresponding triplets
            indices = torch.nonzero(mask, as_tuple = True)[0]
            triplets = kg_source.graphindices[:, indices]
            logging.info(triplets)
            # Add the found triplets to kg_train
            kg_train = kg_train.add_triplets(triplets)

            # Remove the triplets from source_kg
            kg_cleaned = kg_source.remove_triplets(indices)
            if kg_source == kg_validation:
                kg_validation = kg_cleaned
            else:
                kg_test = kg_cleaned

            # Update the list of missing nodes
            nodes_in_triplets = set(triplets[0].tolist() + triplets[1].tolist())
            remaining_nodes = nodes - set(nodes_in_triplets)
            
            return remaining_nodes
        
        return nodes

    # Move triplets from kg_validation then from kg_test
    missing_nodes = find_and_move_triplets(kg_validation, missing_nodes)
    if len(missing_nodes) > 0:
        missing_nodes = find_and_move_triplets(kg_test, missing_nodes)

    # Log the missing nodes that could not be connected
    if len(missing_nodes) > 0:
        for node in missing_nodes:
            logging.info(f"Warning: No triplet found involving node '{node}' in kg_validation or kg_test. Node remains unconnected in kg_train.")

    return kg_train, kg_validation, kg_test


def clean_datasets( kg_train: KnowledgeGraph,
                    kg_second: KnowledgeGraph,
                    known_reverses: List[Tuple[int, int]]
                    ) -> KnowledgeGraph:
    """
    Clean the train knowledge graph by removing reverse duplicate triplets contained
    in the second knowledge graph (test or validation).

    Arguments
    ---------
    kg_train: KnowledgeGraph
        The training knowledge graph subset.
    kg_second: KnowledgeGraph
        The second knowledge graph subset, test or validation.
    known_reverses: List[Tuple[int, int]]
        Each tuple contains two edges (first_edge_type, second_edge_type) that are known reverse edges.

    Returns
    -------
    kg_train: KnowledgeGraph
        The cleaned train knowledge graph subset.
        
    """
    for first_edge_type, second_edge_type in known_reverses:

        logging.info(f"Processing edge pair: ({first_edge_type}, {second_edge_type})")

        # Get (head, tail) pairs, in kg_second, that are related by first_edge_type
        first_edge_type_pairs_in_kg_second = kg_second.get_pairs(first_edge_type, type = "head_tail")
        
        # Get indices of the (head, tail) pairs, in kg_train, that are related by second_edge_type
        indices_to_remove_kg_train = [edge_index
                                        for edge_index, (head, tail)
                                        in enumerate(zip(kg_train.tail_indices, kg_train.head_indices))
                                        if (head.item(), tail.item()) in first_edge_type_pairs_in_kg_second
                                        and kg_train.edge_indices[edge_index].item() == second_edge_type]
        indices_to_remove_kg_train.extend([edge_index
                                            for edge_index, (head, tail)
                                            in enumerate(zip(kg_train.head_indices, kg_train.tail_indices))
                                            if (head.item(), tail.item()) in first_edge_type_pairs_in_kg_second
                                            and kg_train.edge_indices[edge_index].item() == second_edge_type])
        
        # Remove these (head, tail) pairs from kg_train
        kg_train = kg_train.remove_triplets(torch.tensor(indices_to_remove_kg_train, dtype = torch.long))

        logging.info(f"Found {len(indices_to_remove_kg_train)} triplets to remove for edge {second_edge_type} with reverse {first_edge_type}.")

        # Get (head, tail) pairs, in kg_second, that are related by second_edge_type
        second_edge_type_pairs_in_kg_second = kg_second.get_pairs(second_edge_type, type = "head_tail")
        
        # Get indices of the (head, tail) pairs, in kg_train, that are related by first_edge_type
        indices_to_remove_kg_train_reverse = [edge_index
                                                for edge_index, (head, tail)
                                                in enumerate(zip(kg_train.tail_indices, kg_train.head_indices))
                                                if (head.item(), tail.item()) in second_edge_type_pairs_in_kg_second
                                                and kg_train.edge_indices[edge_index].item() == first_edge_type]
        indices_to_remove_kg_train_reverse.extend([edge_index
                                                    for edge_index, (head, tail)
                                                    in enumerate(zip(kg_train.head_indices, kg_train.tail_indices))
                                                    if (head.item(), tail.item()) in second_edge_type_pairs_in_kg_second
                                                    and kg_train.edge_indices[edge_index].item() == first_edge_type])

        # Remove these (head, tail) pairs from kg_train
        kg_train = kg_train.remove_triplets(torch.tensor(indices_to_remove_kg_train_reverse, dtype=torch.long))

        logging.info(f"Found {len(indices_to_remove_kg_train_reverse)} reverse triplets to remove for edge {first_edge_type} with reverse {second_edge_type}.")
    
    return kg_train


def clean_cartesians(kg_first: KnowledgeGraph,
                    kg_second: KnowledgeGraph,
                    known_cartesian: List[int],
                    node_type: str = "head"
                    ) -> Tuple[KnowledgeGraph, KnowledgeGraph]:
    """
    Transfer cartesian product triplets from training set to test set to prevent data leakage.
    For each node (head or tail) involved in a cartesian product edge in the test set,
    all corresponding triplets in the training set are moved to the test set.
    
    Arguments
    ---------
    kg_first: KnowledgeGraph
        Train set knowledge graph to be cleaned.
        Will be modified by removing cartesian product triplets.
    kg_second: KnowledgeGraph
        Test set knowledge graph to be augmented.
        Will receive the transferred cartesian product triplets.
    known_cartesian: List[int]
        List of edge indices that represent cartesian product relationships.
        These are edges where if (head, tail 1, edge) exists, then (head, tail 2, edge) likely exists
        for many other tail nodes 'tail 2' (or vice versa for tail-based cartesian products).
    node_type: str, optional, default to "head"
        Either "head" or "tail" to specify which node type to consider for cartesian products.
    
    Returns
    -------
    kg_first: KnowledgeGraph
        Cleaned train set knowledge graph, with cartesian triplets removed.
    kg_second: KnowledgeGraph
        Augmented test set knowledge graph, with the transferred triplets added.
        
    """
    assert node_type in ["head", "tail"], "node_type must be either 'head' or 'tail'"
    
    for edge_index in known_cartesian:
        # Find all nodes in test set that participate in the cartesian product
        mask = (kg_second.edge_indices == edge_index)
        if node_type == "head":
            cartesian_node_indices = kg_second.head_indices[mask].view(-1,1)
            # Find matching triplets in train set with same head and edge
            all_indices_to_move = []
            for node_index in cartesian_node_indices:
                mask = (kg_first.head_indices == node_index) & (kg_first.edge_indices == edge_index)
                indices = mask.nonzero().squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                all_indices_to_move.extend(indices.tolist())
        else:  # tail
            cartesian_node_indices = kg_second.tail_indices[mask].view(-1,1)
            # Find matching triplets in train set with same tail and edge
            all_indices_to_move = []
            for node_index in cartesian_node_indices:
                mask = (kg_first.tail_indices == node_index) & (kg_first.edge_indices == edge_index)
                indices = mask.nonzero().squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                all_indices_to_move.extend(indices.tolist())
            
        if all_indices_to_move:
            # Extract the triplets to be transferred
            triplets_to_move = torch.stack([
                kg_first.head_indices[all_indices_to_move],
                kg_first.edge_indices[all_indices_to_move],
                kg_first.tail_indices[all_indices_to_move]
            ], dim = 1)
            
            # Remove identified triplets from train set
            kg_first = kg_first.remove_triplets(torch.tensor(all_indices_to_move, dtype = torch.long))
            
            # Add transferred triplets to test set while preserving KG structure
            kg_second_dictionnary = {
                "heads": torch.cat([kg_second.head_indices, triplets_to_move[:, 0]]),
                "tails": torch.cat([kg_second.tail_indices, triplets_to_move[:, 2]]),
                "edges": torch.cat([kg_second.edge_indices, triplets_to_move[:, 1]]),
            }
            
            kg_second = kg_second.__class__(
                kg = kg_second_dictionnary,
                node_to_index = kg_second.node_to_index,
                edge_to_index = kg_first.edge_to_index,
                dict_of_heads = kg_second.dict_of_heads,
                dict_of_tails = kg_second.dict_of_tails,
                dict_of_rels = kg_second.dict_of_rels
            )
            
    return kg_first, kg_second