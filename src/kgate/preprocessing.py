"""Knowledge Graph preprocessing functions to run before any training procedure."""

from pathlib import Path
import logging
import pickle
from typing import Tuple, List, Set

import pandas as pd

import torch
from torch import cat

import torchkge

from .utils import set_random_seeds, compute_triplet_proportions
from .knowledgegraph import KnowledgeGraph

SUPPORTED_SEPARATORS = [",","\t",";"]

def prepare_knowledge_graph(config: dict, 
                            kg: KnowledgeGraph | None, 
                            dataframe: pd.DataFrame | None,
                            metadata: pd.DataFrame | None
                            ) -> Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    """Prepare and clean the knowledge graph.
    
    This function takes an input knowledge graph either as a csv file (from the configuration), an object of type
    `torchkge.KnowledgeGraph` or a pandas `DataFrame`. It is preprocessed by the `clean_knowledge_graph` function
    and saved as a pickle file with the `save_knowledge_graph` function.
    
    Notes
    -----
    The CSV file can have any number of columns but at least three named from, to and rel.

    Arguments
    ---------
    config : dict
        The full configuration, usually parsed from the KGATE configuration file.
    kg : torchKGE.KnowledgeGraph
        The knowledge graph as a single object of class KnowledgeGraph or inheriting the class (KnowledgeGraph inherits the class)
    df : pd.DataFrame
        The knowledge graph as a pandas DataFrame.
        
    Returns
    -------
    kg_train, kg_val, kg_test : KnowledgeGraph
        A tuple containing the preprocessed and split knowledge graph."""

    # Load knowledge graph
    if kg is None and dataframe is None:
        input_file = config["kg_csv"]
        kg_dataframe: pd.DataFrame = None

        for separator in SUPPORTED_SEPARATORS:
            try:
                kg_dataframe = pd.read_csv(input_file, sep=separator, usecols=["from","to","rel"])
                break
            except ValueError:
                continue
        
        if kg_dataframe is None:
            raise ValueError(f"The Knowledge Graph csv file was not found or uses a non supported separator. Supported separators are '{'\', \''.join(SUPPORTED_SEPARATORS)}'.")

        kg = KnowledgeGraph(dataframe=kg_dataframe, metadata=metadata)
    else:
        if kg is not None:
            if isinstance(kg, torchkge.KnowledgeGraph):
                kg_dataframe = kg.get_dataframe()
                kg = KnowledgeGraph(dataframe=kg_dataframe, metadata=metadata)
            elif isinstance(kg, KnowledgeGraph):
                kg = kg
            else:
                raise NotImplementedError(f"Knowledge Graph type {type(kg)} is not supported.")
        elif dataframe is not None:
            kg = KnowledgeGraph(dataframe = dataframe, metadata = metadata)
                
    # Clean and process knowledge graph
    kg_train, kg_validation, kg_test = clean_knowledge_graph(kg, config)

    # Save results
    save_knowledge_graph(config, kg_train, kg_validation, kg_test)

    return kg_train, kg_validation, kg_test

def save_knowledge_graph(config: dict, kg_train: KnowledgeGraph, kg_validation: KnowledgeGraph, kg_test:KnowledgeGraph):
    """Save the knowledge graph to a pickle file.
    
    If the name of a pickle file is specified in the configuration, it will be used. Otherwise, the 
    file will be created in `config["output_directory"]/kg.pkl`.
    
    Arguments
    ---------
    config : dict
        The full configuration, usually parsed from the KGATE configuration file.
    kg_train : KnowledgeGraph
        The training knowledge graph.
    kg_val : KnowledgeGraph
        The validation knowledge graph.
    kg_test : KnowledgeGraph
        The testing knowledge graph."""
    
    if config["kg_pkl"] == "":
        pickle_filename = Path(config["output_directory"], "kg.pkl")
    else:
        pickle_filename = config["kg_pkl"]

    with open(pickle_filename, "wb") as file:
        pickle.dump(kg_train, file)
        pickle.dump(kg_validation, file)
        pickle.dump(kg_test, file)

def load_knowledge_graph(pickle_filename: Path):
    """Load the knowledge graph from a pickle file."""
    with open(pickle_filename, "rb") as file:
        kg_train = pickle.load(file)
        kg_validation = pickle.load(file)
        kg_test = pickle.load(file)
    return kg_train, kg_validation, kg_test

def clean_knowledge_graph(kg: KnowledgeGraph, config: dict) -> Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    """Clean and prepare the knowledge graph according to the configuration."""

    set_random_seeds(config["seed"])

    index_to_edge_name = {value: key for key, value in kg.edge_to_index.items()}

    if config["preprocessing"]["remove_duplicate_triples"]:
        logging.info("Removing duplicated triples...")
        kg = kg.remove_duplicate_triplets()

    duplicated_edges_list = []

    if config["preprocessing"]["flag_near_duplicate_relations"]:
        logging.info("Checking for near duplicates relations...")
        theta_first_edge_type = config["preprocessing"]["params"]["theta1"]
        theta_second_edge_type = config["preprocessing"]["params"]["theta2"]
        duplicate_edges, reverse_duplicate_edges = kg.duplicates(theta_first_edge_type=theta_first_edge_type, theta_second_edge_type=theta_second_edge_type)
        if duplicate_edges:
            logging.info(f"Adding {len(duplicate_edges)} synonymous relations ({[index_to_edge_name[edge] for edge in duplicate_edges]}) to the list of known duplicated relations.")
            duplicated_edges_list.extend(duplicate_edges)
        if reverse_duplicate_edges:
            logging.info(f"Adding {len(reverse_duplicate_edges)} anti-synonymous relations ({[index_to_edge_name[edge] for edge in reverse_duplicate_edges]}) to the list of known duplicated relations.")
            duplicated_edges_list.extend(reverse_duplicate_edges)
    
    if config["preprocessing"]["make_directed"]:
        undirected_edges_names = config["preprocessing"]["make_directed_relations"]
        if len(undirected_edges_names) == 0:
            undirected_edges_names = list(kg.edge_to_index.keys())
        logging.info(f"Adding reverse triplets for relations {undirected_edges_names}...")
        edges_to_process = [kg.edge_to_index[edge_name] for edge_name in undirected_edges_names]
        kg, undirected_edges_list = kg.add_reverse_edges(edges_to_process)
            
        if config["preprocessing"]["flag_near_duplicate_relations"]:
            logging.info(f"Adding created reverses {[(edge_name, edge_name + "_inv") for edge_name in undirected_edges_names]} to the list of known duplicated relations.")
            duplicated_edges_list.extend(undirected_edges_list)

    logging.info("Splitting the dataset into train, validation and test sets...")
    kg_train, kg_validation, kg_test = kg.split_kg(split_proportions=config["preprocessing"]["split"])

    kg_train_ok, _ = verify_node_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")  
    else:
        logging.info("Entity coverage verified successfully.")

    if config["preprocessing"]["clean_train_set"]:
        logging.info("Cleaning the train set to avoid data leakage...")
        logging.info("Step 1: with respect to validation set.")
        kg_train = clean_datasets(kg_train, kg_validation, known_reverses=duplicated_edges_list)
        logging.info("Step 2: with respect to test set.")
        kg_train = clean_datasets(kg_train, kg_test, known_reverses=duplicated_edges_list)

    kg_train_ok, _ = verify_node_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    new_kg_train, new_kg_validation, new_kg_test = ensure_node_coverage(kg_train, kg_validation, kg_test)


    kg_train_ok, missing_nodes = verify_node_coverage(new_kg_train, kg)
    if not kg_train_ok:
        logging.info(f"Entity coverage verification failed. {len(missing_nodes)} entities are missing.")
        logging.info(f"Missing entities: {missing_nodes}")
        raise ValueError("One or more entities are not covered in the training set after ensuring entity coverage...")
    else:
        logging.info("Entity coverage verified successfully.")

    logging.info("Computing triplet proportions...")
    logging.info(compute_triplet_proportions(kg_train, kg_test, kg_validation))

    return new_kg_train, new_kg_validation, new_kg_test

def verify_node_coverage(kg_train: KnowledgeGraph, kg_full: KnowledgeGraph) -> Tuple[bool, List[str]]:
    """
    Verify that all entities in the full knowledge graph are represented in the training set.

    Parameters
    ----------
    train_kg: KnowledgeGraph
        The training knowledge graph.
    full_kg: KnowledgeGraph
        The full knowledge graph.

    Returns
    -------
    tuple
        (bool, list)
        A tuple where the first element is True if all entities in the full knowledge graph are present in the training 
        knowledge graph, and the second element is a list of missing entities (names) if any are missing.
    """
    # Get entity identifiers for the train graph and full graph
    node_indices_train = set(cat((kg_train.head_indices, kg_train.tail_indices)).tolist())
    node_indices_full = set(cat((kg_full.head_indices, kg_full.tail_indices)).tolist())
    
    # Missing entities in the train graph
    missing_node_indices = node_indices_full - node_indices_train
    
    if missing_node_indices:
        # Invert ent2ix dict to get idx: entity_name
        index_to_node = {value: key for key, value in kg_full.node_to_index.items()}
        
        # Get missing entity names from their indices
        missing_node_names = [index_to_node[index] for index in missing_node_indices if index in index_to_node]
        return False, missing_node_names
    else:
        return True, []

def ensure_node_coverage(kg_train: KnowledgeGraph, kg_validation: KnowledgeGraph, kg_test:KnowledgeGraph) -> Tuple[KnowledgeGraph,KnowledgeGraph,KnowledgeGraph]:
    """
    Ensure that all entities in kg_train.ent2ix are present in kg_train as head or tail.
    If an entity is missing, move a triplet involving that entity from kg_val or kg_test to kg_train.

    Parameters
    ----------
    kg_train : torchkge.data_structures.KnowledgeGraph
        The training knowledge graph to ensure entity coverage.
    kg_val : torchkge.data_structures.KnowledgeGraph
        The validation knowledge graph from which to move triplets if needed.
    kg_test : torchkge.data_structures.KnowledgeGraph
        The test knowledge graph from which to move triplets if needed.

    Returns
    -------
    kg_train : torchkge.data_structures.KnowledgeGraph
        The updated training knowledge graph with all entities covered.
    kg_val : torchkge.data_structures.KnowledgeGraph
        The updated validation knowledge graph.
    kg_test : torchkge.data_structures.KnowledgeGraph
        The updated test knowledge graph.
    """

    # Obtenir l"ensemble des entités dans kg_train.ent2ix
    train_nodes = set(kg_train.node_to_index.values())

    # Obtenir l"ensemble des entités présentes dans kg_train comme head ou tail
    present_heads = set(kg_train.head_indices.tolist())
    present_tails = set(kg_train.tail_indices.tolist())
    present_nodes = present_heads.union(present_tails)

    # Identifier les entités manquantes dans kg_train
    missing_nodes = train_nodes - present_nodes

    logging.info(f"Total entities in full kg: {len(train_nodes)}")
    logging.info(f"Entities present in kg_train: {len(present_nodes)}")
    logging.info(f"Missing entities in kg_train: {len(missing_nodes)}")

    def find_and_move_triplets(kg_source: KnowledgeGraph, nodes: Set[int]):
        nonlocal kg_train, kg_validation, kg_test

        # Convert `entities` set to a `Tensor` for compatibility with `torch.isin`
        nodes_tensor = torch.tensor(list(nodes), dtype=kg_source.head_indices.dtype)

        # Create masks for all triplets where the missing entity is present
        mask_heads = torch.isin(kg_source.head_indices, nodes_tensor)
        mask_tails = torch.isin(kg_source.tail_indices, nodes_tensor)
        mask = mask_heads | mask_tails

        if mask.any():
            # Extract the indices and corresponding triplets
            indices = torch.nonzero(mask, as_tuple=True)[0]
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

            # Update the list of missing entities
            nodes_in_triplets = set(triplets[0].tolist() + triplets[1].tolist())
            remaining_nodes = nodes - set(nodes_in_triplets)
            return remaining_nodes
        return nodes

    # Déplacer les triplets depuis kg_val puis depuis kg_test
    missing_nodes = find_and_move_triplets(kg_validation, missing_nodes)
    if len(missing_nodes) > 0:
        missing_nodes = find_and_move_triplets(kg_test, missing_nodes)

    # Loguer les entités restantes non trouvées
    if len(missing_nodes) > 0:
        for node in missing_nodes:
            logging.info(f"Warning: No triplet found involving entity '{node}' in kg_val or kg_test. Entity remains unconnected in kg_train.")

    return kg_train, kg_validation, kg_test


def clean_datasets(kg_train: KnowledgeGraph, kg_second: KnowledgeGraph, known_reverses: List[Tuple[int, int]]) -> KnowledgeGraph:
    """
    Clean the training KG by removing reverse duplicate triples contained in KG2 (test or val KG).

    Parameters
    ----------
    kg_train: torchkge.data_structures.KnowledgeGraph
        The training knowledge graph.
    kg2: torchkge.data_structures.KnowledgeGraph
        The second knowledge graph, test or validation.
    known_reverses: list of tuples
        Each tuple contains two relations (r1, r2) that are known reverse relations.

    Returns
    -------
    torchkge.data_structures.KnowledgeGraph
        The cleaned train knowledge graph.
    """

    for first_edge_type, second_edge_type in known_reverses:

        logging.info(f"Processing relation pair: ({first_edge_type}, {second_edge_type})")

        # Get (h, t) pairs in kg2 related by r1
        first_edge_type_pairs_in_kg_second = kg_second.get_pairs(first_edge_type, type="ht")
        # Get indices of (h, t) in kg_train that are related by r2
        indices_to_remove_kg_train = [edge_index for edge_index, (head, tail) in enumerate(zip(kg_train.tail_indices, kg_train.head_indices)) if (head.item(), tail.item()) in first_edge_type_pairs_in_kg_second and kg_train.edge_indices[edge_index].item() == second_edge_type]
        indices_to_remove_kg_train.extend([edge_index for edge_index, (head, tail) in enumerate(zip(kg_train.head_indices, kg_train.tail_indices)) if (head.item(), tail.item()) in first_edge_type_pairs_in_kg_second and kg_train.edge_indices[edge_index].item() == second_edge_type])
        
        # Remove those (h, t) pairs from kg_train
        kg_train = kg_train.remove_triplets(torch.tensor(indices_to_remove_kg_train, dtype=torch.long))

        logging.info(f"Found {len(indices_to_remove_kg_train)} triplets to remove for relation {second_edge_type} with reverse {first_edge_type}.")

        # Get (h, t) pairs in kg2 related by r2
        second_edge_type_pairs_in_kg_second = kg_second.get_pairs(second_edge_type, type="ht")
        # Get indices of (h, t) in kg_train that are related by r1
        indices_to_remove_kg_train_reverse = [edge_index for edge_index, (head, tail) in enumerate(zip(kg_train.tail_indices, kg_train.head_indices)) if (head.item(), tail.item()) in second_edge_type_pairs_in_kg_second and kg_train.edge_indices[edge_index].item() == first_edge_type]
        indices_to_remove_kg_train_reverse.extend([edge_index for edge_index, (head, tail) in enumerate(zip(kg_train.head_indices, kg_train.tail_indices)) if (head.item(), tail.item()) in second_edge_type_pairs_in_kg_second and kg_train.edge_indices[edge_index].item() == first_edge_type])

        # Remove those (h, t) pairs from kg_train
        kg_train = kg_train.remove_triplets(torch.tensor(indices_to_remove_kg_train_reverse, dtype=torch.long))

        logging.info(f"Found {len(indices_to_remove_kg_train_reverse)} reverse triplets to remove for relation {first_edge_type} with reverse {second_edge_type}.")
    
    return kg_train

def clean_cartesians(kg_first: KnowledgeGraph, kg_second: KnowledgeGraph, known_cartesian: List[int], node_type: str="head") -> Tuple[KnowledgeGraph,KnowledgeGraph]:
    """
    Transfer cartesian product triplets from training set to test set to prevent data leakage.
    For each entity (head or tail) involved in a cartesian product relation in the test set,
    all corresponding triplets in the training set are moved to the test set.
    
    Parameters
    ----------
    kg1 : KnowledgeGraph
        Training set knowledge graph to be cleaned.
        Will be modified by removing cartesian product triplets.
    kg2 : KnowledgeGraph
        Test set knowledge graph to be augmented.
        Will receive the transferred cartesian product triplets.
    known_cartesian : list
        List of relation indices that represent cartesian product relationships.
        These are relations where if (h,r,t1) exists, then (h,r,t2) likely exists
        for many other tail entities t2 (or vice versa for tail-based cartesian products).
    entity_type : str, optional
        Either "head" or "tail" to specify which entity type to consider for cartesian products.
        Default is "head".
    
    Returns
    -------
    tuple (KnowledgeGraph, KnowledgeGraph)
        A pair of (cleaned_train_kg, augmented_test_kg) where:
        - cleaned_train_kg: Training KG with cartesian triplets removed
        - augmented_test_kg: Test KG with the transferred triplets added
    """
    assert node_type in ["head", "tail"], "entity_type must be either 'head' or 'tail'"
    
    for edge_index in known_cartesian:
        # Find all entities in test set that participate in the cartesian relation
        mask = (kg_second.edge_indices == edge_index)
        if node_type == "head":
            cartesian_node_indices = kg_second.head_indices[mask].view(-1,1)
            # Find matching triplets in training set with same head and relation
            all_indices_to_move = []
            for node_index in cartesian_node_indices:
                mask = (kg_first.head_indices == node_index) & (kg_first.edge_indices == edge_index)
                indices = mask.nonzero().squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                all_indices_to_move.extend(indices.tolist())
        else:  # tail
            cartesian_node_indices = kg_second.tail_indices[mask].view(-1,1)
            # Find matching triplets in training set with same tail and relation
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
            ], dim=1)
            
            # Remove identified triplets from training set
            kg_first = kg_first.remove_triplets(torch.tensor(all_indices_to_move, dtype=torch.long))
            
            # Add transferred triplets to test set while preserving KG structure
            kg_second_dictionnary = {
                "heads": torch.cat([kg_second.head_indices, triplets_to_move[:, 0]]),
                "tails": torch.cat([kg_second.tail_indices, triplets_to_move[:, 2]]),
                "relations": torch.cat([kg_second.edge_indices, triplets_to_move[:, 1]]),
            }
            
            kg_second = kg_second.__class__(
                kg=kg_second_dictionnary,
                node_to_index=kg_second.node_to_index,
                edge_to_index=kg_first.edge_to_index,
                dict_of_heads=kg_second.dict_of_heads,
                dict_of_tails=kg_second.dict_of_tails,
                dict_of_rels=kg_second.dict_of_rels
            )
            
    return kg_first, kg_second