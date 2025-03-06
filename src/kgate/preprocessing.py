import pandas as pd
import pickle
from pathlib import Path
import logging
import torch
from torch import cat

from .utils import set_random_seeds, compute_triplet_proportions
from .data_structures import KGATEGraph
from torchkge import KnowledgeGraph
from typing import Tuple, List

def prepare_knowledge_graph(config: dict, kg: KnowledgeGraph | None, df: pd.DataFrame | None) -> Tuple[KGATEGraph, KGATEGraph, KGATEGraph]:
    """Prepare and clean the knowledge graph."""
    # Load knowledge graph
    if kg is None and df is None:
        input_file = config["kg_csv"]
        kg_df = pd.read_csv(input_file, sep="\t", usecols=["from", "to", "rel"]) # QUESTION : \t or , ?

        kg = KGATEGraph(df=kg_df)
    else:
        if kg is not None and isinstance(kg, KnowledgeGraph):
            if isinstance(kg, KGATEGraph):
                kg = kg
            else:
                kg_dict = {
                    "heads": kg.head_idx,
                    "tails": kg.tail_idx,
                    "relations": kg.relations
                }
                kg = KGATEGraph(kg=kg_dict)
        elif df is not None:
            kg = KGATEGraph(df = df)
                
    # Clean and process knowledge graph
    kg_train, kg_val, kg_test = clean_knowledge_graph(kg, config)

    # Save results
    save_knowledge_graph(config, kg_train, kg_val, kg_test)

    return kg_train, kg_val, kg_test

def save_knowledge_graph(config: dict, kg_train: KGATEGraph, kg_val: KGATEGraph, kg_test:KGATEGraph):
    """Save the knowledge graph to files."""
    if config["kg_pkl"] == "":
        pickle_filename = Path(config["output_directory"], "kg.pkl")
    else:
        pickle_filename = config["kg_pkl"]

    with open(pickle_filename, "wb") as file:
        pickle.dump(kg_train, file)
        pickle.dump(kg_val, file)
        pickle.dump(kg_test, file)

def load_knowledge_graph(config: dict):
    """Load the knowledge graph from pickle files."""
    pickle_filename = config["kg_pkl"]
    with open(pickle_filename, "rb") as file:
        kg_train = pickle.load(file)
        kg_val = pickle.load(file)
        kg_test = pickle.load(file)
    return kg_train, kg_val, kg_test

def clean_knowledge_graph(kg: KGATEGraph, config: dict) -> Tuple[KGATEGraph, KGATEGraph, KGATEGraph]:
    """Clean and prepare the knowledge graph according to the configuration."""

    set_random_seeds(config["seed"])

    id_to_rel_name = {v: k for k, v in kg.rel2ix.items()}

    if config["preprocessing"]["remove_duplicate_triples"]:
        logging.info("Removing duplicated triples...")
        kg = kg.remove_duplicate_triples()

    duplicated_relations_list = []

    if config["preprocessing"]["flag_near_duplicate_relations"]:
        logging.info("Checking for near duplicates relations...")
        theta1 = config["preprocessing"]["params"]["theta1"]
        theta2 = config["preprocessing"]["params"]["theta2"]
        duplicates_relations, rev_duplicates_relations = kg.duplicates(theta1=theta1, theta2=theta2)
        if duplicates_relations:
            logging.info(f"Adding {len(duplicates_relations)} synonymous relations ({[id_to_rel_name[rel] for rel in duplicates_relations]}) to the list of known duplicated relations.")
            duplicated_relations_list.extend(duplicates_relations)
        if rev_duplicates_relations:
            logging.info(f"Adding {len(rev_duplicates_relations)} anti-synonymous relations ({[id_to_rel_name[rel] for rel in rev_duplicates_relations]}) to the list of known duplicated relations.")
            duplicated_relations_list.extend(rev_duplicates_relations)
    
    # if config["preprocessing"]["permute_entities"]:
    #     to_permute_relation_names = config["clean_kg"]["permute_kg_params"]
    #     if len(to_permute_relation_names) > 1:
    #         logging.info(f"Making permutations for relations {", ".join([rel for rel in to_permute_relation_names])}...")
    #     for rel in to_permute_relation_names:
    #         logging.info(f"Making permutations for relation {rel} with id {kg.rel2ix[rel]}.")
    #         kg = my_data_redundancy.permute_tails(kg, kg.rel2ix[rel])

    if config["preprocessing"]["make_directed"]:
        undirected_relations_names = config["preprocessing"]["make_directed_relations"]
        relation_names = ", ".join([rel for rel in undirected_relations_names])
        logging.info(f"Adding reverse triplets for relations {relation_names}...")
        kg, undirected_relations_list = kg.add_inverse_relations([kg.rel2ix[key] for key in undirected_relations_names])
            
        if config["preprocessing"]["flag_near_duplicate_relations"]:
            logging.info(f"Adding created reverses {[rel for rel in undirected_relations_names]} to the list of known duplicated relations.")
            duplicated_relations_list.extend(undirected_relations_list)

    logging.info("Splitting the dataset into train, validation and test sets...")
    kg_train, kg_val, kg_test = kg.split_kg(validation=True)

    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")  
    else:
        logging.info("Entity coverage verified successfully.")

    if config["preprocessing"]["clean_train_set"]:
        logging.info("Cleaning the train set to avoid data leakage...")
        logging.info("Step 1: with respect to validation set.")
        kg_train = clean_datasets(kg_train, kg_val, known_reverses=duplicated_relations_list)
        logging.info("Step 2: with respect to test set.")
        kg_train = clean_datasets(kg_train, kg_test, known_reverses=duplicated_relations_list)

    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    new_train, new_val, new_test = ensure_entity_coverage(kg_train, kg_val, kg_test)


    kg_train_ok, missing_entities = verify_entity_coverage(new_train, kg)
    if not kg_train_ok:
        logging.info(f"Entity coverage verification failed. {len(missing_entities)} entities are missing.")
        logging.info(f"Missing entities: {missing_entities}")
        raise ValueError("One or more entities are not covered in the training set after ensuring entity coverage...")
    else:
        logging.info("Entity coverage verified successfully.")

    logging.info("Computing triplet proportions...")
    logging.info(compute_triplet_proportions(kg_train, kg_test, kg_val))

    return new_train, new_val, new_test

def verify_entity_coverage(train_kg: KGATEGraph, full_kg: KGATEGraph) -> Tuple[bool, List[str]]:
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
    train_entities = set(cat((train_kg.head_idx, train_kg.tail_idx)).tolist())
    full_entities = set(cat((full_kg.head_idx, full_kg.tail_idx)).tolist())
    
    # Missing entities in the train graph
    missing_entity_ids = full_entities - train_entities
    
    if missing_entity_ids:
        # Invert ent2ix dict to get idx: entity_name
        ix2ent = {v: k for k, v in full_kg.ent2ix.items()}
        
        # Get missing entity names from their indices
        missing_entities = [ix2ent[idx] for idx in missing_entity_ids if idx in ix2ent]
        return False, missing_entities
    else:
        return True, []

def ensure_entity_coverage(kg_train: KGATEGraph, kg_val: KGATEGraph, kg_test:KGATEGraph) -> Tuple[KGATEGraph,KGATEGraph,KGATEGraph]:
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
    train_entities = set(kg_train.ent2ix.values())

    # Obtenir l"ensemble des entités présentes dans kg_train comme head ou tail
    present_heads = set(kg_train.head_idx.tolist())
    present_tails = set(kg_train.tail_idx.tolist())
    present_entities = present_heads.union(present_tails)

    # Identifier les entités manquantes dans kg_train
    missing_entities = train_entities - present_entities

    logging.info(f"Total entities in full kg: {len(train_entities)}")
    logging.info(f"Entities present in kg_train: {len(present_entities)}")
    logging.info(f"Missing entities in kg_train: {len(missing_entities)}")

    def find_and_move_triplets(source_kg, entities):
        nonlocal kg_train, kg_val, kg_test

        # Convert `entities` set to a `Tensor` for compatibility with `torch.isin`
        entities_tensor = torch.tensor(list(entities), dtype=source_kg.head_idx.dtype)

        # Create masks for all triplets where the missing entity is present
        mask_heads = torch.isin(source_kg.head_idx, entities_tensor)
        mask_tails = torch.isin(source_kg.tail_idx, entities_tensor)
        mask = mask_heads | mask_tails

        if mask.any():
            # Extract the indices and corresponding triplets
            indices = torch.nonzero(mask, as_tuple=True)[0]
            triplets = torch.stack([source_kg.head_idx[indices],
                                    source_kg.tail_idx[indices],
                                    source_kg.relations[indices]], dim=1)

            # Add the found triplets to kg_train
            kg_train = kg_train.add_triples(triplets)

            # Remove the triplets from source_kg
            kg_cleaned = source_kg.remove_triples(indices)
            if source_kg == kg_val:
                kg_val = kg_cleaned
            else:
                kg_test = kg_cleaned

            # Update the list of missing entities
            entities_in_triplets = set(triplets[:, 0].tolist() + triplets[:, 1].tolist())
            remaining_entities = entities - set(entities_in_triplets)
            return remaining_entities
        return entities

    # Déplacer les triplets depuis kg_val puis depuis kg_test
    missing_entities = find_and_move_triplets(kg_val, missing_entities)
    if len(missing_entities) > 0:
        missing_entities = find_and_move_triplets(kg_test, missing_entities)

    # Loguer les entités restantes non trouvées
    if len(missing_entities) > 0:
        for entity in missing_entities:
            logging.info(f"Warning: No triplet found involving entity '{entity}' in kg_val or kg_test. Entity remains unconnected in kg_train.")

    return kg_train, kg_val, kg_test


def clean_datasets(kg_train: KGATEGraph, kg2: KGATEGraph, known_reverses: List[Tuple[int, int]]) -> KGATEGraph:
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

    for r1, r2 in known_reverses:

        logging.info(f"Processing relation pair: ({r1}, {r2})")

        # Get (h, t) pairs in kg2 related by r1
        kg2_ht_r1 = kg2.get_pairs(kg2, r1, type="ht")
        # Get indices of (h, t) in kg_train that are related by r2
        indices_to_remove_kg_train = [i for i, (h, t) in enumerate(zip(kg_train.tail_idx, kg_train.head_idx)) if (h.item(), t.item()) in kg2_ht_r1 and kg_train.relations[i].item() == r2]
        indices_to_remove_kg_train.extend([i for i, (h, t) in enumerate(zip(kg_train.head_idx, kg_train.tail_idx)) if (h.item(), t.item()) in kg2_ht_r1 and kg_train.relations[i].item() == r2])
        
        # Remove those (h, t) pairs from kg_train
        kg_train = kg_train.remove_triples(torch.tensor(indices_to_remove_kg_train, dtype=torch.long))

        logging.info(f"Found {len(indices_to_remove_kg_train)} triplets to remove for relation {r2} with reverse {r1}.")

        # Get (h, t) pairs in kg2 related by r2
        kg2_ht_r2 = kg2.get_pairs(kg2, r2, type="ht")
        # Get indices of (h, t) in kg_train that are related by r1
        indices_to_remove_kg_train_reverse = [i for i, (h, t) in enumerate(zip(kg_train.tail_idx, kg_train.head_idx)) if (h.item(), t.item()) in kg2_ht_r2 and kg_train.relations[i].item() == r1]
        indices_to_remove_kg_train_reverse.extend([i for i, (h, t) in enumerate(zip(kg_train.head_idx, kg_train.tail_idx)) if (h.item(), t.item()) in kg2_ht_r2 and kg_train.relations[i].item() == r1])

        # Remove those (h, t) pairs from kg_train
        kg_train = kg_train.remove_triples(torch.tensor(indices_to_remove_kg_train_reverse, dtype=torch.long))

        logging.info(f"Found {len(indices_to_remove_kg_train_reverse)} reverse triplets to remove for relation {r1} with reverse {r2}.")
    
    return kg_train

def clean_cartesians(kg1: KGATEGraph, kg2: KGATEGraph, known_cartesian: List[int], entity_type: str="head") -> Tuple[KGATEGraph,KGATEGraph]:
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
    assert entity_type in ["head", "tail"], "entity_type must be either 'head' or 'tail'"
    
    for r in known_cartesian:
        # Find all entities in test set that participate in the cartesian relation
        mask = (kg2.relations == r)
        if entity_type == "head":
            cartesian_entities = kg2.head_idx[mask].view(-1,1)
            # Find matching triplets in training set with same head and relation
            all_indices_to_move = []
            for entity in cartesian_entities:
                mask = (kg1.head_idx == entity) & (kg1.relations == r)
                indices = mask.nonzero().squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                all_indices_to_move.extend(indices.tolist())
        else:  # tail
            cartesian_entities = kg2.tail_idx[mask].view(-1,1)
            # Find matching triplets in training set with same tail and relation
            all_indices_to_move = []
            for entity in cartesian_entities:
                mask = (kg1.tail_idx == entity) & (kg1.relations == r)
                indices = mask.nonzero().squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                all_indices_to_move.extend(indices.tolist())
            
        if all_indices_to_move:
            # Extract the triplets to be transferred
            triplets_to_move = torch.stack([
                kg1.head_idx[all_indices_to_move],
                kg1.relations[all_indices_to_move],
                kg1.tail_idx[all_indices_to_move]
            ], dim=1)
            
            # Remove identified triplets from training set
            kg1 = kg1.remove_triples(torch.tensor(all_indices_to_move, dtype=torch.long))
            
            # Add transferred triplets to test set while preserving KG structure
            kg2_dict = {
                "heads": torch.cat([kg2.head_idx, triplets_to_move[:, 0]]),
                "tails": torch.cat([kg2.tail_idx, triplets_to_move[:, 2]]),
                "relations": torch.cat([kg2.relations, triplets_to_move[:, 1]]),
            }
            
            kg2 = kg2.__class__(
                kg=kg2_dict,
                ent2ix=kg2.ent2ix,
                rel2ix=kg1.rel2ix,
                dict_of_heads=kg2.dict_of_heads,
                dict_of_tails=kg2.dict_of_tails,
                dict_of_rels=kg2.dict_of_rels
            )
            
    return kg1, kg2