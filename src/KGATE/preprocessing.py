import pandas as pd
import pickle
from pathlib import Path
import logging
from torch import cat

from .utils import set_random_seeds

def prepare_knowledge_graph(config):
    """Prepare and clean the knowledge graph."""
    # Load knowledge graph
    input_file = config["common"]['input_csv']
    kg_df = pd.read_csv(input_file, sep="\t", usecols=["from", "to", "rel"]) # QUESTION : \t or , ?

    if config.get("clean_kg", {}).get("smaller_kg", False):
        kg_df = kg_df[kg_df['rel'].isin(config["clean_kg"]['keep_relations'])]

    kg = my_knowledge_graph.KnowledgeGraph(df=kg_df)

    # Clean and process knowledge graph
    kg_train, kg_val, kg_test = clean_knowledge_graph(kg, config)

    # Save results
    save_knowledge_graph(config, kg_train, kg_val, kg_test)

    return kg_train, kg_val, kg_test

def save_knowledge_graph(config, kg_train, kg_val, kg_test):
    """Save the knowledge graph to files."""
    if config["kg_pkl"] == "":
        pickle_filename = Path(config["output_directory"], "kg.pkl")
    else:
        pickle_filename = config["kg_pkl"]

    with open(pickle_filename, "wb") as file:
        pickle.dump(kg_train, file)
        pickle.dump(kg_val, file)
        pickle.dump(kg_test, file)

def load_knowledge_graph(config):
    """Load the knowledge graph from pickle files."""
    pickle_filename = config['kg_pkl']
    with open(pickle_filename, 'rb') as file:
        kg_train = pickle.load(file)
        kg_val = pickle.load(file)
        kg_test = pickle.load(file)
    return kg_train, kg_val, kg_test

def clean_knowledge_graph(kg, config):
    """Clean and prepare the knowledge graph according to the configuration."""

    set_random_seeds(config["seed"])

    id_to_rel_name = {v: k for k, v in kg.rel2ix.items()}

    if config["preprocessing"]['remove_duplicates_triples']:
        logging.info("Removing duplicated triples...")
        kg = my_data_redundancy.remove_duplicates_triples(kg)

    duplicated_relations_list = []

    if config['preprocessing']['flag_near_duplicate_relations']:
        logging.info("Checking for near duplicates relations...")
        theta1 = config['preprocessing']['params']['theta1']
        theta2 = config['preprocessing']['params']['theta2']
        duplicates_relations, rev_duplicates_relations = my_data_redundancy.duplicates(kg, theta1=theta1, theta2=theta2)
        if duplicates_relations:
            logging.info(f'Adding {len(duplicates_relations)} synonymous relations ({[id_to_rel_name[rel] for rel in duplicates_relations]}) to the list of known duplicated relations.')
            duplicated_relations_list.extend(duplicates_relations)
        if rev_duplicates_relations:
            logging.info(f'Adding {len(rev_duplicates_relations)} anti-synonymous relations ({[id_to_rel_name[rel] for rel in rev_duplicates_relations]}) to the list of known duplicated relations.')
            duplicated_relations_list.extend(rev_duplicates_relations)
    
    # if config['preprocessing']["permute_entities"]:
    #     to_permute_relation_names = config['clean_kg']["permute_kg_params"]
    #     if len(to_permute_relation_names) > 1:
    #         logging.info(f'Making permutations for relations {", ".join([rel for rel in to_permute_relation_names])}...')
    #     for rel in to_permute_relation_names:
    #         logging.info(f'Making permutations for relation {rel} with id {kg.rel2ix[rel]}.')
    #         kg = my_data_redundancy.permute_tails(kg, kg.rel2ix[rel])

    if config['preprocessing']['make_directed']:
        undirected_relations_names = config['preprocessing']['make_directed_relations']
        relation_names = ", ".join([rel for rel in undirected_relations_names])
        logging.info(f'Adding reverse triplets for relations {relation_names}...')
        kg, undirected_relations_list = my_data_redundancy.add_inverse_relations(kg, [kg.rel2ix[key] for key in undirected_relations_names])
            
        if config['preprocessing']['flag_near_duplicate_relations']:
            logging.info(f'Adding created reverses {[rel for rel in undirected_relations_names]} to the list of known duplicated relations.')
            duplicated_relations_list.extend(undirected_relations_list)

    logging.info("Splitting the dataset into train, validation and test sets...")
    kg_train, kg_val, kg_test = kg.split_kg(validation=True)

    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    if config['preprocessing']['clean_train_set']:
        logging.info("Cleaning the train set to avoid data leakage...")
        logging.info("Step 1: with respect to validation set.")
        kg_train = my_data_redundancy.clean_datasets(kg_train, kg_val, known_reverses=duplicated_relations_list)
        logging.info("Step 2: with respect to test set.")
        kg_train = my_data_redundancy.clean_datasets(kg_train, kg_test, known_reverses=duplicated_relations_list)

    kg_train_ok, _ = verify_entity_coverage(kg_train, kg)
    if not kg_train_ok:
        logging.info("Entity coverage verification failed...")
    else:
        logging.info("Entity coverage verified successfully.")

    new_train, new_val, new_test = my_data_redundancy.ensure_entity_coverage(kg_train, kg_val, kg_test)


    kg_train_ok, missing_entities = verify_entity_coverage(new_train, kg)
    if not kg_train_ok:
        logging.info(f"Entity coverage verification failed. {len(missing_entities)} entities are missing.")
        logging.info(f"Missing entities: {missing_entities}")
        raise ValueError('One or more entities are not covered in the training set after ensuring entity coverage...')
    else:
        logging.info("Entity coverage verified successfully.")

    logging.info("Computing triplet proportions...")
    logging.info(my_data_redundancy.compute_triplet_proportions(kg_train, kg_test, kg_val))

    return new_train, new_val, new_test

def verify_entity_coverage(train_kg, full_kg):
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
