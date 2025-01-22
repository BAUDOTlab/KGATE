import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch import cat
import pandas as pd 
import numpy as np
from pathlib import Path
import tomllib
import random
import logging 
import pickle

log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

# TODO : update, or read from the template
CONFIG_DEFAULTS = {
    "seed":42,
    "kg_csv": "",
    "kg_pkl": "",
    "metadata_csv": "",
    "output_directory": None,
    "verbose": True,
    "run_kg_preprocess": True,
    "run_training": False,
    "run_evaluation": False,
    "preprocessing": {
        "remove_duplicate_triples": True,
        "make_directed": True,
        "make_directed_relations": [],
        "flag_near_duplicate_relations": True,
        "clean_train_set":True
    },
    "model": {
        "encoder": {
            "name": "Default",
            "gnn_layer_number": 1
        },
        "decoder": {
            "name": "TransE",
            "emb_dim": 256,
            "margin": 1
        }
    },
    "sampler": {
        "name":"Positional"
    },
    "optimizer": {
        "name": "Adam",
        "weight_decay": 0.001
    },
    "lr_scheduler": {
        "type": "CosineAnnealingWarmRestarts",
        "T_0": 10,
        "T_mult": 2
    },
    "training": {
        "max_epochs": 2000,
        "patience": 20,
        "train_batch_size": 2048,
        "eval_interval": 10,
        "eval_batch_size": 32
    },
    "evaluation": {
        "made_directed_relations": [],
        "target_relations": [],
        "thresholds": []
    }
}

def parse_config(config_path: str, config_dict: dict) -> dict:
    if config_path != "" and not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    if Path(config_path).exists():
        logging.info(f'Loading parameters from {config_path}')
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    
    # Make the final configuration, using priority orders:
    # 1. Inline configuration (config_dict)
    # 2. Configuration file (config)
    # 3. Default configuration (CONFIG_DEFAULTS)
    # TODO: check for required parameters that aren't set by default
    config = {key: set_config_key(config, CONFIG_DEFAULTS, config_dict, key) for key in CONFIG_DEFAULTS}

    return config

def set_config_key(config, default, inline, key):
    if isinstance(default[key], dict):
        new_value = {}
        for child_key in default[key]:
            new_value.update({child_key: set_config_key(config[key], default[key], inline[key], child_key)})
        return new_value
    
    if key not in inline or inline[key] is None:
        if key not in config or config[key] is None:
            return default[key]
        else:
            return config[key]
    else:
        return inline[key]

def load_knowledge_graph(pickle_filename):
    """Load the knowledge graph from pickle files."""
    logging.info(f'Will not run the preparation step. Using KG stored in: {pickle_filename}')
    with open(pickle_filename, 'rb') as file:
        kg_train = pickle.load(file)
        kg_val = pickle.load(file)
        kg_test = pickle.load(file)
    return kg_train, kg_val, kg_test

def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def my_init_embedding(num_embeddings, emb_dim):
    """Initialize an embedding layer with a normal distribution."""
    embedding = nn.Embedding(num_embeddings, emb_dim)
    nn.init.xavier_uniform_(embedding.weight.data)
    return embedding

def extract_node_type(node_name):
    """Extracts the node type from the node name, based on the string before the first underscore."""
    return node_name.split('_')[0]

def create_hetero_data(kg, mapping_csv):
    df = kg.get_df()
    
    data = HeteroData()
    
    # Dictionary to store mappings
    df_to_hetero_mapping = {}
    hetero_to_df_mapping = {}
    
    df_to_kg_mapping = {} 
    kg_to_df_mapping = {} 
    
    kg_to_hetero_mapping = {}
    hetero_to_kg_mapping = {}

    kg_to_node_type = {} 

    mapping = pd.read_csv(mapping_csv, sep=",", header=1)
    type_mapping = mapping[["type","id"]] # Keep only type and id column

    # 1. Parse node types and IDs
    df = pd.merge(df, type_mapping.add_prefix("from_"), how="left", left_on="from", right_on="from_id")
    df = pd.merge(df, type_mapping.add_prefix("to_"), how="left", left_on="to", right_on="to_id", suffixes=(None, "_to"))
    df.drop([i for i in df.columns if "id" in i],axis=1, inplace=True)

    # 2. Identify all unique node types
    node_types = pd.unique(df[['from_type', 'to_type']].values.ravel('K'))

    # 3. Create mappings for node IDs by type.
    node_dict = {}
    for ntype in node_types:
        # Extract all unique identifiers for each type
        nodes = pd.concat([
            df[df['from_type'] == ntype]['from'],
            df[df['to_type'] == ntype]['to']
        ]).unique()

        node_dict[ntype] = {node: i for i, node in enumerate(nodes)}   

        # Create correspondings for this type of node (DataFrame - HeteroData)
        df_to_hetero_mapping[ntype] = node_dict[ntype]  # Mapping DataFrame -> HeteroData
        hetero_to_df_mapping[ntype] = {v: k for k, v in node_dict[ntype].items()}  # Mapping HeteroData -> DataFrame
        
        # Correspondings between DataFrame and KnowledgeGraph (use kg_train.ent2ix)
        df_to_kg_mapping[ntype] = {node: kg.ent2ix[node] for node in nodes}  # DataFrame -> KG
        kg_to_df_mapping[ntype] = {v: k for k, v in df_to_kg_mapping[ntype].items()}  # KG -> DataFrame
        
        # Mapping KG -> HeteroData via DataFrame
        kg_to_hetero_mapping[ntype] = {df_to_kg_mapping[ntype][k]: df_to_hetero_mapping[ntype][k] for k in node_dict[ntype].keys()}
        hetero_to_kg_mapping[ntype] = {v: k for k, v in kg_to_hetero_mapping[ntype].items()}  # Inverted (HeteroData -> KG)

        # Add node types associated to each ID of the KG
        for kg_id in df_to_kg_mapping[ntype].values():
            kg_to_node_type[kg_id] = ntype

        # Define the number of nodes for this type in HeteroData
        data[ntype].num_nodes = len(node_dict[ntype])

    # 4. Build edge_index for each relation type
    for rel, group in df.groupby('rel'):
        # Identify source and target node type for this group
        src_types = group['from_type'].unique()
        tgt_types = group['to_type'].unique()
        
        for src_type in src_types:
            for tgt_type in tgt_types:
                subset = group[
                    (group['from_type'] == src_type) &
                    (group['to_type'] == tgt_type)
                ]
                
                if subset.empty:
                    continue  # Pass if there are no edges in this group
                
                # Map node identifiers to HeteroData indices
                src = subset['from'].map(node_dict[src_type]).values
                tgt = subset['to'].map(node_dict[tgt_type]).values
                
                # Create edge_index tensor
                edge_index = torch.tensor(np.array([src, tgt]), dtype=torch.long)

                edge_type = (src_type, rel, tgt_type)
                data[(src_type, rel, tgt_type)].edge_index = edge_index
    
    # Return HeteroData object, mappings and node mappings
    return data, kg_to_hetero_mapping, hetero_to_kg_mapping, df_to_kg_mapping, kg_to_node_type

def compute_triplet_proportions(kg_train, kg_test, kg_val):
    """
    Computes the proportion of triples for each relation in each of the KnowledgeGraphs
    (train, test, val) relative to the total number of triples for that relation.

    Parameters
    ----------
    kg_train: KnowledgeGraph
        The training KnowledgeGraph instance.
    kg_test: KnowledgeGraph
        The test KnowledgeGraph instance.
    kg_val: KnowledgeGraph
        The validation KnowledgeGraph instance.

    Returns
    -------
    dict
        A dictionary where keys are relation identifiers and values are sub-dictionaries
        with the respective proportions of each relation in kg_train, kg_test, and kg_val.
    """
     
    # Concatenate relations from all KGs
    all_relations = torch.cat((kg_train.relations, kg_test.relations, kg_val.relations))

    # Compute the number of triples for all relations
    total_counts = torch.bincount(all_relations)

    # Compute occurences of each relations
    train_counts = torch.bincount(kg_train.relations, minlength=len(total_counts))
    test_counts = torch.bincount(kg_test.relations, minlength=len(total_counts))
    val_counts = torch.bincount(kg_val.relations, minlength=len(total_counts))

    # Compute proportions for each KG
    proportions = {}
    for rel_id in range(len(total_counts)):
        if total_counts[rel_id] > 0:
            proportions[rel_id] = {
                'train': train_counts[rel_id].item() / total_counts[rel_id].item(),
                'test': test_counts[rel_id].item() / total_counts[rel_id].item(),
                'val': val_counts[rel_id].item() / total_counts[rel_id].item()
            }

    return proportions

def concat_kgs(kg_tr, kg_val, kg_te):
    h = cat((kg_tr.head_idx, kg_val.head_idx, kg_te.head_idx))
    t = cat((kg_tr.tail_idx, kg_val.tail_idx, kg_te.tail_idx))
    r = cat((kg_tr.relations, kg_val.relations, kg_te.relations))
    return h, t, r

def count_triplets(kg1, kg2, duplicates, rev_duplicates):
    """
    Parameters
    ----------
    kg1: torchkge.data_structures.KnowledgeGraph
    kg2: torchkge.data_structures.KnowledgeGraph
    duplicates: list
        List returned by torchkge.utils.data_redundancy.duplicates.
    rev_duplicates: list
        List returned by torchkge.utils.data_redundancy.duplicates.

    Returns
    -------
    n_duplicates: int
        Number of triplets in kg2 that have their duplicate triplet
        in kg1
    n_rev_duplicates: int
        Number of triplets in kg2 that have their reverse duplicate
        triplet in kg1.
    """
    n_duplicates = 0
    for r1, r2 in duplicates:
        ht_tr = kg1.get_pairs(r2, type='ht')
        ht_te = kg2.get_pairs(r1, type='ht')

        n_duplicates += len(ht_te.intersection(ht_tr))

        ht_tr = kg1.get_pairs(r1, type='ht')
        ht_te = kg2.get_pairs(r2, type='ht')

        n_duplicates += len(ht_te.intersection(ht_tr))

    n_rev_duplicates = 0
    for r1, r2 in rev_duplicates:
        th_tr = kg1.get_pairs(r2, type='th')
        ht_te = kg2.get_pairs(r1, type='ht')

        n_rev_duplicates += len(ht_te.intersection(th_tr))

        th_tr = kg1.get_pairs(r1, type='th')
        ht_te = kg2.get_pairs(r2, type='ht')

        n_rev_duplicates += len(ht_te.intersection(th_tr))

    return n_duplicates, n_rev_duplicates

