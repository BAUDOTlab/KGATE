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
import os
from importlib.resources import open_binary
import matplotlib.pyplot as plt

from typing import List, Tuple
from .data_structures import KGATEGraph

log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

def parse_config(config_path: str, config_dict: dict) -> dict:
    if config_path != "" and not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open_binary("kgate", "config_template.toml") as f:
        default_config = tomllib.load(f)

    config = {}

    if config_path != "":
        logging.info(f'Loading parameters from {config_path}')
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    
    # Make the final configuration, using priority orders:
    # 1. Inline configuration (config_dict)
    # 2. Configuration file (config)
    # 3. Default configuration (default_config)
    # If a default value is None, consider it required and not defaultable
    config = {key: set_config_key(key, default_config, config, config_dict) for key in default_config}

    return config

def set_config_key(key: str, default: dict, config: dict | None = None, inline: dict | None = None) -> str | int | list | dict:
    if inline is not None and key in inline:
        inline_value = inline[key]
    else:
        inline_value = None

    if config is not None and key in config:
        config_value = config[key]
    else: 
        config_value = None

    if isinstance(default[key], dict):
        new_value = {}
        for child_key in default[key]:
            new_value.update({child_key: set_config_key(child_key, default[key], config_value,  inline_value)})
        return new_value
    
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

def load_knowledge_graph(pickle_filename: str):
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

def extract_node_type(node_name: str):
    """Extracts the node type from the node name, based on the string before the first underscore."""
    return node_name.split('_')[0]

def compute_triplet_proportions(kg_train: KGATEGraph, kg_test: KGATEGraph, kg_val: KGATEGraph):
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

def concat_kgs(kg_tr: KGATEGraph, kg_val: KGATEGraph, kg_te: KGATEGraph):
    h = cat((kg_tr.head_idx, kg_val.head_idx, kg_te.head_idx))
    t = cat((kg_tr.tail_idx, kg_val.tail_idx, kg_te.tail_idx))
    r = cat((kg_tr.relations, kg_val.relations, kg_te.relations))
    return h, t, r

def count_triplets(kg1: KGATEGraph, kg2: KGATEGraph, duplicates: List[Tuple[int, int]], rev_duplicates: List[Tuple[int, int]]):
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

def find_best_model(dir: Path):
    try:
        best = max(
            (f for f in os.listdir(dir) if f.startswith('best_model_checkpoint_val_metrics=') and f.endswith('.pt')),
            key=lambda f: float(f.split('val_metrics=')[1].rstrip('.pt')),
            default=None
        )
        return best
    except ValueError:
        return False
    
def init_embedding(num_embeddings: int, emb_dim: int, device:str="cpu"):
    embedding = nn.Embedding(num_embeddings, emb_dim, device=device)
    nn.init.xavier_uniform_(embedding.weight.data)
    return embedding

def read_training_metrics(training_metrics_file: Path):
    df = pd.read_csv(training_metrics_file)

    df = df[~df['Epoch'].astype(str).str.contains('CHECKPOINT RESTART')]

    df['Epoch'] = df['Epoch'].astype(int)
    df = df.sort_values(by='Epoch')

    df = df.drop_duplicates(subset=['Epoch'], keep='last')

    return df

def plot_learning_curves(training_metrics_file: Path, outdir: Path):
    outdir = Path(outdir)
    df = read_training_metrics(training_metrics_file)
    df['Training Loss'] = pd.to_numeric(df['Training Loss'], errors='coerce')
    df['Validation Metric'] = pd.to_numeric(df['Validation Metric'], errors='coerce')
    
    plt.figure(figsize=(12, 5))

    # Plot pour la perte d'entraînement
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Training Loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(outdir.joinpath('training_loss_curve.png'))

    # Plot pour le MRR de validation
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Validation Metric'], label=f'Validation Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Metric')
    plt.title('Validation Metric over Epochs')
    plt.legend()
    plt.savefig(outdir.joinpath('validation_metric_curve.png'))

class HeteroMappings():
    def __init__(self, kg: KGATEGraph, metadata:pd.DataFrame | None):
        df = kg.get_df()
        
        self.data = HeteroData()
        
        # Dictionary to store mappings
        self.df_to_hetero = {}
        self.hetero_to_df = {}
        
        self.df_to_kg = {}
        self.kg_to_df = {}
        
        self.kg_to_hetero_tmp = {}
        self.hetero_to_kg = {}

        if metadata is not None:
            # 1. Parse node types and IDs
            df = pd.merge(df, metadata.add_prefix("from_"), how="left", left_on="from", right_on="from_id")
            df = pd.merge(df, metadata.add_prefix("to_"), how="left", left_on="to", right_on="to_id", suffixes=(None, "_to"))
            df.drop([i for i in df.columns if "id" in i],axis=1, inplace=True)

            # 2. Identify all unique node types
            node_types = pd.unique(df[['from_type', 'to_type']].values.ravel('K'))
        else:
            node_types = ["Node"]
        # 3. Create mappings for node IDs by type.
        node_dict = {}
        kg_to_nt = torch.zeros(kg.n_ent, dtype=torch.int64)
        kg_to_het = torch.zeros(kg.n_ent, dtype=torch.int64)
        
        for i, ntype in enumerate(node_types):
            # Extract all unique identifiers for each type
            if metadata is not None:
                nodes = pd.concat([
                    df[df['from_type'] == ntype]['from'],
                    df[df['to_type'] == ntype]['to']
                ]).unique()
            else:
                nodes = pd.concat([df['from'], df['to']]).unique()

            node_dict[ntype] = {node: i for i, node in enumerate(nodes)}   

            # Create correspondings for this type of node (DataFrame - HeteroData)
            self.df_to_hetero[ntype] = node_dict[ntype]  # Mapping DataFrame -> HeteroData
            self.hetero_to_df[ntype] = {v: k for k, v in node_dict[ntype].items()}  # Mapping HeteroData -> DataFrame
            
            # Correspondings between DataFrame and KnowledgeGraph (use kg_train.ent2ix)
            self.df_to_kg[ntype] = {node: kg.ent2ix[node] for node in nodes}  # DataFrame -> KG
            self.kg_to_df[ntype] = {v: k for k, v in self.df_to_kg[ntype].items()}  # KG -> DataFrame
            
            # Mapping KG -> HeteroData via DataFrame
            self.kg_to_hetero_tmp[ntype] = {self.df_to_kg[ntype][k]: self.df_to_hetero[ntype][k] for k in node_dict[ntype].keys()}
            self.hetero_to_kg[ntype] = {v: k for k, v in self.kg_to_hetero_tmp[ntype].items()}  # Inverted (HeteroData -> KG)

            # Add node types associated to each ID of the KG
            for kg_id, het_id in self.kg_to_hetero_tmp[ntype].items():
                kg_to_nt[kg_id] = i
                kg_to_het[kg_id] = het_id

            # Define the number of nodes for this type in HeteroData
            self.data[ntype].num_nodes = len(node_dict[ntype])

        self.kg_to_node_type = kg_to_nt
        self.kg_to_hetero = kg_to_het

        # 4. Build edge_index for each relation type
        for rel, group in df.groupby('rel'):
            # Identify source and target node type for this group
            if metadata is not None:
                src_types = group['from_type'].unique()
                tgt_types = group['to_type'].unique()
            else:
                src_types = ["Node"]
                tgt_types = ["Node"]
            
            for src_type in src_types:
                for tgt_type in tgt_types:
                    if metadata is not None:
                        subset = group[
                            (group['from_type'] == src_type) &
                            (group['to_type'] == tgt_type)
                        ]
                    else:
                        subset = group
                    
                    if subset.empty:
                        continue  # Pass if there are no edges in this group
                    
                    # Map node identifiers to HeteroData indices
                    src = subset['from'].map(node_dict[src_type]).values
                    tgt = subset['to'].map(node_dict[tgt_type]).values
                    
                    # Create edge_index tensor
                    edge_index = torch.tensor(np.array([src, tgt]), dtype=torch.long)

                    edge_type = (src_type, rel, tgt_type)
                    self.data[edge_type].edge_index = edge_index

    def state_dict(self):
        return {
            "df_to_hetero": self.df_to_hetero,
            "hetero_to_df": self.hetero_to_df,
            "df_to_kg": self.df_to_kg,
            "kg_to_df": self.kg_to_df,
            "kg_to_hetero": self.kg_to_hetero,
            "hetero_to_kg": self.hetero_to_kg,
            "kg_to_node_type": self.kg_to_node_type,
            "data": self.data
        }

    def load_state_dict(self, state_dict):
        self.df_to_hetero = state_dict["df_to_hetero"]
        self.hetero_to_df = state_dict["hetero_to_df"]
        self.df_to_kg = state_dict["df_to_kg"]
        self.kg_to_df = state_dict["kg_to_df"]
        self.kg_to_hetero = state_dict["kg_to_hetero"]
        self.hetero_to_kg = state_dict["hetero_to_kg"]
        self.kg_to_node_type = state_dict["kg_to_node_type"]
        self.data = state_dict["data"]