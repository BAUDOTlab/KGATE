import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch import cat, Tensor
import pandas as pd 
import numpy as np
from pathlib import Path
import tomllib
import tomli_w
import random
import logging 
import pickle
import os
from importlib.resources import open_binary
import matplotlib.pyplot as plt

from typing import List, Tuple, Literal
from .knowledgegraph import KnowledgeGraph

log_level = logging.INFO# if config["common"]["verbose"] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format="%(asctime)s - %(levelname)s - %(message)s" 
)

def parse_config(config_path: str, config_dict: dict) -> dict:
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

    if key in default and isinstance(default[key], dict):
        new_value = {}
        keys = list(default[key].keys())
        if config_value is not None:
            keys += (list(config_value.keys()))
        if inline_value is not None:
            keys += (list(inline_value.keys()))
            
        for child_key in set(keys):
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

def save_config(config:dict, filename:Path | None = None):
    """Saves the Architect configuration as a TOML file.
    
    If no filename is given, it will be created as config.output_directory/kgate_config.toml.
    
    Parameters
    ----------
    config: dict
        The parsed config as a python dictionnary.
    filename: Path (optional)
        The complete path to the configuration file. If one already exists, it will be overwritten."""
        
    config_path = filename or Path(config["output_directory"]).joinpath("kgate_config.toml")

    with open(config_path, "wb") as f:
        tomli_w.dump(config,f)

def load_knowledge_graph(pickle_filename: Path) -> Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph]:
    """Load the knowledge graph from pickle files."""
    logging.info(f"Will not run the preparation step. Using KG stored in: {pickle_filename}")
    with open(pickle_filename, "rb") as file:
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
    return node_name.split("_")[0]

def compute_triplet_proportions(kg_train: KnowledgeGraph, kg_test: KnowledgeGraph, kg_val: KnowledgeGraph):
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
    all_relations = torch.cat((kg_train.triples, kg_test.triples, kg_val.triples))

    # Compute the number of triples for all relations
    total_counts = torch.bincount(all_relations)

    # Compute occurences of each relations
    train_counts = torch.bincount(kg_train.triples, minlength=len(total_counts))
    test_counts = torch.bincount(kg_test.triples, minlength=len(total_counts))
    val_counts = torch.bincount(kg_val.triples, minlength=len(total_counts))

    # Compute proportions for each KG
    proportions = {}
    for rel_id in range(len(total_counts)):
        if total_counts[rel_id] > 0:
            proportions[rel_id] = {
                "train": train_counts[rel_id].item() / total_counts[rel_id].item(),
                "test": test_counts[rel_id].item() / total_counts[rel_id].item(),
                "val": val_counts[rel_id].item() / total_counts[rel_id].item()
            }

    return proportions

def concat_kgs(kg_tr: KnowledgeGraph, kg_val: KnowledgeGraph, kg_te: KnowledgeGraph):
    h = cat((kg_tr.head_idx, kg_val.head_idx, kg_te.head_idx))
    t = cat((kg_tr.tail_idx, kg_val.tail_idx, kg_te.tail_idx))
    r = cat((kg_tr.relations, kg_val.relations, kg_te.relations))
    return h, t, r

def count_triplets(kg1: KnowledgeGraph, kg2: KnowledgeGraph, duplicates: List[Tuple[int, int]], rev_duplicates: List[Tuple[int, int]]):
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
        ht_tr = kg1.get_pairs(r2, type="ht")
        ht_te = kg2.get_pairs(r1, type="ht")

        n_duplicates += len(ht_te.intersection(ht_tr))

        ht_tr = kg1.get_pairs(r1, type="ht")
        ht_te = kg2.get_pairs(r2, type="ht")

        n_duplicates += len(ht_te.intersection(ht_tr))

    n_rev_duplicates = 0
    for r1, r2 in rev_duplicates:
        th_tr = kg1.get_pairs(r2, type="th")
        ht_te = kg2.get_pairs(r1, type="ht")

        n_rev_duplicates += len(ht_te.intersection(th_tr))

        th_tr = kg1.get_pairs(r1, type="th")
        ht_te = kg2.get_pairs(r2, type="ht")

        n_rev_duplicates += len(ht_te.intersection(th_tr))

    return n_duplicates, n_rev_duplicates

def find_best_model(dir: Path):
    return max(
        (f for f in os.listdir(dir) if f.startswith("best_model_checkpoint_val_metrics=") and f.endswith(".pt")),
        key=lambda f: float(f.split("val_metrics=")[1].rstrip(".pt")),
        default=None
    )
    
def init_embedding(num_embeddings: int, emb_dim: int, device:str="cpu"):
    embedding = nn.Embedding(num_embeddings, emb_dim, device=device)
    nn.init.xavier_uniform_(embedding.weight.data)
    return embedding

def read_training_metrics(training_metrics_file: Path):
    df = pd.read_csv(training_metrics_file)

    df = df[~df["Epoch"].astype(str).str.contains("CHECKPOINT RESTART")]

    df["Epoch"] = df["Epoch"].astype(int)
    df = df.sort_values(by="Epoch")

    df = df.drop_duplicates(subset=["Epoch"], keep="last")

    return df

def plot_learning_curves(training_metrics_file: Path, outdir: Path, val_metric: str):
    outdir = Path(outdir)
    df = read_training_metrics(training_metrics_file)
    df["Training Loss"] = pd.to_numeric(df["Training Loss"], errors="coerce")
    df[f"Validation {val_metric}"] = pd.to_numeric(df[f"Validation {val_metric}"], errors="coerce")
    
    plt.figure(figsize=(12, 5))

    # Plot pour la perte d"entraînement
    plt.subplot(1, 2, 1)
    plt.plot(df["Epoch"], df["Training Loss"], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig(outdir.joinpath("training_loss_curve.png"))

    # Plot pour le MRR de validation
    plt.subplot(1, 2, 2)
    plt.plot(df["Epoch"], df[f"Validation {val_metric}"], label=f"Validation {val_metric}")
    plt.xlabel("Epoch")
    plt.ylabel(f"Validation {val_metric}")
    plt.title("Validation Metric over Epochs")
    plt.legend()
    plt.savefig(outdir.joinpath("validation_metric_curve.png"))

def filter_scores(scores: Tensor, edgelist: Tensor, missing: Literal["head","tail","rel"], idx_1: Tensor, idx_2: Tensor, true_idx: Tensor | None):
    """
    Filter a score tensor to ignore the score attributed to true entity or relation except the ones that are being predicted.
    Parameters
    ----------
    scores: Tensor
        Tensor of shape [batch_size,n] where n is the number of entities of relation, depending on what is filtered.
    edgelist: Tensor
        Tensor of shape [4, n_triples] containing every true triple in the KG.
    missing: "head", "tail" or "rel"
        The part of the triple that is currently being predicted.
    idx_1: Tensor
        Tensor containing the index of the heads (if missing is "rel" or "tails") or tails (if missing is "head") that are part of the triple being predicted.
    idx_2: Tensor
        Tensor containing the index of the tails (if missing is "rel") or the relations (if missing is "head" or "tails") that are part of the triple being predicted.
    true_idx: Tensor, optional
        Tensor containing the index of the entities or relations currently being predicted.
        If omitted, every true index will be filtered out.

    Returns
    -------
    filt_scores: Tensor
        Tensor of shape [batch_size, n] with -Inf values for all true entity/relation index except the ones being predicted.
    """
    b_size = scores.shape[0]
    filt_scores = scores.clone()

    if missing == "rel":
        mask_1 = torch.isin(edgelist[0], idx_1)
        mask_2 = torch.isin(edgelist[1], idx_2)
        m_idx = 2
    else:
        e_idx = 0 if missing == "tail" else 1
        m_idx = 1 - e_idx
        mask_1 = torch.isin(edgelist[e_idx], idx_1)
        mask_2 = torch.isin(edgelist[2], idx_2)

    for i in range(b_size):
        if true_idx is None:
            true_mask = torch.zeros(edgelist.size(1), dtype=torch.bool)
        else:
            true_mask = torch.isin(edgelist[m_idx], true_idx[i])

        true_targets = edgelist[m_idx, 
                                    mask_1 & 
                                    mask_2 & 
                                    ~true_mask
                                ]
        filt_scores[i, true_targets] = - float('Inf')

    return filt_scores

def merge_kg(kg_list: List[KnowledgeGraph], complete_edgelist: bool = False) -> KnowledgeGraph:
    """Merge multiple KnowledgeGraph objects into a unique one.
    
    Parameters
    ----------
    kg_list: List[Tensor]
        The list of all KG to be merged
    complete_edgelist: bool, default to False
        Whether or not the removed_triples tensor should be integrated into the final KG's edgelist.
    
    Returns
    -------
    KnowledgeGraph
        The merged KnowledgeGraph object."""
    first_kg = kg_list[0]
    assert all(first_kg.ent2ix == kg.ent2ix for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different ent2ix."
    assert all(first_kg.rel2ix == kg.rel2ix for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different rel2ix."
    assert all(first_kg.nt2ix == kg.nt2ix for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different nt2ix."
    assert all(first_kg.triple_types == kg.triple_types for kg in kg_list[1:]), "Cannot merge KnowledgeGraph with different triple_types."

    new_edgelist = cat([kg.edgelist for kg in kg_list], dim=1)
    if complete_edgelist:
        removed_edgelist = cat([kg.removed_triples for kg in kg_list], dim=1)
        new_edgelist = cat([new_edgelist, removed_edgelist], dim=1)
    
    return first_kg.__class__(
        edgelist=new_edgelist,
        ent2ix=first_kg.ent2ix,
        rel2ix=first_kg.rel2ix,
        nt2ix=first_kg.nt2ix,
        triple_types=first_kg.triple_types
    )

class HeteroMappings():
    def __init__(self, kg: KnowledgeGraph, metadata:pd.DataFrame | None):
        df = kg.get_df()
        
        self.data = HeteroData()
        
        # Dictionary to store mappings
        self.df_to_hetero = {}
        self.hetero_to_df = {}
        
        self.df_to_kg = {}
        self.kg_to_df = {}
        
        self.kg_to_hetero_tmp = {}
        self.hetero_to_kg = []
        self.hetero_node_type = []

        if metadata is not None:
            # 1. Parse node types and IDs
            df = pd.merge(df, metadata.add_prefix("from_"), how="left", left_on="from", right_on="from_id")
            df = pd.merge(df, metadata.add_prefix("to_"), how="left", left_on="to", right_on="to_id", suffixes=(None, "_to"))
            df.drop([i for i in df.columns if "id" in i],axis=1, inplace=True)

            # 2. Identify all unique node types
            node_types = pd.unique(df[["from_type", "to_type"]].values.ravel("K"))
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
                    df[df["from_type"] == ntype]["from"],
                    df[df["to_type"] == ntype]["to"]
                ]).unique()
            else:
                nodes = pd.concat([df["from"], df["to"]], ignore_index=True).unique()

            node_dict[ntype] = {node: i for i, node in enumerate(nodes)}   
            
            # Create correspondings for this type of node (DataFrame - HeteroData)
            self.df_to_hetero[ntype] = node_dict[ntype]  # Mapping DataFrame -> HeteroData
            self.hetero_to_df[ntype] = {v: k for k, v in node_dict[ntype].items()}  # Mapping HeteroData -> DataFrame
            
            # Correspondings between DataFrame and KnowledgeGraph (use kg_train.ent2ix)
            self.df_to_kg[ntype] = {node: kg.ent2ix[node] for node in nodes}  # DataFrame -> KG
            self.kg_to_df[ntype] = {v: k for k, v in self.df_to_kg[ntype].items()}  # KG -> DataFrame
            
            # Mapping KG -> HeteroData via DataFrame
            self.kg_to_hetero_tmp[ntype] = {self.df_to_kg[ntype][k]: self.df_to_hetero[ntype][k] for k in node_dict[ntype].keys()}
            self.hetero_to_kg.append(torch.tensor([k for k in self.kg_to_hetero_tmp[ntype].keys()]))  # Inverted (HeteroData -> KG)
            self.hetero_node_type.append(ntype)
            
            # Add node types associated to each ID of the KG
            for kg_id, het_id in self.kg_to_hetero_tmp[ntype].items():
                kg_to_nt[kg_id] = i
                kg_to_het[kg_id] = het_id

            # Define the number of nodes for this type in HeteroData
            self.data[ntype].num_nodes = len(node_dict[ntype])

        self.kg_to_node_type = kg_to_nt
        self.kg_to_hetero = kg_to_het

        self.relations = []
        # 4. Build edge_index for each relation type
        for rel, group in df.groupby("rel"):
            self.relations.append(rel)
            # Identify source and target node type for this group
            if metadata is not None:
                src_types = group["from_type"].unique()
                tgt_types = group["to_type"].unique()
            else:
                src_types = ["Node"]
                tgt_types = ["Node"]
            
            for src_type in src_types:
                for tgt_type in tgt_types:
                    if metadata is not None:
                        subset = group[
                            (group["from_type"] == src_type) &
                            (group["to_type"] == tgt_type)
                        ]
                    else:
                        subset = group
                    
                    if subset.empty:
                        continue  # Pass if there are no edges in this group
                    
                    # Map node identifiers to HeteroData indices
                    src = subset["from"].map(node_dict[src_type]).values
                    tgt = subset["to"].map(node_dict[tgt_type]).values
                    
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