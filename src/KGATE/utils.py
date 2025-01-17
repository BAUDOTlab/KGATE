import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import pandas as pd 
import numpy as np


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
