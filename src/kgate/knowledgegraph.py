from collections import defaultdict
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Self, Dict, Tuple

import torch
from torch import tensor, Tensor

from torchkge.utils.operations import get_dictionaries

from .utils import init_embedding

class KnowledgeGraph(Dataset):
    def __init__(self, df: pd.DataFrame, kg:Self | None=None,
                 metadata: pd.DataFrame | None=None,
                 ent2ix: Dict[str, int] | None=None, rel2ix: Dict[str, int] | None=None):
        self.edgelist = tensor([])
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.relation_names = []
        self.triple_types = []

        #assert not (df is None and kg is None), "One of df or kg must be given"

        self._df = df

        if ent2ix is None:
            self.ent2ix = get_dictionaries(df, ent=True)
        else:
            self.ent2ix = ent2ix

        if rel2ix is None:
            self.rel2ix = get_dictionaries(df, ent=False)
        else:
            self.rel2ix = rel2ix

        self.node_embedding = tensor([]) 

        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1
        self.n_triples = len(df)
        self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
            

        if metadata is not None:
            assert ["type","id"] in metadata.columns, "The mapping dataframe must have at least the columns `type` and `id`."

            # 1. Parse node types and IDs
            mapping_df = pd.merge(df, metadata.add_prefix("from_"), how="left", left_on="from", right_on="from_id")
            mapping_df = pd.merge(mapping_df, metadata.add_prefix("to_"), how="left", left_on="to", right_on="to_id", suffixes=(None, "_to"))
            mapping_df.drop([i for i in mapping_df.columns if "id" in i],axis=1, inplace=True)

            # 2. Identify all unique node types
            # node_types = pd.unique(mapping_df[["from_type", "to_type"]].values.ravel("K"))
            # node_dict = {}

            # for i, ntype in enumerate(node_types):
            #     nodes: np.ndarray = pd.concat([
            #         mapping_df[mapping_df["from_type"] == ntype]["from"],
            #         mapping_df[mapping_df["to_type"] == ntype]["to"]
            #     ]).unique()
            #     node_dict[ntype] = {node: i for i, node in enumerate(nodes)}
        else:
            mapping_df = df

        i = 0

        for rel, group in mapping_df.groupby("rel"):
            relation = self.rel2ix[rel]
            if metadata is not None:
                src_types = group["from_type"].unique()
                tgt_types = group["to_type"].unique()
            else:
                src_types = tgt_types = ["Node"]

            for src_type in src_types:
                for tgt_type in tgt_types:
                    if metadata is not None:
                        subset = group[
                            (group["from_type"] == src_type) &
                            (group["to_type"] == tgt_type)
                        ]
                    else:
                        subset = group

                    # Skip if there are no edges in this group
                    if subset.empty: 
                        continue 

                    src = subset["from"].map(self.ent2ix).values
                    tgt = subset["to"].map(self.ent2ix).values
                    
                    triplets = torch.cat([
                        tensor(src).long(),
                        tensor(tgt).long(),
                        tensor(relation).repeat(len(subset)).unsqueeze(0),
                        tensor(i).repeat(len(subset)).unsqueeze(0)
                    ], dim=0)

                    self.edgelist = torch.cat([
                        self.edgelist,
                        triplets
                    ], dim=1)

                    edge_type = (src_type, rel, tgt_type)
                    self.triple_types.append(edge_type)
                    i+=1




    def __len__(self):
        return self.n_triples
    
    def __getitem__(self, index) -> Tensor:
        return self.edgelist.T[index]
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @property
    def head_idx(self) -> Tensor:
        return self.edgelist[0]
    
    @property
    def tail_idx(self) -> Tensor:
        return self.edgelist[1]
    
    def get_encoder_input(self, data: Tensor) -> EncoderInput:
        edge_types = data[3]
        node_ids = defaultdict(Tensor)

        edge_indices = {}
        x_dict = {}

        for triple_id in edge_types:
            edge_type = self.triple_types[triple_id]
            h_type, r_type, t_type = edge_type

            mask: Tensor = data[3] == triple_id
            triples = data[:, mask]

            src = triples[0]
            tgt = triples[1]

            node_ids[h_type] = torch.cat([node_ids[h_type], src]).unique()
            node_ids[t_type] = torch.cat([node_ids[t_type], tgt]).unique()

            h_list = src.apply_(lambda x: (node_ids[h_type] == x).nonzero(as_tuple=True)[0])
            t_list = tgt.apply_(lambda x: (node_ids[t_type] == x).nonzero(as_tuple=True)[0])
            
            edge_index = torch.stack([
                h_list,
                t_list
            ], dim=0)

            edge_indices[edge_type] = edge_index

        for ntype, idx in node_ids.items():
            x_dict[ntype] = torch.index_select(self.node_embedding, 0, idx)

        return EncoderInput(x_dict, edge_indices, node_ids)

class EncoderInput:
    def __init__(self, x_dict: Dict[str, Tensor], edge_index: Dict[str,Tensor], mapping:Dict[str,Tensor]):
        self.x_dict = x_dict
        self.edge_index = edge_index
        self.mapping = mapping