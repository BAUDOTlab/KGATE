from math import ceil
from collections import defaultdict
import pandas as pd
import numpy as np
from itertools import combinations
from torch.utils.data import Dataset
from torch.types import Number
from typing import Self, Dict, Tuple, List, Set
from tqdm import tqdm
import logging
import torchkge
from torch_geometric.data import HeteroData

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

import torch
import torch.nn as nn
from torch import tensor, Tensor, cat

from torchkge.utils.operations import get_dictionaries


class EncoderInput:
    def __init__(self, x_dict: Dict[str, Tensor], edge_index: Dict[str,Tensor], mapping:Dict[str,Tensor]):
        self.x_dict = x_dict
        self.edge_index = edge_index
        self.mapping = mapping

    def __repr__(self):
        x_repr = "\n\t".join([
            f"{node}: {{ [{embedding.size(0)},{embedding.size(1)}] }}" 
            for node, embedding in self.x_dict.items()
            ])
        edge_repr = "\n\t".join([
            f"{edge}: {edge_index}"
            for edge, edge_index in self.edge_index.items()
        ])
        map_repr = "\n\t".join([
            f"{node_type}: {idx}"
            for node_type, idx in self.mapping.items()
        ])

        msg = f"""{self.__class__.__name__} (
    x_dict: {{
        {x_repr}
    }}

    edge_index: {{
        {edge_repr}
    }}

    mapping: {{
        {map_repr}
    }})"""

        return msg

class KnowledgeGraph(Dataset):
    def __init__(self, df: pd.DataFrame | None=None,
                 edgelist: Tensor | None=None,
                 metadata: pd.DataFrame | None=None, 
                 triple_types: List[Tuple[str,str,str]] | None = None,
                 ent2ix: Dict[str, int] | None=None, 
                 rel2ix: Dict[str, int] | None=None,
                 nt2ix: Dict[str, int] | None=None,
                 removed_triples: Tensor | None=None):
        
        if df is None:
            assert edgelist is not None and \
                   ent2ix is not None and \
                   rel2ix is not None and \
                   triple_types is not None and \
                   nt2ix is not None, "If df is not given, `edgelist`, `ent2ix`, `rel2ix` and `nt2ix` must be provided."
            self.n_triples = edgelist.size(1)
        else:
            self.n_triples = len(df)

        if edgelist is not None:
            assert edgelist.size(0) == 4, "The `edgelist` parameter must be a 2D tensor of size [4, num_triples]."
            self.edgelist = edgelist.long()
        else:
            self.edgelist = tensor([], dtype=torch.long)

        if removed_triples is not None:
            assert removed_triples.size(0) == 4,  "The `removed_triples` parameter must be a 2D tensor of size [4, num_triples]."
            self.removed_triples = removed_triples
        else:
            self.removed_triples = tensor([], dtype=torch.long)

        self.triple_types: List[Tuple[str,str,str]] = triple_types or []

        self.ent2ix = ent2ix or get_dictionaries(df, ent=True)
        self.nt2ix: Dict[str,int] = nt2ix or {"Node": 0}
        self.rel2ix = rel2ix or get_dictionaries(df, ent=False)

        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1

        if df is None:
            self.n_nodes = cat([self.head_idx, self.tail_idx]).unique().size(0)
            # The mapping is done on the absolute index of nodes. However, subgraphs don't have all the nodes
            # Thus, we must initialize the tensor at -1 to avoid downstream issue with the node_type 0 being 
            # broadcasted to missing nodes in subgraphs.
            self.node_types = torch.ones(self.n_ent, dtype=torch.long).neg()

            for tri_type in self.triples.unique():
                h_t, t_t = self.triple_types[tri_type][0], self.triple_types[tri_type][2]
                triple_edgelist = self.edgelist[:, self.triples == tri_type]
                self.node_types[triple_edgelist[0]] = self.nt2ix[h_t]
                self.node_types[triple_edgelist[1]] = self.nt2ix[t_t]

        else:
            if metadata is not None:
                assert not set(["type","id"]).isdisjoint(list(metadata.columns)), f"The mapping dataframe must have at least the columns `type` and `id`, but found only {",".join(list(metadata.columns))}"

                mapping_df = pd.merge(df, metadata.add_prefix("from_"), how="left", left_on="from", right_on="from_id")
                mapping_df = pd.merge(mapping_df, metadata.add_prefix("to_"), how="left", left_on="to", right_on="to_id", suffixes=(None, "_to"))
                mapping_df.drop([i for i in mapping_df.columns if "id" in i],axis=1, inplace=True)

                df_node_types = list(set(mapping_df['from_type'].unique()).union(set(mapping_df['to_type'].unique())))
                self.nt2ix = {nt: i for i, nt in enumerate(sorted(df_node_types))}
            else:
                mapping_df = df

            i = 0
            self.n_nodes = self.n_ent
            self.node_types = torch.ones(self.n_ent, dtype=torch.long).neg()

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
                        src = tensor(src).unsqueeze(0).long()
                        tgt = subset["to"].map(self.ent2ix).values
                        tgt = tensor(tgt).unsqueeze(0).long()

                        triplets = torch.cat([
                            src,
                            tgt,
                            tensor(relation).repeat(len(subset)).unsqueeze(0),
                            tensor(i).repeat(len(subset)).unsqueeze(0)
                        ], dim=0)

                        self.edgelist = torch.cat([
                            self.edgelist,
                            triplets
                        ], dim=1)

                        self.node_types[src] = self.nt2ix[src_type]
                        self.node_types[tgt] = self.nt2ix[tgt_type]

                        edge_type = (src_type, rel, tgt_type)
                        self.triple_types.append(edge_type)
                        i+=1
        
        self.nt2glob: Dict[str, Tensor] = {}
        self.glob2loc = torch.ones(self.n_ent, dtype=torch.long).neg()

        for i, node_type in enumerate(self.nt2ix):
            glob_idx = (self.node_types == i).nonzero(as_tuple=True)[0]
            self.nt2glob[node_type] = glob_idx
            self.glob2loc[glob_idx] = torch.arange(glob_idx.size(0))


    def __len__(self):
        return self.n_triples
    
    def __getitem__(self, index) -> Tensor:
        return self.edgelist[:, index]
    
    @property
    def head_idx(self) -> Tensor:
        return self.edgelist[0]
    
    @property
    def tail_idx(self) -> Tensor:
        return self.edgelist[1]
    
    @property
    def relations(self) -> Tensor:
        return self.edgelist[2]

    @property
    def triples(self) -> Tensor:
        return self.edgelist[3]

    @property
    def edge_index(self) -> Tensor:
        return self.edgelist[:2]
    
    # torchkge compatibility
    @property
    def n_facts(self) -> int:
        return self.n_triples

    def get_df(self):
        """
        Returns a Pandas DataFrame with columns ['from', 'to', 'rel'].
        """
        ix2ent = {v: k for k, v in self.ent2ix.items()}
        ix2rel = {v: k for k, v in self.rel2ix.items()}

        df = pd.DataFrame(cat((self.head_idx.view(1, -1),
                            self.tail_idx.view(1, -1),
                            self.relations.view(1, -1))).transpose(0, 1).numpy(),
                       columns=['from', 'to', 'rel'])

        df['from'] = df['from'].apply(lambda x: ix2ent[x])
        df['to'] = df['to'].apply(lambda x: ix2ent[x])
        df['rel'] = df['rel'].apply(lambda x: ix2rel[x])

        return df

    def split_kg(self, shares: Tuple[float,float,float]=(0.8,0.1,0.1), 
                 sizes: Tuple[int, int, int] | None=None) -> Tuple[Self, Self, Self]:
        if sizes is not None:
            assert sum(sizes) == self.n_triples, "The sum of provided sizes must match the number of triples."
            
            mask_tr = cat([tensor([1] * sizes[1]),
                           tensor([0 * (sizes[1] + sizes[2])])
            ])
            mask_val = cat([
                tensor([0] * sizes[0]),
                tensor([1] * sizes[1]),
                tensor([0] * sizes[2])
            ])
            mask_te = ~(mask_tr | mask_val)
        else:
            assert sum(shares) == 1, "The sum of provided shares must be equal to 1."
            mask_tr, mask_val, mask_te = self.get_mask(shares)
            
        return (
            self.__class__(
                edgelist = self.edgelist[:, mask_tr], 
                triple_types=self.triple_types,
                ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                nt2ix=self.nt2ix,
                removed_triples=self.removed_triples
            ),
            self.__class__(
                edgelist = self.edgelist[:, mask_val], 
                triple_types=self.triple_types,
                ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                nt2ix=self.nt2ix
            ),
            self.__class__(
                edgelist = self.edgelist[:, mask_te], 
                triple_types=self.triple_types,
                ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                nt2ix=self.nt2ix
            )
        )
            
    def get_mask(self, shares):
        uniques_r, counts_r = self.relations.unique(return_counts=True)
        uniques_e = np.arange(self.n_ent)

        train_mask = torch.zeros_like(self.relations).bool()
        val_mask = torch.zeros_like(self.relations).bool()
        for i,r in enumerate(uniques_r):
            count = counts_r[i].item()
            rand = torch.randperm(count)

            sub_mask = torch.eq(self.relations, r).nonzero(as_tuple=False)[:, 0]

            assert len(sub_mask) == count
            
            train_size = max(1, int(count * shares[0]))
            val_size = min(count - train_size, ceil(count * shares[1]))
            test_size = count - (train_size + val_size)

            assert train_size + val_size + test_size == count

            train_mask[sub_mask[rand[:train_size]]] = True
            val_mask[sub_mask[rand[train_size:train_size + val_size]]] = True

        u = cat([self.head_idx[train_mask], self.tail_idx[train_mask]]).unique()
        if len(u) < self.n_ent:
            missing_entities = tensor(list(set(uniques_e.tolist()) - set(u.tolist())),
                                      dtype=torch.long)
            for e in missing_entities:
                sub_mask = ((self.head_idx == e) |
                            (self.tail_idx == e)).nonzero(as_tuple=False)[:, 0]
                count = len(sub_mask)
                rand = torch.randperm(count)

                train_size = max(1, int(count * shares[0]))
                val_size = min(count - train_size, ceil(count * shares[1]))

                train_mask[sub_mask[rand[:train_size]]] = True
                val_mask[sub_mask[rand[:train_size]]] = False
        
        assert not (train_mask & val_mask).any().item()
        return train_mask, val_mask, ~(train_mask | val_mask)

    def keep_triples(self, indices_to_keep: List[int] | torch.Tensor) -> Self:
        """
        Keeps only the specified triples in the knowledge graph and returns a new
        KnowledgeGraph instance with these triples. Updates the dictionnary of facts.

        Parameters
        ----------
        indices_to_keep : list or torch.Tensor
            Indices of triples to keep in the knowledge graph.

        Returns
        -------
        KnowledgeGraph
            A new instance of KnowledgeGraph with only the specified triples.
        """
        # Create masks for indices to keep
        mask = torch.zeros(self.n_triples, dtype=torch.bool)
        mask[indices_to_keep] = True
        removed_triples = cat([self.removed_triples, self.edgelist[:, ~mask]], dim=1)

        # Create a new KnowledgeGraph instance
        return self.__class__(
            edgelist=self.edgelist[:, mask],
            triple_types=self.triple_types,
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix,
            nt2ix=self.nt2ix,
            removed_triples = removed_triples
        )

    def remove_triples(self, indices_to_remove: List[int] | torch.Tensor) -> Self:
        """
        Removes specified triples from the knowledge graph and returns a new
        KnowledgeGraph instance without these triples.

        Parameters
        ----------
        indices_to_remove : list or torch.Tensor
            Indices of triples to remove from the knowledge graph.

        Returns
        -------
        KnowledgeGraph
            A new instance of KnowledgeGraph without the specified triples.
        """
        # Create masks for indices not to remove
        mask = torch.ones(self.n_triples, dtype=torch.bool)
        mask[indices_to_remove] = False
        removed_triples = cat([self.removed_triples, self.edgelist[:, ~mask]], dim=1)

        return self.__class__(
            edgelist=self.edgelist[:, mask],
            triple_types=self.triple_types,
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix,
            nt2ix=self.nt2ix,
            removed_triples=removed_triples
        )
    
    def add_triples(self, new_triples: torch.Tensor) -> Self:
        """
        Add new triples to the Knowledge Graph

        Parameters
        ----------
        new_triples : torch.Tensor
            Tensor of shape (4, n) where each column represent a triple (head_idx, tail_idx, rel_idx, triple_type).

        Returns
        -------
        KnowledgeGraph
            A new instance of KnowledgeGraph with the updated triples.
        """
        assert new_triples.dim() == 2 and new_triples.size(0) == 4, "new_triples must have shape [4, n]"

        # Check that entities and relations exist in ent2ix and rel2ix
        max_ent_idx = max(new_triples[0].max().item(), new_triples[1].max().item())
        max_triple_idx = new_triples[3].max().item()

        if max_ent_idx >= self.n_ent:
            raise ValueError(f"The maximum entity index ({max_ent_idx}) is superior to the number of entities ({self.n_ent}).")
        if max_triple_idx >= len(self.triple_types):
            raise ValueError(f"The maximum triple index ({max_triple_idx}) is superior to the number of relations ({len(self.triple_types)}).")

        # Concatenate new triples to existing ones
        updated_edgelist = cat([self.edgelist, new_triples], dim=1)
        # Update dict_of_heads, dict_of_tails, dict_of_rels
        # for h, t, r in new_triples.tolist():
        #     self.dict_of_heads[(t, r)].add(h)
        #     self.dict_of_tails[(h, r)].add(t)
        #     self.dict_of_rels[(h, t)].add(r)

        # Create a new instance of the class with updated triples
        return self.__class__(
            edgelist=updated_edgelist,
            triple_types=self.triple_types,
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix,
            nt2ix=self.nt2ix,
            removed_triples=self.removed_triples
        )

    def add_inverse_relations(self, undirected_relations: List[int]) -> Tuple[Self, List[int]]:
        """
        Adds inverse triples for the specified undirected relations in the knowledge graph.
        Updates head_idx, tail_idx, relations with the inverse triples, and updates the dictionaries to include
        both original and inverse facts in all directions.

        Parameters
        ----------
        undirected_relations: list
            List of undirected relations for which inverse triples should be added.

        Returns
        -------
        KnowledgeGraph, list
            The updated KnowledgeGraph with the dictionaries and tensors modified,
            and a list of pairs (old relation ID, new inverse relation ID).
        """

        ix2rel = {v: k for k, v in self.rel2ix.items()}

        reverse_list = []

        # New triples lists
        tmp_edgelist = [self.edgelist]
        tmp_removed = [self.removed_triples]

        for relation_id in undirected_relations:
            inverse_relation = f"{ix2rel[relation_id]}_inv"

            # Check if the inverse relation already exists in the graph
            if relation_id not in self.rel2ix.values():
                logging.info(f"Relation {relation_id} not found in knowledge graph. Skipping...")
                continue

            relation_triples = self.edgelist[:, self.edgelist[2] == relation_id]
            triples_id = relation_triples[3].unique()
            # Create a new ID for the inverse relation
            inverse_relation_id = len(self.rel2ix)
            
            self.rel2ix[inverse_relation] = inverse_relation_id
            for triple_id in triples_id:
                orig_triple = self.triple_types[triple_id]
                inverse_triple_id = len(self.triple_types)

                self.triple_types.append((orig_triple[2], inverse_relation, orig_triple[0]))
                
                mask = (self.edgelist[3] == triple_id)
                subset = self.edgelist[:, mask]

                new_triple = cat([
                        subset[1].unsqueeze(0),
                        subset[0].unsqueeze(0),
                        tensor(inverse_relation_id).repeat(subset.size(1)).unsqueeze(0),
                        tensor(inverse_triple_id).repeat(subset.size(1)).unsqueeze(0)
                    ])
                tmp_edgelist.append(new_triple)
                tmp_removed.append(torch.stack([
                    new_triple[1],
                    new_triple[0],
                    new_triple[2],
                    new_triple[3]
                ]))

            new_edgelist = cat(tmp_edgelist, dim=1)
            new_removed = cat(tmp_removed, dim=1)
            reverse_list.append((relation_id, inverse_relation_id))

        return self.__class__(
                edgelist=new_edgelist,
                triple_types=self.triple_types,
                ent2ix=self.ent2ix,
                rel2ix=self.rel2ix,
                nt2ix=self.nt2ix,
                removed_triples=new_removed
            ), reverse_list

    def remove_duplicate_triples(self) -> Self:
        """
        Remove duplicate triples from a knowledge graph for each relation and keep only unique triples.

        This function processes each relation separately, identifies unique triples based on head and tail indices,
        and retains only the unique triples by filtering out duplicates.

        Returns:
        - KnowledgeGraph: A new instance of the KnowledgeGraph containing only unique triples.
        
        The function also updates a dictionary `T` which holds pairs of head and tail indices for each relation
        along with their original indices in the dataset.

        """
        T = {}  # Dictionary to store pairs for each relation
        keep = torch.tensor([], dtype=torch.long)  # Tensor to store indices of triples to keep

        h, t, r = self.head_idx, self.tail_idx, self.relations

        # Process each relation
        for r_ in tqdm(range(self.n_rel)):
            # Create a mask for the current relation
            mask = (r == r_)

            # Extract pairs of head and tail indices for the current relation
            original_indices = torch.arange(h.size(0))[mask]
            pairs = torch.stack((h[mask], t[mask]), dim=1)
            pairs = torch.sort(pairs, dim=1).values
            pairs = torch.cat([pairs, original_indices.unsqueeze(1)], dim=1)

            # Create a dictionary entry for the relation with pairs
            T[r_] = pairs

            # Identify unique triples and their original indices
            unique, idx, counts = torch.unique(pairs[:, :2], dim=0, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
            first_indices = ind_sorted[cum_sum]

            # Retrieve original indices of first unique entries
            adjusted_indices = pairs[first_indices, 2]

            # Accumulate unique indices globally
            keep = torch.cat((keep, adjusted_indices))

            # Logging duplicate information
            if len(pairs) - len(unique) > 0:
                logging.info(f"{len(pairs) - len(unique)} duplicates found. Keeping {len(unique)} unique triplets for relation {r_}")

        # Return a new KnowledgeGraph instance with only unique triples retained
        return self.keep_triples(keep)

    def get_pairs(self, r: int, type:str="ht") -> Set[Tuple[Number, Number]]:
        mask = (self.relations == r)

        if type == "ht":
            return set((i.item(), j.item()) for i, j in cat(
                (self.head_idx[mask].view(-1, 1),
                self.tail_idx[mask].view(-1, 1)), dim=1))
        else:
            assert type == "th"
            return set((j.item(), i.item()) for i, j in cat(
                (self.head_idx[mask].view(-1, 1),
                self.tail_idx[mask].view(-1, 1)), dim=1))
        
    def duplicates(self, theta1:float = 0.8, theta2:float = 0.8, counts:bool = False, reverses: List[int] | None = None) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Return the duplicate and reverse duplicate relations as explained
        in paper by Akrami et al.

        References
        ----------
        * Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
        `Realistic Re-evaluation of Knowledge Graph Completion Methods:
        An Experimental Study. <https://arxiv.org/pdf/2003.08001.pdf>`_
        SIGMOD’20, June 14–19, 2020, Portland, OR, USA

        Parameters
        ----------
        kg_tr: torchkge.data_structures.KnowledgeGraph
            Train set
        kg_val: torchkge.data_structures.KnowledgeGraph
            Validation set
        kg_te: torchkge.data_structures.KnowledgeGraph
            Test set
        theta1: float
            First threshold (see paper).
        theta2: float
            Second threshold (see paper).
        counts: bool
            Should the triplets involving (reverse) duplicate relations be
            counted in all sets.
        reverses: list
            List of known reverse relations.

        Returns
        -------
        duplicates: list
            List of pairs giving duplicate relations.
        rev_duplicates: list
            List of pairs giving reverse duplicate relations.
        """
        
        if reverses is None:
            reverses = []

        T = dict()
        T_inv = dict()
        lengths = dict()

        h, t, r = self.head_idx, self.tail_idx, self.relations

        for r_ in tqdm(range(self.n_rel)):
            mask = (r == r_)
            lengths[r_] = mask.sum().item()

            pairs = cat((h[mask].view(-1, 1), t[mask].view(-1, 1)), dim=1)

            T[r_] = set([(h_.item(), t_.item()) for h_, t_ in pairs])
            T_inv[r_] = set([(t_.item(), h_.item()) for h_, t_ in pairs])

        logging.info("Finding duplicate relations")

        duplicates: List[Tuple[int, int]] = []
        rev_duplicates: List[Tuple[int, int]] = []

        iter_ = list(combinations(range(self.n_rel), 2))

        for r1, r2 in tqdm(iter_):
            a = len(T[r1].intersection(T[r2])) / lengths[r1]
            b = len(T[r1].intersection(T[r2])) / lengths[r2]

            if a > theta1 and b > theta2:
                duplicates.append((r1, r2))

            if (r1, r2) not in reverses:
                a = len(T[r1].intersection(T_inv[r2])) / lengths[r1]
                b = len(T[r1].intersection(T_inv[r2])) / lengths[r2]

                if a > theta1 and b > theta2:
                    rev_duplicates.append((r1, r2))

        logging.info("Duplicate relations: {}".format(len(duplicates)))
        logging.info("Reverse duplicate relations: "
                "{}\n".format(len(rev_duplicates)))

        return duplicates, rev_duplicates

    def cartesian_product_relations(self, theta: float=0.8) -> List[int]:
        """Return the cartesian product relations as explained in paper by
        Akrami et al.

        References
        ----------
        * Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
        `Realistic Re-evaluation of Knowledge Graph Completion Methods: An
        Experimental Study. <https://arxiv.org/pdf/2003.08001.pdf>`_
        SIGMOD’20, June 14–19, 2020, Portland, OR, USA

        Parameters
        ----------
        kg: torchkge.data_structures.KnowledgeGraph
        theta: float
            Threshold used to compute the cartesian product relations.

        Returns
        -------
        selected_relations: list
            List of relations index that are cartesian product relations
            (see paper for details).

        """
        selected_relations = []

        h, t, r = self.head_idx, self.tail_idx, self.relations

        S = dict()
        O = dict()
        lengths = dict()

        for r_ in tqdm(range(self.n_rel)):
            mask = (r == r_)
            lengths[r_] = mask.sum().item()

            S[r_] = set(h_.item() for h_ in h[mask])
            O[r_] = set(t_.item() for t_ in t[mask])

            if lengths[r_] / (len(S[r_]) * len(O[r_])) > theta:
                selected_relations.append(r_)

        return selected_relations

    def get_encoder_input(self, data: Tensor, node_embedding: nn.ParameterList) -> EncoderInput:
        assert data.device == node_embedding[0].device
        device = data.device

        edge_types = data[3].unique()
        node_ids: Dict[str, Tensor] = defaultdict(Tensor)

        edge_indices = {}
        x_dict = {}

        for triple_id in edge_types:
            edge_type = self.triple_types[triple_id]
            h_type, _, t_type = edge_type

            mask: Tensor = data[3] == triple_id
            # Apparently, PyG convolutional layers crash if the edge_index has less than 3 elements.
            # if mask.sum() < 10:
            #     continue
            triples = data[:, mask]

            src = triples[0]
            tgt = triples[1]

            node_ids[h_type] = torch.cat([node_ids[h_type].to(device), src]).long().unique()
            node_ids[t_type] = torch.cat([node_ids[t_type].to(device), tgt]).long().unique()

            h_sorted_ids, h_sort_idx = torch.sort(node_ids[h_type])
            h_list = h_sort_idx[torch.searchsorted(h_sorted_ids, src)]
            t_sorted_ids, t_sort_idx = torch.sort(node_ids[t_type])
            t_list = t_sort_idx[torch.searchsorted(t_sorted_ids, tgt)]

            edge_index = torch.stack([
                h_list,
                t_list
            ], dim=0)

            edge_indices[edge_type] = edge_index.to(device)
        
        self.glob2loc = self.glob2loc.to(device)
        for ntype, idx in node_ids.items():
            loc_idx = self.glob2loc[idx]
            x_dict[ntype] = node_embedding[self.nt2ix[ntype]][loc_idx] #torch.index_select(node_embedding.weight.data, 0, idx)
            
            # We add self-loops to each nodes, to make sure they are their own neighbors.
            edge_type = (ntype, "self", ntype)
            self_loops = torch.arange(idx.size(0), device=device)
            edge_index_self = torch.stack([self_loops, self_loops], dim=0)
            edge_indices[edge_type] = edge_index_self

        return EncoderInput(x_dict, edge_indices, node_ids)

    def flatten_embeddings(self, node_embeddings: nn.ParameterList) -> Tensor:
        embeddings: torch.Tensor = torch.zeros((self.n_ent, node_embeddings[0].size(1)), device=node_embeddings[0].device, dtype=torch.float)

        for nt_idx in self.nt2ix.values():
            mask = (self.node_types == nt_idx)
            embeddings[mask] = node_embeddings[nt_idx][self.glob2loc[mask]]
        
        return embeddings

    @staticmethod
    def from_hetero_data(hetero_data: HeteroData):
        pass

    @staticmethod
    def from_torchkge(kg: torchkge.KnowledgeGraph, metadata: pd.DataFrame | None = None) -> Self:
        """Create a new KGATE Knowledge Graph instance from the torchKGE format.
        
        Parameters
        ----------
        kg : torchKGE.KnowledgeGraph
            The knowledge graph as a torchKGE KnowledgeGraph object.
        metadata : pd.DataFrame
            The metadata of the knowledge graph, with at least the columns "id" and "type".

        Returns
        -------
        KnowledgeGraph
            The knowledge graph as a KGATE KnowledgeGraph object.
        """
        if metadata is None:
            edgelist = torch.stack([kg.head_idx, kg.tail_idx, kg.relations, tensor(0).repeat(kg.n_facts)], dim=0).long()
            nt2ix = {"Node":0}
            triple_types = [("Node", rel, "Node") for rel in kg.rel2ix]

            return KnowledgeGraph(edgelist=edgelist, triple_types=triple_types, ent2ix=kg.ent2ix, rel2ix=kg.rel2ix, nt2ix=nt2ix)
        else:
            return KnowledgeGraph(df=kg.get_df(), metadata=metadata, ent2ix=kg.ent2ix, rel2ix=kg.rel2ix)