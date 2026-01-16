"""Class to represent a Knowledge Graph in KGATE. Heavily inspired from TorchKGE's Knowledge Graph class, though expanded to take into account triplets and node types."""

from math import ceil
from collections import defaultdict
from itertools import combinations
from typing import Self, Dict, Tuple, List, Set
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import tensor, Tensor, cat
import torch.nn as nn
from torch.utils.data import Dataset
from torch.types import Number

import torchkge
from torchkge.utils.operations import get_dictionaries
from torch_geometric.data import HeteroData

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class EncoderInput:
    def __init__(self, x_dict: Dict[str, Tensor], edge_list: Dict[str,Tensor], mapping:Dict[str,Tensor]):
        self.x_dict = x_dict
        self.edge_list = edge_list
        self.mapping = mapping

    def __repr__(self):
        x_repr = "\n\t".join([
            f"{node}: {{ [{embedding.size(0)},{embedding.size(1)}] }}" 
            for node, embedding in self.x_dict.items()
            ])
        edge_repr = "\n\t".join([
            f"{edge}: {edge_index}"
            for edge, edge_index in self.edge_list.items()
        ])
        mapping_repr = "\n\t".join([
            f"{node_type}: {index}"
            for node_type, index in self.mapping.items()
        ])

        message = f"""{self.__class__.__name__} (
    x_dict: {{
        {x_repr}
    }}

    edge_index: {{
        {edge_repr}
    }}

    mapping: {{
        {mapping_repr}
    }})"""

        return message

class KnowledgeGraph(Dataset):
    def __init__(self, dataframe: pd.DataFrame | None=None,
                 graphindices: Tensor | None=None,
                 metadata: pd.DataFrame | None=None,
                 triplet_types: List[Tuple[str,str,str]] | None = None,
                 node_to_index: Dict[str, int] | None=None, 
                 edge_to_index: Dict[str, int] | None=None,
                 node_type_to_index: Dict[str, int] | None=None,
                 removed_triplets: Tensor | None=None):
        
        if dataframe is None:
            assert graphindices is not None and \
                   node_to_index is not None and \
                   edge_to_index is not None and \
                   triplet_types is not None and \
                   node_type_to_index is not None, "If df is not given, `edgelist`, `triple_types` and `ent2ix`, `rel2ix` and `nt2ix` must be provided."
            self.triplet_count = graphindices.size(1)
        else:
            self.triplet_count = len(dataframe)

        if graphindices is not None:
            assert graphindices.size(0) == 4, "The `edgelist` parameter must be a 2D tensor of size [4, num_triples]."
            self.graphindices = graphindices.long()
        else:
            self.graphindices = tensor([], dtype=torch.long)

        if removed_triplets is not None and removed_triplets.numel() > 0:
            assert removed_triplets.size(0) == 4,  "The `removed_triples` parameter must be a 2D tensor of size [4, num_triples]."
            self.removed_triplets = removed_triplets
        else:
            self.removed_triplets = tensor([], dtype=torch.long)

        self.triplet_types: List[Tuple[str,str,str]] = triplet_types or []

        self.node_to_index = node_to_index or get_dictionaries(dataframe, ent=True)
        self.node_type_to_index: Dict[str,int] = node_type_to_index or {"Node": 0}
        self.edge_to_index = edge_to_index or get_dictionaries(dataframe, ent=False)

        self.node_count = max(self.node_to_index.values()) + 1
        self.edge_count = max(self.edge_to_index.values()) + 1

        self.metadata = None
        if metadata is not None:
            self.add_metadata(metadata)

        if dataframe is None:
            self.node_count = cat([self.head_indices, self.tail_indices]).unique().size(0)
            # The mapping is done on the absolute index of nodes. However, subgraphs don't have all the nodes
            # Thus, we must initialize the tensor at -1 to avoid downstream issue with the node_type 0 being 
            # broadcasted to missing nodes in subgraphs.
            self.node_types = torch.ones(self.node_count, dtype=torch.long).neg()

            for triplet_type in self.triplets.unique():
                head_node_type, tail_node_type = self.triplet_types[triplet_type][0], self.triplet_types[triplet_type][2]
                triple_edgelist = self.graphindices[:, self.triplets == triplet_type]
                self.node_types[triple_edgelist[0]] = self.node_type_to_index[head_node_type]
                self.node_types[triple_edgelist[1]] = self.node_type_to_index[tail_node_type]

        else:
            if metadata is not None:
                mapping_dataframe = pd.merge(dataframe, metadata.add_prefix("from_"), how="left", left_on="from", right_on="from_id")
                mapping_dataframe = pd.merge(mapping_dataframe, metadata.add_prefix("to_"), how="left", left_on="to", right_on="to_id", suffixes=(None, "_to"))
                mapping_dataframe.drop([i for i in mapping_dataframe.columns if "id" in i],axis=1, inplace=True)

                dataframe_node_types = list(set(mapping_dataframe['from_type'].unique()).union(set(mapping_dataframe['to_type'].unique())))
                self.node_type_to_index = {node_type: i for i, node_type in enumerate(sorted(dataframe_node_types))}
                self._identity = "id"
            else:
                mapping_dataframe = dataframe

            triplet_type_counter = 0
            self.node_count = self.node_count
            self.node_types = torch.ones(self.node_count, dtype=torch.long).neg()

            for edge_name, group in mapping_dataframe.groupby("rel"):
                edge_index = self.edge_to_index[edge_name]
                if metadata is not None:
                    source_types = group["from_type"].unique()
                    target_types = group["to_type"].unique()
                else:
                    source_types = target_types = ["Node"]


                for source_type in source_types:
                    for target_type in target_types:
                        if metadata is not None:
                            subset = group[
                                (group["from_type"] == source_type) &
                                (group["to_type"] == target_type)
                            ]
                        else:
                            subset = group

                        # Skip if there are no edges in this group
                        if subset.empty: 
                            continue 

                        source = subset["from"].map(self.node_to_index).values
                        source = tensor(source).unsqueeze(0).long()
                        target = subset["to"].map(self.node_to_index).values
                        target = tensor(target).unsqueeze(0).long()

                        triplets = torch.cat([
                            source,
                            target,
                            tensor(edge_index).repeat(len(subset)).unsqueeze(0),
                            tensor(triplet_type_counter).repeat(len(subset)).unsqueeze(0)
                        ], dim=0)

                        self.graphindices = torch.cat([
                            self.graphindices,
                            triplets
                        ], dim=1)

                        self.node_types[source] = self.node_type_to_index[source_type]
                        self.node_types[target] = self.node_type_to_index[target_type]

                        edge_type = (source_type, edge_name, target_type)
                        self.triplet_types.append(edge_type)
                        triplet_type_counter+=1
        
        self.node_type_to_global: Dict[str, Tensor] = {}
        self.global_to_local_indices = torch.ones(self.node_count, dtype=torch.long).neg()

        for triplet_type_counter, node_type in enumerate(self.node_type_to_index):
            global_index = (self.node_types == triplet_type_counter).nonzero(as_tuple=True)[0]
            self.node_type_to_global[node_type] = global_index
            self.global_to_local_indices[global_index] = torch.arange(global_index.size(0))


    def __len__(self):
        return self.triplet_count
    
    def __getitem__(self, index) -> Tensor:
        return self.graphindices[:, index]
    
    @property
    def head_indices(self) -> Tensor:
        return self.graphindices[0]
    
    @property
    def tail_indices(self) -> Tensor:
        return self.graphindices[1]
    
    @property
    def edge_indices(self) -> Tensor:
        return self.graphindices[2]

    @property
    def triplets(self) -> Tensor:
        return self.graphindices[3]

    @property
    def edge_list(self) -> Tensor:
        return self.graphindices[:2]
    
    # torchkge compatibility
    @property
    def n_facts(self) -> int:
        return self.triplet_count

    @property
    def identity(self) -> pd.DataFrame:
        """Get the DataFrame containing all the identity of the knowledge graph nodes.
        
        The default identity is the node ID, but different values can be set using the `set_identity` method."""
        if self.metadata is not None:
            return self.metadata[self._identity]
        else:
            return pd.DataFrame([])

    def set_identity(self, new_identity: str):
        """Set the identity of the knowledge graph nodes.
        
        To set an identity, there must be a metadata dataframe given to the knowledge graph, and the identity must correspond to a
        column name of this metadata dataframe. Identities are useful to explore the knowledge graph without relying solely on meaningless
        identifiers but on node names instead, for example.
        
        It is best if all values of an identity are unique in order to identify an individual node, though that is not strictly enforced.
        If that is not the case, functions using identities might have unexpected behavior. To get the dataframe corresponding to the current
        identity, call the `identity` property.
        
        Argument
        --------
            new_identity: str
                The name of the new identity, which must exist in the metadata.
        
        Warning
        -------
            If all values are not unique in the new identity, a warning will be issued."""
        assert self.metadata is not None, "You need to add metadata in order to set an identity."
        assert new_identity in self.metadata, f"The given identity is not a valid metadata name. Valid names are: {self.metadata.columns}."

        if not self.metadata[new_identity].is_unique():
            logging.warn(f"All values are not unique across identity {new_identity}, which may introduce ambiguities. Unexpected output may come from inference.")
        
        self._identity = new_identity

    def add_metadata(self, metadata: pd.DataFrame):
        """Add a new metadata dataframe to the existing one or create it.
        
        If there is already a metadata dataframe associated with the knowledge graph, the new one must have an identical "id" column to be valid.
        If there is no metadata, then the given dataframe must contain at least the columns "id" and "type".

        Argument
        --------
            metadata: pd.DataFrame
                The metadata dataframe to associate to the knowledge graph.
        """
        if self.metadata is None:
            assert not set(["type","id"]).isdisjoint(list(metadata.columns)), f"The metadata dataframe must have at least the columns `type` and `id`, but found only {",".join(list(metadata.columns))}"
            assert metadata.shape[0] == self.node_count, f"The number of rows in the metadata dataframe must match the number of entities in the graph, but found {metadata.shape[0]} rows for {self.node_count} entities."
            self.metadata = metadata
        else:
            assert "id" in metadata.columns and metadata["id"] == self.metadata["id"], "The metadata dataframe must have an id column identical to the existing metadata."
            self.metadata = pd.merge(self.metadata, metadata, on="id")

    def get_dataframe(self):
        """
        Returns a Pandas DataFrame with columns ['from', 'to', 'rel'].
        """
        index_to_node = {value: key for key, value in self.node_to_index.items()}
        index_to_edge = {value: key for key, value in self.edge_to_index.items()}

        dataframe = pd.DataFrame(cat((self.head_indices.view(1, -1),
                            self.tail_indices.view(1, -1),
                            self.edge_indices.view(1, -1))).transpose(0, 1).numpy(),
                       columns=['from', 'to', 'rel'])

        dataframe['from'] = dataframe['from'].apply(lambda x: index_to_node[x])
        dataframe['to'] = dataframe['to'].apply(lambda x: index_to_node[x])
        dataframe['rel'] = dataframe['rel'].apply(lambda x: index_to_edge[x])

        return dataframe

    def split_kg(self, split_proportions: Tuple[float,float,float]=(0.8,0.1,0.1), 
                 sizes: Tuple[int, int, int] | None=None) -> Tuple[Self, Self, Self]:
        if sizes is not None:
            assert sum(sizes) == self.triplet_count, "The sum of provided sizes must match the number of triples."
            
            mask_train = cat([tensor([1] * sizes[1]),
                           tensor([0 * (sizes[1] + sizes[2])])
            ])
            mask_validation = cat([
                tensor([0] * sizes[0]),
                tensor([1] * sizes[1]),
                tensor([0] * sizes[2])
            ])
            mask_test = ~(mask_train | mask_validation)
        else:
            assert sum(split_proportions) == 1, "The sum of provided shares must be equal to 1."
            mask_train, mask_validation, mask_test = self.get_mask(split_proportions)
            
        return (
            self.__class__(
                graphindices = self.graphindices[:, mask_train], 
                triplet_types=self.triplet_types,
                node_to_index=self.node_to_index, edge_to_index=self.edge_to_index,
                node_type_to_index=self.node_type_to_index,
                removed_triples=self.removed_triplets
            ),
            self.__class__(
                graphindices = self.graphindices[:, mask_validation], 
                triplet_types=self.triplet_types,
                node_to_index=self.node_to_index, edge_to_index=self.edge_to_index,
                node_type_to_index=self.node_type_to_index
            ),
            self.__class__(
                graphindices = self.graphindices[:, mask_test], 
                triplet_types=self.triplet_types,
                node_to_index=self.node_to_index, edge_to_index=self.edge_to_index,
                node_type_to_index=self.node_type_to_index
            )
        )
            
    def get_mask(self, split_proportions):
        unique_edges, edge_counts = self.edge_indices.unique(return_counts=True)
        unique_nodes = np.arange(self.node_count)

        train_mask = torch.zeros_like(self.edge_indices).bool()
        validation_mask = torch.zeros_like(self.edge_indices).bool()
        for i, edge in enumerate(unique_edges):
            count = edge_counts[i].item()
            random = torch.randperm(count)

            mask_subset = torch.eq(self.edge_indices, edge).nonzero(as_tuple=False)[:, 0]

            assert len(mask_subset) == count
            
            train_set_size = max(1, int(count * split_proportions[0]))
            validation_set_size = min(count - train_set_size, ceil(count * split_proportions[1]))
            test_set_size = count - (train_set_size + validation_set_size)

            assert train_set_size + validation_set_size + test_set_size == count

            train_mask[mask_subset[random[:train_set_size]]] = True
            validation_mask[mask_subset[random[train_set_size:train_set_size + validation_set_size]]] = True

        unique_nodes = cat([self.head_indices[train_mask], self.tail_indices[train_mask]]).unique()
        if len(unique_nodes) < self.node_count:
            missing_nodes = tensor(list(set(unique_nodes.tolist()) - set(unique_nodes.tolist())),
                                      dtype=torch.long)
            for node in missing_nodes:
                mask_subset = ((self.head_indices == node) |
                            (self.tail_indices == node)).nonzero(as_tuple=False)[:, 0]
                count = len(mask_subset)
                random = torch.randperm(count)

                train_set_size = max(1, int(count * split_proportions[0]))
                validation_set_size = min(count - train_set_size, ceil(count * split_proportions[1]))

                train_mask[mask_subset[random[:train_set_size]]] = True
                validation_mask[mask_subset[random[:train_set_size]]] = False
        
        assert not (train_mask & validation_mask).any().item()
        return train_mask, validation_mask, ~(train_mask | validation_mask)

    def keep_triplets(self, indices_to_keep: List[int] | torch.Tensor) -> Self:
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
        mask = torch.zeros(self.triplet_count, dtype=torch.bool)
        mask[indices_to_keep] = True
        removed_triplets = cat([self.removed_triplets, self.graphindices[:, ~mask]], dim=1)

        # Create a new KnowledgeGraph instance
        return self.__class__(
            graphindices=self.graphindices[:, mask],
            triplet_types=self.triplet_types,
            node_to_index=self.node_to_index,
            edge_to_index=self.edge_to_index,
            node_type_to_index=self.node_type_to_index,
            removed_triplets = removed_triplets
        )

    def remove_triplets(self, indices_to_remove: List[int] | torch.Tensor) -> Self:
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
        mask = torch.ones(self.triplet_count, dtype=torch.bool)
        mask[indices_to_remove] = False
        removed_triplets = cat([self.removed_triplets, self.graphindices[:, ~mask]], dim=1)

        return self.__class__(
            graphindices=self.graphindices[:, mask],
            triplet_types=self.triplet_types,
            node_to_index=self.node_to_index,
            edge_to_index=self.edge_to_index,
            node_type_to_index=self.node_type_to_index,
            removed_triplets=removed_triplets
        )
    
    def add_triplets(self, new_triplets: torch.Tensor) -> Self:
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
        assert new_triplets.dim() == 2 and new_triplets.size(0) == 4, "new_triples must have shape [4, n]"

        # Check that entities and relations exist in ent2ix and rel2ix
        max_node_index = max(new_triplets[0].max().item(), new_triplets[1].max().item())
        max_triplet_index = new_triplets[3].max().item()

        if max_node_index >= self.node_count:
            raise ValueError(f"The maximum entity index ({max_node_index}) is superior to the number of entities ({self.node_count}).")
        if max_triplet_index >= len(self.triplet_types):
            raise ValueError(f"The maximum triple index ({max_triplet_index}) is superior to the number of relations ({len(self.triplet_types)}).")

        # Concatenate new triples to existing ones
        updated_graphindices = cat([self.graphindices, new_triplets], dim=1)
        # Update dict_of_heads, dict_of_tails, dict_of_rels
        # for h, t, r in new_triples.tolist():
        #     self.dict_of_heads[(t, r)].add(h)
        #     self.dict_of_tails[(h, r)].add(t)
        #     self.dict_of_rels[(h, t)].add(r)

        # Create a new instance of the class with updated triples
        return self.__class__(
            graphindices=updated_graphindices,
            triplet_types=self.triplet_types,
            node_to_index=self.node_to_index,
            edge_to_index=self.edge_to_index,
            node_type_to_index=self.node_type_to_index,
            removed_triplets=self.removed_triplets
        )

    def add_reverse_edges(self, undirected_edges: List[int]) -> Tuple[Self, List[int]]:
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

        index_to_edge = {value: key for key, value in self.edge_to_index.items()}

        reverse_list = []

        # New triples lists
        graphindices = [self.graphindices]
        removed_triplets = [self.removed_triplets]

        for edge_index in undirected_edges:
            reverse_edge = f"{index_to_edge[edge_index]}_inv"

            # Check if the inverse relation already exists in the graph
            if edge_index not in self.edge_to_index.values():
                logging.info(f"Relation {edge_index} not found in knowledge graph. Skipping...")
                continue

            edge_triplets = self.graphindices[:, self.graphindices[2] == edge_index]
            triplets_indices = edge_triplets[3].unique()
            # Create a new ID for the inverse relation
            reverse_edge_index = len(self.edge_to_index)
            
            self.edge_to_index[reverse_edge] = reverse_edge_index
            for triplet_index in triplets_indices:
                original_triplet = self.triplet_types[triplet_index]
                reverse_triplet_index = len(self.triplet_types)

                self.triplet_types.append((original_triplet[2], reverse_edge, original_triplet[0]))
                
                mask = (self.graphindices[3] == triplet_index)
                subset = self.graphindices[:, mask]

                new_triplet = cat([
                        subset[1].unsqueeze(0),
                        subset[0].unsqueeze(0),
                        tensor(reverse_edge_index).repeat(subset.size(1)).unsqueeze(0),
                        tensor(reverse_triplet_index).repeat(subset.size(1)).unsqueeze(0)
                    ])
                graphindices.append(new_triplet)
                removed_triplets.append(torch.stack([
                    new_triplet[1],
                    new_triplet[0],
                    new_triplet[2],
                    new_triplet[3]
                ]))

            new_graphindices = cat(graphindices, dim=1)
            new_removed_triplets = cat(removed_triplets, dim=1)
            reverse_list.append((edge_index, reverse_edge_index))

        return self.__class__(
                graphindices=new_graphindices,
                triplet_types=self.triplet_types,
                node_to_index=self.node_to_index,
                edge_to_index=self.edge_to_index,
                node_type_to_index=self.node_type_to_index,
                removed_triplets=new_removed_triplets
            ), reverse_list

    def remove_duplicate_triplets(self) -> Self:
        """
        Remove duplicate triples from a knowledge graph for each relation and keep only unique triples.

        This function processes each relation separately, identifies unique triples based on head and tail indices,
        and retains only the unique triples by filtering out duplicates.

        Returns:
        - KnowledgeGraph: A new instance of the KnowledgeGraph containing only unique triples.
        
        The function also updates a dictionary `T` which holds pairs of head and tail indices for each relation
        along with their original indices in the dataset.

        """
        pair_dictionnary = {}  # Dictionary to store pairs for each relation
        indices_to_keep = torch.tensor([], dtype=torch.long)  # Tensor to store indices of triples to keep

        head_indices, tail_indices, edge_indices = self.head_indices, self.tail_indices, self.edge_indices

        # Process each relation
        for edge_type_index in tqdm(range(self.edge_count)):
            # Create a mask for the current relation
            mask = (edge_indices == edge_type_index)

            # Extract pairs of head and tail indices for the current relation
            original_indices = torch.arange(head_indices.size(0))[mask]
            pairs = torch.stack((head_indices[mask], tail_indices[mask]), dim=1)
            pairs = torch.sort(pairs, dim=1).values
            pairs = torch.cat([pairs, original_indices.unsqueeze(1)], dim=1)

            # Create a dictionary entry for the relation with pairs
            pair_dictionnary[edge_type_index] = pairs

            # Identify unique triples and their original indices
            unique_triplets, indices, counts = torch.unique(pairs[:, :2], dim=0, sorted=True, return_inverse=True, return_counts=True)
            _, sorted_indices = torch.sort(indices, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
            first_indices = sorted_indices[cum_sum]

            # Retrieve original indices of first unique entries
            adjusted_indices = pairs[first_indices, 2]

            # Accumulate unique indices globally
            indices_to_keep = torch.cat((indices_to_keep, adjusted_indices))

            # Logging duplicate information
            if len(pairs) - len(unique_triplets) > 0:
                logging.info(f"{len(pairs) - len(unique_triplets)} duplicates found. Keeping {len(unique_triplets)} unique triplets for relation {edge_type_index}")

        # Return a new KnowledgeGraph instance with only unique triples retained
        return self.keep_triplets(indices_to_keep)

    def get_pairs(self, edge_type_index: int, type:str="ht") -> Set[Tuple[Number, Number]]:
        mask = (self.edge_indices == edge_type_index)

        if type == "ht":
            return set((i.item(), j.item()) for i, j in cat(
                (self.head_indices[mask].view(-1, 1),
                self.tail_indices[mask].view(-1, 1)), dim=1))
        else:
            assert type == "th"
            return set((j.item(), i.item()) for i, j in cat(
                (self.head_indices[mask].view(-1, 1),
                self.tail_indices[mask].view(-1, 1)), dim=1))
        
    def duplicates(self, theta_first_edge_type:float = 0.8, theta_second_edge_type:float = 0.8, counts:bool = False, reverse_edges_list: List[int] | None = None) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
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
        
        if reverse_edges_list is None:
            reverse_edges_list = []

        pair_dictionnary = dict()
        reverse_pair_dictionnary = dict()
        triplets_count_per_edge_type = dict()

        head_indices, tail_indices, edge_indices = self.head_indices, self.tail_indices, self.edge_indices

        for edge_type_index in tqdm(range(self.edge_count)):
            mask = (edge_indices == edge_type_index)
            triplets_count_per_edge_type[edge_type_index] = mask.sum().item()

            pairs = cat((head_indices[mask].view(-1, 1), tail_indices[mask].view(-1, 1)), dim=1)

            pair_dictionnary[edge_type_index] = set([(head_index.item(), tail_index.item()) for head_index, tail_index in pairs])
            reverse_pair_dictionnary[edge_type_index] = set([(tail_index.item(), head_index.item()) for head_index, tail_index in pairs])

        logging.info("Finding duplicate relations")

        duplicates: List[Tuple[int, int]] = []
        reverse_duplicates: List[Tuple[int, int]] = []

        list_of_edge_indices = list(combinations(range(self.edge_count), 2))

        for first_edge_type, second_edge_type in tqdm(list_of_edge_indices):
            duplicate_triplets_proportion_first_edge_type = len(pair_dictionnary[first_edge_type].intersection(pair_dictionnary[second_edge_type])) / triplets_count_per_edge_type[first_edge_type]
            duplicate_triplets_proportion_second_edge_type = len(pair_dictionnary[first_edge_type].intersection(pair_dictionnary[second_edge_type])) / triplets_count_per_edge_type[second_edge_type]

            if duplicate_triplets_proportion_first_edge_type > theta_first_edge_type and duplicate_triplets_proportion_second_edge_type > theta_second_edge_type:
                duplicates.append((first_edge_type, second_edge_type))

            if (first_edge_type, second_edge_type) not in reverse_edges_list:
                duplicate_triplets_proportion_first_edge_type = len(pair_dictionnary[first_edge_type].intersection(reverse_pair_dictionnary[second_edge_type])) / triplets_count_per_edge_type[first_edge_type]
                duplicate_triplets_proportion_second_edge_type = len(pair_dictionnary[first_edge_type].intersection(reverse_pair_dictionnary[second_edge_type])) / triplets_count_per_edge_type[second_edge_type]

                if duplicate_triplets_proportion_first_edge_type > theta_first_edge_type and duplicate_triplets_proportion_second_edge_type > theta_second_edge_type:
                    reverse_duplicates.append((first_edge_type, second_edge_type))

        logging.info("Duplicate relations: {}".format(len(duplicates)))
        logging.info("Reverse duplicate relations: "
                "{}\n".format(len(reverse_duplicates)))

        return duplicates, reverse_duplicates

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

        head_indices, tail_indices, edge_indices = self.head_indices, self.tail_indices, self.edge_indices

        head_nodes = dict()
        tail_nodes = dict()
        triplets_count_per_edge_type = dict()

        for edge_type_index in tqdm(range(self.edge_count)):
            mask = (edge_indices == edge_type_index)
            triplets_count_per_edge_type[edge_type_index] = mask.sum().item()

            head_nodes[edge_type_index] = set(head_index.item() for head_index in head_indices[mask])
            tail_nodes[edge_type_index] = set(tail_index.item() for tail_index in tail_indices[mask])

            if triplets_count_per_edge_type[edge_type_index] / (len(head_nodes[edge_type_index]) * len(tail_nodes[edge_type_index])) > theta:
                selected_relations.append(edge_type_index)

        return selected_relations

    def get_encoder_input(self, data: Tensor, node_embedding: nn.ParameterList) -> EncoderInput:
        assert data.device == node_embedding[0].device
        device = data.device

        triplet_type_indices = data[3].unique()
        node_indices: Dict[str, Tensor] = defaultdict(Tensor)

        pyg_edge_index = {}
        x_dict = {}

        for triplet_index in triplet_type_indices:
            triplet_type = self.triplet_types[triplet_index]
            head_node_type, _, tail_node_type = triplet_type

            mask: Tensor = data[3] == triplet_index
            # Apparently, PyG convolutional layers crash if the edge_index has less than 3 elements.
            # if mask.sum() < 10:
            #     continue
            triplets = data[:, mask]

            source_nodes = triplets[0]
            target_nodes = triplets[1]

            node_indices[head_node_type] = torch.cat([node_indices[head_node_type].to(device), source_nodes]).long().unique()
            node_indices[tail_node_type] = torch.cat([node_indices[tail_node_type].to(device), target_nodes]).long().unique()

            head_sorted_identifiers, head_sorted_indices = torch.sort(node_indices[head_node_type])
            head_list = head_sorted_indices[torch.searchsorted(head_sorted_identifiers, source_nodes)]
            tail_sorted_identifiers, tail_sorted_indices = torch.sort(node_indices[tail_node_type])
            tail_list = tail_sorted_indices[torch.searchsorted(tail_sorted_identifiers, target_nodes)]

            edge_list = torch.stack([
                head_list,
                tail_list
            ], dim=0)

            pyg_edge_index[triplet_type] = edge_list.to(device)
        
        self.global_to_local_indices = self.global_to_local_indices.to(device)
        for node_type, index in node_indices.items():
            local_index = self.global_to_local_indices[index]
            x_dict[node_type] = node_embedding[self.node_type_to_index[node_type]][local_index] #torch.index_select(node_embedding.weight.data, 0, idx)
            
            # We add self-loops to each nodes, to make sure they are their own neighbors.
            triplet_type = (node_type, "self", node_type)
            self_loops = torch.arange(index.size(0), device=device)
            edge_index_self = torch.stack([self_loops, self_loops], dim=0)
            pyg_edge_index[triplet_type] = edge_index_self

        return EncoderInput(x_dict, pyg_edge_index, node_indices)

    def flatten_embeddings(self, node_embeddings: nn.ParameterList) -> Tensor:
        embeddings: torch.Tensor = torch.zeros((self.node_count, node_embeddings[0].size(1)), device=node_embeddings[0].device, dtype=torch.float)

        for node_type_index in self.node_type_to_index.values():
            mask = (self.node_types == node_type_index)
            embeddings[mask] = node_embeddings[node_type_index][self.global_to_local_indices[mask]]
        
        return embeddings

    def clean(self):
        self.triplet_types = [triplet for triplet in self.triplet_types if triplet[1] != "self"]

    @staticmethod
    def from_hetero_data(hetero_data: HeteroData):
        pass

    @staticmethod
    def from_torchkge(torchkge_kg: torchkge.KnowledgeGraph, metadata: pd.DataFrame | None = None) -> Self:
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
            graphindices = torch.stack([torchkge_kg.head_idx, torchkge_kg.tail_idx, torchkge_kg.relations, tensor(0).repeat(torchkge_kg.n_facts)], dim=0).long()
            node_type_to_index = {"Node":0}
            triplet_types = [("Node", edge, "Node") for edge in torchkge_kg.rel2ix]

            return KnowledgeGraph(graphindices=graphindices, triplet_types=triplet_types, node_to_index=torchkge_kg.ent2ix, edge_to_index=torchkge_kg.rel2ix, node_type_to_index=node_type_to_index)
        else:
            return KnowledgeGraph(dataframe=torchkge_kg.get_df(), metadata=metadata, node_to_index=torchkge_kg.ent2ix, edge_to_index=torchkge_kg.rel2ix)