from collections import Counter
from itertools import combinations
import random
from torchkge import KnowledgeGraph
from torchkge.exceptions import SizeMismatchError, WrongArgumentsError, SanityError
import torch
from torch import cat, tensor
import logging
import pandas as pd
from typing import Dict, Tuple, List, Self, Set
from torch.types import Number

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class KGATEGraph(KnowledgeGraph):
    """Class extending on torchKGE KnowledgeGraph class, adding functionalities"""
    def __init__(self,
                df: pd.DataFrame | None=None, 
                kg: Dict[str, torch.Tensor] | None=None, 
                ent2ix: Dict[str | int, int] | None=None, 
                rel2ix: Dict[str | int, int] | None=None,
                dict_of_heads: Dict[Tuple[int, int], List[int]] | None=None, 
                dict_of_tails: Dict[Tuple[int, int], List[int]] | None=None, 
                dict_of_rels: Dict[Tuple[int, int], List[int]] | None=None):
        super().__init__(df, kg, ent2ix, rel2ix, dict_of_heads, dict_of_tails, dict_of_rels)
    
    def split_kg(self, share: float=0.8, sizes: Tuple[int, int] | Tuple[int, int, int] | None=None, validation:bool=False) -> Tuple[Self, Self, Self] | Tuple[Self, Self]:
        """Split the knowledge graph into train and test. If `sizes` is
        provided then it is used to split the samples as explained below. If
        only `share` is provided, the split is done at random but it assures
        to keep at least one fact involving each type of entity and relation
        in the training subset.
        Does not update the dictionary of facts.

        Parameters
        ----------
        share: float
            Percentage to allocate to train set.
        sizes: tuple
            Tuple of ints of length 2 or 3.

            * If len(sizes) == 2, then the first sizes[0] values of the
              knowledge graph will be used as training set and the rest as
              test set.

            * If len(sizes) == 3, then the first sizes[0] values of the
              knowledge graph will be used as training set, the following
              sizes[1] as validation set and the last sizes[2] as testing set.
        validation: bool
            Indicate if a validation set should be produced along with train
            and test sets.

        Returns
        -------
        train_kg: torchkge.data_structures.KnowledgeGraph
        val_kg: torchkge.data_structures.KnowledgeGraph, optional
        test_kg: torchkge.data_structures.KnowledgeGraph

        """
        if sizes is not None:
            try:
                if len(sizes) == 3:
                    try:
                        assert (sizes[0] + sizes[1] + sizes[2] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                elif len(sizes) == 2:
                    try:
                        assert (sizes[0] + sizes[1] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the '
                                                  'number of facts.')
                else:
                    raise SizeMismatchError('Tuple `sizes` should be of '
                                            'length 2 or 3.')
            except AssertionError:
                raise SizeMismatchError('Tuple `sizes` should sum up to the '
                                        'number of facts in the knowledge '
                                        'graph.')
        else:
            assert share < 1

        if (sizes is not None and len(sizes) == 3) or (sizes is None and validation):
            # return training, validation and a testing graphs

            if sizes is None:
                mask_tr, mask_val, mask_te = self.get_mask(share,
                                                           validation=True)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1] + sizes[2])])]).bool()
                mask_val = cat([tensor([0 for _ in range(sizes[0])]),
                                tensor([1 for _ in range(sizes[1])]),
                                tensor([0 for _ in range(sizes[2])])]).bool()
                mask_te = ~(mask_tr | mask_val)

            return (self.__class__(
                        kg={'heads': self.head_idx[mask_tr],
                            'tails': self.tail_idx[mask_tr],
                            'relations': self.relations[mask_tr]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    self.__class__(
                        kg={'heads': self.head_idx[mask_val],
                            'tails': self.tail_idx[mask_val],
                            'relations': self.relations[mask_val]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    self.__class__(
                        kg={'heads': self.head_idx[mask_te],
                            'tails': self.tail_idx[mask_te],
                            'relations': self.relations[mask_te]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels))
        else:
            # return training and testing graphs

            assert (sizes is not None and len(sizes) == 2) or (sizes is None and not validation)
            if sizes is None:
                mask_tr, mask_te = self.get_mask(share, validation=False)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1])])]).bool()
                mask_te = ~mask_tr
            return (self.__class__(
                        kg={'heads': self.head_idx[mask_tr],
                            'tails': self.tail_idx[mask_tr],
                            'relations': self.relations[mask_tr]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    self.__class__(
                        kg={'heads': self.head_idx[mask_te],
                            'tails': self.tail_idx[mask_te],
                            'relations': self.relations[mask_te]},
                        ent2ix=self.ent2ix, rel2ix=self.rel2ix,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels))

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
        KGATEGraph
            A new instance of KnowledgeGraph with only the specified triples.
        """
        # Create masks for indices to keep
        mask = torch.zeros(self.n_facts, dtype=torch.bool)
        mask[indices_to_keep] = True
        
        # Use the mask to filter the triples to keep
        new_heads = self.head_idx[mask]
        new_tails = self.tail_idx[mask]
        new_relations = self.relations[mask]


        # Create a new KGATEGraph instance
        return self.__class__(
            kg={'heads': new_heads, 'tails': new_tails, 'relations': new_relations},
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix
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
        mask = torch.ones(self.n_facts, dtype=torch.bool)
        mask[indices_to_remove] = False
        
        # Use the mask to filter out the triples
        new_heads = self.head_idx[mask]
        new_tails = self.tail_idx[mask]
        new_relations = self.relations[mask]

        return self.__class__(
            kg={'heads': new_heads, 'tails': new_tails, 'relations': new_relations},
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix, 
            dict_of_heads=self.dict_of_heads,
            dict_of_tails=self.dict_of_tails,
            dict_of_rels=self.dict_of_rels
        )
    
    def add_triples(self, new_triples: torch.Tensor) -> Self:
        """
        Add new triples to the Knowledge Graph

        Parameters
        ----------
        new_triples : torch.Tensor
            Tensor of shape (n, 3) where each row represent a triple (head_idx, tail_idx, rel_idx).

        Returns
        -------
        KnowledgeGraph
            A new instance of KnowledgeGraph with the updated triples.
        """
        if not isinstance(new_triples, torch.Tensor):
            raise TypeError("new_triples doit être un torch.Tensor.")
        if new_triples.dim() != 2 or new_triples.size(1) != 3:
            raise ValueError("new_triples doit avoir la forme (n, 3).")

        # Check that entities and relations exist in ent2ix and rel2ix
        max_ent_idx = max(new_triples[:, 0].max().item(), new_triples[:, 1].max().item())
        max_rel_idx = new_triples[:, 2].max().item()

        if max_ent_idx >= self.n_ent:
            raise ValueError(f"L'indice d'entité maximal ({max_ent_idx}) dépasse le nombre d'entités ({self.n_ent}).")
        if max_rel_idx >= self.n_rel:
            raise ValueError(f"L'indice de relation maximal ({max_rel_idx}) dépasse le nombre de relations ({self.n_rel}).")

        # Concatenate new triples to existing ones
        updated_head_idx = torch.cat((self.head_idx, new_triples[:, 0]), dim=0)
        updated_tail_idx = torch.cat((self.tail_idx, new_triples[:, 1]), dim=0)
        updated_relations = torch.cat((self.relations, new_triples[:, 2]), dim=0)

        # Update dict_of_heads, dict_of_tails, dict_of_rels
        for h, t, r in new_triples.tolist():
            self.dict_of_heads[(t, r)].add(h)
            self.dict_of_tails[(h, r)].add(t)
            self.dict_of_rels[(h, t)].add(r)

        # Create a new instance of the class with updated triples
        return self.__class__(
            kg={'heads': updated_head_idx, 'tails': updated_tail_idx, 'relations': updated_relations},
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix,
            dict_of_heads=self.dict_of_heads,
            dict_of_tails=self.dict_of_tails,
            dict_of_rels=self.dict_of_rels
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
        new_head_idx, new_tail_idx, new_relations = [], [], []

        for relation_id in undirected_relations:
            inverse_relation = f"{ix2rel[relation_id]}_inv"

            # Check if the inverse relation already exists in the graph
            if relation_id not in self.rel2ix.values():
                logging.info(f"Relation {relation_id} not found in knowledge graph. Skipping...")
                continue

            # Create a new ID for the inverse relation
            inverse_relation_id = len(self.rel2ix)
            self.rel2ix[inverse_relation] = inverse_relation_id

            reverse_list.append((relation_id, inverse_relation_id))

            # Masks for the original relation
            mask = (self.relations == relation_id)

            # Add new inverse triples to the tensors
            new_head_idx.append(self.tail_idx[mask])
            new_tail_idx.append(self.head_idx[mask])
            new_relations.append(torch.full_like(self.relations[mask], inverse_relation_id))

            # Add facts to the dictionnaries
            for i in range(len(mask)):
                if mask[i]:
                    h = self.head_idx[i].item()
                    t = self.tail_idx[i].item()

                    # Add original fact (a, r, b)
                    self.dict_of_heads.setdefault((t, relation_id), set()).add(h)  # (t, r) -> h
                    self.dict_of_tails.setdefault((h, relation_id), set()).add(t)  # (h, r) -> t
                    self.dict_of_rels.setdefault((h, t), set()).add(relation_id)  # (h, t) -> r

                    # Add inverse fact (b, r_inv, a)
                    self.dict_of_heads.setdefault((h, inverse_relation_id), set()).add(t)  # (h, r_inv) -> t
                    self.dict_of_tails.setdefault((t, inverse_relation_id), set()).add(h)  # (t, r_inv) -> h
                    self.dict_of_rels.setdefault((t, h), set()).add(inverse_relation_id)  # (t, h) -> r_inv

                    # Ajouter les combinaisons supplémentaires (b, r, a) et (a, r_inv, b)
                    self.dict_of_heads.setdefault((t, relation_id), set()).add(h)  # (t, r) -> h
                    self.dict_of_tails.setdefault((h, relation_id), set()).add(t)  # (h, r) -> t
                    self.dict_of_rels.setdefault((t, h), set()).add(relation_id)  # (t, h) -> r

                    self.dict_of_heads.setdefault((h, inverse_relation_id), set()).add(t)  # (h, r_inv) -> t
                    self.dict_of_tails.setdefault((t, inverse_relation_id), set()).add(h)  # (t, r_inv) -> h
                    self.dict_of_rels.setdefault((h, t), set()).add(inverse_relation_id)  # (h, t) -> r_inv

        # Concat new triples to existing ones
        if new_head_idx:
            self.head_idx = torch.cat((self.head_idx, *new_head_idx), dim=0)
            self.tail_idx = torch.cat((self.tail_idx, *new_tail_idx), dim=0)
            self.relations = torch.cat((self.relations, *new_relations), dim=0)

        return self.__class__(
                kg={'heads': self.head_idx, 'tails': self.tail_idx, 'relations': self.relations},
                ent2ix=self.ent2ix,
                rel2ix=self.rel2ix,
                dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails,
                dict_of_rels=self.dict_of_rels
            ), reverse_list

    def permute_tails(self, relation_id: int) -> Self:
        """
        Randomly permutes the `tails` for a given relation while maintaining the original degree
        of `heads` and `tails`, ensuring there are no triples of the form (a, rel, a) where `head == tail`.
        Updates the dictionary of facts.

        Parameters
        ----------
        relation_id: int
            The ID of the relation for which `tails` should be permuted.

        Returns
        -------
        KnowledgeGraph
            A new instance of KnowledgeGraph with the `tails` permuted.
        """
        
        # Clone the attributes for the new instance
        new_head_idx = self.head_idx.clone()
        new_tail_idx = self.tail_idx.clone()
        new_relations = self.relations.clone()

        # Mask to filter out all relations except the given one
        mask = (new_relations == relation_id)

        # Extract all head and tail indices for this relation
        heads_for_relation = new_head_idx[mask].tolist()
        tails_for_relation = new_tail_idx[mask].tolist()

        tails_count = Counter(tails_for_relation)

        permuted_tails = tails_for_relation[:]
        random.shuffle(permuted_tails)

        # Correct self-loops while preserving node degree
        for i in range(len(permuted_tails)):
            if heads_for_relation[i] == permuted_tails[i]:
                # If we have a self-loop, look for another tail
                found = False
                for j in range(i + 1, len(permuted_tails)):
                    if heads_for_relation[j] != permuted_tails[i] and heads_for_relation[i] != permuted_tails[j]:
                        # Trade to solve the self-loop
                        permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                        found = True
                        break
                # If no valid trade is found, go back to the beginning
                if not found:
                    for j in range(0, i):
                        if heads_for_relation[j] != permuted_tails[i] and heads_for_relation[i] != permuted_tails[j]:
                            # Trade to solve the self-loop
                            permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                            break

        permuted_tails = torch.tensor(permuted_tails, dtype=new_tail_idx.dtype)

        # Replace original tails with permuted ones
        new_tail_idx[mask] = permuted_tails

        # Check if the node degree is correctly preserved
        assert Counter(new_tail_idx[mask].tolist()) == tails_count, "Tails degree not preserved after petmutation."
        assert all(new_head_idx[i] != new_tail_idx[i] for i in range(len(new_head_idx))), "Some triples have the same `head` and `tail` after permutation (self-loop)."

        return self.__class__(
            kg={'heads': new_head_idx, 'tails': new_tail_idx, 'relations': new_relations},
            ent2ix=self.ent2ix,
            rel2ix=self.rel2ix,
        )


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
                logging.info(f'{len(pairs) - len(unique)} duplicates found. Keeping {len(unique)} unique triplets for relation {r_}')

        # Return a new KnowledgeGraph instance with only unique triples retained
        return self.keep_triples(keep)

    def get_pairs(self, r: int, type:str='ht') -> Set[Tuple[Number, Number]]:
        mask = (self.relations == r)

        if type == 'ht':
            return set((i.item(), j.item()) for i, j in cat(
                (self.head_idx[mask].view(-1, 1),
                self.tail_idx[mask].view(-1, 1)), dim=1))
        else:
            assert type == 'th'
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

        logging.info('Finding duplicate relations')

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

        logging.info('Duplicate relations: {}'.format(len(duplicates)))
        logging.info('Reverse duplicate relations: '
                '{}\n'.format(len(rev_duplicates)))

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