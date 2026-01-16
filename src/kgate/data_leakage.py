import random
from collections import Counter
import torch

from .knowledgegraph import KnowledgeGraph

def permute_tails(kg: KnowledgeGraph, edge: str, preserve_node_degree=True) -> KnowledgeGraph:
    """
    Randomly permutes the `tails` for a given relation while maintaining the original degree
    of `heads` and `tails`, ensuring there are no triples of the form (a, rel, a) where `head == tail`.

    Parameters
    ----------
    kg: KnowledgeGraph
        The KnowledgeGraph instance on which to perform the permutation.
    relation_id: str
        The name of the relation for which `tails` should be permuted.
    preserve_node_degree: bool, optional
        Whether or not the permuted tails should keep the same node degree. Default to True.
    Returns
    -------
    KnowledgeGraph
        A new instance of KnowledgeGraph with the `tails` permuted.
    """
    node_types = kg.node_types
    triplets_types = kg.triplet_types

    index_to_node_type = {value: key for key,value in kg.node_type_to_index.items()}
    edge_index = kg.edge_to_index[edge]

    # Mask only the target relation
    mask = (kg.edge_indices == edge_index)

    # Get head and tail indices for this relation
    heads_for_this_edge = kg.head_indices[mask].tolist()
    tails_for_this_edge = kg.tail_indices[mask].tolist()

    triplets = [0] * len(tails_for_this_edge) if len(kg.node_type_to_index)==1 else []

    # Count the occurence of each tail in the relation
    tails_count = Counter(tails_for_this_edge)

    if preserve_node_degree:
        # Shuffle tails randomly. This might create self-loops
        permuted_tails = tails_for_this_edge[:] # Creating an integral slice is functionnally the same as deepcopy somehow
        random.shuffle(permuted_tails)
    else:
        tail_count = len(tails_for_this_edge)
        node_count = kg.node_count
        permuted_tails = [random.randrange(node_count) for _ in range(tail_count)]



    # Fix self loop and correct node degree
    for i in range(len(permuted_tails)):
        if heads_for_this_edge[i] == permuted_tails[i]:
            # It is a self-loop
            found = False
            for j in range(i + 1, len(permuted_tails)):
                if heads_for_this_edge[j] != permuted_tails[i] and heads_for_this_edge[i] != permuted_tails[j]:
                    # Swap the two permuted tails
                    permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                    found = True
                    break
            # If no valid swap has been found, start again from the beginning
            if not found:
                for j in range(0, i):
                    if heads_for_this_edge[j] != permuted_tails[i] and heads_for_this_edge[i] != permuted_tails[j]:
                        permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                        break

        if len(kg.node_type_to_index) > 1:
            permuted_triplet = (
                    index_to_node_type[node_types[heads_for_this_edge[i]].item()],
                    edge,
                    index_to_node_type[node_types[permuted_tails[i]].item()]
                )
            # Add it if it doesn't already exists
            if not permuted_triplet in triplets_types:
                triplets_types.append(permuted_triplet)
                triplet = len(triplets_types)
            else:
                triplet = triplets_types.index(permuted_triplet)
            triplets.append(triplet)

    permuted_tails = torch.tensor(permuted_tails, dtype=kg.tail_indices.dtype, device=kg.graphindices.device)
    new_triplets = torch.tensor(triplets, device=kg.graphindices.device)

    kg.tail_indices[mask] = permuted_tails
    kg.triplets[mask] = new_triplets

    if preserve_node_degree:
        # Check if node degree is correctly preserved
        assert Counter(kg.tail_indices[mask].tolist()) == tails_count, "`tails` node degree is not conserved after permutation."
        assert all(kg.head_indices[i] != kg.tail_indices[i] for i in range(len(kg.head_indices))), "Self-loops introduced after permutation."

    return kg