import random
from collections import Counter
import torch

from .knowledgegraph import KnowledgeGraph

def permute_tails(kg: KnowledgeGraph, relation: str, preserve_node_degree=True) -> KnowledgeGraph:
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
    triple_types = kg.triple_types

    ix2nt = {v: k for k,v in kg.node_type_to_index.items()}
    relation_id = kg.edge_to_index[relation]

    # Mask only the target relation
    mask = (kg.relations == relation_id)

    # Get head and tail indices for this relation
    heads_for_relation = kg.head_indices[mask].tolist()
    tails_for_relation = kg.tail_indices[mask].tolist()

    triples = [0] * len(tails_for_relation) if len(kg.node_type_to_index)==1 else []

    # Count the occurence of each tail in the relation
    tails_count = Counter(tails_for_relation)

    if preserve_node_degree:
        # Shuffle tails randomly. This might create self-loops
        permuted_tails = tails_for_relation[:] # Creating an integral slice is functionnally the same as deepcopy somehow
        random.shuffle(permuted_tails)
    else:
        num_tails = len(tails_for_relation)
        num_nodes = kg.n_ent
        permuted_tails = [random.randrange(num_nodes) for _ in range(num_tails)]



    # Fix self loop and correct node degree
    for i in range(len(permuted_tails)):
        if heads_for_relation[i] == permuted_tails[i]:
            # It is a self-loop
            found = False
            for j in range(i + 1, len(permuted_tails)):
                if heads_for_relation[j] != permuted_tails[i] and heads_for_relation[i] != permuted_tails[j]:
                    # Swap the two permuted tails
                    permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                    found = True
                    break
            # If no valid swap has been found, start again from the beginning
            if not found:
                for j in range(0, i):
                    if heads_for_relation[j] != permuted_tails[i] and heads_for_relation[i] != permuted_tails[j]:
                        permuted_tails[i], permuted_tails[j] = permuted_tails[j], permuted_tails[i]
                        break

        if len(kg.node_type_to_index) > 1:
            perm_tri = (
                    ix2nt[node_types[heads_for_relation[i]].item()],
                    relation,
                    ix2nt[node_types[permuted_tails[i]].item()]
                )
            # Add it if it doesn't already exists
            if not perm_tri in triple_types:
                triple_types.append(perm_tri)
                triple = len(triple_types)
            else:
                triple = triple_types.index(perm_tri)
            triples.append(triple)

    permuted_tails = torch.tensor(permuted_tails, dtype=kg.tail_indices.dtype, device=kg.edgelist.device)
    new_triples = torch.tensor(triples, device=kg.edgelist.device)

    kg.tail_indices[mask] = permuted_tails
    kg.triples[mask] = new_triples

    if preserve_node_degree:
        # Check if node degree is correctly preserved
        assert Counter(kg.tail_indices[mask].tolist()) == tails_count, "`tails` node degree is not conserved after permutation."
        assert all(kg.head_indices[i] != kg.tail_indices[i] for i in range(len(kg.head_indices))), "Self-loops introduced after permutation."

    return kg