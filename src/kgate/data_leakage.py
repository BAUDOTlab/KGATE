from random import random
from collections import Counter
import torch

from .knowledgegraph import KnowledgeGraph

def permute_tails(kg, relation_id):
    """
    Randomly permutes the `tails` for a given relation while maintaining the original degree
    of `heads` and `tails`, ensuring there are no triples of the form (a, rel, a) where `head == tail`.

    Parameters
    ----------
    kg: KnowledgeGraph
        The KnowledgeGraph instance on which to perform the permutation.
    relation_id: int
        The ID of the relation for which `tails` should be permuted.

    Returns
    -------
    KnowledgeGraph
        A new instance of KnowledgeGraph with the `tails` permuted.
    """
    
    # Copy the attributes for the new instance
    new_head_idx = kg.head_idx.clone()
    new_tail_idx = kg.tail_idx.clone()
    new_relations = kg.relations.clone()
    new_triples = kg.triples.clone()

    # Mask only the target relation
    mask = (new_relations == relation_id)

    # Get head and tail indices for this relation
    heads_for_relation = new_head_idx[mask].tolist()
    tails_for_relation = new_tail_idx[mask].tolist()

    # Count the occurence of each tail in the relation
    tails_count = Counter(tails_for_relation)

    # Shuffle tails randomly. This might create self-loops and break node degree
    permuted_tails = tails_for_relation[:]
    random.shuffle(permuted_tails)

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

    permuted_tails = torch.tensor(permuted_tails, dtype=new_tail_idx.dtype)

    new_tail_idx[mask] = permuted_tails
    new_edgelist = torch.cat([new_head_idx,new_tail_idx,new_relations,new_triples])
    # Check if node degree is correctly preserved
    assert Counter(new_tail_idx[mask].tolist()) == tails_count, "`tails` node degree is not conserved after permutation."
    assert all(new_head_idx[i] != new_tail_idx[i] for i in range(len(new_head_idx))), "Self-loops introduced after permutation."

    return KnowledgeGraph(
        edgelist=new_edgelist,
        ent2ix=kg.ent2ix,
        rel2ix=kg.rel2ix,
        nt2ix=kg.nt2ix,
        triple_types=kg.triple_types,
        removed_triples=kg.removed_triples
    )