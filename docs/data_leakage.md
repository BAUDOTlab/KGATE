# Data Leakage Control

**KGATE** comes with the **data leakage** control procedure described in [Brière et al.](https://www.biorxiv.org/content/10.1101/2025.01.23.634511v1). **Data leakage** is described as an artificial inflation of a model's training performances due to information from the training set leaking into the test set. In graph representation learning, **data leakage** may take multiple forms and be difficult to identify.

**KGATE** addresses the following sources of **data leakage**:

## Redundant relations

This issue is common in **knowledge graphs** assembled from external sources, or built by an automated tool. If two types of relations have a near-total overlap in the node they cover and only one type is present in the test set for a given triplet, the model will easily predict the existence of the other type, which will inflate the performance metrics.

```{hint} Example
In this **knowledge graph**, the relations `interacts` and `PPI` (protein-protein interaction) are used to represent any kind of interaction between two protein nodes. While the first has a broader meaning, every time the `PPI` relation is present, the `interacts` is too. Thus, if in the test set two protein nodes are connected with only the relation `PPI`, the relation `interacts` is correctly predicted, even though the model may not have learned the semantic meaning of this relation.

[IMG]
```

### Fix the knowledge graph

This issue may be solved by modifying directly the input **knowledge graph**, by removing redundant relations with a low relevance from the dataset prior to the training. Before doing that, make sure that the removed relation is indeed purely redundant and bloats the graph, otherwise you risk losing information.

### How KGATE handles it

During the split into train, validation and test set, **KGATE** identifies all redundant relations and make sure that all similar relations for any given triplet are in the same set.

## Carthesian relations

This is a trickier version of the redundant relations. Carthesian relations are edges that, for a given set of nodes, will densely connect them together. Then, if one edge is missing in the test set, its prediction is trivial and leads to the same inflation issue.

```{hint} Example
In the same **knowledge graph**, there are gene and tissue nodes. However, some genes are ubiquitous, which means they are expressed in every single tissue. If the expression relation is not present in the test set, it can be easily predicted.

[IMG]
```

### Fix the knowledge graph

This one is hard to fix directly from the dataset, because carthesian relations may hold relevant semantic information which should not be carelessly removed.

### How KGATE handles it

**KGATE** detects carthesian relations before splitting the knowledge graph into train, validation and test set, and ensure all carthesian relations belong to the training set.

## Other types of data leakage

Other sources of **data leakage** have been identified, but they cannot be addressed automatically and should thus be taken into account by the user.

If a model **uses node degree as an illegitimate feature**, it means that the predictions are driven by the graph topology more than its semantics. Thus, the test metrics could be high but due to not having an understanding of the relation's semantics, the model would perform poorly on inference data. One way to evaluate the use of node degree is to permute randomly the tails of the triplets while preserving the node degree. If there is a **major drop** in the evaluation metrics, then it means node degree is not used, as the model cannot learn effectively without the real information. However, if the metrics show **no change**, then it means the model uses the node degree as its main input feature.

The basics of a split is to keep the same proportions of each node type in the training, validation and test set. However, these proportions may not be **representative of the inference set**. Generalization is an important aspect of any machine learning model, but high performances on a training dataset are not a guarantee that the model can generalize, especially if the dataset it will be used during the inference phase has a very different topology and node type proportions.