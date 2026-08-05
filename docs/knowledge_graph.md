# KGATE's Knowledge Graph Object

The core of KGATE's data management is the `KnowledgeGraph` class. The `KnowledgeGraph` holds all the information about the dataset, including the embeddings. In KGATE, most of the functions and classes apply transformation to the knowledge graph.

## Overview

The graph representation of KGATE is called `graphindices`. It is a pytorch tensor with 4 rows and as many columns as triplets in the graph. The rows are the indices of each elements in the graph like so:
1. Head node indices
2. Tail node indices
3. Edge type indices
4. Triplet type indices

The last 

```{currentmodule} kgate.knowledgegraph
```

```{autoclass} kgate.knowledgegraph.KnowledgeGraph
    :members:
```
