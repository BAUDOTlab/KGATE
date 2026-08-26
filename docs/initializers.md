# Initializers

Initializers are specific components of model training that only run once before the training loop. They are used to generate the initial embeddings for a `Knowledge Graph`. It is possible to manually choose different initializers for different node types using the `initialize_embedding` method, or all of them at once with the `initialize_all_embeddings` method (see the API reference for the signature of each method).

## [Random initialization](./reference/api_initializers.md#defaultinitializer)

This is the default Initializer. The embeddings of each node type and edges are initialized using PyTorch's `[xavier_uniform](https://docs.pytorch.org/docs/2.13/nn.init.html#torch.nn.init.xavier_uniform_)`.

```
from kgate.datasets import load_FB15k_237
from kgate.initializers import Initializer

knowledge_graph = load_FB15k_237()

initializer = Initializer()
knowledge_graph.embeddings = initializer.initialize_all_embeddings(
                knowledge_graph = knowledge_graph,
                node_embedding_dimensions = 128,
                edge_embedding_dimensions = 128,
                device = "cuda"
                )

# Can also be initialized in place
initializer.initialize_all_embeddings(
                knowledge_graph = knowledge_graph,
                node_embedding_dimensions = 128,
                edge_embedding_dimensions = 128,
                device = "cuda",
                inplace = True
                )
```

## [Initial features](./reference/api_initializers.md#featureinitializer)

It is possible to use any kind of numeric features. This is especially useful for biomedical graphs, where biological features can drive a more informed training. Examples of features can be transcriptomic profiles, output of a BERT model, or precomputed scores.

```warning
While you can use features of differente sources for the different node types, they **must** share the same dimensions, unless you use an encoder. In that case, it becomes possible to have heterogeneous features that the encoder will harmonize in the latent space.
```

When using the `initialize_all_embeddings` method, if the dictionary of features doesn't have an entry for a node type, it will be randomly initialized (see above) instead.

<!--Add code example-->

## [Node2Vec initialization](./reference/api_initializers.md#node2vecinitializer)

[Node2Vec](https://arxiv.org/pdf/1607.00653) is a random-walk based machine learning algorithm used to learn a representation of a graph based on its topology. Starting from a seed node, a walker will travel the graph by randomly following an edge each iteration. In KGATE, we use Node2Vec to generate topology-informed initial embeddings. Note that using Node2Vec or any other machine learning-based initializer will significantly increase the embedding generation time.

## Build your own initializer

Adding an initializer to KGATE is fairly straightforward. It is recommended but not mandatory to inherit from the base class `Initializer` to ease type hints and provide a fallback to random initialization in case something goes wrong or the initialization is only partial. In any case, only one method is required :

The `initialize_all_embeddings` method takes as input the `knowledge graph`, `node` and `edge embedding dimensions`, the `device` on which the initial features should be generated, and whether to do it `inplace` or return the tensors. This method should generate feature tensor for at least all nodes and edges present in `knowledge_graph.train_set`.

Another method can be useful to generate individual tensors. `initialize_embedding` should return the initial embeddings of a single tensor, for example a node type or the edge embeddings.