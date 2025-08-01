## Definitions

For the rest of this document, we will consider some terms to be equivalent (in reality, their definition is slightly different, in a way that doesn't matter here):

- A **node**, or **entity**, is a point in the knowledge graph. Nodes may be of different type or not, depending on the input graph.
- An **edge**, or **relation**, is a connection between two **nodes** and holds semantic meaning. One edge always connect one node to one node (both nodes may be the same, in which case it is called a *self-loop*), and two nodes may be connected by more than one edge.
- A **triple**, or **fact**, represents a **head** node, also called **source** node or **subject** node to a **tail** node, also called **target** node or **object** node, connected by an **edge** or **relation**. A single node can be involved in many triples, as *head* and *tail*, but one edge always belongs to only one triple.

Training a KGE model is very easy with KGATE. Using the `Architect` class, KGATE takes care of the data loading, preprocessing and training using mini batches.

## Requirements

All you reall need to run a training procedure is one file: the **Knowledge Graph**, in `.tsv` or `.csv` format, where each row represents a triplet. It must have at least three columns: **from**, with the identifier of the *source* node, **to**; with the identifier of the *target* node; and **rel**, with the *edge type*. Additional columns are ignored. In addition, you can provide additional information:

- A configuration file in [TOML](https://toml.io/en/) format, following the template given here [add link to template]
- A metadata file, in `.tsv` or `.csv` format, giving node-level information such as the **node type**, especially used by GNN. It must have at least two columns: **id**, with the identifier of the node which should be identical to its identifier in the knowledge graph file; and **type**, the type of this node.
- Input features per node type. Instead of starting the training with random embeddings, you can give either precomputed embedding or a list of features to start the training. If you don't set a GNN encoder, all input features must be tensors of the same dimensions than your embedding dimension. If you use a GNN, then you can have features of any dimensions, even different dimensions between node types.

## Train a basic TransE

Let's say you want to train a simple TransE model for 200 epochs, with no encoder and just the knowledge graph structure.

```python
from kgate import Architect

kg_path = "/path/to/your/kg.csv"

model = {"decoder": {"name":"TransE"}} # Default value, explicitly set for the demonstration.
training = {"max_epochs":200}

# Builds the KnowledgeGraph object and preprocess the data.
architect = Architect(kg_csv=kg_path, model=model, training=training)

architect.train_model()

architect.test()
```
