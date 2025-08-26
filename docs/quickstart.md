# Quickstart

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

This code will generate the following files in the folder specified in the `output_directory` of the configuration (by default, the working directory):
  
- `checkpoints`: a folder containing:
  - `checkpoint_XX.pt`: the checkpoints of the last two saved epochs (by default, a checkpoint is saved every 5 epochs).
  - `best_model_checkpoint.pt`: the checkpoint of the model with the best performances on the validation set.
- `evaluation_metics.yaml`: YAML file containing the result of the evaluation on the test set, for all relations at once and individual relations.
- `kgate_config.toml`: TOML file of the exact configuration used for the training, including explicit default parameters.
- `kg.pkl`: only if you did not start from an already existing pickle file. Stores the KGATE representation of a knowledge graph.
- `training_metrics.csv`: CSV file keeping track of the metrics at each epoch, with 4 columns:
  - **Epoch**: the epoch number.
  - **Training Loss**: the mean loss across all batches of this epoch.
  - **Validation Metric**: typically MRR, the result of the latest evaluation on the validation set. By default, updates every 10 epochs.
  - **Learning Rate**: if using a learning rate scheduler, keeps track of the evolution throughout the epochs.
- `validation_metric_curve.png`: plot of the **training loss over epochs** and the **validation metric over epochs** (typically MRR).

## Use the trained model

With the fully trained model, you can then use it to infer new link:

```python
from kgate import Architect

# Load the model if it is not already in memory:
config_path = "/path/to/output_dir/kgate_config.toml"
architect = Architect(config_path=config_path)
architect.load_best_model()

# Find the most probable tail to complete the triplet (p53,INTERACTS,?)
result_df = architect.infer(heads="p53",rels="INTERACTS", top_k=5)
result_df.to_csv("inference_results.csv")

# The output dataframe has 2 columns: "Prediction" with the predicted missing element's name and "Score" with its confidence score.
```

Or get the embeddings for other applications such as node classification:

```python
from kgate import Architect

# Load the model if it is not already in memory:
config_path = "/path/to/output_dir/kgate_config.toml"
architect = Architect(config_path=config_path)
architect.load_best_model()

# Get the embedding dict
embeddings = architect.get_embeddings()

# Get mapping dictionaries to keep track of which embedding corresponds to what
mapping_ix2ent = {v: k for k,v in architect.kg_train.ent2ix.items()}
mapping_ix2rel = {v: k for k,v in architect.kg_train.ent2ix.items()}

# And run downstream tasks with the pretrained embeddings
```

### The embedding dictionary

The object returned by `architect.get_embeddings()` is a python dictionary with at least two elements:

- `entities` containing the entity embeddings as a pytorch tensor of size (n_ent, ent_emb_dim)
- `relations` containing the relation embeddings as a pytorch tensor of size (n_rel, rel_emb_dim)
- `decoder`, containing any additionnal embedding that is required for a decoder, such as RESCAL's relation matrix.

```{warning}
When using decoders with more than one embedding space (such as ComplEx which uses a real space and an imaginary space), they will
both be concatenated in the `entities` tensor, which will then be of size (n_ent, ent_emb_dim * n_embedding_spaces). To retrieve them
separately, split the tensor in part of equal size using `torch.tensor_split(embedding_tensor, n_embedding_spaces, dim=1)`.
```
