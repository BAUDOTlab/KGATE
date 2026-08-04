# Architect

The **Architect** class is the centerpiece when using **KGATE** as a framework. While it is possible to use KGATE as a general toolbox and separate each of its components, most use cases can be handled by the Architect, including the data preparation and model inference.

## Configuration

A configuration file can be fed to the Architect's constructor. See [the configuration page]() for details on how to fill it and what each option does. You can also find a configuration template on the github repository, prefilled with default values. This configuration can also be given programmatically, by giving to the `Architect` a dictionary of kwargs with the same structure or building yourself a `Configuration` object. While not mandatory, it is highly recommended to use a configuration file to ensure reproductibility, and to carefully check the configuration hyperparameters before training a model.

## Structure

The Architect assembles all the building blocks of KGATE in the order of the layers.

### Data Layer

At the base of everything is the `KnowledgeGraph` object. When you create a new Architect, it must be given a knowledge graph either as a csv file (in the configuration or as argument), or a KnowledgeGraph object. It is possible to give a metadata csv file alongside the knowledge graph, which must contain at least the columns "Id" and "Type", corresponding respectively to the identifier of the node as it is in the knowledge graph and its node type. Additional metadata can be given. It won't be used on its own by the Architect, but may be useful for data exploration.

The knowledge graph is the fundation of any KGATE work. All other building blocks can be seen as transformation functions that take the knowledge graph as input and output a transformed version, usually on the embeddings.

```warning
Currently, KGATE does not support edge metadata, although edge feature can be added before training
```

By default, the knowledge graph is automatically processed : 
- If the metadata is given, it is mapped to the knowledge graph.
- All known sources of data leakage are controlled and reduced. See the [preprocessing workflow](./preprocessing_workflow.md) for details on the procedure.
- Masks corresponding to the training, validation and test set are generated in a way that mitigate training biases.

This data preprocessing step can also be run partially by hand, or skipped altogether, in which case you will need to provide your own training, validation and test masks. 

### Model Layer

Once you have your `Architect` object with its knowledge graph, you can build the actual KGE model using the `initialize_model` method. A KGATE model is composed of four parts:

#### Initializer

The `Initializer` is responsible for generating initial embeddings for all nodes and edges in the graph. The most common form of initialization is a random one, though giving more informed features can help the models to learn and converge faster if not better.

- The default `Initializer` generates random embeddings using the `xavier_uniform` function. It is also the fallback when another method doesn't generate embeddings for a set of nodes.
- The `FeatureInitializer` uses user-provided features in the form of a dictionary where the keys are node types and values a pytorch tensor. The tensor must be of the same size of the node type, and entries must match the order of the graph in order to work. The dictionary `KnowledgeGraph.node_to_index` can help to ensure correct mapping. If a node type is not specified, it will be randomly initialized.
```warning
If you don't use an [encoder](#Encoder), all given features must have the same dimensions, otherwise the decoder will be unable to process them.
```

- The `Node2VecInitializer` runs a [Node2Vec (Grover & Leskovec 2016)](https://arxiv.org/pdf/1607.00653) algorithm on the knowledge graph. For each node, the initial embedding is the result of a 100-epoch random walk with that node as seed. The edges are then randomly initialized.

The initializer can be independantly created with the `initialize_initializer` method. To learn how to implement your own intializer, go to [Build your own model: Initializer](./BYOM.md#initializer)

#### Encoder

The `Encoder` is an optional part of the model, able to learn from complex features at a much higher computational cost and training duration. Encoders are able to learn the latent representation of heterogeneous features per node type and homogenize their dimensions in order for the decoder to easily be applied on the whole graph. If you do not have complex features, it is recommended to avoid using an encoder unless you know why you are doing so, as the performance gained from an encoder on a randomly initialized knowledge graph is minimal compared to the increase in training cost.

KGATE encoders are built upon [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)'s GNN. Builtin encoders are simple, using the same convolution for each triplet type, but more advanced architecture can easily be built. See [Build your own model: Encoder](./BYOM.md#encoder) for more details.

#### Decoder

The `Decoder` takes the latent space as input and attempts to reconstruct the original graph from it. The latent space is either the output of the encoder or, if there is none, the knowledge graph's initial embeddings. To do so, the decoder is given the representation of a true triplet and a given number of false triplets, also called negative. It is then tasked with assigning a score to each of them. The objective is for the decoder to score the true triplet high and the false triplets low. The differences between different decoders lie in the way they compute these scores, and what property of the data they model. Comparing different decoders is highly recommended before going further in a project, as there is no telling which decoder will perform best on a specific graph without some benchmarking.

There are three families of decoders:
- **Translational**: sometimes called **geometric**, their objective function is always to make it so the head node's vector is roughly equal to the tail node's vector through a translation by the edge vector. It can be a regular euclidian translation (TransE 2013), a rotation (RotatE), using hyperplanes (TransH) or even more complex methods.
- **Bilinear**: decoders of this family use tensor factorization methods to score the given triplets. The most well-known bilinear decoder is Distmult (Yang et al. 2014), as it is the decoder most deep learning encoders use by default thanks to its simple principle and fast execution.
- **Convolutional**: these decoders use deep convolutional layers, and in turn are less explainable than the other families, but may yield stronger results in some situations.

#### Evaluation Method

Classically, predictions of a KGE models on the completion task are evaluated by ranking the score of all candidates and determining the position of the true triplet in the list. KGATE proposes a second method called **SpherE** which represents the nodes as spheres instead of vectors. Based on RotatE, this representation allows for a new evaluation method, where every sphere intersecting with the head sphere after rotation is considered true, with no ranking. While it removes the interrogation of when does the model consider a triplet to be true, it also have a loss of information resulting in some metrics being unusable. For more detail, read the original paper.

### Hyperparameter Layer

Once the model is built, the hyperparameters and utilities can be created. The Architect creates the optimizer and learning rate scheduler from PyTorch corresponding to the configuration. Then, the **Negative Sampler** prepares the generation of negative triplets. For each true triplet, the negative sampler will create a given number of false triplets according to the given parameters. The false triplets must not be too difficult, e.g. too realistic from the beginning or the decoder will have a hard time discriminating them from the true one. On the other hand, if they are too easy or too few, it may prevent the decoder from learning accurately from the knowledge graph.

### Evaluation Layer

Finally, the objective of the training is set in this layer. Currently, the two available objectives in KGATE are Link Prediction and Triplet Classification. The first one is used to complete a triplet given two components, for example prediction the tail of a (head, edge, ?) triplet. The second one evaluates the plausibility of a complete triplet in the knowledge graph, in other word if this triplet may exist in the data. The metrics used will depend on the objective.