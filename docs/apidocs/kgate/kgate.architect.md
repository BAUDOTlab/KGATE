# {py:mod}`kgate.architect`

```{py:module} kgate.architect
```

```{autodoc2-docstring} kgate.architect
:parser: myst
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Architect <kgate.architect.Architect>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logging_level <kgate.architect.logging_level>`
  - ```{autodoc2-docstring} kgate.architect.logging_level
    :parser: myst
    :summary:
    ```
````

### API

````{py:data} logging_level
:canonical: kgate.architect.logging_level
:value: >
   None

```{autodoc2-docstring} kgate.architect.logging_level
:parser: myst
```

````

`````{py:class} Architect(config_path: str = '', knowledge_graph: kgate.knowledgegraph.KnowledgeGraph | kgate.encoders.Literal[FB15k-237, WN18RR, PrimeKG] | None = None, dataframe: kgate.initializers.pd.DataFrame | None = None, metadata: kgate.initializers.pd.DataFrame | None = None, cudnn_benchmark: bool = True, number_of_cores: int = 0, **kwargs)
:canonical: kgate.architect.Architect

Bases: {py:obj}`torch.nn.Module`

````{py:property} encoder_node_embedding_dimensions
:canonical: kgate.architect.Architect.encoder_node_embedding_dimensions
:type: int

```{autodoc2-docstring} kgate.architect.Architect.encoder_node_embedding_dimensions
:parser: myst
```

````

````{py:property} encoder_edge_embedding_dimensions
:canonical: kgate.architect.Architect.encoder_edge_embedding_dimensions
:type: int

```{autodoc2-docstring} kgate.architect.Architect.encoder_edge_embedding_dimensions
:parser: myst
```

````

````{py:method} set_metadata(metadata: kgate.initializers.pd.DataFrame | os.PathLike | None)
:canonical: kgate.architect.Architect.set_metadata

```{autodoc2-docstring} kgate.architect.Architect.set_metadata
:parser: myst
```

````

````{py:method} initialize_encoder(encoder_name: kgate.encoders.Literal[Default, GCN, GAT, kgate.initializers.Node2Vec, ] = '', gnn_layers: int = 0) -> kgate.encoders.GCNEncoder | kgate.encoders.GATEncoder | None
:canonical: kgate.architect.Architect.initialize_encoder

```{autodoc2-docstring} kgate.architect.Architect.initialize_encoder
:parser: myst
```

````

````{py:method} initialize_decoder(decoder_name: str = '', dissimilarity: kgate.encoders.Literal[L1, L2, torus_L1, torus_L2, torus_eL2, ] = '', margin: int = None, filter_count: int = None) -> kgate.encoders.Tuple[kgate.decoders.BilinearDecoder | kgate.decoders.ConvolutionalDecoder | kgate.decoders.TranslationalDecoder, torchkge.utils.MarginLoss | torchkge.utils.BinaryCrossEntropyLoss]
:canonical: kgate.architect.Architect.initialize_decoder

```{autodoc2-docstring} kgate.architect.Architect.initialize_decoder
:parser: myst
```

````

````{py:method} initialize_optimizer() -> torch.optim.Optimizer
:canonical: kgate.architect.Architect.initialize_optimizer

```{autodoc2-docstring} kgate.architect.Architect.initialize_optimizer
:parser: myst
```

````

````{py:method} initialize_negative_sampler() -> kgate.samplers.NegativeSampler
:canonical: kgate.architect.Architect.initialize_negative_sampler

```{autodoc2-docstring} kgate.architect.Architect.initialize_negative_sampler
:parser: myst
```

````

````{py:method} initialize_learning_rate_scheduler() -> torch.optim.lr_scheduler.LRScheduler | None
:canonical: kgate.architect.Architect.initialize_learning_rate_scheduler

```{autodoc2-docstring} kgate.architect.Architect.initialize_learning_rate_scheduler
:parser: myst
```

````

````{py:method} initialize_evaluator() -> kgate.evaluators.LinkPredictionEvaluator | kgate.evaluators.TripletClassificationEvaluator
:canonical: kgate.architect.Architect.initialize_evaluator

```{autodoc2-docstring} kgate.architect.Architect.initialize_evaluator
:parser: myst
```

````

````{py:method} initialize_initializer() -> kgate.initializers.Initializer
:canonical: kgate.architect.Architect.initialize_initializer

```{autodoc2-docstring} kgate.architect.Architect.initialize_initializer
:parser: myst
```

````

````{py:method} initialize_model(attributes: kgate.encoders.Dict[str, kgate.initializers.pd.DataFrame] = {}, pretrained: kgate.encoders.Path | None = None)
:canonical: kgate.architect.Architect.initialize_model

```{autodoc2-docstring} kgate.architect.Architect.initialize_model
:parser: myst
```

````

````{py:method} train_model(checkpoint_file: kgate.encoders.Path | None = None, attributes: kgate.encoders.Dict[str, kgate.initializers.pd.DataFrame] = {}, dry_run: bool = False)
:canonical: kgate.architect.Architect.train_model

```{autodoc2-docstring} kgate.architect.Architect.train_model
:parser: myst
```

````

````{py:method} test() -> kgate.encoders.Dict[str, float | kgate.encoders.Dict[str, float]]
:canonical: kgate.architect.Architect.test

```{autodoc2-docstring} kgate.architect.Architect.test
:parser: myst
```

````

````{py:method} infer(heads: kgate.encoders.List[str] = [], tails: kgate.encoders.List[str] = [], edges: kgate.encoders.List[str] = [], top_k: int = 100)
:canonical: kgate.architect.Architect.infer

```{autodoc2-docstring} kgate.architect.Architect.infer
:parser: myst
```

````

````{py:method} load_checkpoint(path: kgate.encoders.Path) -> dict
:canonical: kgate.architect.Architect.load_checkpoint

```{autodoc2-docstring} kgate.architect.Architect.load_checkpoint
:parser: myst
```

````

````{py:method} load_best_model() -> None
:canonical: kgate.architect.Architect.load_best_model

```{autodoc2-docstring} kgate.architect.Architect.load_best_model
:parser: myst
```

````

````{py:method} get_batch_embeddings(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, batch: kgate.encoders.Tensor, mask: kgate.encoders.Tensor | None = None) -> kgate.encoders.nn.Parameter
:canonical: kgate.architect.Architect.get_batch_embeddings

```{autodoc2-docstring} kgate.architect.Architect.get_batch_embeddings
:parser: myst
```

````

````{py:method} process_batch(engine: ignite.engine.Engine, batch: kgate.encoders.Tensor) -> kgate.encoders.torch.types.Number
:canonical: kgate.architect.Architect.process_batch

```{autodoc2-docstring} kgate.architect.Architect.process_batch
:parser: myst
```

````

````{py:method} forward(positive_triplets_batch: kgate.encoders.torch.Tensor, negative_triplets_batch: kgate.encoders.torch.Tensor, node_embeddings: kgate.encoders.torch.Tensor) -> kgate.encoders.Tuple[kgate.encoders.Tensor, kgate.encoders.Tensor]
:canonical: kgate.architect.Architect.forward

```{autodoc2-docstring} kgate.architect.Architect.forward
:parser: myst
```

````

````{py:method} scoring_function(batch: kgate.encoders.Tensor, node_embeddings: kgate.encoders.Tensor) -> kgate.encoders.Tensor
:canonical: kgate.architect.Architect.scoring_function

```{autodoc2-docstring} kgate.architect.Architect.scoring_function
:parser: myst
```

````

````{py:method} get_embeddings() -> kgate.encoders.Dict[str, kgate.encoders.Tensor]
:canonical: kgate.architect.Architect.get_embeddings

```{autodoc2-docstring} kgate.architect.Architect.get_embeddings
:parser: myst
```

````

````{py:method} normalize_parameters()
:canonical: kgate.architect.Architect.normalize_parameters

```{autodoc2-docstring} kgate.architect.Architect.normalize_parameters
:parser: myst
```

````

````{py:method} log_metrics_to_csv(engine: ignite.engine.Engine)
:canonical: kgate.architect.Architect.log_metrics_to_csv

```{autodoc2-docstring} kgate.architect.Architect.log_metrics_to_csv
:parser: myst
```

````

````{py:method} clean_memory()
:canonical: kgate.architect.Architect.clean_memory

```{autodoc2-docstring} kgate.architect.Architect.clean_memory
:parser: myst
```

````

````{py:method} evaluate(engine: ignite.engine.Engine)
:canonical: kgate.architect.Architect.evaluate

```{autodoc2-docstring} kgate.architect.Architect.evaluate
:parser: myst
```

````

````{py:method} update_scheduler()
:canonical: kgate.architect.Architect.update_scheduler

```{autodoc2-docstring} kgate.architect.Architect.update_scheduler
:parser: myst
```

````

````{py:method} get_validation_metric(engine: ignite.engine.Engine) -> float
:canonical: kgate.architect.Architect.get_validation_metric

```{autodoc2-docstring} kgate.architect.Architect.get_validation_metric
:parser: myst
```

````

````{py:method} on_training_completed(engine: ignite.engine.Engine)
:canonical: kgate.architect.Architect.on_training_completed

```{autodoc2-docstring} kgate.architect.Architect.on_training_completed
:parser: myst
```

````

````{py:method} calculate_metrics_for_edges(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph | torch.utils.data.Subset[kgate.knowledgegraph.KnowledgeGraph], edge_indices: kgate.encoders.List[str]) -> kgate.encoders.Tuple[float, int, kgate.encoders.Dict[str, float], float]
:canonical: kgate.architect.Architect.calculate_metrics_for_edges

```{autodoc2-docstring} kgate.architect.Architect.calculate_metrics_for_edges
:parser: myst
```

````

````{py:method} calculate_metrics_for_categories(frequent_indices: kgate.encoders.List[int], infrequent_indices: kgate.encoders.List[int]) -> kgate.encoders.Tuple[float, float]
:canonical: kgate.architect.Architect.calculate_metrics_for_categories

```{autodoc2-docstring} kgate.architect.Architect.calculate_metrics_for_categories
:parser: myst
```

````

````{py:method} link_prediction(knowledge_graph_subset: torch.utils.data.Subset[kgate.knowledgegraph.KnowledgeGraph]) -> float
:canonical: kgate.architect.Architect.link_prediction

```{autodoc2-docstring} kgate.architect.Architect.link_prediction
:parser: myst
```

````

````{py:method} triplet_classification() -> float
:canonical: kgate.architect.Architect.triplet_classification

```{autodoc2-docstring} kgate.architect.Architect.triplet_classification
:parser: myst
```

````

`````
