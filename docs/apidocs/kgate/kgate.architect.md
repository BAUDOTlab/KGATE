# {py:mod}`kgate.architect`

```{py:module} kgate.architect
```

```{autodoc2-docstring} kgate.architect
:parser: docstrings_parser
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
    :parser: docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logging_level
:canonical: kgate.architect.logging_level
:value: >
   None

```{autodoc2-docstring} kgate.architect.logging_level
:parser: docstrings_parser
```

````

`````{py:class} Architect(config_path: str = '', kg: kgate.encoders.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph] | kgate.knowledgegraph.KnowledgeGraph | None = None, dataframe: pandas.DataFrame | None = None, metadata: pandas.DataFrame | None = None, cudnn_benchmark: bool = True, number_of_cores: int = 0, **kwargs)
:canonical: kgate.architect.Architect

Bases: {py:obj}`torch.nn.Module`

````{py:property} encoder_node_embedding_dimensions
:canonical: kgate.architect.Architect.encoder_node_embedding_dimensions
:type: int

```{autodoc2-docstring} kgate.architect.Architect.encoder_node_embedding_dimensions
:parser: docstrings_parser
```

````

````{py:property} encoder_edge_embedding_dimensions
:canonical: kgate.architect.Architect.encoder_edge_embedding_dimensions
:type: int

```{autodoc2-docstring} kgate.architect.Architect.encoder_edge_embedding_dimensions
:parser: docstrings_parser
```

````

````{py:method} set_metadata(metadata: pandas.DataFrame | os.PathLike)
:canonical: kgate.architect.Architect.set_metadata

```{autodoc2-docstring} kgate.architect.Architect.set_metadata
:parser: docstrings_parser
```

````

````{py:method} initialize_encoder(encoder_name: kgate.encoders.Literal[Default, GCN, GAT, kgate.encoders.Node2Vec, ] = '', gnn_layers: int = 0) -> kgate.encoders.DefaultEncoder | kgate.encoders.GCNEncoder | kgate.encoders.GATEncoder
:canonical: kgate.architect.Architect.initialize_encoder

```{autodoc2-docstring} kgate.architect.Architect.initialize_encoder
:parser: docstrings_parser
```

````

````{py:method} initialize_decoder(decoder_name: str = '', dissimilarity: kgate.encoders.Literal[L1, L2, torus_L1, torus_L2, torus_eL2, ] = '', margin: int = 0, filter_count: int = 0) -> kgate.encoders.Tuple[kgate.decoders.BilinearDecoder | kgate.decoders.ConvolutionalDecoder | kgate.decoders.TranslationalDecoder, torchkge.utils.MarginLoss | torchkge.utils.BinaryCrossEntropyLoss]
:canonical: kgate.architect.Architect.initialize_decoder

```{autodoc2-docstring} kgate.architect.Architect.initialize_decoder
:parser: docstrings_parser
```

````

````{py:method} initialize_optimizer() -> torch.optim.Optimizer
:canonical: kgate.architect.Architect.initialize_optimizer

```{autodoc2-docstring} kgate.architect.Architect.initialize_optimizer
:parser: docstrings_parser
```

````

````{py:method} initialize_negative_sampler() -> kgate.samplers.NegativeSampler
:canonical: kgate.architect.Architect.initialize_negative_sampler

```{autodoc2-docstring} kgate.architect.Architect.initialize_negative_sampler
:parser: docstrings_parser
```

````

````{py:method} initialize_learning_rate_scheduler() -> kgate.architect.Architect.initialize_learning_rate_scheduler.learning_rate_scheduler | None
:canonical: kgate.architect.Architect.initialize_learning_rate_scheduler

```{autodoc2-docstring} kgate.architect.Architect.initialize_learning_rate_scheduler
:parser: docstrings_parser
```

````

````{py:method} initialize_evaluator() -> kgate.evaluators.LinkPredictionEvaluator | kgate.evaluators.TripletClassificationEvaluator
:canonical: kgate.architect.Architect.initialize_evaluator

```{autodoc2-docstring} kgate.architect.Architect.initialize_evaluator
:parser: docstrings_parser
```

````

````{py:method} initialize_model(attributes: kgate.encoders.Dict[str, pandas.DataFrame] = {}, pretrained: kgate.encoders.Path | None = None)
:canonical: kgate.architect.Architect.initialize_model

```{autodoc2-docstring} kgate.architect.Architect.initialize_model
:parser: docstrings_parser
```

````

````{py:method} train_model(checkpoint_file: kgate.encoders.Path | None = None, attributes: kgate.encoders.Dict[str, pandas.DataFrame] = {}, dry_run: bool = False)
:canonical: kgate.architect.Architect.train_model

```{autodoc2-docstring} kgate.architect.Architect.train_model
:parser: docstrings_parser
```

````

````{py:method} test() -> kgate.encoders.Dict[str, float | kgate.encoders.Dict[str, float]]
:canonical: kgate.architect.Architect.test

```{autodoc2-docstring} kgate.architect.Architect.test
:parser: docstrings_parser
```

````

````{py:method} infer(heads: kgate.encoders.List[str] = [], tails: kgate.encoders.List[str] = [], edges: kgate.encoders.List[str] = [], top_k: int = 100)
:canonical: kgate.architect.Architect.infer

```{autodoc2-docstring} kgate.architect.Architect.infer
:parser: docstrings_parser
```

````

````{py:method} load_checkpoint(path: kgate.encoders.Path) -> dict
:canonical: kgate.architect.Architect.load_checkpoint

```{autodoc2-docstring} kgate.architect.Architect.load_checkpoint
:parser: docstrings_parser
```

````

````{py:method} load_best_model()
:canonical: kgate.architect.Architect.load_best_model

```{autodoc2-docstring} kgate.architect.Architect.load_best_model
:parser: docstrings_parser
```

````

````{py:method} forward(positive_triplets_batch, negative_triplets_batch) -> kgate.encoders.Tuple[kgate.encoders.Tensor, kgate.encoders.Tensor]
:canonical: kgate.architect.Architect.forward

```{autodoc2-docstring} kgate.architect.Architect.forward
:parser: docstrings_parser
```

````

````{py:method} process_batch(engine: ignite.engine.Engine, batch: kgate.encoders.Tensor) -> kgate.encoders.torch.types.Number
:canonical: kgate.architect.Architect.process_batch

```{autodoc2-docstring} kgate.architect.Architect.process_batch
:parser: docstrings_parser
```

````

````{py:method} scoring_function(batch: kgate.encoders.Tensor, kg: kgate.knowledgegraph.KnowledgeGraph) -> kgate.encoders.Tensor
:canonical: kgate.architect.Architect.scoring_function

```{autodoc2-docstring} kgate.architect.Architect.scoring_function
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> kgate.encoders.Dict[str, kgate.encoders.Tensor]
:canonical: kgate.architect.Architect.get_embeddings

```{autodoc2-docstring} kgate.architect.Architect.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters()
:canonical: kgate.architect.Architect.normalize_parameters

```{autodoc2-docstring} kgate.architect.Architect.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} log_metrics_to_csv(engine: ignite.engine.Engine)
:canonical: kgate.architect.Architect.log_metrics_to_csv

```{autodoc2-docstring} kgate.architect.Architect.log_metrics_to_csv
:parser: docstrings_parser
```

````

````{py:method} clean_memory()
:canonical: kgate.architect.Architect.clean_memory

```{autodoc2-docstring} kgate.architect.Architect.clean_memory
:parser: docstrings_parser
```

````

````{py:method} evaluate(engine: ignite.engine.Engine)
:canonical: kgate.architect.Architect.evaluate

```{autodoc2-docstring} kgate.architect.Architect.evaluate
:parser: docstrings_parser
```

````

````{py:method} update_scheduler()
:canonical: kgate.architect.Architect.update_scheduler

```{autodoc2-docstring} kgate.architect.Architect.update_scheduler
:parser: docstrings_parser
```

````

````{py:method} get_validation_metric(engine: ignite.engine.Engine) -> float
:canonical: kgate.architect.Architect.get_validation_metric

```{autodoc2-docstring} kgate.architect.Architect.get_validation_metric
:parser: docstrings_parser
```

````

````{py:method} on_training_completed(engine: ignite.engine.Engine)
:canonical: kgate.architect.Architect.on_training_completed

```{autodoc2-docstring} kgate.architect.Architect.on_training_completed
:parser: docstrings_parser
```

````

````{py:method} categorize_test_nodes(edge_name: str, threshold: int) -> kgate.encoders.Tuple[kgate.encoders.List[int], kgate.encoders.List[int]]
:canonical: kgate.architect.Architect.categorize_test_nodes

```{autodoc2-docstring} kgate.architect.Architect.categorize_test_nodes
:parser: docstrings_parser
```

````

````{py:method} calculate_metrics_for_edges(kg: kgate.knowledgegraph.KnowledgeGraph, edge_indices: kgate.encoders.List[str]) -> kgate.encoders.Tuple[float, int, kgate.encoders.Dict[str, float], float]
:canonical: kgate.architect.Architect.calculate_metrics_for_edges

```{autodoc2-docstring} kgate.architect.Architect.calculate_metrics_for_edges
:parser: docstrings_parser
```

````

````{py:method} calculate_metrics_for_categories(frequent_indices: kgate.encoders.List[int], infrequent_indices: kgate.encoders.List[int]) -> kgate.encoders.Tuple[float, float]
:canonical: kgate.architect.Architect.calculate_metrics_for_categories

```{autodoc2-docstring} kgate.architect.Architect.calculate_metrics_for_categories
:parser: docstrings_parser
```

````

````{py:method} link_prediction(kg: kgate.knowledgegraph.KnowledgeGraph) -> float
:canonical: kgate.architect.Architect.link_prediction

```{autodoc2-docstring} kgate.architect.Architect.link_prediction
:parser: docstrings_parser
```

````

````{py:method} triplet_classification(kg_validation: kgate.knowledgegraph.KnowledgeGraph, kg_test: kgate.knowledgegraph.KnowledgeGraph) -> float
:canonical: kgate.architect.Architect.triplet_classification

```{autodoc2-docstring} kgate.architect.Architect.triplet_classification
:parser: docstrings_parser
```

````

````{py:method} run_data_leakage(attributes: kgate.encoders.Dict[str, pandas.DataFrame] = {})
:canonical: kgate.architect.Architect.run_data_leakage

```{autodoc2-docstring} kgate.architect.Architect.run_data_leakage
:parser: docstrings_parser
```

````

`````
