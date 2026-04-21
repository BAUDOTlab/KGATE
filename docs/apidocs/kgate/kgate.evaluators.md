# {py:mod}`kgate.evaluators`

```{py:module} kgate.evaluators
```

```{autodoc2-docstring} kgate.evaluators
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Predictions <kgate.evaluators.Predictions>`
  - ```{autodoc2-docstring} kgate.evaluators.Predictions
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`LinkPredictionEvaluator <kgate.evaluators.LinkPredictionEvaluator>`
  - ```{autodoc2-docstring} kgate.evaluators.LinkPredictionEvaluator
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`TripletClassificationEvaluator <kgate.evaluators.TripletClassificationEvaluator>`
  - ```{autodoc2-docstring} kgate.evaluators.TripletClassificationEvaluator
    :parser: docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} Predictions(true_predictions_rank: torch.Tensor, filtered_true_predictions_rank: torch.Tensor)
:canonical: kgate.evaluators.Predictions

```{autodoc2-docstring} kgate.evaluators.Predictions
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.evaluators.Predictions.__init__
:parser: docstrings_parser
```

````{py:method} __str__()
:canonical: kgate.evaluators.Predictions.__str__

````

````{py:property} mean_rank
:canonical: kgate.evaluators.Predictions.mean_rank
:type: typing.Tuple[float, float]

```{autodoc2-docstring} kgate.evaluators.Predictions.mean_rank
:parser: docstrings_parser
```

````

````{py:method} hit_at_k(k: int = 10) -> typing.Tuple[float, float]
:canonical: kgate.evaluators.Predictions.hit_at_k

```{autodoc2-docstring} kgate.evaluators.Predictions.hit_at_k
:parser: docstrings_parser
```

````

````{py:property} mrr
:canonical: kgate.evaluators.Predictions.mrr
:type: typing.Tuple[float, float]

```{autodoc2-docstring} kgate.evaluators.Predictions.mrr
:parser: docstrings_parser
```

````

`````

`````{py:class} LinkPredictionEvaluator(full_graphindices: torch.Tensor, embedding_dimensions: int)
:canonical: kgate.evaluators.LinkPredictionEvaluator

```{autodoc2-docstring} kgate.evaluators.LinkPredictionEvaluator
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.evaluators.LinkPredictionEvaluator.__init__
:parser: docstrings_parser
```

````{py:method} evaluate(batch_size: int, encoder: kgate.encoders.DefaultEncoder | kgate.encoders.GNN, decoder: kgate.decoders.BilinearDecoder | kgate.decoders.ConvolutionalDecoder | kgate.decoders.TranslationalDecoder, knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding, verbose: bool = True) -> typing.Tuple[kgate.evaluators.Predictions, kgate.evaluators.Predictions]
:canonical: kgate.evaluators.LinkPredictionEvaluator.evaluate

```{autodoc2-docstring} kgate.evaluators.LinkPredictionEvaluator.evaluate
:parser: docstrings_parser
```

````

`````

`````{py:class} TripletClassificationEvaluator(architect: kgate.architect.Architect, kg_validation: kgate.knowledgegraph.KnowledgeGraph, kg_test: kgate.knowledgegraph.KnowledgeGraph)
:canonical: kgate.evaluators.TripletClassificationEvaluator

```{autodoc2-docstring} kgate.evaluators.TripletClassificationEvaluator
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.evaluators.TripletClassificationEvaluator.__init__
:parser: docstrings_parser
```

````{py:method} get_scores(heads: torch.Tensor, tails: torch.Tensor, edges: torch.Tensor, batch_size: int) -> torch.Tensor
:canonical: kgate.evaluators.TripletClassificationEvaluator.get_scores

```{autodoc2-docstring} kgate.evaluators.TripletClassificationEvaluator.get_scores
:parser: docstrings_parser
```

````

````{py:method} evaluate(batch_size: int, kg: kgate.knowledgegraph.KnowledgeGraph)
:canonical: kgate.evaluators.TripletClassificationEvaluator.evaluate

```{autodoc2-docstring} kgate.evaluators.TripletClassificationEvaluator.evaluate
:parser: docstrings_parser
```

````

````{py:method} accuracy(batch_size: int, kg_test: kgate.knowledgegraph.KnowledgeGraph, kg_validation: kgate.knowledgegraph.KnowledgeGraph | None = None) -> float
:canonical: kgate.evaluators.TripletClassificationEvaluator.accuracy

```{autodoc2-docstring} kgate.evaluators.TripletClassificationEvaluator.accuracy
:parser: docstrings_parser
```

````

`````
