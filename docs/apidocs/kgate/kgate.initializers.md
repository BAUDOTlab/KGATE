# {py:mod}`kgate.initializers`

```{py:module} kgate.initializers
```

```{autodoc2-docstring} kgate.initializers
:parser: myst
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Initializer <kgate.initializers.Initializer>`
  - ```{autodoc2-docstring} kgate.initializers.Initializer
    :parser: myst
    :summary:
    ```
* - {py:obj}`FeatureInitializer <kgate.initializers.FeatureInitializer>`
  - ```{autodoc2-docstring} kgate.initializers.FeatureInitializer
    :parser: myst
    :summary:
    ```
* - {py:obj}`Node2VecInitializer <kgate.initializers.Node2VecInitializer>`
  - ```{autodoc2-docstring} kgate.initializers.Node2VecInitializer
    :parser: myst
    :summary:
    ```
````

### API

`````{py:class} Initializer
:canonical: kgate.initializers.Initializer

```{autodoc2-docstring} kgate.initializers.Initializer
:parser: myst
```

````{py:method} initialize_embedding(sample_count: int, embedding_dimensions: int, device: torch.device | str) -> torch.nn.Parameter
:canonical: kgate.initializers.Initializer.initialize_embedding

```{autodoc2-docstring} kgate.initializers.Initializer.initialize_embedding
:parser: myst
```

````

````{py:method} initialize_all_embeddings(knowledge_graph: kgate.KnowledgeGraph, *, node_embedding_dimensions: int, edge_embedding_dimensions: int, device: torch.device | str = 'cpu', inplace: bool = False) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Parameter] | None
:canonical: kgate.initializers.Initializer.initialize_all_embeddings

```{autodoc2-docstring} kgate.initializers.Initializer.initialize_all_embeddings
:parser: myst
```

````

`````

`````{py:class} FeatureInitializer(node_features: typing.Dict[str, pandas.DataFrame], edge_features: pandas.DataFrame)
:canonical: kgate.initializers.FeatureInitializer

Bases: {py:obj}`kgate.initializers.Initializer`

```{autodoc2-docstring} kgate.initializers.FeatureInitializer
:parser: myst
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.initializers.FeatureInitializer.__init__
:parser: myst
```

````{py:method} initialize_embeddings(features: torch.Tensor, knowledge_graph: kgate.KnowledgeGraph, node_type: str, device: torch.device | str) -> torch.nn.Parameter
:canonical: kgate.initializers.FeatureInitializer.initialize_embeddings

```{autodoc2-docstring} kgate.initializers.FeatureInitializer.initialize_embeddings
:parser: myst
```

````

````{py:method} initialize_all_embeddings(knowledge_graph: kgate.KnowledgeGraph, *, node_embedding_dimensions: int, edge_embedding_dimensions: int, device: torch.device | str = 'cpu', inplace: bool = False) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Parameter] | None
:canonical: kgate.initializers.FeatureInitializer.initialize_all_embeddings

```{autodoc2-docstring} kgate.initializers.FeatureInitializer.initialize_all_embeddings
:parser: myst
```

````

`````

`````{py:class} Node2VecInitializer(edge_indices: torch.Tensor, embedding_dimensions: int, walk_length: int, context_size: int, output_directory: pathlib.Path, device: torch.device | str = 'cuda', **node2vec_kwargs)
:canonical: kgate.initializers.Node2VecInitializer

Bases: {py:obj}`kgate.initializers.Initializer`

```{autodoc2-docstring} kgate.initializers.Node2VecInitializer
:parser: myst
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.initializers.Node2VecInitializer.__init__
:parser: myst
```

````{py:method} generate_all_embeddings(knowledge_graph: kgate.KnowledgeGraph, *, device: torch.device | str = 'cpu', inplace: bool = False, **_) -> typing.Any | None
:canonical: kgate.initializers.Node2VecInitializer.generate_all_embeddings

```{autodoc2-docstring} kgate.initializers.Node2VecInitializer.generate_all_embeddings
:parser: myst
```

````

`````
