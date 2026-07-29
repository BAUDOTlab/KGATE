# {py:mod}`kgate.encoders`

```{py:module} kgate.encoders
```

```{autodoc2-docstring} kgate.encoders
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DefaultEncoder <kgate.encoders.DefaultEncoder>`
  -
* - {py:obj}`GNN <kgate.encoders.GNN>`
  -
* - {py:obj}`GATEncoder <kgate.encoders.GATEncoder>`
  -
* - {py:obj}`GCNEncoder <kgate.encoders.GCNEncoder>`
  -
* - {py:obj}`Node2VecEncoder <kgate.encoders.Node2VecEncoder>`
  - ```{autodoc2-docstring} kgate.encoders.Node2VecEncoder
    :parser: docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logging_level <kgate.encoders.logging_level>`
  - ```{autodoc2-docstring} kgate.encoders.logging_level
    :parser: docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logging_level
:canonical: kgate.encoders.logging_level
:value: >
   None

```{autodoc2-docstring} kgate.encoders.logging_level
:parser: docstrings_parser
```

````

```{py:class} DefaultEncoder()
:canonical: kgate.encoders.DefaultEncoder

Bases: {py:obj}`torch.nn.Module`

```

`````{py:class} GNN(edge_types: typing.List[typing.Tuple[str, str, str]], aggregation: typing.Literal[sum, mean, min, max, cat, None] = 'sum')
:canonical: kgate.encoders.GNN

Bases: {py:obj}`torch.nn.Module`

````{py:method} forward(x_dict: typing.Dict[str, torch.Tensor], edge_index_dict: typing.Dict[typing.Tuple[str, str, str], torch.Tensor]) -> typing.Dict[str, torch.Tensor]
:canonical: kgate.encoders.GNN.forward

```{autodoc2-docstring} kgate.encoders.GNN.forward
:parser: docstrings_parser
```

````

`````

```{py:class} GATEncoder(edge_types: typing.List[typing.Tuple[str, str, str]], embedding_dimensions: int, gat_layer_count: int = 2, aggregation: typing.Literal[sum, mean, min, max, cat, None] = 'sum', device: torch.device | typing.Literal[cuda, cpu] = 'cuda')
:canonical: kgate.encoders.GATEncoder

Bases: {py:obj}`kgate.encoders.GNN`

```

```{py:class} GCNEncoder(edge_types: typing.List[typing.Tuple[str, str, str]], embedding_dimensions: int, gcn_layer_count: int = 2, aggregation: typing.Literal[sum, mean, min, max, cat, None] = 'sum', device: torch.device | typing.Literal[cuda, cpu] = 'cuda')
:canonical: kgate.encoders.GCNEncoder

Bases: {py:obj}`kgate.encoders.GNN`

```

`````{py:class} Node2VecEncoder(edge_indices: torch.Tensor, embedding_dimensions: int, walk_length: int, context_size: int, output_directory: pathlib.Path, device: torch.device | typing.Literal[cuda, cpu] = 'cuda', **node2vec_kwargs)
:canonical: kgate.encoders.Node2VecEncoder

```{autodoc2-docstring} kgate.encoders.Node2VecEncoder
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.encoders.Node2VecEncoder.__init__
:parser: docstrings_parser
```

````{py:method} generate_embeddings()
:canonical: kgate.encoders.Node2VecEncoder.generate_embeddings

```{autodoc2-docstring} kgate.encoders.Node2VecEncoder.generate_embeddings
:parser: docstrings_parser
```

````

`````
