# {py:mod}`kgate.encoders`

```{py:module} kgate.encoders
```

```{autodoc2-docstring} kgate.encoders
:parser: myst
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GNN <kgate.encoders.GNN>`
  -
* - {py:obj}`GATEncoder <kgate.encoders.GATEncoder>`
  -
* - {py:obj}`GCNEncoder <kgate.encoders.GCNEncoder>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logging_level <kgate.encoders.logging_level>`
  - ```{autodoc2-docstring} kgate.encoders.logging_level
    :parser: myst
    :summary:
    ```
````

### API

````{py:data} logging_level
:canonical: kgate.encoders.logging_level
:value: >
   None

```{autodoc2-docstring} kgate.encoders.logging_level
:parser: myst
```

````

`````{py:class} GNN(edge_types: typing.List[typing.Tuple[str, str, str]], aggregation: typing.Literal[sum, mean, min, max, cat, None] = 'sum')
:canonical: kgate.encoders.GNN

Bases: {py:obj}`torch.nn.Module`

````{py:method} forward(x_dict: typing.Dict[str, torch.Tensor], edge_index_dict: typing.Dict[typing.Tuple[str, str, str], torch.Tensor]) -> typing.Dict[str, torch.Tensor]
:canonical: kgate.encoders.GNN.forward

```{autodoc2-docstring} kgate.encoders.GNN.forward
:parser: myst
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
