# {py:mod}`kgate.knowledgegraph`

```{py:module} kgate.knowledgegraph
```

```{autodoc2-docstring} kgate.knowledgegraph
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EncoderInput <kgate.knowledgegraph.EncoderInput>`
  - ```{autodoc2-docstring} kgate.knowledgegraph.EncoderInput
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`KnowledgeGraph <kgate.knowledgegraph.KnowledgeGraph>`
  -
````

### API

`````{py:class} EncoderInput(x_dict: typing.Dict[str, torch.Tensor], edge_list: typing.Dict[typing.Tuple[str, str, str], torch.Tensor], mapping: typing.Dict[str, torch.Tensor])
:canonical: kgate.knowledgegraph.EncoderInput

```{autodoc2-docstring} kgate.knowledgegraph.EncoderInput
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.knowledgegraph.EncoderInput.__init__
:parser: docstrings_parser
```

````{py:method} __repr__()
:canonical: kgate.knowledgegraph.EncoderInput.__repr__

````

`````

`````{py:class} KnowledgeGraph(dataframe: pandas.DataFrame | None = None, graphindices: torch.Tensor | None = None, metadata: pandas.DataFrame | None = None, triplet_types: typing.List[typing.Tuple[str, str, str]] | None = None, node_to_index: typing.Dict[str, int] | None = None, edge_to_index: typing.Dict[str, int] | None = None, node_type_to_index: typing.Dict[str, int] | None = None, removed_triplets: torch.Tensor | None = None)
:canonical: kgate.knowledgegraph.KnowledgeGraph

Bases: {py:obj}`torch.utils.data.Dataset`

````{py:method} __len__()
:canonical: kgate.knowledgegraph.KnowledgeGraph.__len__

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.__len__
:parser: docstrings_parser
```

````

````{py:method} __getitem__(index) -> torch.Tensor
:canonical: kgate.knowledgegraph.KnowledgeGraph.__getitem__

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.__getitem__
:parser: docstrings_parser
```

````

````{py:property} head_idx
:canonical: kgate.knowledgegraph.KnowledgeGraph.head_idx
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.head_idx
:parser: docstrings_parser
```

````

````{py:property} tail_idx
:canonical: kgate.knowledgegraph.KnowledgeGraph.tail_idx
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.tail_idx
:parser: docstrings_parser
```

````

````{py:property} relations
:canonical: kgate.knowledgegraph.KnowledgeGraph.relations
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.relations
:parser: docstrings_parser
```

````

````{py:property} head_indices
:canonical: kgate.knowledgegraph.KnowledgeGraph.head_indices
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.head_indices
:parser: docstrings_parser
```

````

````{py:property} tail_indices
:canonical: kgate.knowledgegraph.KnowledgeGraph.tail_indices
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.tail_indices
:parser: docstrings_parser
```

````

````{py:property} edge_indices
:canonical: kgate.knowledgegraph.KnowledgeGraph.edge_indices
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.edge_indices
:parser: docstrings_parser
```

````

````{py:property} triplets
:canonical: kgate.knowledgegraph.KnowledgeGraph.triplets
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.triplets
:parser: docstrings_parser
```

````

````{py:property} edge_list
:canonical: kgate.knowledgegraph.KnowledgeGraph.edge_list
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.edge_list
:parser: docstrings_parser
```

````

````{py:property} n_facts
:canonical: kgate.knowledgegraph.KnowledgeGraph.n_facts
:type: int

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.n_facts
:parser: docstrings_parser
```

````

````{py:property} identity
:canonical: kgate.knowledgegraph.KnowledgeGraph.identity
:type: pandas.DataFrame

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.identity
:parser: docstrings_parser
```

````

````{py:method} set_identity(new_identity: str)
:canonical: kgate.knowledgegraph.KnowledgeGraph.set_identity

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.set_identity
:parser: docstrings_parser
```

````

````{py:method} add_metadata(metadata: pandas.DataFrame)
:canonical: kgate.knowledgegraph.KnowledgeGraph.add_metadata

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.add_metadata
:parser: docstrings_parser
```

````

````{py:method} get_dataframe()
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_dataframe

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_dataframe
:parser: docstrings_parser
```

````

````{py:method} split_kg(split_proportions: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1), sizes: typing.Tuple[int, int, int] | None = None) -> typing.Tuple[typing.Self, typing.Self, typing.Self]
:canonical: kgate.knowledgegraph.KnowledgeGraph.split_kg

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.split_kg
:parser: docstrings_parser
```

````

````{py:method} get_mask(split_proportions: typing.Tuple[float, float, float]) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_mask

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_mask
:parser: docstrings_parser
```

````

````{py:method} keep_triplets(indices_to_keep: typing.List[int] | torch.Tensor) -> typing.Self
:canonical: kgate.knowledgegraph.KnowledgeGraph.keep_triplets

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.keep_triplets
:parser: docstrings_parser
```

````

````{py:method} remove_triplets(indices_to_remove: typing.List[int] | torch.Tensor) -> typing.Self
:canonical: kgate.knowledgegraph.KnowledgeGraph.remove_triplets

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.remove_triplets
:parser: docstrings_parser
```

````

````{py:method} add_triplets(new_triplets: torch.Tensor) -> typing.Self
:canonical: kgate.knowledgegraph.KnowledgeGraph.add_triplets

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.add_triplets
:parser: docstrings_parser
```

````

````{py:method} add_reverse_edges(undirected_edges: typing.List[int]) -> typing.Tuple[typing.Self, typing.List[int]]
:canonical: kgate.knowledgegraph.KnowledgeGraph.add_reverse_edges

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.add_reverse_edges
:parser: docstrings_parser
```

````

````{py:method} remove_duplicate_triplets() -> typing.Self
:canonical: kgate.knowledgegraph.KnowledgeGraph.remove_duplicate_triplets

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.remove_duplicate_triplets
:parser: docstrings_parser
```

````

````{py:method} get_pairs(edge_type_index: int, type: typing.Literal[head_tail, tail_head] = 'head_tail') -> typing.Set[typing.Tuple[torch.types.Number, torch.types.Number]]
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_pairs

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_pairs
:parser: docstrings_parser
```

````

````{py:method} duplicates(theta_first_edge_type: float = 0.8, theta_second_edge_type: float = 0.8, reverse_edges_list: typing.List[int] | None = None) -> typing.Tuple[typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]]]
:canonical: kgate.knowledgegraph.KnowledgeGraph.duplicates

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.duplicates
:parser: docstrings_parser
```

````

````{py:method} cartesian_product_edges(theta: float = 0.8) -> typing.List[int]
:canonical: kgate.knowledgegraph.KnowledgeGraph.cartesian_product_edges

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.cartesian_product_edges
:parser: docstrings_parser
```

````

````{py:method} get_encoder_input(data: torch.Tensor, node_embedding: torch.nn.ParameterList) -> kgate.knowledgegraph.EncoderInput
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_encoder_input

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_encoder_input
:parser: docstrings_parser
```

````

````{py:method} flatten_embeddings(node_embeddings: torch.nn.ParameterList) -> torch.Tensor
:canonical: kgate.knowledgegraph.KnowledgeGraph.flatten_embeddings

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.flatten_embeddings
:parser: docstrings_parser
```

````

````{py:method} clean()
:canonical: kgate.knowledgegraph.KnowledgeGraph.clean

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.clean
:parser: docstrings_parser
```

````

````{py:method} from_hetero_data(hetero_data: torch_geometric.data.HeteroData)
:canonical: kgate.knowledgegraph.KnowledgeGraph.from_hetero_data
:staticmethod:

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.from_hetero_data
:parser: docstrings_parser
```

````

````{py:method} from_torchkge(torchkge_kg: torchkge.KnowledgeGraph, metadata: pandas.DataFrame | None = None) -> typing.Self
:canonical: kgate.knowledgegraph.KnowledgeGraph.from_torchkge
:staticmethod:

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.from_torchkge
:parser: docstrings_parser
```

````

`````
