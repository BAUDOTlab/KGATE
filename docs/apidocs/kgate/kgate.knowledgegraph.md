# {py:mod}`kgate.knowledgegraph`

```{py:module} kgate.knowledgegraph
```

```{autodoc2-docstring} kgate.knowledgegraph
:parser: myst
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EncoderInput <kgate.knowledgegraph.EncoderInput>`
  - ```{autodoc2-docstring} kgate.knowledgegraph.EncoderInput
    :parser: myst
    :summary:
    ```
* - {py:obj}`KnowledgeGraphEmbeddings <kgate.knowledgegraph.KnowledgeGraphEmbeddings>`
  -
* - {py:obj}`KnowledgeGraph <kgate.knowledgegraph.KnowledgeGraph>`
  - ```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph
    :parser: myst
    :summary:
    ```
````

### API

`````{py:class} EncoderInput(x_dict: typing.Dict[str, torch.Tensor], edge_list: typing.Dict[typing.Tuple[str, str, str], torch.Tensor], node_mapping: typing.Dict[str, torch.Tensor], seed_mapping: typing.Dict[str, torch.Tensor])
:canonical: kgate.knowledgegraph.EncoderInput

```{autodoc2-docstring} kgate.knowledgegraph.EncoderInput
:parser: myst
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.knowledgegraph.EncoderInput.__init__
:parser: myst
```

````{py:method} __repr__()
:canonical: kgate.knowledgegraph.EncoderInput.__repr__

````

`````

`````{py:class} KnowledgeGraphEmbeddings(*args: typing.Any, **kwargs: typing.Any)
:canonical: kgate.knowledgegraph.KnowledgeGraphEmbeddings

Bases: {py:obj}`torch.nn.Module`

````{py:attribute} node_embeddings
:canonical: kgate.knowledgegraph.KnowledgeGraphEmbeddings.node_embeddings
:type: torch.nn.ParameterList
:value: >
   None

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraphEmbeddings.node_embeddings
:parser: myst
```

````

````{py:attribute} edge_embeddings
:canonical: kgate.knowledgegraph.KnowledgeGraphEmbeddings.edge_embeddings
:type: torch.nn.Parameter
:value: >
   None

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraphEmbeddings.edge_embeddings
:parser: myst
```

````

`````

`````{py:class} KnowledgeGraph(dataframe: pandas.DataFrame | None = None, graphindices: torch.Tensor | None = None, metadata: pandas.DataFrame | None = None, triplet_types: typing.List[typing.Tuple[str, str, str]] | None = None, node_to_index: typing.Dict[str, int] | None = None, edge_to_index: typing.Dict[str, int] | None = None, node_type_to_index: typing.Dict[str, int] | None = None)
:canonical: kgate.knowledgegraph.KnowledgeGraph

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph
:parser: myst
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.__init__
:parser: myst
```

````{py:method} __len__()
:canonical: kgate.knowledgegraph.KnowledgeGraph.__len__

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.__len__
:parser: myst
```

````

````{py:method} __getitem__(index) -> torch.Tensor
:canonical: kgate.knowledgegraph.KnowledgeGraph.__getitem__

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.__getitem__
:parser: myst
```

````

````{py:property} embeddings
:canonical: kgate.knowledgegraph.KnowledgeGraph.embeddings
:type: kgate.knowledgegraph.KnowledgeGraphEmbeddings

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.embeddings
:parser: myst
```

````

````{py:property} node_embeddings
:canonical: kgate.knowledgegraph.KnowledgeGraph.node_embeddings
:type: torch.nn.ParameterList

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.node_embeddings
:parser: myst
```

````

````{py:property} edge_embeddings
:canonical: kgate.knowledgegraph.KnowledgeGraph.edge_embeddings
:type: torch.nn.Parameter

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.edge_embeddings
:parser: myst
```

````

````{py:property} tail_idx
:canonical: kgate.knowledgegraph.KnowledgeGraph.tail_idx
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.tail_idx
:parser: myst
```

````

````{py:property} relations
:canonical: kgate.knowledgegraph.KnowledgeGraph.relations
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.relations
:parser: myst
```

````

````{py:property} head_indices
:canonical: kgate.knowledgegraph.KnowledgeGraph.head_indices
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.head_indices
:parser: myst
```

````

````{py:property} tail_indices
:canonical: kgate.knowledgegraph.KnowledgeGraph.tail_indices
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.tail_indices
:parser: myst
```

````

````{py:property} edge_indices
:canonical: kgate.knowledgegraph.KnowledgeGraph.edge_indices
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.edge_indices
:parser: myst
```

````

````{py:property} triplets
:canonical: kgate.knowledgegraph.KnowledgeGraph.triplets
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.triplets
:parser: myst
```

````

````{py:property} edge_list
:canonical: kgate.knowledgegraph.KnowledgeGraph.edge_list
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.edge_list
:parser: myst
```

````

````{py:property} train_set
:canonical: kgate.knowledgegraph.KnowledgeGraph.train_set
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.train_set
:parser: myst
```

````

````{py:property} validation_set
:canonical: kgate.knowledgegraph.KnowledgeGraph.validation_set
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.validation_set
:parser: myst
```

````

````{py:property} test_set
:canonical: kgate.knowledgegraph.KnowledgeGraph.test_set
:type: torch.Tensor

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.test_set
:parser: myst
```

````

````{py:property} triplet_count
:canonical: kgate.knowledgegraph.KnowledgeGraph.triplet_count
:type: int

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.triplet_count
:parser: myst
```

````

````{py:property} node_count
:canonical: kgate.knowledgegraph.KnowledgeGraph.node_count
:type: int

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.node_count
:parser: myst
```

````

````{py:property} edge_count
:canonical: kgate.knowledgegraph.KnowledgeGraph.edge_count
:type: int

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.edge_count
:parser: myst
```

````

````{py:property} identity
:canonical: kgate.knowledgegraph.KnowledgeGraph.identity
:type: pandas.DataFrame

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.identity
:parser: myst
```

````

````{py:method} set_identity(new_identity: str) -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.set_identity

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.set_identity
:parser: myst
```

````

````{py:method} add_metadata(metadata: pandas.DataFrame) -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.add_metadata

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.add_metadata
:parser: myst
```

````

````{py:method} get_dataframe(include_splits: bool = False) -> pandas.DataFrame
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_dataframe

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_dataframe
:parser: myst
```

````

````{py:method} generate_masks(split_proportions: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1), sizes: typing.Tuple[int, int, int] | None = None) -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.generate_masks

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.generate_masks
:parser: myst
```

````

````{py:method} get_mask(split_proportions: typing.Tuple[float, float, float]) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_mask

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_mask
:parser: myst
```

````

````{py:method} delete_triplets(indices_to_delete: typing.List[int] | torch.Tensor) -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.delete_triplets

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.delete_triplets
:parser: myst
```

````

````{py:method} remove_triplets_from_training(indices_to_remove: typing.List[int] | torch.Tensor) -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.remove_triplets_from_training

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.remove_triplets_from_training
:parser: myst
```

````

````{py:method} add_triplets(new_triplets: torch.Tensor, split: typing.Literal[train, validation, test] | None = None) -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.add_triplets

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.add_triplets
:parser: myst
```

````

````{py:method} add_reverse_edges(undirected_edges: typing.List[int]) -> typing.List[int]
:canonical: kgate.knowledgegraph.KnowledgeGraph.add_reverse_edges

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.add_reverse_edges
:parser: myst
```

````

````{py:method} remove_duplicate_triplets() -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.remove_duplicate_triplets

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.remove_duplicate_triplets
:parser: myst
```

````

````{py:method} get_pairs(edge_type_index: int, split: typing.Literal[train, validation, test] | None = None) -> torch.Tensor
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_pairs

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_pairs
:parser: myst
```

````

````{py:method} duplicates(theta_first_edge_type: float = 0.8, theta_second_edge_type: float = 0.8, reverse_edges_list: typing.List[int] | None = None) -> typing.Tuple[typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]]]
:canonical: kgate.knowledgegraph.KnowledgeGraph.duplicates

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.duplicates
:parser: myst
```

````

````{py:method} cartesian_product_edges(theta: float = 0.8) -> typing.List[int]
:canonical: kgate.knowledgegraph.KnowledgeGraph.cartesian_product_edges

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.cartesian_product_edges
:parser: myst
```

````

````{py:method} get_encoder_input(*, seed_nodes: torch.Tensor, hop_count: int, mask: torch.Tensor | None = None) -> kgate.knowledgegraph.EncoderInput
:canonical: kgate.knowledgegraph.KnowledgeGraph.get_encoder_input

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.get_encoder_input
:parser: myst
```

````

````{py:method} flatten_embeddings() -> torch.Tensor
:canonical: kgate.knowledgegraph.KnowledgeGraph.flatten_embeddings

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.flatten_embeddings
:parser: myst
```

````

````{py:method} clean() -> None
:canonical: kgate.knowledgegraph.KnowledgeGraph.clean

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.clean
:parser: myst
```

````

````{py:method} from_hetero_data(hetero_data: torch_geometric.data.HeteroData) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.knowledgegraph.KnowledgeGraph.from_hetero_data
:classmethod:

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.from_hetero_data
:parser: myst
```

````

````{py:method} from_torchkge(torchkge_kg: torchkge.KnowledgeGraph, metadata: pandas.DataFrame | None = None) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.knowledgegraph.KnowledgeGraph.from_torchkge
:classmethod:

```{autodoc2-docstring} kgate.knowledgegraph.KnowledgeGraph.from_torchkge
:parser: myst
```

````

`````
