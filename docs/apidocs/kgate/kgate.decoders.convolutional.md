# {py:mod}`kgate.decoders.convolutional`

```{py:module} kgate.decoders.convolutional
```

```{autodoc2-docstring} kgate.decoders.convolutional
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConvolutionalDecoder <kgate.decoders.convolutional.ConvolutionalDecoder>`
  -
* - {py:obj}`ConvKB <kgate.decoders.convolutional.ConvKB>`
  -
````

### API

`````{py:class} ConvolutionalDecoder()
:canonical: kgate.decoders.convolutional.ConvolutionalDecoder

Bases: {py:obj}`torch.nn.Module`

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.convolutional.ConvolutionalDecoder.score
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.convolutional.ConvolutionalDecoder.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding] | None
:canonical: kgate.decoders.convolutional.ConvolutionalDecoder.normalize_parameters

```{autodoc2-docstring} kgate.decoders.convolutional.ConvolutionalDecoder.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> typing.Dict[str, torch.Tensor] | None
:canonical: kgate.decoders.convolutional.ConvolutionalDecoder.get_embeddings

```{autodoc2-docstring} kgate.decoders.convolutional.ConvolutionalDecoder.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.convolutional.ConvolutionalDecoder.inference_prepare_candidates
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.convolutional.ConvolutionalDecoder.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} inference_score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.convolutional.ConvolutionalDecoder.inference_score
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.convolutional.ConvolutionalDecoder.inference_score
:parser: docstrings_parser
```

````

`````

`````{py:class} ConvKB(node_count: int, edge_count: int, embedding_dimensions: int, filter_count: int)
:canonical: kgate.decoders.convolutional.ConvKB

Bases: {py:obj}`kgate.decoders.convolutional.ConvolutionalDecoder`

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.convolutional.ConvKB.score

```{autodoc2-docstring} kgate.decoders.convolutional.ConvKB.score
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.convolutional.ConvKB.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.convolutional.ConvKB.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} inference_score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.convolutional.ConvKB.inference_score

```{autodoc2-docstring} kgate.decoders.convolutional.ConvKB.inference_score
:parser: docstrings_parser
```

````

`````
