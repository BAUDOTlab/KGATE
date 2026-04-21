# {py:mod}`kgate.decoders.bilinear`

```{py:module} kgate.decoders.bilinear
```

```{autodoc2-docstring} kgate.decoders.bilinear
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BilinearDecoder <kgate.decoders.bilinear.BilinearDecoder>`
  - ```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`RESCAL <kgate.decoders.bilinear.RESCAL>`
  - ```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`DistMult <kgate.decoders.bilinear.DistMult>`
  - ```{autodoc2-docstring} kgate.decoders.bilinear.DistMult
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`ComplEx <kgate.decoders.bilinear.ComplEx>`
  - ```{autodoc2-docstring} kgate.decoders.bilinear.ComplEx
    :parser: docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} BilinearDecoder()
:canonical: kgate.decoders.bilinear.BilinearDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.bilinear.BilinearDecoder.score
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding] | None
:canonical: kgate.decoders.bilinear.BilinearDecoder.normalize_parameters

```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> typing.Dict[str, torch.Tensor] | None
:canonical: kgate.decoders.bilinear.BilinearDecoder.get_embeddings

```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor | typing.Tuple, torch.Tensor | typing.Tuple, torch.Tensor | typing.Tuple, torch.Tensor | typing.Tuple]
:canonical: kgate.decoders.bilinear.BilinearDecoder.inference_prepare_candidates
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} inference_score(*, head_embeddings: torch.Tensor | typing.Tuple, tail_embeddings: torch.Tensor | typing.Tuple, edge_embeddings: torch.Tensor | typing.Tuple) -> torch.Tensor
:canonical: kgate.decoders.bilinear.BilinearDecoder.inference_score
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.bilinear.BilinearDecoder.inference_score
:parser: docstrings_parser
```

````

`````

`````{py:class} RESCAL(node_count: int, edge_count: int, embedding_dimensions: int)
:canonical: kgate.decoders.bilinear.RESCAL

Bases: {py:obj}`kgate.decoders.bilinear.BilinearDecoder`

```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_indices: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.bilinear.RESCAL.score

```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding]
:canonical: kgate.decoders.bilinear.RESCAL.normalize_parameters

```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> typing.Dict[str, torch.Tensor]
:canonical: kgate.decoders.bilinear.RESCAL.get_embeddings

```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.bilinear.RESCAL.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} inference_score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.bilinear.RESCAL.inference_score

```{autodoc2-docstring} kgate.decoders.bilinear.RESCAL.inference_score
:parser: docstrings_parser
```

````

`````

`````{py:class} DistMult(node_count: int, edge_count: int, embedding_dimensions: int)
:canonical: kgate.decoders.bilinear.DistMult

Bases: {py:obj}`kgate.decoders.bilinear.BilinearDecoder`

```{autodoc2-docstring} kgate.decoders.bilinear.DistMult
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.bilinear.DistMult.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.bilinear.DistMult.score

```{autodoc2-docstring} kgate.decoders.bilinear.DistMult.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding]
:canonical: kgate.decoders.bilinear.DistMult.normalize_parameters

```{autodoc2-docstring} kgate.decoders.bilinear.DistMult.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.bilinear.DistMult.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.bilinear.DistMult.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} inference_score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.bilinear.DistMult.inference_score

```{autodoc2-docstring} kgate.decoders.bilinear.DistMult.inference_score
:parser: docstrings_parser
```

````

`````

`````{py:class} ComplEx(embedding_dimensions: int)
:canonical: kgate.decoders.bilinear.ComplEx

Bases: {py:obj}`kgate.decoders.bilinear.BilinearDecoder`

```{autodoc2-docstring} kgate.decoders.bilinear.ComplEx
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.bilinear.ComplEx.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.bilinear.ComplEx.score

```{autodoc2-docstring} kgate.decoders.bilinear.ComplEx.score
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: kgate.decoders.bilinear.ComplEx.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.bilinear.ComplEx.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} inference_score(*, head_embeddings: typing.Tuple[torch.Tensor, torch.Tensor], tail_embeddings: typing.Tuple[torch.Tensor, torch.Tensor], edge_embeddings: typing.Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor
:canonical: kgate.decoders.bilinear.ComplEx.inference_score

```{autodoc2-docstring} kgate.decoders.bilinear.ComplEx.inference_score
:parser: docstrings_parser
```

````

`````
