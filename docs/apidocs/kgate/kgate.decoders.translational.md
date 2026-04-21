# {py:mod}`kgate.decoders.translational`

```{py:module} kgate.decoders.translational
```

```{autodoc2-docstring} kgate.decoders.translational
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TranslationalDecoder <kgate.decoders.translational.TranslationalDecoder>`
  - ```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`TransE <kgate.decoders.translational.TransE>`
  - ```{autodoc2-docstring} kgate.decoders.translational.TransE
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`TransH <kgate.decoders.translational.TransH>`
  - ```{autodoc2-docstring} kgate.decoders.translational.TransH
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`TransR <kgate.decoders.translational.TransR>`
  - ```{autodoc2-docstring} kgate.decoders.translational.TransR
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`TransD <kgate.decoders.translational.TransD>`
  - ```{autodoc2-docstring} kgate.decoders.translational.TransD
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`TorusE <kgate.decoders.translational.TorusE>`
  - ```{autodoc2-docstring} kgate.decoders.translational.TorusE
    :parser: docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} TranslationalDecoder()
:canonical: kgate.decoders.translational.TranslationalDecoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.translational.TranslationalDecoder.score
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding] | None
:canonical: kgate.decoders.translational.TranslationalDecoder.normalize_parameters

```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> typing.Dict[str, torch.Tensor] | None
:canonical: kgate.decoders.translational.TranslationalDecoder.get_embeddings

```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.translational.TranslationalDecoder.inference_prepare_candidates
:abstractmethod:

```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} inference_score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.translational.TranslationalDecoder.inference_score

```{autodoc2-docstring} kgate.decoders.translational.TranslationalDecoder.inference_score
:parser: docstrings_parser
```

````

`````

`````{py:class} TransE(dissimilarity_type: typing.Literal[L1, L2] = 'L2')
:canonical: kgate.decoders.translational.TransE

Bases: {py:obj}`kgate.decoders.translational.TranslationalDecoder`

```{autodoc2-docstring} kgate.decoders.translational.TransE
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.translational.TransE.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.translational.TransE.score

```{autodoc2-docstring} kgate.decoders.translational.TransE.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding]
:canonical: kgate.decoders.translational.TransE.normalize_parameters

```{autodoc2-docstring} kgate.decoders.translational.TransE.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.translational.TransE.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.translational.TransE.inference_prepare_candidates
:parser: docstrings_parser
```

````

`````

`````{py:class} TransH(embedding_dimensions: int, node_count: int, edge_count: int)
:canonical: kgate.decoders.translational.TransH

Bases: {py:obj}`kgate.decoders.translational.TranslationalDecoder`

```{autodoc2-docstring} kgate.decoders.translational.TransH
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.translational.TransH.__init__
:parser: docstrings_parser
```

````{py:method} project(nodes: torch.Tensor, normal_vector: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.translational.TransH.project
:staticmethod:

```{autodoc2-docstring} kgate.decoders.translational.TransH.project
:parser: docstrings_parser
```

````

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, edge_indices: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.translational.TransH.score

```{autodoc2-docstring} kgate.decoders.translational.TransH.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding]
:canonical: kgate.decoders.translational.TransH.normalize_parameters

```{autodoc2-docstring} kgate.decoders.translational.TransH.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> typing.Dict[str, torch.Tensor]
:canonical: kgate.decoders.translational.TransH.get_embeddings

```{autodoc2-docstring} kgate.decoders.translational.TransH.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.translational.TransH.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.translational.TransH.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} evaluate_projections(node_embeddings: torch.Tensor)
:canonical: kgate.decoders.translational.TransH.evaluate_projections

```{autodoc2-docstring} kgate.decoders.translational.TransH.evaluate_projections
:parser: docstrings_parser
```

````

`````

`````{py:class} TransR(node_count: int, edge_count: int, node_embedding_dimensions: int, edge_embedding_dimensions: int)
:canonical: kgate.decoders.translational.TransR

Bases: {py:obj}`kgate.decoders.translational.TranslationalDecoder`

```{autodoc2-docstring} kgate.decoders.translational.TransR
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.translational.TransR.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, edge_indices: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.translational.TransR.score

```{autodoc2-docstring} kgate.decoders.translational.TransR.score
:parser: docstrings_parser
```

````

````{py:method} project(nodes: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.translational.TransR.project

```{autodoc2-docstring} kgate.decoders.translational.TransR.project
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding]
:canonical: kgate.decoders.translational.TransR.normalize_parameters

```{autodoc2-docstring} kgate.decoders.translational.TransR.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> typing.Dict[str, torch.Tensor]
:canonical: kgate.decoders.translational.TransR.get_embeddings

```{autodoc2-docstring} kgate.decoders.translational.TransR.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.translational.TransR.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.translational.TransR.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} evaluate_projections(node_embeddings: torch.Tensor)
:canonical: kgate.decoders.translational.TransR.evaluate_projections

```{autodoc2-docstring} kgate.decoders.translational.TransR.evaluate_projections
:parser: docstrings_parser
```

````

`````

`````{py:class} TransD(node_count: int, edge_count: int, node_embedding_dimensions: int, edge_embedding_dimensions: int)
:canonical: kgate.decoders.translational.TransD

Bases: {py:obj}`kgate.decoders.translational.TranslationalDecoder`

```{autodoc2-docstring} kgate.decoders.translational.TransD
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.translational.TransD.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.translational.TransD.score

```{autodoc2-docstring} kgate.decoders.translational.TransD.score
:parser: docstrings_parser
```

````

````{py:method} project(nodes: torch.Tensor, node_projection_vector: torch.Tensor, edge_projection_vector: torch.Tensor) -> torch.Tensor
:canonical: kgate.decoders.translational.TransD.project

```{autodoc2-docstring} kgate.decoders.translational.TransD.project
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding]
:canonical: kgate.decoders.translational.TransD.normalize_parameters

```{autodoc2-docstring} kgate.decoders.translational.TransD.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} get_embeddings() -> typing.Dict[str, torch.Tensor]
:canonical: kgate.decoders.translational.TransD.get_embeddings

```{autodoc2-docstring} kgate.decoders.translational.TransD.get_embeddings
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.translational.TransD.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.translational.TransD.inference_prepare_candidates
:parser: docstrings_parser
```

````

````{py:method} evaluate_projections(node_embeddings: torch.Tensor)
:canonical: kgate.decoders.translational.TransD.evaluate_projections

```{autodoc2-docstring} kgate.decoders.translational.TransD.evaluate_projections
:parser: docstrings_parser
```

````

`````

`````{py:class} TorusE(dissimilarity_type: typing.Literal[L1, torus_L1, torus_L2, torus_eL2])
:canonical: kgate.decoders.translational.TorusE

Bases: {py:obj}`kgate.decoders.translational.TranslationalDecoder`

```{autodoc2-docstring} kgate.decoders.translational.TorusE
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.decoders.translational.TorusE.__init__
:parser: docstrings_parser
```

````{py:method} score(*, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, **_) -> torch.Tensor
:canonical: kgate.decoders.translational.TorusE.score

```{autodoc2-docstring} kgate.decoders.translational.TorusE.score
:parser: docstrings_parser
```

````

````{py:method} normalize_parameters(node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding) -> typing.Tuple[torch.nn.ParameterList, torch.nn.Embedding]
:canonical: kgate.decoders.translational.TorusE.normalize_parameters

```{autodoc2-docstring} kgate.decoders.translational.TorusE.normalize_parameters
:parser: docstrings_parser
```

````

````{py:method} inference_prepare_candidates(*, node_embeddings: torch.Tensor, edge_embeddings: torch.nn.Embedding, head_indices: torch.Tensor, tail_indices: torch.Tensor, edge_indices: torch.Tensor, node_inference: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.decoders.translational.TorusE.inference_prepare_candidates

```{autodoc2-docstring} kgate.decoders.translational.TorusE.inference_prepare_candidates
:parser: docstrings_parser
```

````

`````
