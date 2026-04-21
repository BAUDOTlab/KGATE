# {py:mod}`kgate.samplers`

```{py:module} kgate.samplers
```

```{autodoc2-docstring} kgate.samplers
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NegativeSampler <kgate.samplers.NegativeSampler>`
  - ```{autodoc2-docstring} kgate.samplers.NegativeSampler
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`UniformNegativeSampler <kgate.samplers.UniformNegativeSampler>`
  - ```{autodoc2-docstring} kgate.samplers.UniformNegativeSampler
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`BernoulliNegativeSampler <kgate.samplers.BernoulliNegativeSampler>`
  - ```{autodoc2-docstring} kgate.samplers.BernoulliNegativeSampler
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`PositionalNegativeSampler <kgate.samplers.PositionalNegativeSampler>`
  - ```{autodoc2-docstring} kgate.samplers.PositionalNegativeSampler
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`MixedNegativeSampler <kgate.samplers.MixedNegativeSampler>`
  - ```{autodoc2-docstring} kgate.samplers.MixedNegativeSampler
    :parser: docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} NegativeSampler
:canonical: kgate.samplers.NegativeSampler

```{autodoc2-docstring} kgate.samplers.NegativeSampler
:parser: docstrings_parser
```

````{py:method} corrupt_batch(batch: torch.Tensor, negative_triplet_count=None) -> torch.types.Tensor
:canonical: kgate.samplers.NegativeSampler.corrupt_batch
:abstractmethod:

```{autodoc2-docstring} kgate.samplers.NegativeSampler.corrupt_batch
:parser: docstrings_parser
```

````

`````

`````{py:class} UniformNegativeSampler(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, negative_triplet_count=1)
:canonical: kgate.samplers.UniformNegativeSampler

Bases: {py:obj}`kgate.samplers.NegativeSampler`

```{autodoc2-docstring} kgate.samplers.UniformNegativeSampler
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.samplers.UniformNegativeSampler.__init__
:parser: docstrings_parser
```

````{py:method} corrupt_batch(batch: torch.Tensor, negative_triplet_count=None) -> torch.types.Tensor
:canonical: kgate.samplers.UniformNegativeSampler.corrupt_batch

```{autodoc2-docstring} kgate.samplers.UniformNegativeSampler.corrupt_batch
:parser: docstrings_parser
```

````

`````

`````{py:class} BernoulliNegativeSampler(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, negative_triplet_count=1)
:canonical: kgate.samplers.BernoulliNegativeSampler

Bases: {py:obj}`kgate.samplers.NegativeSampler`

```{autodoc2-docstring} kgate.samplers.BernoulliNegativeSampler
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.samplers.BernoulliNegativeSampler.__init__
:parser: docstrings_parser
```

````{py:method} evaluate_bernoulli_probabilities() -> torch.Tensor
:canonical: kgate.samplers.BernoulliNegativeSampler.evaluate_bernoulli_probabilities

```{autodoc2-docstring} kgate.samplers.BernoulliNegativeSampler.evaluate_bernoulli_probabilities
:parser: docstrings_parser
```

````

````{py:method} corrupt_batch(batch: torch.LongTensor, negative_triplet_count: int | None = None)
:canonical: kgate.samplers.BernoulliNegativeSampler.corrupt_batch

```{autodoc2-docstring} kgate.samplers.BernoulliNegativeSampler.corrupt_batch
:parser: docstrings_parser
```

````

`````

`````{py:class} PositionalNegativeSampler(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph)
:canonical: kgate.samplers.PositionalNegativeSampler

Bases: {py:obj}`kgate.samplers.BernoulliNegativeSampler`

```{autodoc2-docstring} kgate.samplers.PositionalNegativeSampler
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.samplers.PositionalNegativeSampler.__init__
:parser: docstrings_parser
```

````{py:method} find_possibilities() -> typing.Tuple[typing.Dict[int, typing.List[int]], typing.Dict[int, typing.List[int]], torch.types.Tensor, torch.types.Tensor]
:canonical: kgate.samplers.PositionalNegativeSampler.find_possibilities

```{autodoc2-docstring} kgate.samplers.PositionalNegativeSampler.find_possibilities
:parser: docstrings_parser
```

````

````{py:method} corrupt_batch(batch: torch.types.Tensor, _: int = 1) -> torch.types.Tensor
:canonical: kgate.samplers.PositionalNegativeSampler.corrupt_batch

```{autodoc2-docstring} kgate.samplers.PositionalNegativeSampler.corrupt_batch
:parser: docstrings_parser
```

````

`````

`````{py:class} MixedNegativeSampler(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, negative_triplet_count=1)
:canonical: kgate.samplers.MixedNegativeSampler

Bases: {py:obj}`kgate.samplers.NegativeSampler`

```{autodoc2-docstring} kgate.samplers.MixedNegativeSampler
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.samplers.MixedNegativeSampler.__init__
:parser: docstrings_parser
```

````{py:method} corrupt_batch(batch: torch.LongTensor, negative_triplet_count: int = 1)
:canonical: kgate.samplers.MixedNegativeSampler.corrupt_batch

```{autodoc2-docstring} kgate.samplers.MixedNegativeSampler.corrupt_batch
:parser: docstrings_parser
```

````

`````
