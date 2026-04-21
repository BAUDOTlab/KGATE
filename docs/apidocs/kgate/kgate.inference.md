# {py:mod}`kgate.inference`

```{py:module} kgate.inference
```

```{autodoc2-docstring} kgate.inference
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Inference_KG <kgate.inference.Inference_KG>`
  - ```{autodoc2-docstring} kgate.inference.Inference_KG
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`EdgeInference <kgate.inference.EdgeInference>`
  - ```{autodoc2-docstring} kgate.inference.EdgeInference
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`NodeInference <kgate.inference.NodeInference>`
  - ```{autodoc2-docstring} kgate.inference.NodeInference
    :parser: docstrings_parser
    :summary:
    ```
````

### API

`````{py:class} Inference_KG(first_index_tensor: torch.Tensor, second_index_tensor: torch.Tensor)
:canonical: kgate.inference.Inference_KG

Bases: {py:obj}`torch.utils.data.Dataset`

```{autodoc2-docstring} kgate.inference.Inference_KG
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.inference.Inference_KG.__init__
:parser: docstrings_parser
```

````{py:method} __len__()
:canonical: kgate.inference.Inference_KG.__len__

```{autodoc2-docstring} kgate.inference.Inference_KG.__len__
:parser: docstrings_parser
```

````

````{py:method} __getitem__(index: int)
:canonical: kgate.inference.Inference_KG.__getitem__

```{autodoc2-docstring} kgate.inference.Inference_KG.__getitem__
:parser: docstrings_parser
```

````

`````

`````{py:class} EdgeInference(kg: kgate.knowledgegraph.KnowledgeGraph)
:canonical: kgate.inference.EdgeInference

```{autodoc2-docstring} kgate.inference.EdgeInference
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.inference.EdgeInference.__init__
:parser: docstrings_parser
```

````{py:method} evaluate(head_indices: torch.Tensor, tail_indices: torch.Tensor, *, top_k: int, batch_size: int, encoder: kgate.encoders.DefaultEncoder | kgate.encoders.GNN, decoder: kgate.decoders.TranslationalDecoder | kgate.decoders.BilinearDecoder | kgate.decoders.ConvolutionalDecoder, node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding, verbose: bool = True, **_)
:canonical: kgate.inference.EdgeInference.evaluate

```{autodoc2-docstring} kgate.inference.EdgeInference.evaluate
:parser: docstrings_parser
```

````

`````

`````{py:class} NodeInference(kg: kgate.knowledgegraph.KnowledgeGraph)
:canonical: kgate.inference.NodeInference

```{autodoc2-docstring} kgate.inference.NodeInference
:parser: docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} kgate.inference.NodeInference.__init__
:parser: docstrings_parser
```

````{py:method} evaluate(node_indices: torch.Tensor, edge_indices: torch.Tensor, *, top_k: int, missing_triplet_part: typing.Literal[head, tail], batch_size: int, encoder: kgate.encoders.DefaultEncoder | kgate.encoders.GNN, decoder: kgate.decoders.TranslationalDecoder | kgate.decoders.BilinearDecoder | kgate.decoders.ConvolutionalDecoder, node_embeddings: torch.nn.ParameterList, edge_embeddings: torch.nn.Embedding, verbose: bool = True, **_)
:canonical: kgate.inference.NodeInference.evaluate

```{autodoc2-docstring} kgate.inference.NodeInference.evaluate
:parser: docstrings_parser
```

````

`````
