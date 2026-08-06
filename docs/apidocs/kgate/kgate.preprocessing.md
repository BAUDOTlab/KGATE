# {py:mod}`kgate.preprocessing`

```{py:module} kgate.preprocessing
```

```{autodoc2-docstring} kgate.preprocessing
:parser: myst
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`prepare_knowledge_graph <kgate.preprocessing.prepare_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.prepare_knowledge_graph
    :parser: myst
    :summary:
    ```
* - {py:obj}`save_knowledge_graph <kgate.preprocessing.save_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.save_knowledge_graph
    :parser: myst
    :summary:
    ```
* - {py:obj}`load_knowledge_graph <kgate.preprocessing.load_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.load_knowledge_graph
    :parser: myst
    :summary:
    ```
* - {py:obj}`clean_knowledge_graph <kgate.preprocessing.clean_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.clean_knowledge_graph
    :parser: myst
    :summary:
    ```
* - {py:obj}`verify_node_coverage <kgate.preprocessing.verify_node_coverage>`
  - ```{autodoc2-docstring} kgate.preprocessing.verify_node_coverage
    :parser: myst
    :summary:
    ```
* - {py:obj}`ensure_node_coverage <kgate.preprocessing.ensure_node_coverage>`
  - ```{autodoc2-docstring} kgate.preprocessing.ensure_node_coverage
    :parser: myst
    :summary:
    ```
* - {py:obj}`clean_datasets <kgate.preprocessing.clean_datasets>`
  - ```{autodoc2-docstring} kgate.preprocessing.clean_datasets
    :parser: myst
    :summary:
    ```
* - {py:obj}`clean_cartesians <kgate.preprocessing.clean_cartesians>`
  - ```{autodoc2-docstring} kgate.preprocessing.clean_cartesians
    :parser: myst
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SUPPORTED_SEPARATORS <kgate.preprocessing.SUPPORTED_SEPARATORS>`
  - ```{autodoc2-docstring} kgate.preprocessing.SUPPORTED_SEPARATORS
    :parser: myst
    :summary:
    ```
````

### API

````{py:data} SUPPORTED_SEPARATORS
:canonical: kgate.preprocessing.SUPPORTED_SEPARATORS
:value: >
   [',', '\t', ';']

```{autodoc2-docstring} kgate.preprocessing.SUPPORTED_SEPARATORS
:parser: myst
```

````

````{py:function} prepare_knowledge_graph(config: dict, kg: kgate.knowledgegraph.KnowledgeGraph | None = None, dataframe: pandas.DataFrame | None = None, metadata: pandas.DataFrame | None = None) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.preprocessing.prepare_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.prepare_knowledge_graph
:parser: myst
```
````

````{py:function} save_knowledge_graph(config: dict, knowledge_graph: kgate.knowledgegraph.KnowledgeGraph)
:canonical: kgate.preprocessing.save_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.save_knowledge_graph
:parser: myst
```
````

````{py:function} load_knowledge_graph(pickle_filename: pathlib.Path) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.preprocessing.load_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.load_knowledge_graph
:parser: myst
```
````

````{py:function} clean_knowledge_graph(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, config: dict) -> None
:canonical: kgate.preprocessing.clean_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.clean_knowledge_graph
:parser: myst
```
````

````{py:function} verify_node_coverage(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph) -> typing.Tuple[bool, typing.List[str]]
:canonical: kgate.preprocessing.verify_node_coverage

```{autodoc2-docstring} kgate.preprocessing.verify_node_coverage
:parser: myst
```
````

````{py:function} ensure_node_coverage(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph) -> None
:canonical: kgate.preprocessing.ensure_node_coverage

```{autodoc2-docstring} kgate.preprocessing.ensure_node_coverage
:parser: myst
```
````

````{py:function} clean_datasets(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, known_reverses: typing.List[typing.Tuple[int, int]]) -> None
:canonical: kgate.preprocessing.clean_datasets

```{autodoc2-docstring} kgate.preprocessing.clean_datasets
:parser: myst
```
````

````{py:function} clean_cartesians(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph, known_cartesian: typing.List[int], node_position: typing.Literal[head, tail] = 'head') -> None
:canonical: kgate.preprocessing.clean_cartesians

```{autodoc2-docstring} kgate.preprocessing.clean_cartesians
:parser: myst
```
````
