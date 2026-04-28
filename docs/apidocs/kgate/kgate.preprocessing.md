# {py:mod}`kgate.preprocessing`

```{py:module} kgate.preprocessing
```

```{autodoc2-docstring} kgate.preprocessing
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`prepare_knowledge_graph <kgate.preprocessing.prepare_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.prepare_knowledge_graph
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`save_knowledge_graph <kgate.preprocessing.save_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.save_knowledge_graph
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`load_knowledge_graph <kgate.preprocessing.load_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.load_knowledge_graph
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`clean_knowledge_graph <kgate.preprocessing.clean_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.preprocessing.clean_knowledge_graph
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`verify_node_coverage <kgate.preprocessing.verify_node_coverage>`
  - ```{autodoc2-docstring} kgate.preprocessing.verify_node_coverage
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`ensure_node_coverage <kgate.preprocessing.ensure_node_coverage>`
  - ```{autodoc2-docstring} kgate.preprocessing.ensure_node_coverage
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`clean_datasets <kgate.preprocessing.clean_datasets>`
  - ```{autodoc2-docstring} kgate.preprocessing.clean_datasets
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`clean_cartesians <kgate.preprocessing.clean_cartesians>`
  - ```{autodoc2-docstring} kgate.preprocessing.clean_cartesians
    :parser: docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SUPPORTED_SEPARATORS <kgate.preprocessing.SUPPORTED_SEPARATORS>`
  - ```{autodoc2-docstring} kgate.preprocessing.SUPPORTED_SEPARATORS
    :parser: docstrings_parser
    :summary:
    ```
````

### API

````{py:data} SUPPORTED_SEPARATORS
:canonical: kgate.preprocessing.SUPPORTED_SEPARATORS
:value: >
   [',', '\t', ';']

```{autodoc2-docstring} kgate.preprocessing.SUPPORTED_SEPARATORS
:parser: docstrings_parser
```

````

````{py:function} prepare_knowledge_graph(config: dict, kg: kgate.knowledgegraph.KnowledgeGraph | None = None, dataframe: pandas.DataFrame | None = None, metadata: pandas.DataFrame | None = None) -> typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph]
:canonical: kgate.preprocessing.prepare_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.prepare_knowledge_graph
:parser: docstrings_parser
```
````

````{py:function} save_knowledge_graph(config: dict, kg_train: kgate.knowledgegraph.KnowledgeGraph, kg_validation: kgate.knowledgegraph.KnowledgeGraph, kg_test: kgate.knowledgegraph.KnowledgeGraph)
:canonical: kgate.preprocessing.save_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.save_knowledge_graph
:parser: docstrings_parser
```
````

````{py:function} load_knowledge_graph(pickle_filename: pathlib.Path) -> typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph]
:canonical: kgate.preprocessing.load_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.load_knowledge_graph
:parser: docstrings_parser
```
````

````{py:function} clean_knowledge_graph(kg: kgate.knowledgegraph.KnowledgeGraph, config: dict) -> typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph]
:canonical: kgate.preprocessing.clean_knowledge_graph

```{autodoc2-docstring} kgate.preprocessing.clean_knowledge_graph
:parser: docstrings_parser
```
````

````{py:function} verify_node_coverage(kg_train: kgate.knowledgegraph.KnowledgeGraph, kg_full: kgate.knowledgegraph.KnowledgeGraph) -> typing.Tuple[bool, typing.List[str]]
:canonical: kgate.preprocessing.verify_node_coverage

```{autodoc2-docstring} kgate.preprocessing.verify_node_coverage
:parser: docstrings_parser
```
````

````{py:function} ensure_node_coverage(kg_train: kgate.knowledgegraph.KnowledgeGraph, kg_validation: kgate.knowledgegraph.KnowledgeGraph, kg_test: kgate.knowledgegraph.KnowledgeGraph) -> typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph]
:canonical: kgate.preprocessing.ensure_node_coverage

```{autodoc2-docstring} kgate.preprocessing.ensure_node_coverage
:parser: docstrings_parser
```
````

````{py:function} clean_datasets(kg_train: kgate.knowledgegraph.KnowledgeGraph, kg_second: kgate.knowledgegraph.KnowledgeGraph, known_reverses: typing.List[typing.Tuple[int, int]]) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.preprocessing.clean_datasets

```{autodoc2-docstring} kgate.preprocessing.clean_datasets
:parser: docstrings_parser
```
````

````{py:function} clean_cartesians(kg_first: kgate.knowledgegraph.KnowledgeGraph, kg_second: kgate.knowledgegraph.KnowledgeGraph, known_cartesian: typing.List[int], node_type: str = 'head') -> typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph]
:canonical: kgate.preprocessing.clean_cartesians

```{autodoc2-docstring} kgate.preprocessing.clean_cartesians
:parser: docstrings_parser
```
````
