# {py:mod}`kgate.utils`

```{py:module} kgate.utils
```

```{autodoc2-docstring} kgate.utils
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_config <kgate.utils.parse_config>`
  - ```{autodoc2-docstring} kgate.utils.parse_config
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`set_config_key <kgate.utils.set_config_key>`
  - ```{autodoc2-docstring} kgate.utils.set_config_key
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`save_config <kgate.utils.save_config>`
  - ```{autodoc2-docstring} kgate.utils.save_config
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`load_knowledge_graph <kgate.utils.load_knowledge_graph>`
  - ```{autodoc2-docstring} kgate.utils.load_knowledge_graph
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`set_random_seeds <kgate.utils.set_random_seeds>`
  - ```{autodoc2-docstring} kgate.utils.set_random_seeds
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`compute_triplet_proportions <kgate.utils.compute_triplet_proportions>`
  - ```{autodoc2-docstring} kgate.utils.compute_triplet_proportions
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`concat_kgs <kgate.utils.concat_kgs>`
  - ```{autodoc2-docstring} kgate.utils.concat_kgs
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`count_triplets <kgate.utils.count_triplets>`
  - ```{autodoc2-docstring} kgate.utils.count_triplets
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`find_best_model <kgate.utils.find_best_model>`
  - ```{autodoc2-docstring} kgate.utils.find_best_model
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`initialize_embedding <kgate.utils.initialize_embedding>`
  - ```{autodoc2-docstring} kgate.utils.initialize_embedding
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`read_train_metrics <kgate.utils.read_train_metrics>`
  - ```{autodoc2-docstring} kgate.utils.read_train_metrics
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`plot_learning_curves <kgate.utils.plot_learning_curves>`
  - ```{autodoc2-docstring} kgate.utils.plot_learning_curves
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`filter_scores <kgate.utils.filter_scores>`
  - ```{autodoc2-docstring} kgate.utils.filter_scores
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`merge_kg <kgate.utils.merge_kg>`
  - ```{autodoc2-docstring} kgate.utils.merge_kg
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`get_dictionary_mapping <kgate.utils.get_dictionary_mapping>`
  - ```{autodoc2-docstring} kgate.utils.get_dictionary_mapping
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`get_average_heads_per_tail <kgate.utils.get_average_heads_per_tail>`
  - ```{autodoc2-docstring} kgate.utils.get_average_heads_per_tail
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`get_average_tails_per_head <kgate.utils.get_average_tails_per_head>`
  - ```{autodoc2-docstring} kgate.utils.get_average_tails_per_head
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`get_bernoulli_probabilities <kgate.utils.get_bernoulli_probabilities>`
  - ```{autodoc2-docstring} kgate.utils.get_bernoulli_probabilities
    :parser: docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logging_level <kgate.utils.logging_level>`
  - ```{autodoc2-docstring} kgate.utils.logging_level
    :parser: docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logging_level
:canonical: kgate.utils.logging_level
:value: >
   None

```{autodoc2-docstring} kgate.utils.logging_level
:parser: docstrings_parser
```

````

````{py:function} parse_config(config_path: str, config_dictionnary: dict) -> dict
:canonical: kgate.utils.parse_config

```{autodoc2-docstring} kgate.utils.parse_config
:parser: docstrings_parser
```
````

````{py:function} set_config_key(key: str, default: dict, config: dict | None = None, inline: dict | None = None) -> str | int | list | dict
:canonical: kgate.utils.set_config_key

```{autodoc2-docstring} kgate.utils.set_config_key
:parser: docstrings_parser
```
````

````{py:function} save_config(config: dict, filename: pathlib.Path | None = None)
:canonical: kgate.utils.save_config

```{autodoc2-docstring} kgate.utils.save_config
:parser: docstrings_parser
```
````

````{py:function} load_knowledge_graph(pickle_filename: pathlib.Path) -> typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph]
:canonical: kgate.utils.load_knowledge_graph

```{autodoc2-docstring} kgate.utils.load_knowledge_graph
:parser: docstrings_parser
```
````

````{py:function} set_random_seeds(seed: int) -> None
:canonical: kgate.utils.set_random_seeds

```{autodoc2-docstring} kgate.utils.set_random_seeds
:parser: docstrings_parser
```
````

````{py:function} compute_triplet_proportions(kg_train: kgate.knowledgegraph.KnowledgeGraph, kg_test: kgate.knowledgegraph.KnowledgeGraph, kg_validation: kgate.knowledgegraph.KnowledgeGraph) -> dict
:canonical: kgate.utils.compute_triplet_proportions

```{autodoc2-docstring} kgate.utils.compute_triplet_proportions
:parser: docstrings_parser
```
````

````{py:function} concat_kgs(kg_train: kgate.knowledgegraph.KnowledgeGraph, kg_validation: kgate.knowledgegraph.KnowledgeGraph, kg_test: kgate.knowledgegraph.KnowledgeGraph) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: kgate.utils.concat_kgs

```{autodoc2-docstring} kgate.utils.concat_kgs
:parser: docstrings_parser
```
````

````{py:function} count_triplets(kg1: kgate.knowledgegraph.KnowledgeGraph, kg2: kgate.knowledgegraph.KnowledgeGraph, duplicates: typing.List[typing.Tuple[int, int]], reverse_duplicates: typing.List[typing.Tuple[int, int]]) -> typing.Tuple[int, int]
:canonical: kgate.utils.count_triplets

```{autodoc2-docstring} kgate.utils.count_triplets
:parser: docstrings_parser
```
````

````{py:function} find_best_model(directory: pathlib.Path) -> pathlib.Path | None
:canonical: kgate.utils.find_best_model

```{autodoc2-docstring} kgate.utils.find_best_model
:parser: docstrings_parser
```
````

````{py:function} initialize_embedding(embedding_count: int, embedding_dimensions: int, device: str = 'cpu') -> torch.nn.Embedding
:canonical: kgate.utils.initialize_embedding

```{autodoc2-docstring} kgate.utils.initialize_embedding
:parser: docstrings_parser
```
````

````{py:function} read_train_metrics(train_metrics_file: pathlib.Path) -> pandas.DataFrame
:canonical: kgate.utils.read_train_metrics

```{autodoc2-docstring} kgate.utils.read_train_metrics
:parser: docstrings_parser
```
````

````{py:function} plot_learning_curves(train_metrics_file: pathlib.Path, output_directory: pathlib.Path, validation_metric_value: str)
:canonical: kgate.utils.plot_learning_curves

```{autodoc2-docstring} kgate.utils.plot_learning_curves
:parser: docstrings_parser
```
````

````{py:function} filter_scores(scores: torch.Tensor, graphindices: torch.Tensor, missing: typing.Literal[head, tail, edge], first_index: torch.Tensor, second_index: torch.Tensor, true_index: torch.Tensor | None) -> torch.Tensor
:canonical: kgate.utils.filter_scores

```{autodoc2-docstring} kgate.utils.filter_scores
:parser: docstrings_parser
```
````

````{py:function} merge_kg(kg_list: typing.List[kgate.knowledgegraph.KnowledgeGraph], complete_graphindices: bool = False) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.utils.merge_kg

```{autodoc2-docstring} kgate.utils.merge_kg
:parser: docstrings_parser
```
````

````{py:function} get_dictionary_mapping(dataframe: pandas.DataFrame, nodes=True) -> typing.Dict[str, int]
:canonical: kgate.utils.get_dictionary_mapping

```{autodoc2-docstring} kgate.utils.get_dictionary_mapping
:parser: docstrings_parser
```
````

````{py:function} get_average_heads_per_tail(graphindices: torch.Tensor) -> typing.Dict[float, float]
:canonical: kgate.utils.get_average_heads_per_tail

```{autodoc2-docstring} kgate.utils.get_average_heads_per_tail
:parser: docstrings_parser
```
````

````{py:function} get_average_tails_per_head(graphindices: torch.Tensor) -> typing.Dict[float, float]
:canonical: kgate.utils.get_average_tails_per_head

```{autodoc2-docstring} kgate.utils.get_average_tails_per_head
:parser: docstrings_parser
```
````

````{py:function} get_bernoulli_probabilities(knowledge_graph: kgate.knowledgegraph.KnowledgeGraph) -> typing.Dict[float, float]
:canonical: kgate.utils.get_bernoulli_probabilities

```{autodoc2-docstring} kgate.utils.get_bernoulli_probabilities
:parser: docstrings_parser
```
````
