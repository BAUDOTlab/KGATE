# {py:mod}`kgate.grid_search`

```{py:module} kgate.grid_search
```

```{autodoc2-docstring} kgate.grid_search
:parser: docstrings_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_grid_search <kgate.grid_search.run_grid_search>`
  - ```{autodoc2-docstring} kgate.grid_search.run_grid_search
    :parser: docstrings_parser
    :summary:
    ```
* - {py:obj}`suggest_value <kgate.grid_search.suggest_value>`
  - ```{autodoc2-docstring} kgate.grid_search.suggest_value
    :parser: docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logging_level <kgate.grid_search.logging_level>`
  - ```{autodoc2-docstring} kgate.grid_search.logging_level
    :parser: docstrings_parser
    :summary:
    ```
````

### API

````{py:data} logging_level
:canonical: kgate.grid_search.logging_level
:value: >
   None

```{autodoc2-docstring} kgate.grid_search.logging_level
:parser: docstrings_parser
```

````

````{py:function} run_grid_search(config_path: str, number_of_trials: int = 10, kg: typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph] | kgate.knowledgegraph.KnowledgeGraph | None = None, dataframe: pandas.DataFrame | None = None)
:canonical: kgate.grid_search.run_grid_search

```{autodoc2-docstring} kgate.grid_search.run_grid_search
:parser: docstrings_parser
```
````

````{py:function} suggest_value(trial: optuna.trial.Trial, value_name: str, value: int | float | list) -> int | float | list
:canonical: kgate.grid_search.suggest_value

```{autodoc2-docstring} kgate.grid_search.suggest_value
:parser: docstrings_parser
```
````
