# {py:mod}`kgate.grid_search`

```{py:module} kgate.grid_search
```

```{autodoc2-docstring} kgate.grid_search
:parser: myst
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_grid_search <kgate.grid_search.run_grid_search>`
  - ```{autodoc2-docstring} kgate.grid_search.run_grid_search
    :parser: myst
    :summary:
    ```
* - {py:obj}`suggest_value <kgate.grid_search.suggest_value>`
  - ```{autodoc2-docstring} kgate.grid_search.suggest_value
    :parser: myst
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logging_level <kgate.grid_search.logging_level>`
  - ```{autodoc2-docstring} kgate.grid_search.logging_level
    :parser: myst
    :summary:
    ```
````

### API

````{py:data} logging_level
:canonical: kgate.grid_search.logging_level
:value: >
   None

```{autodoc2-docstring} kgate.grid_search.logging_level
:parser: myst
```

````

````{py:function} run_grid_search(config_path: str, number_of_trials: int = 10, kg: typing.Tuple[kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph, kgate.knowledgegraph.KnowledgeGraph] | kgate.knowledgegraph.KnowledgeGraph | None = None, dataframe: pandas.DataFrame | None = None)
:canonical: kgate.grid_search.run_grid_search

```{autodoc2-docstring} kgate.grid_search.run_grid_search
:parser: myst
```
````

````{py:function} suggest_value(trial: optuna.trial.Trial, value_name: str, value: int | float | list) -> int | float | list
:canonical: kgate.grid_search.suggest_value

```{autodoc2-docstring} kgate.grid_search.suggest_value
:parser: myst
```
````
