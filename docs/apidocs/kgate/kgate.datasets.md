# {py:mod}`kgate.datasets`

```{py:module} kgate.datasets
```

```{autodoc2-docstring} kgate.datasets
:parser: myst
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_data_root_directory <kgate.datasets.get_data_root_directory>`
  - ```{autodoc2-docstring} kgate.datasets.get_data_root_directory
    :parser: myst
    :summary:
    ```
* - {py:obj}`load_FB15k_237 <kgate.datasets.load_FB15k_237>`
  - ```{autodoc2-docstring} kgate.datasets.load_FB15k_237
    :parser: myst
    :summary:
    ```
* - {py:obj}`load_WN18RR <kgate.datasets.load_WN18RR>`
  - ```{autodoc2-docstring} kgate.datasets.load_WN18RR
    :parser: myst
    :summary:
    ```
* - {py:obj}`load_PrimeKG <kgate.datasets.load_PrimeKG>`
  - ```{autodoc2-docstring} kgate.datasets.load_PrimeKG
    :parser: myst
    :summary:
    ```
````

### API

````{py:function} get_data_root_directory() -> pathlib.Path
:canonical: kgate.datasets.get_data_root_directory

```{autodoc2-docstring} kgate.datasets.get_data_root_directory
:parser: myst
```
````

````{py:function} load_FB15k_237(data_directory: os.PathLike = None, keep_split: bool = False) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.datasets.load_FB15k_237

```{autodoc2-docstring} kgate.datasets.load_FB15k_237
:parser: myst
```
````

````{py:function} load_WN18RR(data_directory: os.PathLike = None, keep_split: bool = False) -> kgate.knowledgegraph.KnowledgeGraph
:canonical: kgate.datasets.load_WN18RR

```{autodoc2-docstring} kgate.datasets.load_WN18RR
:parser: myst
```
````

````{py:function} load_PrimeKG()
:canonical: kgate.datasets.load_PrimeKG

```{autodoc2-docstring} kgate.datasets.load_PrimeKG
:parser: myst
```
````
