# Knowledge Graph Autoencoder Training Environment (KGATE)

KGATE is a knowledge graph embedding library bridging the encoders from [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and the decoders from [TorchKGE](https://github.com/torchkge-team/torchkge).

This tool relies heavily on the performances of TorchKGE and its numerous implemented modules for link prediction, negative sampling and model evaluation. The main goal here is to address the lack of encoders in the original library, who is unfortunately not maintained anymore.

## Installation

It is recommended to download the [configuration template](src/kgate/config_template.toml) alongside your installation (see [Usage](usage) below).

A Python version > 3.10 is required, you can download one [here](https://www.python.org/downloads/).

### With pip

```bash
pip install kgate
```

<details>
<summary>If you don't have pip, you must install it first:</summary>
On [the official pip website](https://bootstrap.pypa.io/get-pip.py) download `get-pip.py`.

Then run this on Linux or MacOS:
<pre>$ python get-pip.py</pre>

Or this on Windows:
<pre>$ py get-pip.py</pre>
</details>

### From source

Clone this repository and install it in a virtual environment like so:

```bash
git clone https://github.com/BAUDOTlab/KGATE.git
python -m venv kge_env
source kge_env/bin/activate
```

<details>
<summary>For SSH:</summary>
<pre>$ git clone git@github.com:BAUDOTlab/KGATE.git<br>python -m venv kge_env<br>source kge_env/bin/activate</pre>
</details>

<details>
<summary>For Windows:</summary>
<pre>$ git clone https://github.com/BAUDOTlab/KGATE.git<br>python -m venv kge_env<br>kge_env\Scripts\activate</pre>
</details>

### Join the development

KGATE is developed using [Poetry](https://python-poetry.org/). If you want to contribute to KGATE or make your own modifications, follow these steps:

#### 1. Install Poetry

```bash
pip install poetry
```

<details>
<summary>If you don't have pip, you must install it first:</summary>
On [the official pip website](https://bootstrap.pypa.io/get-pip.py) download `get-pip.py`.

Then run this on Linux or MacOS:
<pre>$ python get-pip.py</pre>

Or this on Windows:
<pre>$ py get-pip.py</pre>
</details>

#### 2. Clone the repository

```bash
git clone git@github.com:BAUDOTlab/KGATE.git
```

<details>
<summary>For SSH:</summary>
<pre>$ git clone git@github.com:BAUDOTlab/KGATE.git</pre>
</details>

#### 3. Install dependencies

```bash
cd KGATE
poetry install
```

## Usage

KGATE is meant to be a self-sufficient training environment for knowledge graph embedding that requires very little code to work but can easily be expanded or modified. Everything stems from the **Architect** class, which holds all the necessary attributes and methods to fully train and test a KGE model following the autoencoder architecture, as well as run inference.

The configuration file lets you iterate quickly without changing your code. See the [template](src/kgate/config_template.toml) to learn what the different options do.

At the very least, KGATE expects the Knowledge Graph to be given as a pandas dataframe or a CSV file with the columns "head", "tail" and "edge", corresponding respectively to the head nodes, tail nodes and edge types of the triplets, with one triplet per row. Any extra columns are ignored. In addition, a metadata dataframe can be submitted (can also be a CSV) to map each node with their type, requiring the columns "id" and "type". Extra columns are likewise ignored. 

```python
from kgate import Architect

config_path = "/path/to/your/config.toml"

architect = Architect(config_path = config_path)

# Train the model using KG and hyperparameters specified in the configuration
architect.train_model()

# Test the trained model, using the best checkpoint
architect.test()

# Run KG completion task, the empty list is the element that will be predicted
known_heads = []
known_edges = []
known_tails = []
architect.infer(known_heads, known_tails, known_edges)
```

For a more detailed example and specific methods that are available in the package, see the upcoming readthedocs documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
