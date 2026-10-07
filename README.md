# Knowledge Graph Autoencoder Training Environment (KGATE)

KGATE is a knowledge graph embedding library bridging the encoders from [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and the decoders from [TorchKGE](https://github.com/torchkge-team/torchkge).

This tool relies heavily on the performances of TorchKGE and its numerous implemented modules for link prediction, negative sampling and model evaluation. The main goal here is to address the lack of encoders in the original library, who is unfortunately not maintained anymore.

## Installation

It is recommended to download the [configuration template](src/kgate/config_template.toml) alongside your installation (see [Usage](usage) below).

### With pip

```bash
pip install kgate
```

### From source

Clone this repository and install it in a virtual environment like so:

```bash
git clone git@github.com:BAUDOTlab/KGATE.git
python -m venv kge_env
source kge_env/bin/activate
```

### Join the development

KGATE is developed using [Poetry](https://python-poetry.org/). If you want to contribute to KGATE or make your own modifications, follow these steps:

#### 1. Install Poetry

```bash
pip install poetry
```

#### 2. Clone the repository

```bash
git clone git@github.com:BAUDOTlab/KGATE.git
```

#### 3. Install dependencies

```bash
cd KGATE
poetry install
```

(usage)=
## Usage

KGATE is meant to be a self-sufficient training environment for knowledge graph embedding that requires very little code to work but can easily be expanded or modified. Everything stems from the **Architect** class, which holds all the necessary attributes and methods to fully train and test a KGE model following the autoencoder architecture, as well as run inference.

### Command line

After installation, the `kgate` command trains and tests a model in a single step (or use `python -m kgate`):

```bash
# The most common case: a CSV file and an output directory. That's enough.
kgate my_kg.csv --output runs/my_run

# Tweak the model: a different decoder, embedding size, training budget
kgate my_kg.csv --output runs/my_run --decoder ComplEx --dim 50 --epochs 30

# Start from a configuration file and override a few options on the command line
kgate config.toml --lr 0.0005 --batch-size 4096

# Check that a setup is valid without training
kgate my_kg.csv --output runs/my_run --dry-run
```

- The first argument can be either a TOML configuration file or directly the knowledge graph CSV (equivalent to `--kg-csv`). Command line options always override the configuration file values.
- Every option of the configuration [template](src/kgate/config_template.toml) is available as a command line option; run `kgate --help` for the full list with plain-language explanations, allowed values and defaults.
- Results (checkpoints, metrics, learning curves) are written to the output directory, and the resolved configuration is saved there as `kgate_config.toml` for reproducibility.

### Python API

The configuration file lets you iterate quickly without changing your code. See the [template](src/kgate/config_template.toml) to learn what the different options do.

At the very least, KGATE expects the Knowledge Graph to be given as a pandas dataframe or a CSV file with the columns "head", "tail" and "edge", corresponding respectively to the head nodes, tail nodes and relation types of the triplets, with one triplet per row. Any extra columns are ignored. In addition, a metadata dataframe can be submitted (can also be a CSV) to map each node with their type, requiring the columns "id" and "type". Extra columns are likewise ignored. 

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
known_relations = []
known_tails = []
architect.infer(known_heads, known_relations, known_tails)
```

For a more detailed example and specific methods that are available in the package, see the upcoming readthedocs documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
