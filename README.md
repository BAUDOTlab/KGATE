# Knowledge Graph Autoencoder Training Environment (KGATE)

KGATE is a knowledge graph embedding library bridging the encoders from [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and the decoders from [TorchKGE](https://github.com/torchkge-team/torchkge).

This tool relies heavily on the performances of TorchKGE and its numerous implemented modules for link prediction, negative sampling and model evaluation. The main goal here is to address the lack of encoders in the original library, who is unfortunately not maintained anymore.

## Installation

To join in the development, clone this repository and install Poetry:

```pip install poetry```

Install the dependencies with:

```poetry install```

## Usage

KGATE is meant to be a self-sufficient training environment for knowledge graph embedding that requires very little code to work but can easily be expanded or modified. Everything stems from the **Architect** class, which holds all the necessary attributes and methods to fully train and test a KGE model following the autoencoder architecture, as well as run inference.

```python
from kgate import Architect

config_path = "/path/to/your/config.toml"

architect = Architect(config = config_path)

# Train the model using KG and hyperparameters specified in the configuration
architect.train()

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

This project is licensed under the MIT License. See the [LICENSE](#license) file for details.