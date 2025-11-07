# Encoders

KGATE implements several encoders and allows you to use any model using [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)'s models. In KGATE, an **encoder** is a module that takes any kind of numeric features as input and transform it into embeddings of the desired size that the **decoder** uses to reconstruct the graph. If you are looking for models to learn an initial representation of the graph before launching the autoencoder training, head over to the **[preprocessors](./preprocessors.md)** page.

```{currentmodule} kgate.encoders
```

```{autosummary}
   :nosignatures:
   :toctree: ../generated

   ConvKB
```

## Building your own encoder

KGATE encoders inherits from the (GNN)[#GNN] class, which holds the convolution layer in the `convs` property, adds self-loops on all node types and implements the `forward` method. The encoders simply create the architecture of the encoder. While the out-of-the-box encoders are rather simple, you can easily create a more complex model fitting your needs.

Register your custom encoder like so:

```python
from kgate import Architect
from kgate.encoders import GNN
from typing import List, Tuple
from torch_geometric.nn import HeteroConv, GATConv

# We want to test the original GATConv layer and not the GATv2Conv used by KGATE
class MyCustomEncoder(GNN):
   def __init__(self, edge_types: List[Tuple[str,str,str]], emb_dim: int, num_gnn_layers: int=2, aggr: str="sum", device: str="cuda", add_self_loops: bool = True):
      super().__init__(edge_types, add_self_loops, aggr)
      self.n_layers = num_gnn_layers

      for layer in range(num_gnn_layers):
         conv = HeteroConv(
         {edge_type: GATConv(in_channels=-1, out_channels=emb_dim, add_self_loops=False) for edge_type in self.edge_types},
               aggr=self.aggr
         ).to(device)
         self.convs.append(conv)

architect = Architect(config_path="my/super/config.toml")

edge_types = architect.kg_train.triple_types
gnn_layers = architect.config["model"]["encoder"]["gnn_layer_number"]

architect.encoder = MyCustomEncoder(edge_types=edge_types, emb_dim=architect.enc_emb_dim, num_gnn_layers=gnn_layers)

# The custom encoder is registered and will be used regardless of the model set in the configuration
architect.train_model()
```
