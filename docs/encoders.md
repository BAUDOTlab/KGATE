# Encoders

KGATE implements several encoders and allows you to use any model using [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)'s models. In KGATE, an **encoder** is a module that takes any kind of numeric features as input and transform it into embeddings of the desired size that the **decoder** uses to reconstruct the graph. If you are looking for models to learn an initial representation of the graph before launching the autoencoder training, head over to the **[preprocessors](./preprocessors.md)** page.

Currently implemented encoders:
* [DefaultEncoder](./reference/api_encoders.md#defaultencoder)
* [GNN](./reference/api_encoders.md#gnn)
* [GATEncoder](./reference/api_encoders.md#gatencoder)
* [GCNEncoder](./reference/api_encoders.md#gcnencoder)
* [Node2VecEncoder](./reference/api_encoders.md#node2vecencoder)
<!--[NewEncoderName](./reference/api_encoders.md#newencodername)-->


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
   def __init__(self,
               edge_types: List[Tuple[str, str, str]],
               embedding_dimensions: int,
               gnn_layer_count: int = 2,
               aggregation: : Literal["sum", "mean", "min", "max", "cat", None] = "sum",
               device: torch.device | Literal["cuda", "cpu"] = "cuda",
               add_self_loops: bool = True):
      super().__init__(edge_types, add_self_loops, aggregation)
      self.layer_count = gnn_layer_count

      for layer in range(gnn_layer_count):
         conv = HeteroConv(
         {edge_type: GATConv(   in_channels = -1,
                                out_channels = embedding_dimensions,
                                add_self_loops = False)
                             for edge_type in self.edge_types},
                             aggregation = self.aggregation
         ).to(device)
         self.convs.append(conv)

architect = Architect(config_path = "my/super/config.toml")

edge_types = architect.kg_train.triplet_types
gnn_layers = architect.config["model"]["encoder"]["gnn_layer_number"]

architect.encoder = MyCustomEncoder(edge_types = edge_types,
                                    embedding_dimensions = architect.encoder_node_embedding_dimensions,
                                    gnn_layer_count = gnn_layers)

# The custom encoder is registered and will be used regardless of the model set in the configuration
architect.train_model()
```
