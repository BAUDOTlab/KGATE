import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv
from torch_geometric.data import HeteroData

from typing import List, Dict, Tuple
from .utils import HeteroMappings

from copy import deepcopy
import logging
log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format="%(asctime)s - %(levelname)s - %(message)s" 
)
class DefaultEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep = False

    def forward(self, node_embeddings: nn.ModuleList, node_types: List[str]):
        device = node_embeddings[0].weight.device
        return {node_type: embedding.weight.data.to(device) for node_type, embedding in zip(mappings.hetero_node_type, node_embeddings)}

class GNN(nn.Module):
    def __init__(self, aggr:str="sum"):
        super().__init__()
        self.deep = True
        self.device = "cuda"
        # Define HeteroConv aggregation
        self.aggr = aggr
        self.convs = nn.ModuleList()

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str,], Tensor]):
        # x_dict = {node_type: embedding.weight.to(self.device) for node_type, embedding in zip(mappings.hetero_node_type, node_embeddings)}
        # edge_index_dict = {key: edge_index.to(self.device) for key, edge_index in mappings.data.edge_index_dict.items()}  # Move edges
        for conv in self.convs:
            x_dict = conv(x_dict=x_dict, edge_index_dict=edge_index_dict)

        return x_dict
    

class GATEncoder(GNN):
    def __init__(self, edge_types: List[Tuple[str,str,str]], emb_dim: int, num_gat_layers: int=2, aggr: str="sum", device: str="cuda"):
        super().__init__(aggr)
        self.n_layers = num_gat_layers

        for layer in range(num_gat_layers):
            # Add_self_loops doesn"t work on heterogeneous graphs as per https://github.com/pyg-team/pytorch_geometric/issues/8121#issuecomment-1751129825  
            conv = HeteroConv(
                {edge_type: GATv2Conv(in_channels=-1, out_channels=emb_dim, add_self_loops=False) for edge_type in edge_types},
                aggr=self.aggr
            ).to(device)
            self.convs.append(conv)
        
class GCNEncoder(GNN):
    def __init__(self, edge_types: List[Tuple[str,str,str]], emb_dim: int, num_gcn_layers: int=2, aggr: str="sum", device: str="cuda"):
        super().__init__(aggr)
        self.n_layers = num_gcn_layers
        
        for layer in range(num_gcn_layers):
            conv = HeteroConv(
                {edge_type: SAGEConv(in_channels=-1, out_channels=emb_dim, aggr="mean") for edge_type in edge_types},
                aggr=self.aggr
            ).to(device)
            self.convs.append(conv)
