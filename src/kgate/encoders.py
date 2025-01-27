import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv

# Utility?
class DefaultEncoder(nn.Module):
    def __init__(self):
        self.deep = False

class GNN(nn.Module):
    def __init__(self, node_embeddings, aggr='sum'):
        self.deep = True

        # Define HeteroConv aggregation
        self.aggr = aggr
        self.convs = nn.ModuleList()
        # Initialize the embedding dict
        self.x_dict = {node_type: embedding.weight for node_type, embedding in node_embeddings.items()}

    def forward(self, hetero_data):
        x_dict = self.x_dict

        for _, conv in enumerate(self.convs):
                x_dict = conv(x_dict, hetero_data.edge_index_dict)

        return x_dict
    

class GATEncoder(GNN):
    def __init__(self, node_embeddings, hetero_data, emb_dim, num_gat_layers=2, aggr='sum'):
        super().__init__(node_embeddings, aggr)
        
        # Définition des couches GCN multiples pour chaque type d'arête
        for layer in range(num_gat_layers):
            conv = HeteroConv(
                {edge_type: GATv2Conv(emb_dim, emb_dim) for edge_type in hetero_data.edge_types},
                aggr=self.aggr
            )
            self.convs.append(conv)
        
class GCNEncoder(GNN):
    def __init__(self, node_embeddings, hetero_data, emb_dim, num_gcn_layers=2, aggr="sum"):
        super().__init__(node_embeddings, aggr)
        for layer in range(num_gcn_layers):
            conv = HeteroConv(
                {edge_type: SAGEConv(emb_dim, emb_dim, aggr="mean") for edge_type in hetero_data.edge_types},
                aggr=self.aggr
            )
            self.convs.append(conv)
