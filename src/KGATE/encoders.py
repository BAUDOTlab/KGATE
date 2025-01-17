import torch.nn as nn
from torch.nn.functional import normalize
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv
from torchkge.models import BilinearModel
import logging
from utils import my_init_embedding, create_hetero_data

class DefaultEncoder():
    def __init__(self, n_entities, n_relations, emb_dim):
        self.deep = False
        self.node_embeddings = my_init_embedding(n_entities, emb_dim)
        self.rel_emb = my_init_embedding(n_relations, emb_dim)

class GNN:
    def __init__(self, emb_dim, n_relations, kg, device, mapping_csv, aggr='sum'):
        self.deep = True
        self.hetero_data, self.kg2het, self.het2kg, _, self.kg2nodetype = create_hetero_data(kg, mapping_csv)
        self.hetero_data = self.hetero_data.to(device)
        self.rel_emb = my_init_embedding(n_relations, emb_dim)

        # Initialize initial embeddings for every node type
        self.node_embeddings = nn.ModuleDict()
        for node_type in self.hetero_data.node_types:
            num_nodes = self.hetero_data[node_type].num_nodes
            self.node_embeddings[node_type] = my_init_embedding(num_nodes, emb_dim)

        # Define HeteroConv aggregation
        self.aggr = aggr
        self.convs = nn.ModuleList()
        # Initialize the embedding dict
        self.x_dict = {node_type: embedding.weight for node_type, embedding in self.node_embeddings.items()}

    def forward(self):
        x_dict = self.x_dict

        for _, conv in enumerate(self.convs):
                x_dict = conv(x_dict, self.hetero_data.edge_index_dict)

        return x_dict
    

class GATEncoder(GNN):
    def __init__(self, emb_dim, n_relations, kg, device, mapping_csv, num_gat_layers=2, aggr='sum'):
        super().__init__(emb_dim, n_relations, kg, device, mapping_csv, aggr)
        
        # Définition des couches GCN multiples pour chaque type d'arête
        for layer in range(num_gat_layers):
            conv = HeteroConv(
                {edge_type: GATv2Conv(emb_dim, emb_dim) for edge_type in self.hetero_data.edge_types},
                aggr=self.aggr
            )
            self.convs.append(conv)
        
class GCNEncoder(GNN):
    def __init__(self, emb_dim, n_relations, kg, device, mapping_csv, num_gcn_layers=2, aggr="sum"):
        super().__init__(emb_dim, n_relations, kg, device, mapping_csv, aggr)
        for layer in range(num_gcn_layers):
            conv = HeteroConv(
                {edge_type: SAGEConv(emb_dim, emb_dim, aggr="mean") for edge_type in self.hetero_data.edge_types},
                aggr=self.aggr
            )
            self.convs.append(conv)
