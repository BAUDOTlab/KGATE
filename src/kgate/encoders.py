"""Collections of encoder classes to embed the graph structure into a latent space."""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, HeteroConv, Node2Vec, SAGEConv


logging_level = logging.INFO
logging.basicConfig(
    level = logging_level,  
    format = "%(asctime)s - %(levelname)s - %(message)s" 
)


class DefaultEncoder(nn.Module):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Attributes
    ----------
    deep: bool, defaul to False
        TODO.What_that_variable_is_or_does
    TODO.inherited_attributes
    
    """
    def __init__(self):
        
        super().__init__()
        self.deep = False



class GNN(nn.Module):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    edge_types: List[Tuple[str, str, str]]
        TODO.What_that_argument_is_or_does
    aggregation: str, default to "sum"
        TODO.What_that_argument_is_or_does

    Attributes
    ----------
    deep: bool, default to True
        TODO.What_that_variable_is_or_does
    device: str, default to "cuda"
        TODO.What_that_variable_is_or_does
    aggregation: str, default to "sum"
        TODO.What_that_variable_is_or_does
    convolutions: nn.ModuleList()
        TODO.What_that_variable_is_or_does
    edge_types: List[Tuple[str, str, str]]
        TODO.What_that_variable_is_or_does
    TODO.inherited_attributes
    
    """
    def __init__(self,
                edge_types: List[Tuple[str, str, str]],
                aggregation: str = "sum"):
        
        super().__init__()
        self.deep = True
        self.device = "cuda"
        # Define HeteroConv aggregation
        self.aggregation = aggregation
        self.convolutions = nn.ModuleList()

        if edge_types is not None:
            node_types = []
            for triplet in edge_types:
                node_types += [triplet[0], triplet[2]]
            for node_type in set(node_types):
                edge_types.append((node_type, "self", node_type))
        self.edge_types = edge_types


    def forward(self,
                x_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str,], Tensor]
                ) -> Dict[str, Tensor]:
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO

        Arguments
        ---------
        x_dict: Dict[str, Tensor]
            TODO.What_that_argument_is_or_does
        edge_index_dict: Dict[Tuple[str, str, str,], Tensor]
            TODO.What_that_argument_is_or_does

        Returns
        -------
        x_dict: Dict[str, Tensor]
            TODO.What_that_variable_is_or_does
            
        """
        for _, conv in enumerate(self.convolutions):
            x_dict = conv(x_dict = x_dict, edge_index_dict = edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        return x_dict
    


class GATEncoder(GNN):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    edge_types: List[Tuple[str, str, str]]
        TODO.What_that_argument_is_or_does
    embedding_dimensions: int
        TODO.What_that_argument_is_or_does
    gat_layer_count: int, default to 2
        TODO.What_that_argument_is_or_does
    aggregation: str, default to "sum"
        TODO.What_that_argument_is_or_does
    device: str, default to "cuda"
        TODO.What_that_argument_is_or_does
    add_self_loops: bool, default to True
        TODO.What_that_argument_is_or_does

    Attributes
    ----------
    layer_count: TODO.type
        TODO.What_that_variable_is_or_does
    edge_types: TODO.type
        TODO.What_that_variable_is_or_does
    aggregation: TODO.type
        TODO.What_that_variable_is_or_does
    convolutions: TODO.type
        TODO.What_that_variable_is_or_does
    TODO.inherited_attributes
    
    """
    def __init__(self,
                edge_types: List[Tuple[str, str, str]],
                embedding_dimensions: int,
                gat_layer_count: int = 2,
                aggregation: str = "sum",
                device: str = "cuda",
                add_self_loops: bool = True):
        
        super().__init__(edge_types, add_self_loops, aggregation)
        self.layer_count = gat_layer_count

        for _ in range(gat_layer_count):
            # Add_self_loops doesn't work on heterogeneous graphs as per https://github.com/pyg-team/pytorch_geometric/issues/8121#issuecomment-1751129825  
            convolution = HeteroConv(
                {edge_type: GATv2Conv(  in_channels = -1,
                                        out_channels = embedding_dimensions,
                                        add_self_loops = False)
                    for edge_type in self.edge_types},
                aggr = self.aggregation
                ).to(device)
            self.convolutions.append(convolution)



class GCNEncoder(GNN):
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    edge_types: List[Tuple[str, str, str]]
        TODO.What_that_argument_is_or_does
    embedding_dimensions: int
        TODO.What_that_argument_is_or_does
    gcn_layer_count: int, default to 2
        TODO.What_that_argument_is_or_does
    aggregation: str, default to "sum"
        TODO.What_that_argument_is_or_does
    device: str, default to "cuda"
        TODO.What_that_argument_is_or_does
    add_self_loops: bool, default to True
        TODO.What_that_argument_is_or_does

    Attributes
    ----------
    layer_count: TODO.type
        TODO.What_that_variable_is_or_does
    edge_types: TODO.type
        TODO.What_that_variable_is_or_does
    aggregation: TODO.type
        TODO.What_that_variable_is_or_does
    convolutions: TODO.type
        TODO.What_that_variable_is_or_does
    TODO.inherited_attributes
    
    """
    def __init__(self,
                edge_types: List[Tuple[str, str, str]],
                embedding_dimensions: int,
                gcn_layer_count: int = 2,
                aggregation: str = "sum",
                device: str = "cuda",
                add_self_loops: bool = True):
        
        super().__init__(edge_types, add_self_loops, aggregation)
        self.layer_count = gcn_layer_count
        
        for _ in range(gcn_layer_count):
            convolution = HeteroConv(
                {edge_type: SAGEConv(   in_channels = -1,
                                        out_channels = embedding_dimensions,
                                        aggregation = "mean")
                    for edge_type in self.edge_types},
                aggr = self.aggregation
                ).to(device)
            self.convolutions.append(convolution)



class Node2VecEncoder:
    """
    TODO.What_the_class_is_about_globally

    References
    ----------
    TODO

    Arguments
    ---------
    edge_index: torch.Tensor
        TODO.What_that_argument_is_or_does
    embedding_dimensions: int
        TODO.What_that_argument_is_or_does
    walk_length: int
        TODO.What_that_argument_is_or_does
    context_size: int
        TODO.What_that_argument_is_or_does
    device: torch.device
        TODO.What_that_argument_is_or_does
    output_directory: Path
        TODO.What_that_argument_is_or_does

    Attributes
    ----------
    device: torch.device
        TODO.What_that_variable_is_or_does
    output_directory: Path
        TODO.What_that_variable_is_or_does
    model: TODO.type
        TODO.What_that_variable_is_or_does
    loader: TODO.type
        TODO.What_that_variable_is_or_does
    optimizer: TODO.type
        TODO.What_that_variable_is_or_does
    
    """
    def __init__(self,
                edge_index: Tensor,
                embedding_dimensions: int,
                walk_length: int,
                context_size: int,
                device: torch.device,
                output_directory: Path,
                **node2vec_kwargs):
        self.device = device
        self.output_directory = output_directory
        self.model = Node2Vec(
            edge_index = edge_index,
            embedding_dim = embedding_dimensions,
            walk_length = walk_length,
            context_size = context_size,
            **node2vec_kwargs
            ).to(device)

        workers_count = 4 if sys.platform == 'linux' else 0
        self.loader = self.model.loader(batch_size = 128, shuffle = True, num_workers = workers_count)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr = 0.01)
    
    
    def generate_embeddings(self):
        """
        TODO.What_the_function_does_about_globally

        References
        ----------
        TODO
        
        """
        for epoch in range(1,101):
            epoch_loss = 0
            for positive_random_walk, negative_random_walk in tqdm(self.loader):
                self.optimizer.zero_grad()
                loss = self.model.loss(positive_random_walk.to(self.device), negative_random_walk.to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            logging.info(f"Epoch {epoch: 03d}, Embedding Loss: {loss: .4f}")

        torch.save(self.model.embedding, self.output_directory.joinpath("embeddings_node2vec.pt"))
        logging.info(f"Embedding fully generated, saved in {self.output_directory}")