from kgate import KnowledgeGraph

import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
import logging
from pathlib import Path
from torch_geometric.nn import Node2Vec
import sys
from tqdm import tqdm

class Initializer:
    """Base class for initializing node and edge embedding."""
    def initialize_embedding(self, 
                            sample_count: int,
                            embedding_dimensions: int,
                            device: torch.device | str) -> nn.Parameter:
        """
        Initialize embeddings with number of nodes/edges and embedding dimensions.
        
        Use of a Xavier uniform distribution.
        See PyTorch documentation: https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        
        Arguments
        ---------
        sample_count: int
            Number of nodes/edges in the embedding.
        embedding_dimensions: int
            Dimensions of embeddings.
        device: torch.device or str, default to "cpu"
            Indicate if data should be sent to GPU or CPU.
            GPU is referenced to as Cuda.
            
        Returns
        -------
        embedding: nn.Embedding
            Embedding object with given parameters.
        
        """
        embedding = nn.Parameter(torch.empty((sample_count, embedding_dimensions), device = device))
        nn.init.xavier_uniform_(embedding.data)
        
        return embedding

    def initialize_all_embeddings(self, 
                                    knowledge_graph: KnowledgeGraph,
                                    *,
                                    node_embedding_dimensions: int, 
                                    edge_embedding_dimensions: int,
                                    device: torch.device | str = "cpu",
                                    inplace: bool = False
                                ) -> Tuple[nn.ParameterList, nn.Parameter] | None:
        """Initialize all node and edge embeddings.

        This is a generic function that calls the initializer.initialize_embedding(). Refer to the 
        initializer's `initialize_embedding` method for the actual initialization logic.

        Arguments
        ---------
        knowledge_graph: KnowledgeGraph
            The knowledge graph for which the embeddings are initialized.
        node_embedding_dimensions: int
            The embedding dimensions of the nodes.
        edge_embedding_dimensions: int
            The embedding dimensions of the edges. For most models, this is the same as above.
        device: torch.device or str, optional, defaults to "cpu"
            The PyTorch device where the embeddings should be created in.
        inplace: bool, optional, defaults to False
            Whether the embeddings should be returned or directly applied to the knowledge graph.

        Returns
        -------
        Only if inplace = False
        node_embeddings: nn.ParameterList
            The generated node embeddings
        edge_embeddings: nn.Parameter
            The generated edge embeddings
        """
        node_embeddings = nn.ParameterList()
        index_to_node_type = {value: key for key,value in knowledge_graph.node_type_to_index.items()}
        for node_type in knowledge_graph.node_type_to_global:
            node_count = knowledge_graph.node_type_to_global[node_type].size(0)

            node_embeddings.append(self.initialize_embedding(node_count, node_embedding_dimensions, device))
    
        edge_embeddings = self.initialize_embedding(knowledge_graph.edge_count, edge_embedding_dimensions, device)

        if inplace:
            knowledge_graph.embeddings = node_embeddings, edge_embeddings
        else:
            return node_embeddings, edge_embeddings
    

class FeatureInitializer(Initializer):
    """
    Initializer using user-supplied features
    """
    def __init__(self, node_features: Dict[str, pd.DataFrame], edge_features: pd.DataFrame) -> None:
        self.node_features = node_features
        self.edge_features = edge_features

    def initialize_all_embeddings(self,
                                    knowledge_graph: KnowledgeGraph,
                                    *,
                                    node_embedding_dimensions: int, 
                                    edge_embedding_dimensions: int,
                                    device: torch.device | str = "cpu",
                                    inplace: bool = False
                                ) -> Tuple[nn.ParameterList, nn.Parameter] | None:
        """
        Initialize all node and edge embeddings using user-supplied features.

        If a node or edge type was not provided with any feature, its embeddings will be 
        randomly initialized.

        Arguments
        ---------
        knowledge_graph: KnowledgeGraph
            The knowledge graph for which the embeddings are initialized.
        node_embedding_dimensions: int
            The embedding dimensions of the nodes.
        edge_embedding_dimensions: int
            The embedding dimensions of the edges. For most models, this is the same as above.
        device: torch.device or str, optional, defaults to "cpu"
            The PyTorch device where the embeddings should be created in.
        inplace: bool, optional, defaults to False
            Whether the embeddings should be returned or directly applied to the knowledge graph.

        Returns
        -------
        Only if inplace = False
        node_embeddings: nn.ParameterList
            The generated node embeddings
        edge_embeddings: nn.Parameter
            The generated edge embeddings
        """
        node_embeddings = nn.ParameterList()
        index_to_node_type = {value: key for key,value in knowledge_graph.node_type_to_index.items()}
        for node_type in knowledge_graph.node_type_to_global:
            node_count = knowledge_graph.node_type_to_global[node_type].size(0)

            if node_type in self.node_features:
                current_feature: pd.DataFrame = self.node_features[node_type]
                
                assert current_feature.shape[0] == node_count, f"The length of the given attribute ({current_feature.shape[0]}) must match the number of nodes of this type ({node_count})."
                input_features = torch.empty((node_count, current_feature.shape[1]), dtype = torch.float, device = device)
                
                for node in current_feature.index:
                    node_index = knowledge_graph.node_to_index[node]
                    node_type_index = knowledge_graph.node_types[node_index]
                    local_index = knowledge_graph.global_to_local_indices[node_index]
                    assert node_type_index == knowledge_graph.node_type_to_index[node_type], f"The node {node} is given as {node_type} but registered as {index_to_node_type[str(node_type_index)]} in the KG."

                    input_features[local_index] = torch.tensor(current_feature.loc[node], dtype = torch.float, device = device)
                
                node_embeddings.append(nn.Parameter(input_features))
            else:
                logging.warning(f"Node type {node_type} was not given any feature, will initialize random embeddings.")
                embeddings: nn.Parameter = self.initialize_embedding(node_count, node_embedding_dimensions, device)
                node_embeddings.append(embeddings)

        if self.edge_features is not None:
            assert self.edge_features.shape[0] == knowledge_graph.edge_count, f"The length of the edge features ({self.edge_features.shape[0]}) must match the number of edges in the graph ({knowledge_graph.edge_count})."
            edge_embeddings = torch.empty((knowledge_graph.edge_count, self.edge_features.shape[1]), dtype = torch.float, device = device)

            for edge in self.edge_features.index:
                edge_index = knowledge_graph.edge_to_index[edge]
                edge_embeddings[edge_index] = torch.tensor(self.edge_features.loc[edge], dtype = torch.float, device = device)
        else:
            edge_embeddings: nn.Parameter = self.initialize_embedding(knowledge_graph.edge_count, edge_embedding_dimensions, device)
        if inplace:
            knowledge_graph.embeddings = node_embeddings, edge_embeddings
        else:
            return node_embeddings, edge_embeddings

class Node2VecInitializer(Initializer):
    """
    Implementation of node2vec model detailed in the paper referenced below.

    References
    ----------
    Aditya Grover, Jure Leskovec
    `node2vec: Scalable Feature Learning for Networks`
    https://arxiv.org/pdf/1607.00653
    In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.

    Arguments
    ---------
    edge_indices: torch.Tensor
        Indices of edges.
    embedding_dimensions: int
        Dimensions of embedding, both of nodes and edges.
    walk_length: int
        The walk length.
    context_size: int
        The actual context size which is considered for positive samples.
        This parameter increases the effective sampling rate by reusing samples across different source nodes.
    device: torch.device or Literal["cuda", "cpu"]
        Indicate if data should be sent to GPU or CPU.
        GPU is referenced to as Cuda.
    output_directory: Path
        Path to the directory where files will be created.

    Attributes
    ----------
    device: torch.device or Literal["cuda", "cpu"]
        Indicate if data should be sent to GPU or CPU.
        GPU is referenced to as Cuda.
    output_directory: Path
        Path to the directory where files will be created.
    model: torch_geometric.nn.Node2Vec
        The Node2Vec model object.
        Node2Vec documentation: https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.models.Node2Vec.html
    loader: TODO.type
        TODO.What_that_variable_is_or_does
    optimizer: TODO.type
        TODO.What_that_variable_is_or_does
    
    """
    def __init__(self,
                edge_indices: torch.Tensor,
                embedding_dimensions: int,
                walk_length: int,
                context_size: int,
                output_directory: Path,
                device: torch.device | str = "cuda",
                **node2vec_kwargs) -> None:
        self.device = device
        self.output_directory = output_directory
        self.model = Node2Vec(
            edge_index = edge_indices,
            embedding_dim = embedding_dimensions,
            walk_length = walk_length,
            context_size = context_size,
            **node2vec_kwargs
            ).to(device)

        workers_count = 4 if sys.platform == 'linux' else 0
        self.loader = self.model.loader(batch_size = 128, shuffle = True, num_workers = workers_count)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr = 0.01)
    
    
    def generate_all_embeddings(self,
                                knowledge_graph: KnowledgeGraph,
                                *,
                                device: torch.device | str = "cpu",
                                inplace: bool = False,
                                **_) -> Any | None:
        """
        Generate initial embeddings using random walk
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

        if inplace:
            #TODO
            pass