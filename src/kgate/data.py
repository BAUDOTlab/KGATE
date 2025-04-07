from .data_structures import KGATEGraph
from .utils import HeteroMappings
from torch_geometric.data import HeteroData
from torchkge.utils.data import get_n_batches

import torch
from torch import tensor, cat
class KGLoader:
    def __init__(self, data: HeteroData, mappings: HeteroMappings, batch_size: int, use_cuda: str=""):

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        
        self.data = data.cpu()

        self.mappings = mappings
        self.edgelist = tensor([], dtype=torch.int64)
        for i, edge_type in enumerate(self.data.edge_types):
            triplet_index = cat([
                self.data[edge_type].edge_index,
                tensor(i).repeat(self.data[edge_type].num_edges).unsqueeze(0)
            ], dim=0)

            self.edgelist = cat([
                self.edgelist,
                triplet_index
            ], dim=1)

        if use_cuda == "all":
            self.edgelist.cuda()

    def __len__(self):
        return get_n_batches(len(self.edgelist), self.batch_size)

    def __iter__(self):
        return _KGLoaderIter(self)

class _KGLoaderIter:
    def __init__(self, loader):
        self.edgelist = loader.edgelist
        self.data: HeteroData = loader.data
        self.mappings: HeteroMappings = loader.mappings

        self.use_cuda = loader.use_cuda
        self.batch_size = loader.batch_size

        self.n_batches = get_n_batches(len(self.edgelist), self.batch_size)
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            edges = self.data.edge_types
            batch_data = HeteroData()

            # Tensor of shape (3,batch_size), where the first row is the head idx, the second the tail idx,
            # and the third the relation idx
            batch_triplets: torch.Tensor = self.edgelist[i * self.batch_size: (i+1) * self.batch_size]
            for edge_type_id in batch_triplets[2].unique():
                # Keep only the triplets of the same type
                mask: torch.Tensor = batch_triplets[2] == edge_type_id
                edge_type_triplets = batch_triplets[:, mask.squeeze(0)]

                # Get unique head and tail idx
                heads = edge_type_triplets[0].unique()
                tails = edge_type_triplets[1].unique()
                
                # Retrieve names of nodes and relation type
                edge_type = edges[edge_type_id]
                h_type = edge_type[0]
                t_type = edge_type[2]

                # Populate the batch HeteroData with original data
                if h_type not in batch_data.node_types:
                    batch_data[h_type].x = self.data[h_type].x[heads]
                else:
                    batch_data[h_type].x = cat([batch_data[h_type].x, self.data[h_type].x[heads]])
                if t_type not in batch_data.node_types:
                    batch_data[t_type].x = self.data[t_type].x[tails]
                else:
                    batch_data[t_type].x = cat([batch_data[t_type].x, self.data[t_type].x[tails]])

                # Copy the edge index from the batch info
                batch_data[edge_type].edge_index = edge_type_triplets[0:2]
            
            # Get the corresponding index for each h, r, t triple (required for negative sampling)
            r = torch.tensor([self.mappings.relations.index(edges[triple[2]][1]) for triple in batch_triplets.T])
            # h = torch.tensor([self.mappings.hetero_to_kg[edges[triple[2]][0]][triple[0]] for i, triple in enumerate(batch_triplets.T)])
            # t = torch.tensor([self.mappings.hetero_to_kg[edges[triple[2]][1]][triple[1]] for triple in batch_triplets.T])
            return batch_data.cuda()
            # if self.use_cuda == "batch":
            #     return batch_data.cuda(), h.cuda(), t.cuda(), r.cuda()
            # else:
            #     return batch_data, h, t, r

    def __iter__(self):
        return self
