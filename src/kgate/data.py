from .data_structures import KGATEGraph
from .utils import HeteroMappings
from torch_geometric.data import HeteroData
from torchkge.utils.data import get_n_batches
from collections import defaultdict
from typing import Tuple, List


import torch
from torch import tensor, cat
class KGLoader:
    def __init__(self, data: HeteroData, mappings: HeteroMappings, batch_size: int, use_cuda: str=""):

        self.use_cuda = use_cuda
        self.batch_size = batch_size
        
        self.data = data.cpu()

        self.mappings = mappings
        self.edgelist = tensor([], dtype=torch.int64)
        self.edge_type_to_indices = defaultdict(list)
        self.edge_type_indices = []

        for i, edge_type in enumerate(self.data.edge_types):
            triplet_index = cat([
                self.data[edge_type].edge_index,
                tensor(i).repeat(self.data[edge_type].num_edges).unsqueeze(0)
            ], dim=0)

            self.edgelist = cat([
                self.edgelist,
                triplet_index
            ], dim=1)

            self.edge_type_indices += [i] * self.data[edge_type].num_edges
        # for i, r in enumerate(self.edgelist[2]):
        #     self.edge_type_to_indices[r.item()].append(i)
        # for k in self.edge_type_to_indices:
        #     self.edge_type_to_indices[k] = torch.tensor(self.edge_type_to_indices[k], dtype=torch.long)


        if use_cuda == "all":
            self.edgelist.cuda()

    def __len__(self):
        return get_n_batches(self.edgelist.size(1), self.batch_size)

    def __iter__(self):
        return _KGLoaderIter(self)

class _KGLoaderIter:
    def __init__(self, loader: KGLoader):
        self.edgelist: torch.Tensor = loader.edgelist
        self.data: HeteroData = loader.data
        self.mappings: HeteroMappings = loader.mappings
        self.edge_type_indices: List[int] = loader.edge_type_indices

        self.use_cuda = loader.use_cuda
        self.batch_size = loader.batch_size

        self.n_batches = get_n_batches(self.edgelist.size(1), self.batch_size)
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            edge_type_ids = range(self.edge_type_indices[i * self.batch_size], self.edge_type_indices[min(len(self.edge_type_indices) -1, (i+1) * self.batch_size)] + 1)
            edges = self.data.edge_types
            batch_data = HeteroData()
            node_ids = defaultdict(set)
            batch_to_kg = {}
            h = torch.tensor([], dtype=torch.int64)
            t = torch.tensor([], dtype=torch.int64)
            r = torch.tensor([], dtype=torch.int64)
            # Tensor of shape (3,batch_size), where the first row is the head idx, the second the tail idx,
            # and the third the relation idx
            batch_triplets: torch.Tensor = self.edgelist[:, i * self.batch_size: (i+1) * self.batch_size]
            for edge_type_id in edge_type_ids:
                # Keep only the triplets of the same type
                mask: torch.Tensor = batch_triplets[2] == edge_type_id
                triplets = batch_triplets[:, mask]
                
                # Retrieve names of nodes and relation type
                src = triplets[0]
                dst = triplets[1]
                src_list = src.tolist()
                dst_list = dst.tolist()
                edge_type = edges[edge_type_id]
                h_type, r_type, t_type = edge_type

                node_ids[h_type].update(src_list)
                node_ids[t_type].update(dst_list)
                
                # Can probably be optimized
                # id_maps = {ntype: {global_id.item(): i for i, global_id in enumerate(torch.tensor(list(idx), dtype=torch.long))}
                # for ntype, idx in node_ids.items()}

                h_list = [list(node_ids[h_type]).index(i) for i in src_list]
                t_list = [list(node_ids[t_type]).index(i) for i in dst_list]
                edge_index = torch.stack([
                    torch.tensor(h_list),
                    torch.tensor(t_list)
                ], dim=0)
                
                batch_data[edge_type].edge_index = edge_index

                h_type_id = self.mappings.hetero_node_type.index(h_type)
                t_type_id = self.mappings.hetero_node_type.index(t_type)

                h = cat([h, self.mappings.hetero_to_kg[h_type_id][h_list]])
                t = cat([t, self.mappings.hetero_to_kg[t_type_id][t_list]])
                r = cat([r, tensor([self.mappings.relations.index(r_type)]).repeat(triplets.size(1))])

            for ntype, ids in node_ids.items():
                idx = torch.tensor(list(ids), dtype=torch.long)
                batch_data[ntype].x = torch.index_select(self.data[ntype].x, 0, idx)
                node_ids[ntype] = idx

            if self.use_cuda == "batch":
                return batch_data.cuda(), h.cuda(), t.cuda(), r.cuda()
            else:
                return batch_data, h, t, r

    def __iter__(self):
        return self

def hetero_collate_fn(
        batch: Tuple[torch.Tensor,torch.Tensor,torch.Tensor],
        data: HeteroData, 
        mappings: HeteroMappings
        ) -> HeteroData:
    h_idx, t_idx, r_idx = zip(*batch)
    h_idx = torch.tensor(h_idx)
    t_idx = torch.tensor(t_idx)
    r_idx = torch.tensor(r_idx)
    
    batch_data = HeteroData()

    h_node_types = mappings.kg_to_node_type[h_idx]
    t_node_types = mappings.kg_to_node_type[t_idx]
    h_het_idx = mappings.kg_to_hetero[h_idx]
    t_het_idx = mappings.kg_to_hetero[t_idx]

    node_ids = defaultdict(set)
    edge_type_triplets = defaultdict(list)

    for hi, ti, ri, hnt, tnt, hhi, thi in zip(h_idx, t_idx, r_idx, h_node_types, t_node_types, h_het_idx, t_het_idx):
        rt = mappings.relations[ri]
        ht = mappings.hetero_node_type[hnt]
        tt = mappings.hetero_node_type[tnt]
        edge_type = (ht, rt, tt)
        
        edge_type_triplets[edge_type].append((hhi.item(), thi.item()))

        node_ids[edge_type[0]].add(hhi.item())
        node_ids[edge_type[2]].add(thi.item())

    id_maps = {}
    for ntype, global_ids in node_ids.items():
        idx = torch.tensor(sorted(global_ids), dtype=torch.long)
        batch_data[ntype].x = data[ntype].x[idx]
        id_maps[ntype] = {gid.item(): i for i, gid in enumerate(idx)}

    for edge_type, pairs in edge_type_triplets.items():
        h_type, rel, t_type = edge_type
        src = [id_maps[h_type][h] for h, _ in pairs]
        dst = [id_maps[t_type][t] for _, t in pairs]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        batch_data[edge_type].edge_index = edge_index

    return batch_data

class HeteroCollator:
    def __init__(self, data:HeteroData, mappings:HeteroMappings):
        self.data = data
        self.mappings = mappings

    def __call__(self, batch):
        return hetero_collate_fn(batch, self.data, self.mappings)