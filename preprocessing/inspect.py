import torch
import dgl

dataset = torch.load('tox21_3d_egnn_dataset.pt', weights_only=False)

print(f'Total molecules: {len(dataset)}')

g, y = dataset[0]
print(f'\nFirst molecule:')
print(f'Nodes: {g.num_nodes()}')
print(f'Edges: {g.num_edges()}')
print(f'Node features: {g.ndata["x"].shape}')
print(f'Positions: {g.ndata["pos"].shape}')
print(f'Edge attrs: {g.edata["edge_attr"].shape}')
print(f'Labels: {y}')

is_fgn = g.ndata['x'][:, -1]
print(f'\nReal atoms: {(is_fgn == 0).sum().item()}')
print(f'FGN nodes:  {(is_fgn == 1).sum().item()}')