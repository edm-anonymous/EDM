import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
import torch.nn as nn
import torch.nn.functional as F

class MultiConv(MessagePassing):
    def __init__(self, in_channels):
        super(MultiConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.gate = torch.nn.Linear(2*in_channels, 1)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_i, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))

        return norm.view(-1, 1) * (x_j) *alpha_g

    def update(self, aggr_out):
        return aggr_out  

class directConv(MessagePassing):
    def __init__(self, in_channels):
        super(directConv, self).__init__(aggr='add')  
        self.gate = torch.nn.Linear(2*in_channels, 1)
        
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)
        
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        return aggr_out
