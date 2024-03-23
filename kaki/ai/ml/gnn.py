import networkx as nx 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

"""
requirements:
    - torch
    - networkx
    - numpy
    - scikit-learn
"""

# define the Graph Conv Layers
class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self, hidden_channels, dataset) -> None:
        super.__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_idx):
        x = self.conv1(x, edge_idx)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_idx)
        return x

model = GraphConvolutionLayer(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 
loss = torch.nn.CrossEntropyLoss()

def train(data, model, optim, loss_fct):
    model.train()
    optimizer.zero_grad()
    """
    TODO: adapt to data we are using
    """
    out = model(data.x, data.edge_index)
    loss = loss_fct(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss
