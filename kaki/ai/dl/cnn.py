from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as opt

class EarlyStopping:
    def __init__(self, patience=7, path='./model1.pth'):
        self.counter = 0
        self.patience = patience
        self.path = path
        self.best_loss = None
        self.early_stopping = False
    
    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            torch.save(model.state_dict(), self.path)
        
        elif loss < self.best_loss:
            self.best_loss = loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping = True
                print('早停')


class net(nn.Module):
    def __init__(self, input_dim, label_dim):
        super(net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, label_dim)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, data):
        out = self.fc1(data)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out