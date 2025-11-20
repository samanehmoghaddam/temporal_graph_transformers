import torch
import torch.nn as nn


class NodeDataset(torch.utils.data.Dataset):
    """
    Baseline dataset: treat each node as a sample, with the graph label
    broadcast to all its nodes.
    """
    def __init__(self, graphs):
        self.features = []
        self.labels = []
        for g in graphs:
            self.features.append(g.x)
            num_nodes = g.x.size(0)
            self.labels.append(torch.full((num_nodes,), g.y.item(), dtype=torch.float))

        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class NodeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))
