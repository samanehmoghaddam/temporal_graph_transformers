import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .edge_transformer import EdgeTypeTransformerConv


class TemporalGraphTransformer(nn.Module):
    def __init__(self, in_dim, hid_dim: int = 64, num_layers: int = 3, num_heads: int = 4, num_classes: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(EdgeTypeTransformerConv(in_dim, hid_dim, num_heads=num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(EdgeTypeTransformerConv(hid_dim * num_heads, hid_dim, num_heads=num_heads))

        self.temporal_rnn = nn.GRU(hid_dim * num_heads, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_type_ids):
        for conv in self.layers:
            x = F.elu(conv(x, edge_index, edge_type_ids))

        graph_embed = global_mean_pool(x, batch).unsqueeze(1)  # [batch_size, 1, feat_dim]
        out, _ = self.temporal_rnn(graph_embed)
        out = out[:, -1, :]
        return self.fc(out)
