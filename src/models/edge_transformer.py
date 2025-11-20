import torch.nn as nn
import torch_scatter
from torch_geometric.nn import TransformerConv


class EdgeTypeTransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads: int = 4, num_edge_types: int = 4, dropout: float = 0.2):
        super().__init__()
        self.edge_embeddings = nn.Embedding(num_edge_types, in_channels)
        self.conv = TransformerConv(in_channels, out_channels, heads=num_heads, dropout=dropout)

    def forward(self, x, edge_index, edge_type_ids):
        edge_feat = self.edge_embeddings(edge_type_ids)
        src_nodes = edge_index[0]
        x = x + torch_scatter.scatter(edge_feat, src_nodes, dim=0, dim_size=x.size(0), reduce="mean")
        return self.conv(x, edge_index)
