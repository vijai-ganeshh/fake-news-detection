import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)   # 768 → 256
        self.conv2 = GCNConv(hidden_dim, output_dim)  # 256 → 128

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
