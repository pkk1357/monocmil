import torch.nn as nn
from models.abmil import GatedAttention
from models.mlp_head import ProjectionHead

class MonoCMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, z_dim=256):
        super(MonoCMIL, self).__init__()
        self.abmil = GatedAttention(input_dim=input_dim, hidden_dim=hidden_dim)
        self.mlp_head = ProjectionHead(input_dim=hidden_dim, hidden_dim=hidden_dim, z_dim=z_dim)

    def forward(self, x):
        f_bag, A = self.abmil(x)
        z = self.mlp_head(f_bag)
        return z, f_bag, A