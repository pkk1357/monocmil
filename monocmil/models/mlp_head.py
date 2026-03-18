import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, z_dim=256):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x):
        return self.head(x)