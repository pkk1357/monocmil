import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, z_dim=256):
        super(ProjectionHead, self).__init__()
        # Table 5에서 성능이 가장 좋았던 d=256으로 매핑 [cite: 280, 797]
        self.phi = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, z_dim)
        )

    def forward(self, f_bag):
        # z = phi(f_bag) 
        z = self.phi(f_bag)
        return z