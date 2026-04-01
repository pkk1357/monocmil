import torch.nn as nn
from models.abmil import GatedAttention
from models.mlp_head import ProjectionHead

class MonoCMIL_Network(nn.Module):
    def __init__(self, input_dim=512, d=256):
        super().__init__()
        # 1. Frozen Feature Extractor (CONCH 등) - 보통 외부에서 처리 [cite: 279]
        
        # 2. Trainable MIL Aggregator (ABMIL with Gated Attention) [cite: 279]
        self.mil_aggregator = GatedAttention(input_dim=input_dim, hidden_dim=input_dim)
        
        # 3. Trainable MLP Head (phi) -> z (Eq. 4.1) [cite: 280]
        # d = 256이 성능이 가장 좋음 (Table 5 근거) 
        self.mlp_head = ProjectionHead(input_dim=input_dim, z_dim=d)

    def forward(self, x):
        """
        f_bag = MIL({f_i}) [cite: 168]
        z = phi(f_bag) [cite: 280]
        """
        f_bag, A = self.mil_aggregator(x) # Slide-level representation [cite: 279]
        z = self.mlp_head(f_bag)          # Projected representation z [cite: 280]
        return z, f_bag, A