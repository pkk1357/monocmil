import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512):
        super(GatedAttention, self).__init__()
        self.L = 128  # 논문 관례 및 코드 구조상의 차원
        self.M = hidden_dim
        
        # Gated Attention 구조 (Eq. from Ilse et al.)
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, 1)

    def forward(self, h):
        """
        h: Instance embeddings (N, M) 또는 (1, N, M)
        """
        # 만약 배치 차원(1)이 포함되어 있다면 제거하여 (N, M)으로 만듭니다. [cite: 167]
        if h.dim() == 3:
            h = h.squeeze(0) # (1, N, 512) -> (N, 512)

        A_V = self.attention_V(h)  # Tanh branch
        A_U = self.attention_U(h)  # Sigmoid branch
    
        # Element-wise multiplication (Gating)
        A = self.attention_w(A_V * A_U) # (N, 1)
        A = torch.transpose(A, 1, 0)     # (1, N)
        A = F.softmax(A, dim=1)         # Attention weights (Eq. 279 관련)

        # 행렬 곱셈: (1, N) * (N, M) = (1, M) [cite: 168, 868]
        f_bag = torch.mm(A, h)  
        return f_bag, A