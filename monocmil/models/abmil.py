import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512):
        super(GatedAttention, self).__init__()
        self.M = hidden_dim
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, self.M),
            nn.ReLU(),
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(0)

        H = self.feature_extractor(x)
        
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        f_bag = torch.mm(A, H)
        return f_bag, A