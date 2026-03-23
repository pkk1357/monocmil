import torch
import torch.nn.functional as F

def generate_orthogonal_anchor(dim, existing_anchors):
    while True:
        epsilon = torch.randn(dim)
        a_t = F.normalize(epsilon, p=2, dim=0)
    
        if not existing_anchors:
            return a_t

        sims = [F.cosine_similarity(a_t.unsqueeze(0), ea.unsqueeze(0)).item() for ea in existing_anchors]

        if all(-0.1 <= s <= 0.1 for s in sims):
            return a_t