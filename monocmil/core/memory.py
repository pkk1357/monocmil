import torch
import torch.nn.functional as F

class FeatureMemoryBank:
    def __init__(self, z_dim=256):
        self.anchors = {}
        self.mu = {}
        self.sigma = {}
        self.z_dim = z_dim

    def generate_orthogonal_anchor(self, task_id, device):
        while True:
            epsilon = torch.randn(self.z_dim, device=device)
            a_t = F.normalize(epsilon, p=2, dim=0)
            
            if task_id == 0:
                self.anchors[task_id] = a_t
                return a_t
                
            existing_anchors = list(self.anchors.values())
            sims = [F.cosine_similarity(a_t.unsqueeze(0), ea.unsqueeze(0)).item() for ea in existing_anchors]
            
            if all(-0.1 <= s <= 0.1 for s in sims):
                self.anchors[task_id] = a_t
                return a_t

    def update_statistics(self, task_id, f_bags):
        self.mu[task_id] = torch.mean(f_bags, dim=0)
        self.sigma[task_id] = torch.std(f_bags, dim=0) + 1e-6

    def sample_past_features(self, task_id, batch_size, f_cur=None):
        past_features = {}
        
        if task_id == 0 and f_cur is not None:
            noise = torch.randn_like(f_cur) * 0.2
            f_past = f_cur + noise
            f_past = (f_past - f_past.mean(dim=0)) / (f_past.std(dim=0) + 1e-6)
            past_features[0] = f_past
            return past_features

        for past_id in range(task_id):
            v_random = torch.randn(batch_size, self.mu[past_id].size(0), device=self.mu[past_id].device)
            f_past = v_random * self.sigma[past_id] + self.mu[past_id]
            past_features[past_id] = f_past
            
        return past_features