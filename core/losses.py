import torch
import torch.nn as nn
import torch.nn.functional as F

class MonoCMILLoss(nn.Module):
    def __init__(self, beta=1.0, temp=0.1):
        super(MonoCMILLoss, self).__init__()
        self.beta = beta    # Scaling hyperparameter for recency weight
        self.temp = temp    # Temperature scaling for directional alignment
        self.eps = 1e-8

    # ---------------------------------------------------------
    # Phase 1: MIL Training
    # ---------------------------------------------------------
    def forward_phase1(self, z_i_t, c_t):
        # L_MIL = sum ||z_i,t - c_t||^2
        L_MIL = torch.mean(torch.norm(z_i_t - c_t, p=2, dim=1)**2)
        return L_MIL

    # ---------------------------------------------------------
    # Phase 2: Three-stage MLP Training
    # ---------------------------------------------------------
    def forward_phase2(self, z_cur, z_past_list, a_t, stage, t):
        device = z_cur.device
        B = z_cur.size(0) # Batch size [cite: 399]

        # --- Stage 1: Directional Alignment (Eq. 5) [cite: 401] ---
        # S_cur_c: Average cosine similarity of current-class w.r.t anchor a(t) (Eq. 6) [cite: 402]
        S_cur_c = F.cosine_similarity(z_cur, a_t, dim=-1).mean()
        
        # S_past_c: Average cosine similarity of past-classes w.r.t anchor a(t) (Eq. 7) [cite: 405, 411, 457]
        if len(z_past_list) > 0:
            S_past_c_list = [F.cosine_similarity(z_p, a_t, dim=-1).mean() for z_p in z_past_list]
            S_past_c = torch.stack(S_past_c_list).mean()
        else:
            S_past_c = torch.tensor(-1.0, device=device)

        # L_Stage1 calculation (Eq. 5) [cite: 401]
        exp_S_cur = torch.exp(S_cur_c / self.temp)
        exp_S_past = torch.exp(S_past_c / self.temp)
        L_Stage1 = -torch.log(exp_S_cur / (exp_S_cur + exp_S_past + self.eps))

        if stage == 1:
            return L_Stage1

        # --- Stage 2: Intra-class Compactness (Eq. 11) [cite: 484] ---
        # D_anchor_cur: Mean Euclidean distance to current anchor (Eq. 8) [cite: 462, 476]
        D_anchor_cur = torch.norm(z_cur - a_t, p=2, dim=1).mean()

        # D_cur: Pairwise-distance compactness for current class (Eq. 9) [cite: 480]
        dist_cur = torch.cdist(z_cur, z_cur, p=2)
        D_cur = torch.sum(torch.triu(dist_cur, diagonal=1)) / (B * (B - 1) / 2 + self.eps)

        # D_past: Pairwise-distance compactness for past classes (Eq. 10) [cite: 480, 481]
        D_past_list = []
        for z_p in z_past_list:
            dist_p = torch.cdist(z_p, z_p, p=2)
            D_past_list.append(torch.sum(torch.triu(dist_p, diagonal=1)) / (B * (B - 1) / 2 + self.eps))
        D_past = torch.stack(D_past_list).mean() if D_past_list else torch.tensor(0.0, device=device)

        # L_Compact (Eq. 11) [cite: 484]
        L_Compact = D_past + D_cur + D_anchor_cur

        if stage == 2:
            # L_Stage2 = L_Stage1 + L_Compact (Eq. 12) [cite: 496, 503]
            return L_Stage1 + L_Compact

        # --- Stage 3: Structured Separation (Eq. 14) [cite: 472, 474] ---
        # D_past_anchor_j: Average anchor distance for past class j (Eq. 15) [cite: 473, 475]
        # w_j: Recency weight (Eq. 13) [cite: 468, 469]
        weighted_D_past_anchor_sum = 0.0
        for j_idx, z_p in enumerate(z_past_list):
            j = j_idx + 1 # Class index j < t [cite: 467]
            w_j = (t - j) * self.beta # Recency weight [cite: 468]
            D_past_anchor_j = torch.norm(z_p - a_t, p=2, dim=1).mean() # Eq. 15
            weighted_D_past_anchor_sum += D_past_anchor_j * w_j

        # Mean of weighted distances (Numerator of log term in Eq. 14)
        term_weighted_dist = weighted_D_past_anchor_sum / (t - 1 + self.eps)

        # L_Stage3 calculation (Eq. 14) [cite: 472]
        dist_ratio = (term_weighted_dist + self.eps) / (term_weighted_dist + L_Compact + self.eps)
        L_Stage3 = L_Stage1 - torch.log(dist_ratio)

        return L_Stage3