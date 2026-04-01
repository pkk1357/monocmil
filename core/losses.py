import torch
import torch.nn as nn
import torch.nn.functional as F

def cos_sim(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean()

def euclid_dist(a, b=None):
    if b is None:
        # Intra-class tightness: 배치 내 샘플 간 평균 거리
        B = a.size(0)
        if B <= 1: return torch.tensor(0.0, device=a.device)
        dist_matrix = torch.cdist(a, a, p=2)
        return torch.sum(torch.triu(dist_matrix, diagonal=1)) / (B * (B - 1) / 2 + 1e-8)
    else:
        # Anchor distance: 샘플과 앵커 간 평균 거리
        return torch.norm(a - b, p=2, dim=1).mean()

class MonoCMIL_Loss(nn.Module):
    def __init__(self, beta=1.0, temp=0.1):
        super().__init__()
        self.beta = beta
        self.temp = temp
        self.eps = 1e-8

    def forward(self, MLP_model, x_RT_batch, y_RT, y_past_list, anchor, stats_list, step, step1, step2, task_id):
        device = y_RT.device
        
        # 1. Contrastive Loss (Stage 1) - Log-Sum-Exp 적용
        cos_RT = cos_sim(y_RT, anchor)
        
        # 과거 데이터 유사도 합산
        cos_sum = []
        if len(y_past_list) > 0:
            for y_past in y_past_list:
                cos_sum.append(cos_sim(y_past, anchor))
            cos_SP_avg = torch.stack(cos_sum).mean()
        else:
            cos_SP_avg = torch.tensor(-1.0, device=device)

        # 수치 안정성을 위한 log_softmax 구조
        logits = torch.stack([cos_RT, cos_SP_avg]) / self.temp
        contrastive_loss = -F.log_softmax(logits, dim=0)[0]

        if step < step1:
            return contrastive_loss

        # 2. Stage 2 & 3: Compactness & Separation
        eu_RT_tight = euclid_dist(y_RT)
        eu_RT_anchor = euclid_dist(y_RT, anchor)
        
        eu_tight_sum = []
        eu_anchor_sum = []
        
        for past_idx, y_past in enumerate(y_past_list):
            weight = (task_id + 1 - (past_idx + 1)) * 0.1 # ref_spacing
            
            eu_tight_sum.append(euclid_dist(y_past))
            if step >= step2:
                eu_anchor_sum.append(euclid_dist(y_past, anchor) * weight)

        eu_SP_tight_avg = torch.stack(eu_tight_sum).mean() if eu_tight_sum else torch.tensor(0.0, device=device)
        
        if step1 <= step < step2:
            return contrastive_loss + eu_RT_tight + eu_SP_tight_avg + eu_RT_anchor
        
        else: # step >= step2 (Structured Separation)
            eu_SP_anchor_avg = torch.stack(eu_anchor_sum).mean() if eu_anchor_sum else torch.tensor(0.0, device=device)
            # dist_ratio가 0이 되지 않게 eps 추가
            denom = eu_SP_anchor_avg + eu_RT_tight + eu_SP_tight_avg + eu_RT_anchor + self.eps
            dist_ratio = (eu_SP_anchor_avg + self.eps) / denom
            return contrastive_loss - torch.log(dist_ratio)