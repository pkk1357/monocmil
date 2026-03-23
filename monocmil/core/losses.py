import torch
import torch.nn as nn
import torch.nn.functional as F

class HypersphereLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(HypersphereLoss, self).__init__()
        self.reduction = reduction

    def forward(self, z, labels):
        unique_classes = torch.unique(labels)
        total_loss = 0.0
        centers = {}

        for cls in unique_classes:
            class_mask = (labels == cls)
            z_c = z[class_mask]
            
            c = torch.mean(z_c, dim=0)
            centers[cls.item()] = c
            
            squared_distances = torch.sum((z_c - c) ** 2, dim=1)
            
            if self.reduction == 'mean':
                total_loss += torch.mean(squared_distances)
            else:
                total_loss += torch.sum(squared_distances)
                
        if self.reduction == 'mean':
            total_loss = total_loss / len(unique_classes)
            
        return total_loss, centers
    
    
class Phase2Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(Phase2Loss, self).__init__()
        self.beta = beta

    def forward(self, z_cur, z_past_dict, current_anchor, stage_idx, task_id):
        B = z_cur.shape[0]
        
        # Stage 1
        sim_cur = F.cosine_similarity(z_cur, current_anchor.unsqueeze(0))
        S_cur = torch.mean(sim_cur)
        
        S_past = 0.0
        t = task_id + 1
        if len(z_past_dict) > 0:
            total_past_sim = 0.0
            for past_id, z_past in z_past_dict.items():
                sim_past = F.cosine_similarity(z_past, current_anchor.unsqueeze(0))
                total_past_sim += torch.mean(sim_past)
            S_past = total_past_sim / len(z_past_dict)
            
        loss_stage1 = -torch.log(torch.exp(S_cur) / (torch.exp(S_cur) + torch.exp(S_past) + 1e-8))
        
        if stage_idx == 1 or len(z_past_dict) == 0:
            return loss_stage1
        
        # Stage 2    
        D_anchor_cur = torch.mean(torch.norm(z_cur - current_anchor.unsqueeze(0), p=2, dim=1))
        
        D_cur, D_past = 0.0, 0.0
        if B > 1:
            dist_matrix = torch.cdist(z_cur, z_cur, p=2)
            D_cur = torch.sum(torch.triu(dist_matrix, diagonal=1)) * (2.0 / (B * (B - 1)))
            
            if len(z_past_dict) > 0:
                for past_id, z_past in z_past_dict.items():
                    dist_matrix_past = torch.cdist(z_past, z_past, p=2)
                    D_past += torch.sum(torch.triu(dist_matrix_past, diagonal=1)) * (2.0 / (B * (B - 1)))
                D_past = D_past / len(z_past_dict)
                
        L_compact = D_past + D_cur + D_anchor_cur
        loss_stage2 = loss_stage1 + L_compact
        
        if stage_idx == 2:
            return loss_stage2
        
        # Stage 3    
        weighted_D_anchor_past = 0.0
        
        for past_id, z_past in z_past_dict.items():
            j = past_id + 1
            w_j = (t - j) * self.beta
            
            D_anchor_j_past = torch.mean(torch.norm(z_past - current_anchor.unsqueeze(0), p=2, dim=1))
            weighted_D_anchor_past += D_anchor_j_past * w_j
            
        weighted_D_anchor_past = weighted_D_anchor_past / len(z_past_dict)
        
        loss_stage3 = loss_stage1 - torch.log((weighted_D_anchor_past + 1e-8) / (weighted_D_anchor_past + L_compact + 1e-8))
        
        return loss_stage3