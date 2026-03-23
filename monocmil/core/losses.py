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
    def __init__(self):
        super(Phase2Loss, self).__init__()

    def forward(self, z_cur, z_past_dict, current_anchor):

        sim_cur = F.cosine_similarity(z_cur, current_anchor.unsqueeze(0))
        S_cur = torch.mean(sim_cur)
        
        S_past = 0.0
        if len(z_past_dict) > 0:
            total_past_sim = 0.0
            for past_id, z_past in z_past_dict.items():
                sim_past = F.cosine_similarity(z_past, current_anchor.unsqueeze(0))
                total_past_sim += torch.mean(sim_past)
            S_past = total_past_sim / len(z_past_dict)
            
        loss_stage1 = -torch.log(torch.exp(S_cur) / (torch.exp(S_cur) + torch.exp(S_past) + 1e-8))
        
        return loss_stage1