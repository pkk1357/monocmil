import torch
import torch.nn as nn

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