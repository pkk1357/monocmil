import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from data.dataset import MonoCMILDataset, collate_MIL

@torch.no_grad()
def evaluate_continual_learning(t_idx, tasks, all_class_files, adapter_list, MLP_model, existing_anchors, device):
    MLP_model.eval()
    for adapter in adapter_list:
        adapter.eval()
        
    all_preds = []
    all_labels = []
    
    for true_task_idx in range(t_idx + 1):
        test_files = []
        for cid in tasks[true_task_idx]: 
            test_files.extend(all_class_files[cid])
            
        # 평가 속도를 위해 Task당 30개로 제한 (실제 최종 평가 시에는 제한 해제)
        test_files = test_files[:30] 
        
        test_dataset = MonoCMILDataset(test_files, true_task_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_MIL)
        
        for feat_list, _ in test_loader:
            feat = feat_list[0].to(device)
            
            sims = []
            for i, adapter in enumerate(adapter_list):
                z, _ = adapter(feat.unsqueeze(0))
                y = MLP_model(z)
                
                anchor_eval = existing_anchors[i].to(device)
                sim = F.cosine_similarity(y, anchor_eval.unsqueeze(0), dim=1).item()
                sims.append(sim)
            
            pred_task = np.argmax(sims)
            all_preds.append(pred_task)
            all_labels.append(true_task_idx)
            
    acc = accuracy_score(all_labels, all_preds) * 100
    bacc = balanced_accuracy_score(all_labels, all_preds) * 100
    wf1 = f1_score(all_labels, all_preds, average='weighted')
    
    return acc, bacc, wf1