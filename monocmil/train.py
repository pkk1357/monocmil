import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from data.dataset import scan_all_tcga_classes, MonoCMILDataset, collate_MIL
from models.abmil import GatedAttention
from models.mlp_head import ProjectionHead
from core.memory import FeatureMemoryBank
from core.losses import MonoCMIL_Loss
from core.anchor import generate_orthogonal_anchor
from utils.metrics import evaluate_continual_learning

# 1. 학습 유틸리티 정의
def progressive_patch_masking(features, step, total_steps, max_ratio=0.3):
    ratio = max_ratio * (step / total_steps)
    N = features.shape[0]
    indices = torch.randperm(N)[:max(1, int(N * (1 - ratio)))].to(features.device)
    return features[indices, :]

class Center(nn.Module):
    def __init__(self, init_tensor):
        super().__init__()
        self.c = nn.Parameter(init_tensor.squeeze(0))
    def forward(self):
        return self.c

def orth_loss(cur_adapter, adapter_list):
    loss = 0.0
    for prev_adapter in adapter_list:
        cur_w = cur_adapter.attention_w.weight
        prev_w = prev_adapter.attention_w.weight
        loss += torch.abs(torch.mean(torch.matmul(cur_w, prev_w.transpose(1, 0))))
    return loss

# 2. 설정 및 데이터 로드
SCENARIO = "A1"
BASE_PATH = "/data3/jinsol/TCGA/features_conch_256/pt_files"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR = 0.001
BATCH_SIZE = 10
STEPS_P1 = 80 
STEPS_P2 = 120 

all_class_files = scan_all_tcga_classes(BASE_PATH)
tasks = [[i] for i in range(25)] if SCENARIO == "A1" else [[0], [1], [2, 3], [4, 5], [6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17], [18, 19], [20, 21], [22], [23], [24]]

# 3. 모델 및 글로벌 상태 초기화
adapter_list = nn.ModuleList()
MLP_model = ProjectionHead(input_dim=512, hidden_dim=512, z_dim=256).to(DEVICE)
memory_bank = FeatureMemoryBank(z_dim=512)
criterion_p2 = MonoCMIL_Loss(beta=1.0, temp=0.1).to(DEVICE)
existing_anchors = [] 

# 4. 메인 학습 루프
for t_idx, class_ids in enumerate(tasks):
    print(f"\n{'='*70}\n [Task {t_idx+1}] 학습 시작 (Class IDs: {class_ids})\n{'='*70}")
    
    cur_adapter = GatedAttention(input_dim=512, hidden_dim=512).to(DEVICE)
    cur_adapter.requires_grad_(True)
    
    task_files = []
    for cid in class_ids: task_files.extend(all_class_files[cid])
    loader = DataLoader(MonoCMILDataset(task_files, t_idx), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_MIL)
    data_iter = iter(loader)

    # --- Center 초기화 ---
    cur_adapter.eval()
    init_z = []
    with torch.no_grad():
        for feat_list, _ in loader:
            for feat in feat_list:
                z, _ = cur_adapter(feat.to(DEVICE).unsqueeze(0))
                init_z.append(z.squeeze(0))
                if len(init_z) >= 10: break
            if len(init_z) >= 10: break
            
    center = Center(torch.stack(init_z).mean(dim=0, keepdim=True)).to(DEVICE)

    optimizer_MIL = optim.Adam(list(cur_adapter.parameters()) + list(center.parameters()), lr=LR)
    optimizer_MLP = optim.Adam(MLP_model.parameters(), lr=LR)
    scheduler_MIL = get_cosine_schedule_with_warmup(optimizer_MIL, num_warmup_steps=int(STEPS_P1*0.1), num_training_steps=STEPS_P1)
    scheduler_MLP = get_cosine_schedule_with_warmup(optimizer_MLP, num_warmup_steps=int(STEPS_P2*0.1), num_training_steps=STEPS_P2)

    # [Phase 1] MIL Aggregator Training (80 Steps)
    print("▶ Phase 1: MIL Adapter & Center 최적화 진행 중...")
    cur_adapter.train()
    center.train()
    
    for step in range(STEPS_P1):
        try: feat_list, _ = next(data_iter)
        except StopIteration: data_iter = iter(loader); feat_list, _ = next(data_iter)
        
        optimizer_MIL.zero_grad()
        accum_loss = 0.0
        
        for feat in feat_list:
            feat_masked = progressive_patch_masking(feat.to(DEVICE), step, STEPS_P1)
            aggregated_vec, _ = cur_adapter(feat_masked.unsqueeze(0)) 
            c = center()
            
            loss = torch.mean((aggregated_vec.squeeze(0) - c) ** 2)
            if len(adapter_list) > 0:
                loss += orth_loss(cur_adapter, adapter_list) * 0.001 * torch.exp(torch.tensor(-(t_idx+1.0), device=DEVICE))
            accum_loss += loss / len(feat_list)
            
        accum_loss.backward()
        optimizer_MIL.step()
        scheduler_MIL.step()
        
        if (step+1) % 20 == 0: print(f"   [Step {step+1}/{STEPS_P1}] MIL Loss: {accum_loss.item():.4f}")

    # --- 어댑터 저장 및 통계/앵커 업데이트 ---
    cur_adapter.requires_grad_(False)
    adapter_list.append(copy.deepcopy(cur_adapter))
    
    with torch.no_grad():
        clean_feat_list, _ = next(iter(loader))
        clean_z_list = []
        for clean_feat in clean_feat_list:
            clean_f_bag, _ = cur_adapter(clean_feat.to(DEVICE).unsqueeze(0))
            clean_z_list.append(clean_f_bag.squeeze(0))
            
        memory_bank.update_statistics(t_idx, torch.stack(clean_z_list))
        
        anchor = generate_orthogonal_anchor(256, existing_anchors).to(DEVICE)
        existing_anchors.append(anchor.cpu())

    # [Phase 2] MLP Head Training (120 Steps, 3-Stage)
    print("\n▶ Phase 2: MLP 3-Stage 공간 분리 최적화 진행 중...")
    cur_adapter.eval()
    MLP_model.train()
    
    step1, step2 = STEPS_P2 // 3, (STEPS_P2 * 2) // 3 
    stats_list = [] 
    
    for step in range(STEPS_P2):
        optimizer_MLP.zero_grad()
        try: feat_list, _ = next(data_iter)
        except StopIteration: data_iter = iter(loader); feat_list, _ = next(data_iter)
        
        z_RT_list = []
        for feat in feat_list:
            z_RT, _ = cur_adapter(feat.to(DEVICE).unsqueeze(0))
            z_RT_list.append(z_RT.squeeze(0))
        x_RT_batch = torch.stack(z_RT_list).detach()
        y_RT = MLP_model(x_RT_batch)
        
        # 딕셔너리 언패킹 로직 적용
        y_past_list = []
        if t_idx > 0:
            # 1. 함수를 한 번만 호출해서 모든 과거 데이터 딕셔너리를 받아옵니다.
            past_features_dict = memory_bank.sample_past_features(t_idx, len(feat_list), f_cur=x_RT_batch)
            
            # 2. 딕셔너리에서 키값(past_idx)으로 텐서를 하나씩 꺼내 씁니다.
            for past_idx in range(t_idx):
                x_past_fake = past_features_dict[past_idx].to(DEVICE)
                y_past = MLP_model(x_past_fake)
                y_past_list.append(y_past)
            
        loss_p2 = criterion_p2(
            MLP_model=MLP_model, x_RT_batch=x_RT_batch, y_RT=y_RT, 
            y_past_list=y_past_list, anchor=anchor, stats_list=stats_list, 
            step=step, step1=step1, step2=step2, task_id=t_idx
        )
        loss_p2.backward()
        optimizer_MLP.step()
        scheduler_MLP.step()

        if (step+1) % 40 == 0:
            stage = (step // 40) + 1
            print(f"   [Stage {stage} 완료] Step {step+1} Loss: {loss_p2.item():.4f}")

    # [평가] metrics.py의 함수를 호출하여 간단하게 실행
    print(f"\n[Task {t_idx+1}] 모델 평가 진행 중 (지금까지 배운 모든 Task 대상)...")
    
    acc, bacc, wf1 = evaluate_continual_learning(
        t_idx=t_idx, 
        tasks=tasks, 
        all_class_files=all_class_files, 
        adapter_list=adapter_list, 
        MLP_model=MLP_model, 
        existing_anchors=existing_anchors, 
        device=DEVICE
    )
    
    print(f"[Task {t_idx+1} 누적 평가 결과]")
    print(f"   - ACC (정확도)      : {acc:.2f}%")
    print(f"   - BACC (균형 정확도): {bacc:.2f}%")
    print(f"   - WF1 (가중치 F1)   : {wf1:.4f}")

print("\n전체 클래스 학습 완료")