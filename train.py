import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. 사용자 정의 모듈 임포트
from models.network import MonoCMIL_Network
from core.losses import MonoCMILLoss
from core.anchor import generate_orthogonal_anchor
from core.memory import FeatureMemoryBank
# data/dataset.py에 scan_all_tcga_classes와 MonoCMILDataset이 있다고 가정합니다.
from data.dataset import MonoCMILDataset, collate_MIL, scan_all_tcga_classes, progressive_patch_masking

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n" + "="*50)
print(f"▶ 현재 사용 장치: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"▶ GPU 모델: {torch.cuda.get_device_name(0)}")
    print(f"▶ 사용 가능 GPU 개수: {torch.cuda.device_count()}")
print("="*50 + "\n")

# 2. 경로 및 하이퍼파라미터 설정 (보내주신 이미지 근거)
# 이미지 상의 경로: data/pt_files 내부에 TCGA-XXX 폴더들이 존재
BASE_PATH = "/media/jinsol/data/pt_files" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001
BATCH_SIZE = 10
STEPS_P1 = 80   # Phase 1: 80 steps [cite: 981]
STEPS_P2 = 120  # Phase 2: 120 steps [cite: 981]
Z_DIM = 256     # Table 5 최적 값 [cite: 797]

# 3. 데이터 스캔 및 Task(클래스) 리스트 생성
# 사용자님의 scan_all_tcga_classes 함수를 통해 {0: [파일들], 1: [...]} 딕셔너리 생성
all_label_files = scan_all_tcga_classes(BASE_PATH)

# 4. 모델 및 핵심 객체 초기화
model = MonoCMIL_Network(input_dim=512, d=Z_DIM).to(DEVICE)
criterion = MonoCMILLoss(beta=1.0, temp=0.1).to(DEVICE)
memory_bank = FeatureMemoryBank(d=512)
existing_anchors = []

# --- [메인 증분 학습 루프: 25개 클래스 순차 학습] ---
for t_idx in range(25):
    class_files = all_label_files[t_idx]
    if not class_files: # 데이터가 없는 클래스는 건너뜀
        continue
        
    t = t_idx + 1 
    print(f"\n{'='*30}\n[Task {t}] Class ID {t_idx} 학습 시작\n{'='*30}")

    # 데이터셋 및 로더 생성
    dataset = MonoCMILDataset(file_list=class_files, label_idx=t_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_MIL)
    
    # 새로운 클래스를 위한 고유 앵커 생성 (Eq. 2) [cite: 376]
    a_t = generate_orthogonal_anchor(d=Z_DIM, existing_anchors=existing_anchors).to(DEVICE)
    existing_anchors.append(a_t)

    # ---------------------------------------------------------
    # [Phase 1] MIL Aggregator Training (80 Steps)
    # 목적: 새로운 클래스의 특징을 센터 c(t)로 응집 [cite: 288]
    # ---------------------------------------------------------
    print("▶ Phase 1: MIL Aggregator 최적화 진행 중 (MLP 고정)...")
    model.mil_aggregator.train()
    model.mlp_head.requires_grad_(False) # MLP 가중치 고정 [cite: 981]
    
    optimizer_mil = optim.Adam(model.mil_aggregator.parameters(), lr=LR)
    data_iter = iter(loader)

    for step in range(STEPS_P1):
        try: features_list, labels = next(data_iter)
        except StopIteration: data_iter = iter(loader); features_list, labels = next(data_iter)
        
        # [Appendix A2] 점진적 패치 마스킹 적용 [cite: 981]
        masked_features = [progressive_patch_masking(f.to(DEVICE), step, STEPS_P1) for f in features_list]
        
        optimizer_mil.zero_grad()
        
        # 슬라이드별 임베딩 z 추출
        z_list = []
        for f in masked_features:
            # f: (N, 512) -> unsqueeze -> (1, N, 512)
            z, _, _ = model(f.unsqueeze(0))
            z_list.append(z)
        z_i_t = torch.cat(z_list, dim=0) # (Batch, Z_DIM)
        
        # c_t: 현재 클래스의 데이터 중심점 (Empirical mean) [cite: 294]
        c_t = z_i_t.mean(dim=0, keepdim=True).detach()
        
        # L_MIL 계산 (Eq. 1) [cite: 290]
        loss_p1 = criterion.forward_phase1(z_i_t, c_t)
        loss_p1.backward()
        optimizer_mil.step()
        
        if (step+1) % 20 == 0:
            print(f"   [Step {step+1}/{STEPS_P1}] MIL Loss: {loss_p1.item():.4f}")

    # ---------------------------------------------------------
    # [Phase 2] MLP Head Training (120 Steps, 3-Stage)
    # 목적: 앵커를 활용한 공간 분리 및 과거 지식 복습 [cite: 397]
    # ---------------------------------------------------------
    print("▶ Phase 2: MLP Head 3단계 최적화 진행 중 (MIL 고정)...")
    model.mil_aggregator.eval() 
    model.mil_aggregator.requires_grad_(False) # MIL 가중치 고정 [cite: 133]
    model.mlp_head.requires_grad_(True)
    
    optimizer_mlp = optim.Adam(model.mlp_head.parameters(), lr=LR)

    for step in range(STEPS_P2):
        # 40스텝마다 Stage 1->2->3 변경 [cite: 981]
        stage = (step // 40) + 1 
        
        try: features_list, labels = next(data_iter)
        except StopIteration: data_iter = iter(loader); features_list, labels = next(data_iter)
        
        optimizer_mlp.zero_grad()

        # 1. 현재 데이터의 특징(f_bag) 추출 및 투사(z)
        f_bag_list = []
        with torch.no_grad():
            for f in features_list:
                f_bag, A = model.mil_aggregator(f.to(DEVICE).unsqueeze(0))
                f_bag_list.append(f_bag)
        z_cur = model.mlp_head(torch.cat(f_bag_list, dim=0))
        
        # 2. 과거 데이터 합성 (Generative Feature Replay) [cite: 388]
        z_past_list = []
        if t > 1:
            # j=1...t-1 까지의 통계량에서 샘플링 [cite: 388]
            f_past_dict = memory_bank.sample_past_features(t_idx, BATCH_SIZE)
            z_past_list = [model.mlp_head(f_p.to(DEVICE)) for f_p in f_past_dict.values()]

        # 3. 논문의 3단계 로스 적용 (Eq. 5 ~ 15) [cite: 401, 484, 472]
        loss_p2 = criterion.forward_phase2(z_cur, z_past_list, a_t, stage, t)
        loss_p2.backward()
        optimizer_mlp.step()

        if (step+1) % 40 == 0:
            print(f"   [Stage {stage} 완료] Step {step+1} Loss: {loss_p2.item():.4f}")

    # --- Task 종료 후 처리: 다음 복습을 위한 통계량(mu, sigma) 저장 [cite: 387] ---
    with torch.no_grad():
        all_f_bags = []
        for feat_list, _ in loader:
            for f in feat_list:
                f_bag, A = model.mil_aggregator(f.to(DEVICE).unsqueeze(0))
                all_f_bags.append(f_bag)
        memory_bank.update_statistics(t_idx, torch.cat(all_f_bags, dim=0))

print("\n[학습 종료] 모든 TCGA 클래스 학습이 완료되었습니다! 🎉")