import os
import torch
from torch.utils.data import Dataset

def collate_MIL(batch):
    img = [item[0] for item in batch]
    label = [item[1] for item in batch]
    return [img, torch.LongTensor(label)]

class MonoCMILDataset(Dataset):
    def __init__(self, file_list, label_idx):
        self.file_list = file_list
        self.label_idx = label_idx
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # WSI 피처 로드
        path = self.file_list[idx]
        data = torch.load(path, map_location="cpu")
        return data, self.label_idx

def get_tcga_label_info(file_path):
    # 파일명 바코드의 14-15번째 자리로 Tumor/Normal 구분
    file_name = os.path.basename(file_path)
    parts = file_name.split('-')
    if len(parts) < 4: return None
    code = int(parts[3][:2])
    return "Tumor" if code < 10 else "Normal"

def scan_all_tcga_classes(base_path):
    # Table A1의 25개 클래스를 자동으로 스캔하여 분류
    tumor_to_organ = {
        'ACC': 'Adrenal', 'BLCA': 'Bladder', 'BRCA': 'Breast', 'CHOL': 'Bileduct',
        'COAD': 'Colon', 'DLBC': 'Lymph', 'GBM': 'Brain', 'KICH': 'Kidney',
        'LGG': 'Brain', 'LUAD': 'Lung', 'LUSC': 'Lung', 'MESO': 'Mesothelium',
        'OV': 'Ovary', 'PCPG': 'Adrenal-multi', 'PRAD': 'Prostate',
        'TGCT': 'Testis', 'UCS': 'Uterus'
    }
    
    label_to_files = {i: [] for i in range(25)}
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path): continue
        
        tumor = folder.split('-')[-1]
        if tumor not in tumor_to_organ: continue
        organ = tumor_to_organ[tumor]
        
        for f in os.listdir(folder_path):
            if not f.endswith('.pt'): continue
            path = os.path.join(folder_path, f)
            sample_type = get_tcga_label_info(path)
            is_tumor = (sample_type == "Tumor")
            
            # 논문 Table A1 매핑
            cid = -1
            if organ == 'Adrenal' and is_tumor: cid = 0
            elif organ == 'Adrenal-multi' and is_tumor: cid = 1
            elif organ == 'Bileduct': cid = 3 if is_tumor else 2
            elif organ == 'Bladder': cid = 5 if is_tumor else 4
            elif organ == 'Brain': cid = 6 if tumor == 'LGG' else 7
            elif organ == 'Breast': cid = 9 if is_tumor else 8
            elif organ == 'Colon': cid = 11 if is_tumor else 10
            elif organ == 'Kidney': cid = 13 if is_tumor else 12
            elif organ == 'Lung': 
                if not is_tumor: cid = 14
                else: cid = 15 if tumor == 'LUSC' else 16
            elif organ == 'Lymph': cid = 17
            elif organ == 'Mesothelium': cid = 18
            elif organ == 'Ovary': cid = 20 if is_tumor else 19
            elif organ == 'Prostate': cid = 22 if is_tumor else 21
            elif organ == 'Testis': cid = 23
            elif organ == 'Uterus': cid = 24
            
            if cid != -1: label_to_files[cid].append(path)
            
    return label_to_files

def progressive_patch_masking(features, step, total_steps=80, max_ratio=0.3):
    """
    학습 안정화를 위해 패치의 일부를 무작위로 제거합니다.
    """
    ratio = max_ratio * (step / total_steps)
    n_patches = features.shape[0]
    # 남길 패치 개수 계산
    n_keep = max(1, int(n_patches * (1 - ratio)))
    
    indices = torch.randperm(n_patches)[:n_keep].to(features.device)
    return features[indices, :]