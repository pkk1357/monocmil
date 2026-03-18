import torch
from torch.utils.data import Dataset

class TCGABags(Dataset):
    """
    실제 학습에 사용될 TCGA WSI Feature Bag 데이터셋.
    """
    def __init__(self, data_list):
        """
        Args:
            data_list (list): [(pt_file_path, class_label), ...] 형태의 리스트
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_path, label = self.data_list[index]
        
        # [N, 512] 형태의 패치 피처 로드
        bag_features = torch.load(file_path)
        
        # 라벨을 텐서로 변환
        label = torch.tensor(label, dtype=torch.long)

        return bag_features, label