import torch
from torch.utils.data import Dataset

class TCGABags(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_path, label = self.data_list[index]
        
        bag_features = torch.load(file_path)
        
        label = torch.tensor(label, dtype=torch.long)

        return bag_features, label