import torch

class FeatureMemoryBank:
    def __init__(self, d=512):
        self.stats = {} # {class_idx: (mu_j, sigma_j)} [cite: 387]

    def update_statistics(self, class_idx, f_bag_list):
        """
        학습이 끝난 후 해당 클래스의 평균(mu)과 표준편차(sigma) 저장 [cite: 387]
        """
        mu_j = torch.mean(f_bag_list, dim=0)
        sigma_j = torch.std(f_bag_list, dim=0)
        self.stats[class_idx] = (mu_j, sigma_j)

    def sample_past_features(self, current_t, batch_size):
        """
        f_past_j = v_random * sigma_j + mu_j (Eq. 3) [cite: 389]
        """
        past_features_dict = {}
        for j in range(current_t): # j = 1, ..., t-1 [cite: 388]
            if j in self.stats:
                mu_j, sigma_j = self.stats[j]
                v_random = torch.randn(batch_size, mu_j.size(0)).to(mu_j.device)
                # Eq. 3 적용 [cite: 389]
                f_past_j = v_random * sigma_j + mu_j
                past_features_dict[j] = f_past_j
        return past_features_dict