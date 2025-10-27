import torch
from torch.utils.data import Dataset

class FluorescenceDataset(Dataset):

    def __init__(self, data):
        """
        fluorescence: (N, F, H, W)
        mu_a, mu_s: (N, H, W)
        concentration_fluor: (N, H, W)
        depth: (N, H, W)
        """
        super().__init__()
        self.fluorescence = torch.tensor(data['fluorescence'], dtype=torch.float32)
        self.mu_a = torch.tensor(data['mu_a'], dtype=torch.float32)  
        self.mu_s = torch.tensor(data['mu_s'], dtype=torch.float32)
        self.concentration = torch.tensor(data['concentration_fluor'], dtype=torch.float32)
        self.depth = torch.tensor(data['depth'], dtype=torch.float32)

        print("Dataset shapes:", self.fluorescence.shape, self.mu_a.shape, self.mu_s.shape, self.concentration.shape, self.depth.shape)

        self.fluorescence = self.fluorescence.permute(0,3,1,2).unsqueeze(1)
        self.op = torch.cat([self.mu_a.unsqueeze(1), self.mu_s.unsqueeze(1)], dim=1)
        self.concentration = self.concentration.unsqueeze(1)
        self.depth = self.depth.unsqueeze(1)

    def __len__(self):
        return self.fluorescence.shape[0]

    def __getitem__(self, idx):
        return self.fluorescence[idx], self.op[idx], self.concentration[idx], self.depth[idx]
