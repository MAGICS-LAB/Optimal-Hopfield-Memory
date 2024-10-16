import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, sample_size, sample_dim):
        self.sample_size = sample_size
        self.sample_dim = sample_dim
        self.data = torch.randn(sample_size, sample_dim)
        self.data = torch.nn.functional.normalize(self.data, dim=-1)
        #self.data = self.data / torch.norm(self.data, dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample, 0
