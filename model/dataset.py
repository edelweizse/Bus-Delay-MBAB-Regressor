import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, features, delays):
        self.features = torch.FloatTensor(features)
        self.delays = torch.FloatTensor(delays)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.delays[idx]