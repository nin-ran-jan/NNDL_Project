from torch.utils.data import Dataset
import torch
import numpy as np

class AccidentFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(np.array(self.features[idx], dtype=np.float32), dtype=torch.float32), \
               torch.tensor(np.array(self.labels[idx], dtype=np.float32), dtype=torch.float32)