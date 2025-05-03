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
        features = np.array(self.features[idx], dtype=np.float32)
        labels = np.array(self.labels[idx], dtype=np.float32)
        binary_label = torch.tensor(1.0 if np.max(self.labels[idx]) > 0.5 else 0.0, dtype=torch.float32)
        return torch.tensor(features, dtype=torch.float32), \
               torch.tensor(labels, dtype=torch.float32), \
               binary_label