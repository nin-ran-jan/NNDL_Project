import torch
import numpy as np
import os
import csv

from models.ResNetLSTM import ResNetLSTM
from datasets.feature_dataset import AccidentFeatureDataset
from torch.utils.data import DataLoader

TIMESTAMP = "0305051105"
MODEL_TIMESTAMP = "0305011141"
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

test_features = np.load(f"ResNet_Features/test_features_{TIMESTAMP}.npy", allow_pickle=True)
test_ids = np.load(f"ResNet_Features/test_ids_{TIMESTAMP}.npy", allow_pickle=True)

dummy_frame_labels = np.zeros((len(test_features), test_features.shape[1], 1), dtype=np.float32)
test_dataset = AccidentFeatureDataset(test_features, dummy_frame_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ResNetLSTM(dropout=0).to(device)
model.load_state_dict(torch.load(f"checkpoints/ResNetLSTM_best_{MODEL_TIMESTAMP}.pth", map_location=device))
model.eval()

all_scores = []
with torch.no_grad():
    for sequences, _, _ in test_loader:
        sequences = sequences.to(device)
        _, binary_preds = model(sequences)
        scores = binary_preds.squeeze().cpu().numpy()
        all_scores.extend(scores.tolist())

submission_path = f"submissions/submission_{TIMESTAMP}.csv"
with open(submission_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'score'])
    for vid_id, score in zip(test_ids, all_scores):
        writer.writerow([vid_id, f"{score:.4f}"])

print(f"Saved predictions to {submission_path}")
