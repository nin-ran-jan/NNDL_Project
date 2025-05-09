import torch
import numpy as np
import os
import csv

from models.ResNetLSTM import ResNetLSTM
from datasets.feature_dataset import AccidentFeatureDataset
from torch.utils.data import DataLoader

TEST_TIMESTAMP = "0305101417"     
MODEL_TIMESTAMP = "0305080728"    
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

feature_dir = f"CLIP_ViT_Features_Test_clip-vit-large-patch14/run_250509_185359"
checkpoint_path = f"checkpoints/ViTLSTM_best_{MODEL_TIMESTAMP}.pth"

# batch_indices = sorted(set(
#     f.split("_")[2][5:] for f in os.listdir(feature_dir)
#     if f.startswith("test_features_batch") and f.endswith(".npy")
# ))
batch_indices = sorted(list(set(
    f.split("_")[4].split(".")[0] 
    for f in os.listdir(feature_dir)
    if f.startswith("train_features_saving_batch") and f.endswith(".npy")
)))

model = ResNetLSTM(input_dim=768, dropout=0).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

all_scores = []
all_ids = []

for batch_idx in batch_indices:
    # feature_path = os.path.join(feature_dir, f"test_features_batch{batch_idx}_{TEST_TIMESTAMP}.npy")
    # ids_path = os.path.join(feature_dir, f"test_ids_batch{batch_idx}_{TEST_TIMESTAMP}.npy")

    feature_file = f"test_features_saving_batch_{batch_idx}.npy"
    test_id_file = f"test_ids_saving_batch_{batch_idx}.npy"

    test_features = np.load(os.path.join(feature_dir, feature_file), allow_pickle=True)
    test_ids = np.load(os.path.join(feature_dir, test_id_file), allow_pickle=True)

    # test_features = np.load(feature_path, allow_pickle=True)
    # test_ids = np.load(ids_path, allow_pickle=True)

    dummy_frame_labels = np.zeros((len(test_features), test_features[0].shape[0], 1), dtype=np.float32)
    test_dataset = AccidentFeatureDataset(test_features, dummy_frame_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for sequences, _, _ in test_loader:
            sequences = sequences.to(device)
            _, binary_preds = model(sequences)
            scores = binary_preds.squeeze().cpu().numpy()
            all_scores.extend(scores.tolist())

    all_ids.extend(test_ids)

submission_path = f"submissions/submission_{TEST_TIMESTAMP}.csv"
os.makedirs("submissions", exist_ok=True)

with open(submission_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'score'])
    for vid_id, score in zip(all_ids, all_scores):
        writer.writerow([vid_id, f"{score:.4f}"])

print(f"Saved predictions to {submission_path}")
