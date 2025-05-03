import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from datasets.feature_dataset import AccidentFeatureDataset
from models.ResNetLSTM import ResNetLSTM

TIMESTAMP = "0305080728"
EPOCHS = 100
ALPHA = 0.5
VAL_SPLIT = 0.25
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

feature_dir = f"ResNet_Features/batch_{TIMESTAMP}"
checkpoint_path = f"checkpoints/ResNetLSTM_best_{TIMESTAMP}.pth"

model = ResNetLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion_frame = nn.BCELoss(reduction="sum")
criterion_binary = nn.BCELoss(reduction="sum")
best_val_loss = float("inf")

batch_indices = sorted(set(f.split("_")[2][5:] for f in os.listdir(feature_dir) if f.endswith(".npy")))

model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    val_loss = 0.0
    total_train_samples = 0
    total_val_samples = 0

    for batch_idx in batch_indices:
        features = np.load(os.path.join(feature_dir, f"train_features_batch{batch_idx}_{TIMESTAMP}.npy"), allow_pickle=True)
        labels = np.load(os.path.join(feature_dir, f"train_labels_batch{batch_idx}_{TIMESTAMP}.npy"), allow_pickle=True)

        dataset = AccidentFeatureDataset(features, labels)
        val_size = int(VAL_SPLIT * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        for sequences, frame_labels, binary_labels in train_loader:
            model.train()
            sequences = sequences.to(device)
            frame_labels = frame_labels.to(device)
            binary_labels = binary_labels.to(device)

            total_train_samples += len(sequences)

            frame_preds, binary_pred = model(sequences)
            loss_frame = criterion_frame(frame_preds, frame_labels)
            loss_binary = criterion_binary(binary_pred, binary_labels)
            loss = ALPHA * loss_frame + (1 - ALPHA) * loss_binary

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for sequences, frame_labels, binary_labels in val_loader:
                sequences = sequences.to(device)
                frame_labels = frame_labels.to(device)
                binary_labels = binary_labels.to(device)

                total_val_samples += len(sequences)

                frame_preds, binary_pred = model(sequences)
                loss_frame = criterion_frame(frame_preds, frame_labels)
                loss_binary = criterion_binary(binary_pred, binary_labels)
                loss = ALPHA * loss_frame + (1 - ALPHA) * loss_binary
                val_loss += loss.item()

    avg_train_loss = running_loss / total_train_samples
    avg_val_loss = val_loss / total_val_samples
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("Saved best model so far.")

print("Training complete.")
