import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import numpy as np

from datasets.feature_dataset import AccidentFeatureDataset
from models.ResNetLSTM import ResNetLSTM

TIMESTAMP = "0305011141"
EPOCHS = 100
ALPHA = 0.5
VAL_SPLIT = 0.2
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

features = np.load(f"ResNet_Features/train_features_{TIMESTAMP}.npy", allow_pickle=True)
labels = np.load(f"ResNet_Features/train_labels_{TIMESTAMP}.npy", allow_pickle=True)

dataset = AccidentFeatureDataset(features, labels)

val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = ResNetLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion_frame = nn.BCELoss()
criterion_binary = nn.BCELoss()

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for sequences, frame_labels, binary_labels in train_loader:
        sequences = sequences.to(device)
        frame_labels = frame_labels.to(device)
        binary_labels = binary_labels.to(device)

        frame_preds, binary_pred = model(sequences)
        loss_frame = criterion_frame(frame_preds, frame_labels)
        loss_binary = criterion_binary(binary_pred, binary_labels)
        loss = ALPHA * loss_frame + (1 - ALPHA) * loss_binary

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, frame_labels, binary_labels in val_loader:
            sequences = sequences.to(device)
            frame_labels = frame_labels.to(device)
            binary_labels = binary_labels.to(device)

            frame_preds, binary_pred = model(sequences)
            loss_frame = criterion_frame(frame_preds, frame_labels)
            loss_binary = criterion_binary(binary_pred, binary_labels)
            loss = ALPHA * loss_frame + (1 - ALPHA) * loss_binary
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"checkpoints/ResNetLSTM_best_{TIMESTAMP}.pth")
        print("Saved best model so far.")

print("Training complete.")
