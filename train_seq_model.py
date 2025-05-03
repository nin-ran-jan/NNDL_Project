import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np

from datasets.feature_dataset import AccidentFeatureDataset
from models.ResNetLSTM import ResNetLSTM

TIMESTAMP = "0305011141"
device = "cuda" if torch.cuda.is_available() else "cpu"

features = np.load(f"ResNet_Features/train_features_{TIMESTAMP}.npy", allow_pickle=True)
labels = np.load(f"ResNet_Features/train_labels_{TIMESTAMP}.npy", allow_pickle=True)

dataset = AccidentFeatureDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ResNetLSTM().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for sequences, label_seqs in dataloader:
        sequences = sequences.to(device)
        label_seqs = label_seqs.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, label_seqs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), f"checkpoints/ResNetLSTM_{TIMESTAMP}.pth")
print("Training complete. Model saved.")