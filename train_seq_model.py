from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from datasets.feature_dataset import AccidentFeatureDataset

features = np.load("train_features.npy")
labels = np.load("train_labels.npy")

dataset = AccidentFeatureDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ResNetLSTM().to(device)
criterion = nn.BCELoss()  # Because labels are soft (0 to 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for sequences, label_seqs in dataloader:
        sequences = sequences.to(device)  # (B, 5, 512)
        label_seqs = label_seqs.to(device)  # (B, 5)

        outputs = model(sequences)  # (B, 5)
        loss = criterion(outputs, label_seqs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
