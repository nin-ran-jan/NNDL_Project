import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
from torchvision import transforms
from datasets.video_dataset import FrameCollector
from PIL import Image
from tqdm import tqdm  # ✅ added

# ==== Configuration ====
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_NAME = "MCG-NJU/videomae-base"
CSV_PATH = "nexar-collision-prediction/train.csv"
VIDEO_DIR = "nexar-collision-prediction/train"
FPS = 3
SEQUENCE_LENGTH = 30
BATCH_SIZE = 4
EPOCHS = 10
VAL_SPLIT = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Checkpoint helpers ====
def save_checkpoint(model, optimizer, epoch, val_acc, filename="best_checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }, os.path.join(CHECKPOINT_DIR, filename))

def load_checkpoint(model, optimizer, filename="best_checkpoint.pt"):
    path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1, checkpoint['val_acc']
    return 0, 0.0

# ==== Preprocessing ====
feature_extractor = VideoMAEFeatureExtractor.from_pretrained(MODEL_NAME)
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

df = pd.read_csv(CSV_PATH)
df["id"] = df["id"].apply(lambda x: str(x).zfill(5))
collector = FrameCollector(df, VIDEO_DIR, fps_target=FPS, sequence_length=SEQUENCE_LENGTH)
frames_per_video, _, labels_per_video = collector.collect()

# ==== Prepare data ====
def collate_fn(batch):
    pixel_values = []
    labels = []
    for frames, label in batch:
        # Preprocess the video
        clip = [video_transform(f) for f in frames]
        clip = torch.stack(clip)  # [T, C, H, W]
        inputs = feature_extractor(clip, return_tensors="pt")
        pixel_values.append(inputs["pixel_values"].squeeze(0))
        labels.append(label)
    return torch.stack(pixel_values), torch.tensor(labels)

# Determine binary label per video (1 if max frame label > 0.5)
binary_labels = [1 if np.max(lbls) > 0.5 else 0 for lbls in labels_per_video]
dataset = list(zip(frames_per_video, binary_labels))
val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ==== Model ====
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# ==== Training ====
start_epoch, best_val_acc = load_checkpoint(model, optimizer)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    # ✅ tqdm progress bar added here
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for pixel_values, labels in train_bar:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())  # live loss display

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values)
            _, preds = torch.max(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.4f}")

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, optimizer, epoch, val_acc)
        print("New best checkpoint saved.")
    else:
        print("No improvement.")

print("Training complete.")
