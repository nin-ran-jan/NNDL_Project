import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import time

from models.Vit_Transformer import ViTTransformer 
from datasets.feature_dataset import AccidentFeatureDataset 

# AI Prompts:
# Improve formatting
# And sensible outputs when runing the script
# Add control statements for errors
# Saving and config related code is generated using AI.

# config
TRAIN_RUN_TIMESTAMP = time.strftime("%Y%m%d%H%M%S") 
DATA_TIMESTAMP = "250509_162201" 

FEATURE_DIR_BASE = "processed_data/CLIP_ViT_Features_clip-vit-large-patch14"  
FEATURE_SUBDIR = f"run_{DATA_TIMESTAMP}" 
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH_TEMPLATE = os.path.join(CHECKPOINT_DIR, f"ViTTransformer_best_{TRAIN_RUN_TIMESTAMP}.pth")

# model hyperparameters
VIT_FEATURE_DIM = 768      
MODEL_DIM =  128           
N_HEADS =  1               
NUM_ENCODER_LAYERS = 2     
DIM_FEEDFORWARD = 640    
DROPOUT = 0.5

# training hyperparameters
EPOCHS = 50 
ALPHA = 0.9                
VAL_SPLIT = 0.20           
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4        

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Training run timestamp: {TRAIN_RUN_TIMESTAMP}")
print(f"Loading features from data timestamp: {DATA_TIMESTAMP}")
print(f"Current Dropout: {DROPOUT}, Weight Decay: {WEIGHT_DECAY}")


def load_all_features_and_labels(feature_dir_path):
    all_features = []
    all_labels = []
    
    try:
        batch_indices = sorted(list(set(
            f.split("_")[4].split(".")[0] 
            for f in os.listdir(feature_dir_path)
            if f.startswith("train_features_saving_batch") and f.endswith(".npy")
        )))
    except Exception as e:
        print(f"Error discovering feature batches: {e}")
        return None, None

    for batch_idx in batch_indices:
        feature_file = f"train_features_saving_batch_{batch_idx}.npy"
        label_file = f"train_labels_saving_batch_{batch_idx}.npy"
        
        feature_path = os.path.join(feature_dir_path, feature_file)
        label_path = os.path.join(feature_dir_path, label_file)

        try:
            features = np.load(feature_path, allow_pickle=True)
            labels = np.load(label_path, allow_pickle=True)
            all_features.extend(list(features)) 
            all_labels.extend(list(labels))     
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
            continue
            
    print(f"Loaded {len(all_features)} total sequences.")
    return all_features, all_labels

def collate_sequences(batch):
    sequences = [item[0] for item in batch]
    frame_labels_list = [item[1] for item in batch]
    binary_labels_list = [item[2] for item in batch]

    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    frame_labels_padded = pad_sequence(frame_labels_list, batch_first=True, padding_value=0.0) 

    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = sequences_padded.size(1)
    padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None] 

    binary_labels_batch = torch.stack(binary_labels_list)

    return sequences_padded, frame_labels_padded, binary_labels_batch, padding_mask

full_feature_dir = os.path.join(FEATURE_DIR_BASE, FEATURE_SUBDIR)
print(f"Attempting to load data from: {full_feature_dir}")

all_train_features, all_train_labels = load_all_features_and_labels(full_feature_dir)

if all_train_features is None or all_train_labels is None:
    print("Exiting due to data loading issues.")
    exit()

full_dataset = AccidentFeatureDataset(all_train_features, all_train_labels)

val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences)

model = ViTTransformer(
    feature_dim=VIT_FEATURE_DIM,
    model_dim=MODEL_DIM,
    nhead=N_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

criterion_frame = nn.BCELoss(reduction='none') 
criterion_binary = nn.BCELoss() 

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    total_train_sequences = 0

    for sequences, frame_labels, binary_labels, padding_mask in train_loader:
        sequences = sequences.to(device)
        frame_labels = frame_labels.to(device) 
        binary_labels = binary_labels.to(device) 
        padding_mask = padding_mask.to(device) 

        optimizer.zero_grad()
        frame_preds, binary_pred = model(sequences, src_key_padding_mask=padding_mask)
        
        loss_binary = criterion_binary(binary_pred, binary_labels)
        frame_loss_unreduced = criterion_frame(frame_preds, frame_labels.squeeze(-1)) 
        
        active_frames_mask = ~padding_mask 
        masked_frame_loss_elements = frame_loss_unreduced * active_frames_mask.float()
        
        num_active_frames_in_batch = active_frames_mask.sum()
        if num_active_frames_in_batch > 0:
            loss_frame = masked_frame_loss_elements.sum() / num_active_frames_in_batch
        else: 
            loss_frame = torch.tensor(0.0, device=device)

        combined_loss = ALPHA * loss_frame + (1 - ALPHA) * loss_binary
        
        combined_loss.backward()
        optimizer.step()

        running_train_loss += combined_loss.item() * sequences.size(0) 
        total_train_sequences += sequences.size(0)

    avg_train_loss = running_train_loss / total_train_sequences if total_train_sequences > 0 else 0

    model.eval()
    running_val_loss = 0.0
    total_val_sequences = 0

    with torch.no_grad():
        for sequences, frame_labels, binary_labels, padding_mask in val_loader:
            sequences = sequences.to(device)
            frame_labels = frame_labels.to(device)
            binary_labels = binary_labels.to(device)
            padding_mask = padding_mask.to(device)

            frame_preds, binary_pred = model(sequences, src_key_padding_mask=padding_mask)

            loss_binary_val = criterion_binary(binary_pred, binary_labels)
            
            frame_loss_unreduced_val = criterion_frame(frame_preds, frame_labels.squeeze(-1))
            active_frames_mask_val = ~padding_mask
            masked_frame_loss_elements_val = frame_loss_unreduced_val * active_frames_mask_val.float()
            
            num_active_frames_in_batch_val = active_frames_mask_val.sum()
            if num_active_frames_in_batch_val > 0:
                loss_frame_val = masked_frame_loss_elements_val.sum() / num_active_frames_in_batch_val
            else:
                loss_frame_val = torch.tensor(0.0, device=device)

            combined_loss_val = ALPHA * loss_frame_val + (1 - ALPHA) * loss_binary_val
            
            running_val_loss += combined_loss_val.item() * sequences.size(0)
            total_val_sequences += sequences.size(0)

    avg_val_loss = running_val_loss / total_val_sequences if total_val_sequences > 0 else 0

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        current_checkpoint_path = CHECKPOINT_PATH_TEMPLATE 
        torch.save(model.state_dict(), current_checkpoint_path)
        print(f"Saved best model to {current_checkpoint_path}")

print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved with timestamp {TRAIN_RUN_TIMESTAMP} if validation improved.")