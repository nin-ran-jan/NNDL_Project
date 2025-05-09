import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import time # For unique timestamps

# Import your models and dataset
from datasets.feature_dataset import AccidentFeatureDataset
from models.Vit_Transformer import ViTTransformer # Make sure this path is correct

# --- Configuration Section ---
# Timestamps and Paths
# Generate a unique timestamp for this training run
TRAIN_RUN_TIMESTAMP = time.strftime("%Y%m%d%H%M%S") 
# Timestamp or identifier for the ViT features you are using (e.g., when they were generated)
# This should match the suffix of your feature files if they have one.
# If your feature files are just "train_features_batchX.npy", you can simplify the loading logic.
# For this example, I'll assume there's a DATA_TIMESTAMP associated with the feature set.
DATA_TIMESTAMP = "250509_162201" # !!! UPDATE this to your feature set's timestamp or ID

FEATURE_DIR_BASE = "processed_data/CLIP_ViT_Features_clip-vit-large-patch14"  
# The script will look for files like "train_features_batch0_DATA_TIMESTAMP.npy"
# If your files are just "train_features_batch0.npy", adjust load_all_features_and_labels
FEATURE_SUBDIR = f"run_{DATA_TIMESTAMP}" 
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH_TEMPLATE = os.path.join(CHECKPOINT_DIR, f"ViTTransformer_best_{TRAIN_RUN_TIMESTAMP}.pth")

VIT_FEATURE_DIM = 768      # Dimension of features from ViT (e.g., 768 for ViT-Base)
MODEL_DIM = 512            # Transformer model's internal dimension (d_model)
N_HEADS = 8                # Number of attention heads
NUM_ENCODER_LAYERS = 4     # Number of Transformer encoder layers
DIM_FEEDFORWARD = 1024     # Dimension of the feedforward network
DROPOUT = 0.1              # Dropout rate

# Training Hyperparameters
EPOCHS = 50 # Reduced for quicker example, adjust as needed
ALPHA = 0.5                # Weight for frame loss vs binary loss
VAL_SPLIT = 0.20           # Proportion of data for validation
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
# --- End Configuration Section ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Training run timestamp: {TRAIN_RUN_TIMESTAMP}")
print(f"Loading features from data timestamp: {DATA_TIMESTAMP}")


def load_all_features_and_labels(feature_dir_path):
    """
    Loads all features and labels from .npy files in the specified directory.
    Assumes files are named like:
    train_features_saving_batch_<INDEX>.npy
    train_labels_saving_batch_<INDEX>.npy

    Adjust file naming pattern if yours is different.
    """
    all_features = []
    all_labels = []
    
    if not os.path.isdir(feature_dir_path):
        print(f"Error: Feature directory not found: {feature_dir_path}")
        return None, None

    # Discover batch indices based on feature files
    try:
        batch_indices = sorted(list(set(
            f.split("_")[4].split(".")[0]  # Extracts 'X' from 'train_features_saving_batch_X.npy'
            for f in os.listdir(feature_dir_path)
            if f.startswith("train_features_saving_batch") and f.endswith(".npy")
        )))
        if not batch_indices:
            print(f"Error: No feature batch files found in {feature_dir_path} with suffix '.npy'.")
            print(f"Example expected name: train_features_saving_batch0.npy")
            return None, None
        print(f"Found feature batch indices: {batch_indices}")
    except Exception as e:
        print(f"Error discovering feature batches: {e}")
        return None, None

    for batch_idx in batch_indices:
        feature_file = f"train_features_saving_batch_{batch_idx}.npy"
        label_file = f"train_labels_saving_batch_{batch_idx}.npy"
        
        feature_path = os.path.join(feature_dir_path, feature_file)
        label_path = os.path.join(feature_dir_path, label_file)

        if not os.path.exists(feature_path):
            print(f"Warning: Feature file not found: {feature_path}. Skipping.")
            continue
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}. Skipping.")
            continue
            
        try:
            features = np.load(feature_path, allow_pickle=True)
            labels = np.load(label_path, allow_pickle=True)
            all_features.extend(list(features)) # features is expected to be a list/array of sequences
            all_labels.extend(list(labels))     # labels is expected to be a list/array of label sequences
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
            continue
            
    if not all_features or not all_labels:
        print("No data loaded. Please check feature directory and file naming.")
        return None, None
        
    print(f"Loaded {len(all_features)} total sequences.")
    return all_features, all_labels

def collate_sequences(batch):
    """
    Collate function to pad sequences in a batch and create padding masks.
    Args:
        batch: A list of tuples, where each tuple is (sequence_features, frame_labels, binary_label).
               - sequence_features: Tensor of shape (seq_len, feature_dim)
               - frame_labels: Tensor of shape (seq_len, 1)
               - binary_label: Scalar tensor
    Returns:
        sequences_padded: Tensor of shape (batch_size, max_seq_len, feature_dim)
        frame_labels_padded: Tensor of shape (batch_size, max_seq_len, 1)
        binary_labels_batch: Tensor of shape (batch_size)
        padding_mask: Bool tensor of shape (batch_size, max_seq_len), True for padded elements.
    """
    # Unzip the batch
    sequences = [item[0] for item in batch]
    frame_labels_list = [item[1] for item in batch]
    binary_labels_list = [item[2] for item in batch]

    # Pad sequences (features)
    # pad_sequence expects a list of tensors, batch_first=True makes output (B, T, F)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Pad frame labels
    frame_labels_padded = pad_sequence(frame_labels_list, batch_first=True, padding_value=0.0) # Pad with 0 for labels

    # Create padding mask for sequences
    # True where padded, False where actual data
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = sequences_padded.size(1)
    padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None] # (B, S_max)

    # Stack binary labels
    binary_labels_batch = torch.stack(binary_labels_list)

    return sequences_padded, frame_labels_padded, binary_labels_batch, padding_mask

# --- Data Loading ---
# Construct the full path to the directory containing batched feature files
full_feature_dir = os.path.join(FEATURE_DIR_BASE, FEATURE_SUBDIR)
print(f"Attempting to load data from: {full_feature_dir}")

# Load all data
# Pass DATA_TIMESTAMP to match file names like "train_features_batch0_DATA_TIMESTAMP.npy"
# If your files don't have this timestamp suffix, pass an empty string or modify load_all_features_and_labels
all_train_features, all_train_labels = load_all_features_and_labels(full_feature_dir)

if all_train_features is None or all_train_labels is None:
    print("Exiting due to data loading issues.")
    exit()

# Create the full dataset
full_dataset = AccidentFeatureDataset(all_train_features, all_train_labels)

# Split dataset
val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Create DataLoaders with the collate function
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences)

# --- Model, Optimizer, Criterion ---
model = ViTTransformer(
    feature_dim=VIT_FEATURE_DIM,
    model_dim=MODEL_DIM,
    nhead=N_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # AdamW is often preferred for Transformers

# Use reduction='none' for frame loss to apply masking correctly before summing/meaning
criterion_frame = nn.BCELoss(reduction='none') 
criterion_binary = nn.BCELoss() # Default reduction is 'mean'

best_val_loss = float("inf")

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    total_train_active_frames = 0 # For averaging frame loss correctly
    total_train_sequences = 0

    for sequences, frame_labels, binary_labels, padding_mask in train_loader:
        sequences = sequences.to(device)
        frame_labels = frame_labels.to(device) # (B, S, 1)
        binary_labels = binary_labels.to(device) # (B)
        padding_mask = padding_mask.to(device) # (B, S), True for padded

        optimizer.zero_grad()

        frame_preds, binary_pred = model(sequences, src_key_padding_mask=padding_mask)
        # frame_preds: (B, S), binary_pred: (B)

        print(f"frame_preds: {frame_preds.shape}, binary_pred: {binary_pred.shape}")
        print(f"frame_labels: {frame_labels.shape}, binary_labels: {binary_labels.shape}")
        print(f"binary_labels: {binary_labels}")
        print(f"binary_pred: {binary_pred}")
        print(f"frame_preds: {frame_preds[0]}")
        print(f"frame_labels: {frame_labels.squeeze(-1)[0]}")

        # Binary loss (sequence-level)
        loss_binary = criterion_binary(binary_pred, binary_labels)

        # Frame loss (masked)
        # Squeeze frame_labels from (B, S, 1) to (B, S) to match frame_preds
        frame_loss_unreduced = criterion_frame(frame_preds, frame_labels.squeeze(-1)) # (B, S)
        
        # Mask for active (non-padded) frames: True for non-padded, False for padded
        active_frames_mask = ~padding_mask # (B, S)
        
        masked_frame_loss_elements = frame_loss_unreduced * active_frames_mask.float()
        
        num_active_frames_in_batch = active_frames_mask.sum()
        if num_active_frames_in_batch > 0:
            loss_frame = masked_frame_loss_elements.sum() / num_active_frames_in_batch
        else: # Avoid division by zero if a batch has all empty sequences (should not happen with real data)
            loss_frame = torch.tensor(0.0, device=device)

        # Combined loss
        # Note: loss_binary is already mean, loss_frame is now mean over active frames
        # So, their direct combination with ALPHA is a weighted average of two mean losses.
        combined_loss = ALPHA * loss_frame + (1 - ALPHA) * loss_binary
        
        combined_loss.backward()
        optimizer.step()

        running_train_loss += combined_loss.item() * sequences.size(0) # Accumulate loss scaled by batch size
        total_train_sequences += sequences.size(0)


    avg_train_loss = running_train_loss / total_train_sequences if total_train_sequences > 0 else 0

    # --- Validation Loop ---
    model.eval()
    running_val_loss = 0.0
    total_val_active_frames = 0
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

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Use the template for checkpoint path, ensuring it uses the TRAIN_RUN_TIMESTAMP
        current_checkpoint_path = CHECKPOINT_PATH_TEMPLATE 
        torch.save(model.state_dict(), current_checkpoint_path)
        print(f"Saved best model to {current_checkpoint_path}")

print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved with timestamp {TRAIN_RUN_TIMESTAMP} if validation improved.")
