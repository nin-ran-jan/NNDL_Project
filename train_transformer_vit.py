import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import time # For unique timestamps

# Import your models and dataset
from models.Vit_Transformer import ViTTransformer 
from datasets.feature_dataset import AccidentFeatureDataset 

# --- Configuration Section ---
# Timestamps and Paths
TRAIN_RUN_TIMESTAMP = time.strftime("%Y%m%d%H%M%S") 
DATA_TIMESTAMP = "250509_162201" # From your previous script

FEATURE_DIR_BASE = "processed_data/CLIP_ViT_Features_clip-vit-large-patch14"  
FEATURE_SUBDIR = f"run_{DATA_TIMESTAMP}" 
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH_TEMPLATE = os.path.join(CHECKPOINT_DIR, f"ViTTransformer_best_{TRAIN_RUN_TIMESTAMP}.pth")

# --- Hyperparameters for Experimentation ---

# 1. Model Architecture (Reduce complexity to fight overfitting)
VIT_FEATURE_DIM = 768      # Dimension of features from ViT (fixed by your features)
MODEL_DIM = 256            # Example: Reduced from 512. Try [128, 256, 384]
N_HEADS = 4                # Example: Reduced from 8. Ensure MODEL_DIM % N_HEADS == 0. Try [2, 4]
NUM_ENCODER_LAYERS = 2     # Example: Reduced from 4. Try [1, 2, 3]
DIM_FEEDFORWARD = 512      # Example: Reduced from 1024. Often 2x or 4x MODEL_DIM. Try [256, 512, 1024]

# 2. Regularization
DROPOUT = 0.3              # Example: Increased from 0.25. Try [0.2, 0.3, 0.4, 0.5]
WEIGHT_DECAY = 5e-4        # Example: Increased from 1e-4. Try [1e-5, 1e-4, 5e-4, 1e-3, 1e-2]

# 3. Training Hyperparameters
EPOCHS = 100 # Increased max epochs, early stopping will handle actual duration
ALPHA = 0.5                # Weight for frame loss vs binary loss. Could also be tuned.
VAL_SPLIT = 0.20           
BATCH_SIZE = 32
LEARNING_RATE = 1e-4       # Initial learning rate. Scheduler will adjust it.

# 4. Learning Rate Scheduler
SCHEDULER_PATIENCE = 7     # Patience for ReduceLROnPlateau (epochs)
SCHEDULER_FACTOR = 0.1     # Factor by which LR is reduced

# 5. Early Stopping
EARLY_STOPPING_PATIENCE = 15 # Number of epochs to wait for val_loss improvement before stopping
                             # Should be greater than scheduler patience to allow LR reduction to take effect.
# 6. Gradient Clipping
GRADIENT_CLIP_VALUE = 1.0  # Max norm of the gradients. Set to None to disable.

# --- End Hyperparameter Section ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Training Run: {TRAIN_RUN_TIMESTAMP} ---")
print(f"Using device: {device}")
print(f"Features: {FEATURE_DIR_BASE}/{FEATURE_SUBDIR} (Data Timestamp: {DATA_TIMESTAMP})")
print(f"Model Config: MODEL_DIM={MODEL_DIM}, N_HEADS={N_HEADS}, LAYERS={NUM_ENCODER_LAYERS}, DFF={DIM_FEEDFORWARD}")
print(f"Regularization: DROPOUT={DROPOUT}, WEIGHT_DECAY={WEIGHT_DECAY}")
print(f"Training Params: LR={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, ALPHA={ALPHA}")
print(f"Scheduler: ReduceLROnPlateau (Patience={SCHEDULER_PATIENCE}, Factor={SCHEDULER_FACTOR})")
print(f"Early Stopping: Patience={EARLY_STOPPING_PATIENCE}")
print(f"Gradient Clipping: {GRADIENT_CLIP_VALUE if GRADIENT_CLIP_VALUE else 'Disabled'}")


def load_all_features_and_labels(feature_dir_path):
    all_features = []
    all_labels = []
    if not os.path.isdir(feature_dir_path):
        print(f"Error: Feature directory not found: {feature_dir_path}")
        return None, None
    try:
        batch_indices = sorted(list(set(
            f.split("_")[4].split(".")[0] 
            for f in os.listdir(feature_dir_path)
            if f.startswith("train_features_saving_batch") and f.endswith(".npy")
        )))
        if not batch_indices:
            print(f"Error: No feature batch files found in {feature_dir_path} matching 'train_features_saving_batch_X.npy'.")
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

        if not os.path.exists(feature_path) or not os.path.exists(label_path):
            print(f"Warning: Missing files for batch {batch_idx}. Skipping.")
            continue
        try:
            features = np.load(feature_path, allow_pickle=True)
            labels = np.load(label_path, allow_pickle=True)
            # Basic validation for loaded features/labels per file
            if len(features) != len(labels):
                print(f"Warning: Mismatch in feature ({len(features)}) and label ({len(labels)}) counts in batch {batch_idx}. Skipping this batch file.")
                continue
            
            valid_features_in_batch = []
            valid_labels_in_batch = []
            for i in range(len(features)):
                f_seq = features[i]
                l_seq = labels[i]
                if isinstance(f_seq, np.ndarray) and f_seq.ndim == 2 and f_seq.shape[0] > 0 and f_seq.shape[1] == VIT_FEATURE_DIM:
                    if isinstance(l_seq, np.ndarray) and l_seq.ndim == 2 and l_seq.shape[0] == f_seq.shape[0]: # Ensure label seq length matches feature seq length
                        valid_features_in_batch.append(f_seq)
                        valid_labels_in_batch.append(l_seq)
                    else:
                        print(f"Warning: Invalid label sequence or length mismatch for item {i} in batch {batch_idx}. Feature shape: {f_seq.shape}, Label type: {type(l_seq)}, Label shape: {l_seq.shape if isinstance(l_seq, np.ndarray) else 'N/A'}. Skipping item.")
                else:
                    print(f"Warning: Invalid feature sequence for item {i} in batch {batch_idx} (shape: {f_seq.shape if isinstance(f_seq, np.ndarray) else 'N/A'}, type: {type(f_seq)}). Skipping item.")

            all_features.extend(valid_features_in_batch)
            all_labels.extend(valid_labels_in_batch)
        except Exception as e:
            print(f"Error loading or processing batch {batch_idx}: {e}")
            continue
            
    if not all_features or not all_labels:
        print("No valid data loaded after filtering. Please check feature directory and file naming/content.")
        return None, None
        
    print(f"Loaded {len(all_features)} total valid sequences.")
    return all_features, all_labels

def collate_sequences(batch):
    sequences = [item[0] for item in batch]
    frame_labels_list = [item[1] for item in batch]
    binary_labels_list = [item[2] for item in batch]

    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    frame_labels_padded = pad_sequence(frame_labels_list, batch_first=True, padding_value=0.0) 

    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = sequences_padded.size(1)
    if max_len == 0: # Should not happen if load_all_features_and_labels filters correctly
        print("Warning: collate_sequences received batch yielding max_len=0.")
        # Create minimal non-empty tensors to avoid downstream errors if this somehow happens
        # This assumes batch_size is at least 1
        batch_size_actual = len(sequences)
        sequences_padded = torch.zeros((batch_size_actual, 1, VIT_FEATURE_DIM), dtype=torch.float32)
        frame_labels_padded = torch.zeros((batch_size_actual, 1, 1), dtype=torch.float32)
        padding_mask = torch.ones((batch_size_actual, 1), dtype=torch.bool) # Mask everything
    else:
        padding_mask = torch.arange(max_len)[None, :] >= lengths[:, None] 

    binary_labels_batch = torch.stack(binary_labels_list)
    return sequences_padded, frame_labels_padded, binary_labels_batch, padding_mask

# --- Data Loading ---
full_feature_dir = os.path.join(FEATURE_DIR_BASE, FEATURE_SUBDIR)
print(f"Attempting to load data from: {full_feature_dir}")
all_train_features, all_train_labels = load_all_features_and_labels(full_feature_dir)

if all_train_features is None or not all_train_features:
    print("Exiting: No valid training features loaded.")
    exit()

full_dataset = AccidentFeatureDataset(all_train_features, all_train_labels)
val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_sequences)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences)

# --- Model, Optimizer, Criterion ---
model = ViTTransformer(
    feature_dim=VIT_FEATURE_DIM,
    model_dim=MODEL_DIM,
    nhead=N_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT 
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)

criterion_frame = nn.BCELoss(reduction='none') 
criterion_binary = nn.BCELoss() 

best_val_loss = float("inf")
epochs_no_improve = 0 # Counter for early stopping

# --- Training Loop ---
print("\n--- Starting Training ---")
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    total_train_sequences = 0

    for sequences, frame_labels, binary_labels, padding_mask in train_loader:
        if sequences.shape[1] == 0: # Skip if batch somehow ended up with 0 sequence length
            print("Warning: Skipping a training batch with 0 sequence length.")
            continue
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
        if GRADIENT_CLIP_VALUE is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
        optimizer.step()

        running_train_loss += combined_loss.item() * sequences.size(0) 
        total_train_sequences += sequences.size(0)

    avg_train_loss = running_train_loss / total_train_sequences if total_train_sequences > 0 else 0

    # --- Validation Loop ---
    model.eval()
    running_val_loss = 0.0
    total_val_sequences = 0

    with torch.no_grad():
        for sequences, frame_labels, binary_labels, padding_mask in val_loader:
            if sequences.shape[1] == 0: # Skip if batch somehow ended up with 0 sequence length
                print("Warning: Skipping a validation batch with 0 sequence length.")
                continue
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

    avg_val_loss = running_val_loss / total_val_sequences if total_val_sequences > 0 else float('inf') # Handle case of no val samples

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        current_checkpoint_path = CHECKPOINT_PATH_TEMPLATE 
        torch.save(model.state_dict(), current_checkpoint_path)
        print(f"Validation loss improved. Saved best model to {current_checkpoint_path}")
        epochs_no_improve = 0 # Reset counter
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
        break

print(f"\n--- Training Complete ---")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved to {CHECKPOINT_PATH_TEMPLATE} (timestamp: {TRAIN_RUN_TIMESTAMP}) if validation improved.")
print("--- Note on Hyperparameter Tuning ---")
print("To find optimal hyperparameters, you would typically run multiple experiments,")
print("varying one or a few parameters at a time (e.g., using grid search, random search,")
print("or more advanced techniques like Bayesian optimization with tools like Optuna).")
print("Monitor training/validation loss curves and your target metric (e.g., mAP on a hold-out set).")

