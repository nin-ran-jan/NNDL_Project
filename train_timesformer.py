# timesformer_pipeline/train_timesformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import pandas as pd
import time
import json
from tqdm import tqdm

from datasets.timeformer_video_dataset import TimesformerVideoDataset, collate_fn_timesformer_videos, compute_frame_label # if compute_frame_label is there
from models.custom_timesformer import CustomTimeSformer
from utils.video_transforms import get_timesformer_video_transform # Make sure this path is correct

# --- Configuration ---
TRAIN_RUN_TIMESTAMP = time.strftime("%Y%m%d%H%M%S")
BASE_DATA_DIR = "../nexar-collision-prediction" # Adjust path as per your structure
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "train.csv")
TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train")

# TimeSformer specific params
NUM_CLIP_FRAMES = 8        # Number of frames per clip for TimeSformer input
TIMESFORMER_MODEL_NAME = "timesformer_base_patch16_224" # From PyTorchVideo
BACKBONE_FEATURE_DIM = 768 # For timesformer_base_patch16_224, output feature dim is 768
TARGET_SPATIAL_SIZE = (224, 224) # Match model

# Dataset and DataLoader params
TARGET_PROCESSING_FPS = 3  # FPS for initial window sampling by dataset
SEQUENCE_WINDOW_SECONDS = 10.0 # Duration of the window from which clip is sampled
BATCH_SIZE = 8             # Adjust based on GPU memory (TimeSformer is memory intensive)
VAL_SPLIT = 0.15
DATALOADER_NUM_WORKERS = max(0, os.cpu_count() // 2 -1 if os.cpu_count() else 0) # Reduce if memory issues

# Training Hyperparameters
EPOCHS = 30
LEARNING_RATE = 1e-5       # Lower LR for fine-tuning pretrained models
WEIGHT_DECAY = 1e-4
ALPHA_LOSS = 0.5           # For combining frame and sequence loss
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.2

CHECKPOINT_DIR = os.path.join("checkpoints", f"run_{TRAIN_RUN_TIMESTAMP}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if DATALOADER_NUM_WORKERS > 0 and device == 'cuda':
    current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
    if current_start_method != 'spawn':
        try: torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError: print(f"Warning: Could not set start method to 'spawn'.")

def main():
    print(f"Training run timestamp: {TRAIN_RUN_TIMESTAMP}")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

    # --- Load Data ---
    df_full = pd.read_csv(TRAIN_CSV_PATH)
    df_full["id"] = df_full["id"].astype(str).str.zfill(5)
    # df_full = df_full.head(100) # For debugging with a small dataset

    # --- Transforms ---
    train_transforms = get_timesformer_video_transform(
        is_train=True,
        num_frames_to_sample=NUM_CLIP_FRAMES,
        target_spatial_size=TARGET_SPATIAL_SIZE
    )
    val_transforms = get_timesformer_video_transform(
        is_train=False,
        num_frames_to_sample=NUM_CLIP_FRAMES,
        target_spatial_size=TARGET_SPATIAL_SIZE
    )

    # --- Datasets and DataLoaders ---
    full_dataset = TimesformerVideoDataset(
        df=df_full,
        video_dir=TRAIN_VIDEO_DIR,
        video_transforms=None, # Will apply transforms per split
        num_clip_frames=NUM_CLIP_FRAMES,
        target_processing_fps=TARGET_PROCESSING_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS
    )

    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size],
                                           generator=torch.Generator().manual_seed(42)) # For reproducibility

    # Assign transforms to subsets
    train_subset.dataset.video_transforms = train_transforms
    val_subset.dataset.video_transforms = val_transforms
    
    print(f"Train dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn_timesformer_videos,
                              pin_memory=True if device == 'cuda' else False,
                              persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn_timesformer_videos,
                            pin_memory=True if device == 'cuda' else False,
                            persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False)

    # --- Model ---
    model = CustomTimeSformer(
        num_input_clip_frames=NUM_CLIP_FRAMES,
        backbone_name=TIMESFORMER_MODEL_NAME,
        pretrained=True,
        backbone_feature_dim=BACKBONE_FEATURE_DIM
    ).to(device)
    
    # You might want to compile the model for speed on newer PyTorch versions
    # model = torch.compile(model)


    # --- Optimizer, Scheduler, Criterion ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    # Using BCEWithLogitsLoss is generally more numerically stable than Sigmoid + BCELoss
    criterion_frame = nn.BCEWithLogitsLoss(reduction='none') # For applying mask later
    criterion_binary = nn.BCEWithLogitsLoss()
    # If you use BCEWithLogitsLoss, your model should output raw logits, not probabilities (remove final sigmoids in model)

    best_val_loss = float("inf")

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for batch_idx, batch in enumerate(progress_bar_train):
            if batch is None: # Skip if collate_fn returned None due to all invalid samples
                print(f"Skipping empty batch {batch_idx} in training.")
                continue

            video_clips = batch["video_clip"].to(device)
            frame_labels = batch["frame_labels"].to(device) # (B, T_clip)
            binary_labels = batch["binary_label"].to(device) # (B)

            optimizer.zero_grad()

            # Assuming model outputs logits if using BCEWithLogitsLoss
            frame_logits, seq_logits = model(video_clips) # (B, T_clip), (B)

            loss_binary = criterion_binary(seq_logits, binary_labels)
            
            # Frame loss - simple mean for now, can be masked later if needed
            # (BCEWithLogitsLoss reduction='none' gives per-element loss)
            loss_frame_unreduced = criterion_frame(frame_logits, frame_labels)
            loss_frame = loss_frame_unreduced.mean() # Mean over all frames in batch

            combined_loss = ALPHA_LOSS * loss_frame + (1 - ALPHA_LOSS) * loss_binary
            
            combined_loss.backward()
            # Gradient clipping (optional, but can help stabilize training)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += combined_loss.item() * video_clips.size(0)
            total_train_samples += video_clips.size(0)
            progress_bar_train.set_postfix(loss=running_train_loss/total_train_samples if total_train_samples > 0 else 0)
        
        avg_train_loss = running_train_loss / total_train_samples if total_train_samples > 0 else 0

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        total_val_samples = 0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

        with torch.no_grad():
            for batch in progress_bar_val:
                if batch is None:
                    print(f"Skipping empty batch in validation.")
                    continue
                video_clips = batch["video_clip"].to(device)
                frame_labels = batch["frame_labels"].to(device)
                binary_labels = batch["binary_label"].to(device)

                frame_logits, seq_logits = model(video_clips)

                loss_binary_val = criterion_binary(seq_logits, binary_labels)
                loss_frame_val_unreduced = criterion_frame(frame_logits, frame_labels)
                loss_frame_val = loss_frame_val_unreduced.mean()

                combined_loss_val = ALPHA_LOSS * loss_frame_val + (1 - ALPHA_LOSS) * loss_binary_val
                
                running_val_loss += combined_loss_val.item() * video_clips.size(0)
                total_val_samples += video_clips.size(0)
                progress_bar_val.set_postfix(loss=running_val_loss/total_val_samples if total_val_samples > 0 else 0)

        avg_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"timesformer_best_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save metadata about the run
        run_meta = {
            "epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss, "lr": current_lr,
            "config": { # Log key parameters
                "NUM_CLIP_FRAMES": NUM_CLIP_FRAMES, "TARGET_PROCESSING_FPS": TARGET_PROCESSING_FPS,
                "BATCH_SIZE": BATCH_SIZE, "LEARNING_RATE": LEARNING_RATE, "ALPHA_LOSS": ALPHA_LOSS,
                "TIMESFORMER_MODEL_NAME": TIMESFORMER_MODEL_NAME
            }
        }
        with open(os.path.join(CHECKPOINT_DIR, "training_log.jsonl"), "a") as f:
            f.write(json.dumps(run_meta) + "\n")


    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()