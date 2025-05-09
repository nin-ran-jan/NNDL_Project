# hf_timesformer_pipeline/train_hf_timesformer.py
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
import sys # For path manipulation if needed

# Ensure the custom modules can be imported
# If train_hf_timesformer.py is in hf_timesformer_pipeline/
# and datasets/models are subdirectories:
# Option 1: Add parent directory to path (if running script directly from its location)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir) # If script is one level down from project root
# sys.path.append(parent_dir) # Add project root to allow imports like from datasets.hf_video_dataset

# Option 2: Assume Python's import resolution handles it (e.g. if hf_timesformer_pipeline is a package or in PYTHONPATH)
from datasets.timesformer_dataset import HFVideoDataset, collate_fn_hf_videos
from models.custom_timesformer import HFCustomTimeSformer

# --- Configuration ---
# Path Configuration
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in hf_timesformer_pipeline
BASE_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "../nexar-collision-prediction") # Adjust if your data is elsewhere
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "train.csv")
TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train")

# Run Configuration
TRAIN_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
CHECKPOINT_SUBDIR = f"run_{TRAIN_RUN_TIMESTAMP}"
CHECKPOINT_DIR = os.path.join(BASE_PROJECT_DIR, "checkpoints_hf", CHECKPOINT_SUBDIR)

# Hugging Face Model & Processor Configuration
HF_PROCESSOR_NAME = "facebook/timesformer-base-finetuned-k400" # Or specific processor for your model
HF_MODEL_NAME = "facebook/timesformer-base-finetuned-k400"     # Or just "facebook/timesformer-base"
BACKBONE_FEATURE_DIM = 768 # For "base" TimeSformer models (e.g., ViT-B). Check model card for others.

# Dataset & DataLoader Parameters
NUM_CLIP_FRAMES = 8        # Number of frames per clip. TimeSformer base often uses 8. HR (High Resolution) might use more.
TARGET_PROCESSING_FPS = 3  # Your internal FPS for windowing logic & label alignment
SEQUENCE_WINDOW_SECONDS = 10.0 # Duration of the video segment from which the clip is sampled

# Training Hyperparameters
BATCH_SIZE = 8             # Adjust based on GPU memory. Start small (4 or 8).
EPOCHS = 30
LEARNING_RATE = 3e-5       # Common starting LR for fine-tuning HF transformer models
WEIGHT_DECAY = 1e-4
ALPHA_LOSS = 0.5           # Weight for combining frame and sequence loss
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.2
GRADIENT_CLIP_VAL = 1.0    # Max norm for gradient clipping

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# DataLoader Workers
DATALOADER_NUM_WORKERS = min(os.cpu_count() // 2 if os.cpu_count() else 0, 4) # Keep it modest to avoid issues

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Experiment Configuration ---")
    print(f"Timestamp: {TRAIN_RUN_TIMESTAMP}")
    print(f"Using device: {device}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    print(f"Hugging Face Processor: {HF_PROCESSOR_NAME}")
    print(f"Hugging Face Model: {HF_MODEL_NAME}")
    print(f"Num Clip Frames: {NUM_CLIP_FRAMES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Num Workers: {DATALOADER_NUM_WORKERS}")
    print(f"-------------------------------")

    # Handle multiprocessing start method for CUDA
    if DATALOADER_NUM_WORKERS > 0 and device == 'cuda':
        current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
                print("Set PyTorch multiprocessing start method to 'spawn'.")
            except RuntimeError as e:
                print(f"Warning: Could not set start method to 'spawn': {e}. Using default: {current_start_method}")

    # --- Load Data ---
    try:
        df_full = pd.read_csv(TRAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Training CSV not found at {TRAIN_CSV_PATH}")
        return
    df_full["id"] = df_full["id"].astype(str).str.zfill(5)
    # For quick debugging:
    # df_full = df_full.sample(n=min(200, len(df_full)), random_state=SEED).reset_index(drop=True)
    # print(f"Using a subset of {len(df_full)} samples for debugging.")


    # --- Datasets and DataLoaders ---
    print("Initializing datasets...")
    full_dataset = HFVideoDataset(
        df=df_full,
        video_dir=TRAIN_VIDEO_DIR,
        hf_processor_name=HF_PROCESSOR_NAME,
        num_clip_frames=NUM_CLIP_FRAMES,
        target_processing_fps=TARGET_PROCESSING_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS
    )

    # Splitting dataset (ensure transforms are applied after split if they differ)
    val_split_ratio = 0.15 # Use a fixed validation split ratio
    val_size = int(val_split_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    if train_size == 0 or val_size == 0:
        print(f"ERROR: Dataset too small for splitting. Train size: {train_size}, Val size: {val_size}. Exiting.")
        return
        
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                           generator=torch.Generator().manual_seed(SEED))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn_hf_videos,
                              pin_memory=True if device == 'cuda' else False,
                              persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False,
                              drop_last=True) # Drop last incomplete batch for stability
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn_hf_videos,
                            pin_memory=True if device == 'cuda' else False,
                            persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False)

    # --- Model ---
    print("Initializing model...")
    model = HFCustomTimeSformer(
        hf_model_name=HF_MODEL_NAME,
        num_frames_input_clip=NUM_CLIP_FRAMES,
        backbone_feature_dim=BACKBONE_FEATURE_DIM,
        pretrained=True
    ).to(device)
    
    # Optional: Compile the model for potential speedup (PyTorch 2.0+)
    # try:
    #     model = torch.compile(model)
    #     print("Model compiled successfully.")
    # except Exception as e:
    #     print(f"Model compilation failed: {e}. Using uncompiled model.")


    # --- Optimizer, Scheduler, Criterion ---
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    # Using BCEWithLogitsLoss as it's more numerically stable. Model should output raw logits.
    criterion_frame = nn.BCEWithLogitsLoss(reduction='mean') # Already averages over all elements if not 'none'
    criterion_binary = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    training_log = []

    print("Starting training...")
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        total_train_batches = 0

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch", leave=False)
        for batch_idx, batch in enumerate(progress_bar_train):
            if batch is None: # Handle empty batches from collate_fn
                print(f"Warning: Skipping empty batch {batch_idx+1}/{len(train_loader)} in training.")
                continue

            pixel_values = batch["pixel_values"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            binary_labels = batch["binary_label"].to(device)

            optimizer.zero_grad()

            frame_logits, seq_logits = model(pixel_values) # Model outputs raw logits

            loss_binary = criterion_binary(seq_logits, binary_labels)
            loss_frame = criterion_frame(frame_logits, frame_labels)

            combined_loss = ALPHA_LOSS * loss_frame + (1 - ALPHA_LOSS) * loss_binary
            
            combined_loss.backward()
            if GRADIENT_CLIP_VAL > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
            optimizer.step()

            running_train_loss += combined_loss.item()
            total_train_batches += 1
            progress_bar_train.set_postfix(loss=running_train_loss/total_train_batches if total_train_batches > 0 else 0.0)
        
        avg_train_loss = running_train_loss / total_train_batches if total_train_batches > 0 else 0.0

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        total_val_batches = 0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch", leave=False)

        with torch.no_grad():
            for batch in progress_bar_val:
                if batch is None:
                    print(f"Warning: Skipping empty batch in validation.")
                    continue
                pixel_values = batch["pixel_values"].to(device)
                frame_labels = batch["frame_labels"].to(device)
                binary_labels = batch["binary_label"].to(device)

                frame_logits, seq_logits = model(pixel_values)

                loss_binary_val = criterion_binary(seq_logits, binary_labels)
                loss_frame_val = criterion_frame(frame_logits, frame_labels)

                combined_loss_val = ALPHA_LOSS * loss_frame_val + (1 - ALPHA_LOSS) * loss_binary_val
                
                running_val_loss += combined_loss_val.item()
                total_val_batches +=1
                progress_bar_val.set_postfix(loss=running_val_loss/total_val_batches if total_val_batches > 0 else 0.0)

        avg_val_loss = running_val_loss / total_val_batches if total_val_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        epoch_log = {
            "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "lr": current_lr
        }
        training_log.append(epoch_log)
        with open(os.path.join(CHECKPOINT_DIR, "training_log.jsonl"), "a") as f:
            f.write(json.dumps(epoch_log) + "\n")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(), # Or model.module.state_dict() if using DataParallel/DDP or torch.compile
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': { # Save key config params for reproducibility
                    "HF_MODEL_NAME": HF_MODEL_NAME, "HF_PROCESSOR_NAME": HF_PROCESSOR_NAME,
                    "NUM_CLIP_FRAMES": NUM_CLIP_FRAMES, "BACKBONE_FEATURE_DIM": BACKBONE_FEATURE_DIM,
                    "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, "ALPHA_LOSS": ALPHA_LOSS,
                }
            }, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
        
        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
        torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss, 'current_val_loss': avg_val_loss,
            }, latest_checkpoint_path)


    print(f"Training complete. Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Checkpoints and logs saved in: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    # Store script and key configurations used for this run
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Ensure it exists before trying to copy
    try:
        # Save a copy of the training script
        script_path = os.path.abspath(__file__)
        destination_script_path = os.path.join(CHECKPOINT_DIR, os.path.basename(script_path))
        with open(script_path, 'r') as source_file, open(destination_script_path, 'w') as dest_file:
            dest_file.write(source_file.read())
        print(f"Saved training script to {destination_script_path}")
        
        # Save main config parameters to a json file
        config_summary = {
            "BASE_DATA_DIR": BASE_DATA_DIR, "TRAIN_CSV_PATH": TRAIN_CSV_PATH, "TRAIN_VIDEO_DIR": TRAIN_VIDEO_DIR,
            "TRAIN_RUN_TIMESTAMP": TRAIN_RUN_TIMESTAMP, "CHECKPOINT_DIR": CHECKPOINT_DIR,
            "HF_PROCESSOR_NAME": HF_PROCESSOR_NAME, "HF_MODEL_NAME": HF_MODEL_NAME,
            "BACKBONE_FEATURE_DIM": BACKBONE_FEATURE_DIM, "NUM_CLIP_FRAMES": NUM_CLIP_FRAMES,
            "TARGET_PROCESSING_FPS": TARGET_PROCESSING_FPS, "SEQUENCE_WINDOW_SECONDS": SEQUENCE_WINDOW_SECONDS,
            "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY, "ALPHA_LOSS": ALPHA_LOSS, "SCHEDULER_PATIENCE": SCHEDULER_PATIENCE,
            "SCHEDULER_FACTOR": SCHEDULER_FACTOR, "GRADIENT_CLIP_VAL": GRADIENT_CLIP_VAL, "SEED": SEED,
            "DATALOADER_NUM_WORKERS": DATALOADER_NUM_WORKERS
        }
        with open(os.path.join(CHECKPOINT_DIR, "run_config.json"), "w") as f:
            json.dump(config_summary, f, indent=4)
        print(f"Saved run configuration to {os.path.join(CHECKPOINT_DIR, 'run_config.json')}")

    except Exception as e:
        print(f"Error saving script/config: {e}")

    main()