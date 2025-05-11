import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import numpy as np
import os
import pandas as pd
import time
import json
from tqdm import tqdm

from datasets.timesformer_dataset import HFVideoDataset, collate_fn_hf_videos
from models.custom_timesformer import HFCustomTimeSformer
# utils.video_transforms_augmented is used within HFVideoDataset

# --- Configuration ---
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "nexar-collision-prediction")
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "train.csv")
TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train")

TRAIN_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
CHECKPOINT_SUBDIR = f"run_{TRAIN_RUN_TIMESTAMP}"
CHECKPOINT_DIR = os.path.join(BASE_PROJECT_DIR, "checkpoints_hf", CHECKPOINT_SUBDIR)

HF_PROCESSOR_NAME = "facebook/timesformer-base-finetuned-k400"
HF_MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
BACKBONE_FEATURE_DIM = 768 # For "base" TimeSformer (actual dim taken from model.config later)

NUM_CLIP_FRAMES = 8
TARGET_PROCESSING_FPS = 3
SEQUENCE_WINDOW_SECONDS = 10.0
AUGMENTATION_SPATIAL_SIZE = (224, 224) # For RandomResizedCrop etc. in augmentation

BATCH_SIZE = 4
EPOCHS = 30 # Total epochs
LEARNING_RATE = 3e-5 # Initial LR
WEIGHT_DECAY = 1e-3 # Increased weight decay
ALPHA_LOSS = 0.5
GRADIENT_CLIP_VAL = 1.0
DROPOUT_RATE = 0.3 # Dropout for custom heads

# New Training Strategy Configs
FREEZE_BACKBONE_EPOCHS = 3 # Number of epochs to train with backbone frozen (0 to disable)
UNFREEZE_LR_FACTOR = 0.1 # Factor to reduce LR when unfreezing backbone (e.g., LR * 0.1)
SCHEDULER_TYPE = "CosineAnnealing" # "ReduceLROnPlateau" or "CosineAnnealing" or "LinearWarmup"
R_PLATEAU_PATIENCE = 3 # For ReduceLROnPlateau
R_PLATEAU_FACTOR = 0.2 # For ReduceLROnPlateau
COSINE_T_MAX_RATIO = 1.0 # Ratio of total steps for CosineAnnealing T_max (1.0 means T_max = total_steps)
WARMUP_STEPS_RATIO = 0.1 # Ratio of total steps for warmup (if using LinearWarmup or Cosine with warmup)

EARLY_STOPPING_PATIENCE = 7 # Stop if val_loss doesn't improve for this many epochs

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DATALOADER_NUM_WORKERS = 16

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ... (print config section from previous train script) ...
    config_summary = { # Save all key parameters
        # ... (all parameters from above) ...
        "HF_PROCESSOR_NAME": HF_PROCESSOR_NAME, "HF_MODEL_NAME": HF_MODEL_NAME,
        "NUM_CLIP_FRAMES": NUM_CLIP_FRAMES, "TARGET_PROCESSING_FPS": TARGET_PROCESSING_FPS,
        "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY, "ALPHA_LOSS": ALPHA_LOSS, "DROPOUT_RATE": DROPOUT_RATE,
        "FREEZE_BACKBONE_EPOCHS": FREEZE_BACKBONE_EPOCHS, "UNFREEZE_LR_FACTOR": UNFREEZE_LR_FACTOR,
        "SCHEDULER_TYPE": SCHEDULER_TYPE, "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "GRADIENT_CLIP_VAL": GRADIENT_CLIP_VAL, "SEED": SEED
    }
    with open(os.path.join(CHECKPOINT_DIR, "run_config.json"), "w") as f:
        json.dump(config_summary, f, indent=4)
    print(f"Saved run configuration to {os.path.join(CHECKPOINT_DIR, 'run_config.json')}")
    print(f"Using device: {device}")


    if DATALOADER_NUM_WORKERS > 0 and device == 'cuda':
        # ... (multiprocessing start method logic) ...
        current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try: torch.multiprocessing.set_start_method('spawn', force=True); print("Set mp start method to spawn.")
            except RuntimeError as e: print(f"Warning: Could not set start method to 'spawn': {e}")

    df_full = pd.read_csv(TRAIN_CSV_PATH)
    df_full["id"] = df_full["id"].astype(str).str.zfill(5)
    # df_full = df_full.sample(n=min(100, len(df_full)), random_state=SEED).reset_index(drop=True) # Debug subset

    # --- Datasets with Augmentation ---
    print("Initializing datasets with augmentations...")
    train_dataset_full = HFVideoDataset(
        df=df_full, video_dir=TRAIN_VIDEO_DIR, hf_processor_name=HF_PROCESSOR_NAME,
        num_clip_frames=NUM_CLIP_FRAMES, target_processing_fps=TARGET_PROCESSING_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS, is_train=True, # Enable train augmentations
        augmentation_spatial_size=AUGMENTATION_SPATIAL_SIZE
    )
    # For validation, use a dataset instance with is_train=False
    val_dataset_full = HFVideoDataset(
        df=df_full, video_dir=TRAIN_VIDEO_DIR, hf_processor_name=HF_PROCESSOR_NAME,
        num_clip_frames=NUM_CLIP_FRAMES, target_processing_fps=TARGET_PROCESSING_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS, is_train=False, # Disable train augmentations for val
        augmentation_spatial_size=AUGMENTATION_SPATIAL_SIZE # Usually eval uses center crop of this
    )
    
    val_split_ratio = 0.15
    val_size = int(val_split_ratio * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size
    if train_size == 0 or val_size == 0: print("ERROR: Dataset too small for splitting."); return

    # Create subsets using indices from the original df_full to ensure val set uses val_transforms
    indices = list(range(len(train_dataset_full)))
    np.random.shuffle(indices) # Shuffle once for consistent split
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices) # Uses val_dataset_full with is_train=False

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn_hf_videos,
                              pin_memory=True if device == 'cuda' else False,
                              persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn_hf_videos,
                            pin_memory=True if device == 'cuda' else False,
                            persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False)

    print("Initializing model...")
    model = HFCustomTimeSformer(
        hf_model_name=HF_MODEL_NAME,
        num_frames_input_clip=NUM_CLIP_FRAMES,
        backbone_feature_dim_config=BACKBONE_FEATURE_DIM, # Initial guess, model will use its actual
        pretrained=True,
        dropout_rate=DROPOUT_RATE,
        freeze_backbone=(FREEZE_BACKBONE_EPOCHS > 0) # Freeze if FREEZE_BACKBONE_EPOCHS is positive
    ).to(device)
    print(f"Model actual backbone feature dim: {model.backbone_actual_feature_dim}")

    # --- Optimizer, Scheduler, Criterion ---
    # Initially, optimizer for potentially only the head
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(WARMUP_STEPS_RATIO * total_steps) if SCHEDULER_TYPE in ["CosineAnnealing", "LinearWarmup"] else 0


    if SCHEDULER_TYPE == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=R_PLATEAU_FACTOR, patience=R_PLATEAU_PATIENCE, verbose=True)
    elif SCHEDULER_TYPE == "CosineAnnealing":
        t_max_steps = int(COSINE_T_MAX_RATIO * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_max_steps)
        print(f"Using CosineAnnealing scheduler with {num_warmup_steps} warmup steps and T_max={t_max_steps} total steps.")
    elif SCHEDULER_TYPE == "LinearWarmup":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        print(f"Using LinearWarmup scheduler with {num_warmup_steps} warmup steps and {total_steps} total steps.")
    else:
        scheduler = None # No scheduler or default
        print("No specific LR scheduler or invalid type specified.")


    criterion_frame = nn.BCEWithLogitsLoss(reduction='mean')
    criterion_binary = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_log = []
    backbone_is_frozen = (FREEZE_BACKBONE_EPOCHS > 0)

    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- Backbone Unfreezing Logic ---
        if backbone_is_frozen and epoch >= FREEZE_BACKBONE_EPOCHS:
            model.unfreeze_backbone()
            backbone_is_frozen = False
            print(f"Backbone unfrozen at epoch {epoch + 1}. Re-initializing optimizer with new LR.")
            # Re-initialize optimizer with all parameters and potentially adjusted LR
            new_lr = LEARNING_RATE * (UNFREEZE_LR_FACTOR if UNFREEZE_LR_FACTOR > 0 else 1.0)
            optimizer = optim.AdamW(model.parameters(), lr=new_lr, weight_decay=WEIGHT_DECAY)
            print(f"Optimizer re-initialized. New LR for full model: {new_lr:.2e}")
            # Re-initialize scheduler with the new optimizer
            if SCHEDULER_TYPE == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=R_PLATEAU_FACTOR, patience=R_PLATEAU_PATIENCE, verbose=True)
            elif SCHEDULER_TYPE == "CosineAnnealing": # Recalculate remaining steps
                remaining_epochs = EPOCHS - epoch
                remaining_steps = len(train_loader) * remaining_epochs
                current_warmup_steps = max(0, num_warmup_steps - (len(train_loader) * epoch)) # Adjust warmup if unfreezing happens mid-warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=current_warmup_steps, num_training_steps=remaining_steps)
            elif SCHEDULER_TYPE == "LinearWarmup":
                remaining_epochs = EPOCHS - epoch
                remaining_steps = len(train_loader) * EPOCHS # Total steps for linear decay should be total training steps
                current_total_steps_done = len(train_loader) * epoch
                # The num_training_steps for get_linear_schedule_with_warmup is the *total* number of steps, not remaining.
                # So, re-initializing it might reset its state. It's often better to let it run with the initial optimizer,
                # but if optimizer params change, it needs re-init.
                # For simplicity, let's re-init, assuming warmup is likely done.
                current_warmup_steps = max(0, num_warmup_steps - current_total_steps_done)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=current_warmup_steps, num_training_steps=total_steps)


        model.train()
        running_train_loss = 0.0; total_train_batches = 0
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch", leave=False)
        for batch_idx, batch in enumerate(progress_bar_train):
            # ... (training batch loop from previous train script: pixel_values, labels, optimizer.zero_grad(), model forward, loss, backward, step)
            if batch is None: continue
            pixel_values = batch["pixel_values"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            binary_labels = batch["binary_label"].to(device)
            optimizer.zero_grad()
            frame_logits, seq_logits = model(pixel_values)
            loss_binary = criterion_binary(seq_logits, binary_labels)
            loss_frame = criterion_frame(frame_logits, frame_labels)
            combined_loss = ALPHA_LOSS * loss_frame + (1 - ALPHA_LOSS) * loss_binary
            combined_loss.backward()
            if GRADIENT_CLIP_VAL > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
            optimizer.step()
            if SCHEDULER_TYPE in ["CosineAnnealing", "LinearWarmup"] and scheduler: # Step per batch for these
                 scheduler.step()

            running_train_loss += combined_loss.item(); total_train_batches += 1
            progress_bar_train.set_postfix(loss=running_train_loss/total_train_batches if total_train_batches > 0 else 0.0)

        avg_train_loss = running_train_loss / total_train_batches if total_train_batches > 0 else 0.0

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0; total_val_batches = 0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch", leave=False)
        with torch.no_grad():
            for batch in progress_bar_val:
                # ... (validation batch loop from previous train script: pixel_values, labels, model forward, loss)
                if batch is None: continue
                pixel_values = batch["pixel_values"].to(device)
                frame_labels = batch["frame_labels"].to(device)
                binary_labels = batch["binary_label"].to(device)
                frame_logits, seq_logits = model(pixel_values)
                loss_binary_val = criterion_binary(seq_logits, binary_labels)
                loss_frame_val = criterion_frame(frame_logits, frame_labels)
                combined_loss_val = ALPHA_LOSS * loss_frame_val + (1 - ALPHA_LOSS) * loss_binary_val
                running_val_loss += combined_loss_val.item(); total_val_batches +=1
                progress_bar_val.set_postfix(loss=running_val_loss/total_val_batches if total_val_batches > 0 else 0.0)

        avg_val_loss = running_val_loss / total_val_batches if total_val_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        epoch_log = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": current_lr}
        training_log.append(epoch_log)
        with open(os.path.join(CHECKPOINT_DIR, "training_log.jsonl"), "a") as f: f.write(json.dumps(epoch_log) + "\n")

        if SCHEDULER_TYPE == "ReduceLROnPlateau" and scheduler:
            scheduler.step(avg_val_loss)
        
        # Checkpointing and Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_best.pth")
            # ... (save checkpoint logic from previous train script, include config_summary) ...
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss, 'config': config_summary
            }, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # Save latest model
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
        torch.save({ 'epoch': epoch + 1, 'model_state_dict': model.state_dict(), # ... (include other relevant states)
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_loss': best_val_loss, 'current_val_loss': avg_val_loss, 'config': config_summary }, latest_checkpoint_path)


        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement on validation loss.")
            break
            
    print(f"Training complete. Best validation loss achieved: {best_val_loss:.4f}")
    # ... (save script copy logic from previous train script) ...
    try:
        script_path = os.path.abspath(__file__)
        destination_script_path = os.path.join(CHECKPOINT_DIR, os.path.basename(script_path))
        with open(script_path, 'r') as source_file, open(destination_script_path, 'w') as dest_file:
            dest_file.write(source_file.read())
    except Exception as e: print(f"Error saving training script: {e}")


if __name__ == "__main__":
    main()