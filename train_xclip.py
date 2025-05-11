import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers.models.x_clip import XCLIPProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
import numpy as np
import os
import pandas as pd
import time
import json
from tqdm import tqdm

from datasets.xclip_dataset import XCLIPVideoDataset, collate_fn_xclip
from models.xclip import CustomXCLIPModel
import torch.cuda.amp as amp

# --- Configuration ---
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "nexar-collision-prediction")
TRAIN_CSV_PATH = os.path.join(BASE_DATA_DIR, "train.csv")
TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train")

TRAIN_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
CHECKPOINT_SUBDIR = f"xclip_run_{TRAIN_RUN_TIMESTAMP}"
CHECKPOINT_DIR = os.path.join(BASE_PROJECT_DIR, "checkpoints_xclip", CHECKPOINT_SUBDIR)

# X-CLIP Configuration
XCLIP_MODEL_NAME = "microsoft/xclip-base-patch32"
NUM_FRAMES = 8
TARGET_FPS = 3
SEQUENCE_WINDOW_SECONDS = 10.0

# Training Configuration
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
ALPHA_LOSS = 0.5  # Weight between frame and sequence loss
DROPOUT_RATE = 0.4
GRADIENT_CLIP_VAL = 1.0

# Progressive Unfreezing Strategy
FREEZE_BACKBONE_EPOCHS = 5  # Train with frozen backbone for this many epochs
UNFREEZE_LR_FACTOR = 0.1  # Lower LR by this factor when unfreezing
MIXED_PRECISION = True  # Use mixed precision training

# Augmentation Configuration
AUGMENTATION_PARAMS = {
    'strength': 0.6,  # Strength of augmentations
    'color_jitter_prob': 0.8,  # Probability of applying color jitter
    'gray_scale_prob': 0.2,  # Probability of grayscale
    'random_crop_scale': (0.6, 1.0),  # Random resized crop scale
    'random_crop_ratio': (0.75, 1.33)  # Random resized crop aspect ratio
}

# Scheduler Configuration
SCHEDULER_TYPE = "CosineAnnealing"
WARMUP_RATIO = 0.1  # Percentage of total steps for warmup

# Early stopping
EARLY_STOPPING_PATIENCE = 7

# Seed and Workers
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DATALOADER_NUM_WORKERS = 4
PIN_MEMORY = True

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Save config
    config_summary = {
        "XCLIP_MODEL_NAME": XCLIP_MODEL_NAME,
        "NUM_FRAMES": NUM_FRAMES,
        "TARGET_FPS": TARGET_FPS,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "ALPHA_LOSS": ALPHA_LOSS,
        "DROPOUT_RATE": DROPOUT_RATE,
        "FREEZE_BACKBONE_EPOCHS": FREEZE_BACKBONE_EPOCHS,
        "UNFREEZE_LR_FACTOR": UNFREEZE_LR_FACTOR,
        "AUGMENTATION_PARAMS": AUGMENTATION_PARAMS,
        "SCHEDULER_TYPE": SCHEDULER_TYPE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "GRADIENT_CLIP_VAL": GRADIENT_CLIP_VAL,
        "SEED": SEED,
        "MIXED_PRECISION": MIXED_PRECISION
    }
    
    with open(os.path.join(CHECKPOINT_DIR, "run_config.json"), "w") as f:
        json.dump(config_summary, f, indent=4)
    
    print(f"Saved run configuration to {os.path.join(CHECKPOINT_DIR, 'run_config.json')}")
    print(f"Using device: {device}")
    
    # Load data
    df_full = pd.read_csv(TRAIN_CSV_PATH)
    df_full["id"] = df_full["id"].astype(str).str.zfill(5)
    
    # Initialize processor
    processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_NAME)
    
    # Create datasets
    print("Initializing datasets with augmentations...")
    train_dataset_full = XCLIPVideoDataset(
        df=df_full,
        video_dir=TRAIN_VIDEO_DIR,
        processor_name=XCLIP_MODEL_NAME,
        num_frames=NUM_FRAMES,
        target_fps=TARGET_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS,
        is_train=True,
        augmentation_params=AUGMENTATION_PARAMS
    )
    
    val_dataset_full = XCLIPVideoDataset(
        df=df_full,
        video_dir=TRAIN_VIDEO_DIR,
        processor_name=XCLIP_MODEL_NAME,
        num_frames=NUM_FRAMES,
        target_fps=TARGET_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS,
        is_train=False
    )
    
    # Split datasets
    val_split_ratio = 0.15
    val_size = int(val_split_ratio * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size
    
    if train_size == 0 or val_size == 0:
        print("ERROR: Dataset too small for splitting.")
        return
    
    # Create indices for train/val split
    indices = list(range(len(train_dataset_full)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn_xclip,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn_xclip,
        pin_memory=PIN_MEMORY
    )
    
    # Initialize model
    print("Initializing model...")
    model = CustomXCLIPModel(
        model_name=XCLIP_MODEL_NAME,
        num_frames=NUM_FRAMES,
        dropout_rate=DROPOUT_RATE,
        pretrained=True,
        freeze_backbone=(FREEZE_BACKBONE_EPOCHS > 0),
        freeze_text_model=True,
        freeze_projection=True
    ).to(device)
    
    # Initialize text features
    model.initialize_text_features(processor.tokenizer)
    
    # Optimizer (only train non-frozen parameters)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Count trainable parameters
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_param_count:,} / {total_param_count:,} ({trainable_param_count/total_param_count:.2%})")
    
    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    
    if SCHEDULER_TYPE == "CosineAnnealing":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        print(f"Using CosineAnnealing scheduler with {warmup_steps} warmup steps")
    else:
        scheduler = None
        print("No scheduler specified")
    
    # Loss functions
    criterion_frame = nn.BCEWithLogitsLoss(reduction='mean')
    criterion_binary = nn.BCEWithLogitsLoss()
    
    # Training loop variables
    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_log = []
    backbone_is_frozen = (FREEZE_BACKBONE_EPOCHS > 0)
    
    # Scaler for mixed precision
    scaler = amp.GradScaler() if MIXED_PRECISION and device == "cuda" else None
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Unfreeze backbone if needed
        if backbone_is_frozen and epoch >= FREEZE_BACKBONE_EPOCHS:
            print(f"Unfreezing backbone at epoch {epoch + 1}")
            model.unfreeze_vision_model()
            backbone_is_frozen = False
            
            # Re-initialize optimizer with adjusted learning rate
            new_lr = LEARNING_RATE * UNFREEZE_LR_FACTOR
            print(f"Adjusting learning rate to {new_lr:.2e}")
            
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=new_lr,
                weight_decay=WEIGHT_DECAY
            )
            
            # Count trainable parameters after unfreezing
            trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters after unfreezing: {trainable_param_count:,} ({trainable_param_count/total_param_count:.2%})")
            
            # Re-initialize scheduler
            if SCHEDULER_TYPE == "CosineAnnealing":
                remaining_epochs = EPOCHS - epoch
                remaining_steps = len(train_loader) * remaining_epochs
                current_warmup_steps = max(0, int(WARMUP_RATIO * remaining_steps))
                
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=current_warmup_steps,
                    num_training_steps=remaining_steps
                )
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        total_train_batches = 0
        
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar_train):
            if batch is None:
                continue
                
            pixel_values = batch["pixel_values"].to(device)
            frame_labels = batch["frame_labels"].to(device)
            binary_labels = batch["binary_label"].to(device)
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with amp.autocast():
                    frame_logits, seq_logits = model(pixel_values)
                    loss_binary = criterion_binary(seq_logits, binary_labels)
                    loss_frame = criterion_frame(frame_logits, frame_labels)
                    combined_loss = ALPHA_LOSS * loss_frame + (1 - ALPHA_LOSS) * loss_binary
                
                # Backward pass with scaler
                optimizer.zero_grad()
                scaler.scale(combined_loss).backward()
                
                if GRADIENT_CLIP_VAL > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                frame_logits, seq_logits = model(pixel_values)
                loss_binary = criterion_binary(seq_logits, binary_labels)
                loss_frame = criterion_frame(frame_logits, frame_labels)
                combined_loss = ALPHA_LOSS * loss_frame + (1 - ALPHA_LOSS) * loss_binary
                
                # Standard backward pass
                optimizer.zero_grad()
                combined_loss.backward()
                
                if GRADIENT_CLIP_VAL > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
                
                optimizer.step()
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
            
            running_train_loss += combined_loss.item()
            total_train_batches += 1
            
            progress_bar_train.set_postfix(
                loss=running_train_loss/total_train_batches if total_train_batches > 0 else 0.0
            )
        
        avg_train_loss = running_train_loss / total_train_batches if total_train_batches > 0 else 0.0
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_binary_acc = 0.0
        total_val_batches = 0
        total_val_samples = 0
        
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar_val:
                if batch is None:
                    continue
                    
                pixel_values = batch["pixel_values"].to(device)
                frame_labels = batch["frame_labels"].to(device)
                binary_labels = batch["binary_label"].to(device)
                
                frame_logits, seq_logits = model(pixel_values)
                loss_binary = criterion_binary(seq_logits, binary_labels)
                loss_frame = criterion_frame(frame_logits, frame_labels)
                combined_loss = ALPHA_LOSS * loss_frame + (1 - ALPHA_LOSS) * loss_binary
                
                # Calculate binary accuracy
                binary_preds = (torch.sigmoid(seq_logits) > 0.5).float()
                binary_correct = (binary_preds == binary_labels).float().sum()
                
                running_val_loss += combined_loss.item()
                running_binary_acc += binary_correct.item()
                total_val_batches += 1
                total_val_samples += binary_labels.size(0)
                
                progress_bar_val.set_postfix(
                    loss=running_val_loss/total_val_batches if total_val_batches > 0 else 0.0
                )
        
        avg_val_loss = running_val_loss / total_val_batches if total_val_batches > 0 else 0.0
        avg_binary_acc = running_binary_acc / total_val_samples if total_val_samples > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_binary_acc:.4f} | LR: {current_lr:.2e}")
        
        # Log epoch results
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": avg_binary_acc,
            "lr": current_lr
        }
        
        training_log.append(epoch_log)
        
        with open(os.path.join(CHECKPOINT_DIR, "training_log.jsonl"), "a") as f:
            f.write(json.dumps(epoch_log) + "\n")
        
        # Check for improvement and save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            # Save best model
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'val_acc': avg_binary_acc,
                'config': config_summary
            }, checkpoint_path)
            
            print(f"Saved new best model to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
        
        # Save latest model
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'current_val_loss': avg_val_loss,
            'val_acc': avg_binary_acc,
            'config': config_summary
        }, latest_checkpoint_path)
        
        # Early stopping check
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    
    # Save a copy of the training script
    try:
        script_path = os.path.abspath(__file__)
        destination_script_path = os.path.join(CHECKPOINT_DIR, os.path.basename(script_path))
        with open(script_path, 'r') as source_file, open(destination_script_path, 'w') as dest_file:
            dest_file.write(source_file.read())
    except Exception as e:
        print(f"Error saving training script: {e}")

if __name__ == "__main__":
    main()