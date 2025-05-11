import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import sys
import json
from datetime import datetime

from datasets.timesformer_dataset import HFVideoDataset, collate_fn_hf_videos
from models.custom_timesformer import HFCustomTimeSformer

# --- Configuration ---
# Path Configuration
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp dynamically at runtime
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in hf_timesformer_pipeline
BASE_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "nexar-collision-prediction") # Adjust if your data is elsewhere
BEST_RUN_DIR = "run_20250511_020430"

TEST_CSV_PATH = os.path.join(BASE_DATA_DIR, "test.csv")
TEST_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "test")
BEST_MODEL_PATH = f"{BASE_PROJECT_DIR}/checkpoints_hf/{BEST_RUN_DIR}/model_best.pth" # <--- REPLACE THIS
OUTPUT_SUBMISSION_FILE = f"{BASE_PROJECT_DIR}/submissions/submission_timesformer_{TIMESTAMP}.csv"
# === END USER SETTINGS ===

HF_PROCESSOR_NAME = "facebook/timesformer-base-finetuned-k400" # Or the one used for training
HF_MODEL_NAME = "facebook/timesformer-base-finetuned-k400"     # Or the one used for training
BACKBONE_FEATURE_DIM = 768 # For "base" TimeSformer
NUM_CLIP_FRAMES = 8        # Number of frames per clip for TimeSformer input

# Dataset & DataLoader Parameters for Inference
TARGET_PROCESSING_FPS = 3  # Should be consistent with how dataset processes videos if relevant for windowing
SEQUENCE_WINDOW_SECONDS = 10.0 # Duration of video segment to process
BATCH_SIZE = 4            # Adjust based on GPU memory for inference
DATALOADER_NUM_WORKERS = min(os.cpu_count() // 2 if os.cpu_count() else 0, 4)
DROPOUT_RATE_INFERENCE = 0.3

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Prediction Configuration ---")
    print(f"Using device: {device}")
    print(f"Test CSV: {TEST_CSV_PATH}")
    print(f"Test Video Directory: {TEST_VIDEO_DIR}")
    print(f"Loading model from: {BEST_MODEL_PATH}")
    print(f"Output submission file: {OUTPUT_SUBMISSION_FILE}")
    print(f"Hugging Face Processor: {HF_PROCESSOR_NAME}")
    print(f"Hugging Face Model Name (for backbone): {HF_MODEL_NAME}")
    print(f"Num Clip Frames: {NUM_CLIP_FRAMES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"-------------------------------")

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"ERROR: Best model path not found: {BEST_MODEL_PATH}")
        print("Please update BEST_MODEL_PATH in the script.")
        return

    # --- Load Test Data Info ---
    try:
        df_test = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Test CSV not found at {TEST_CSV_PATH}")
        return
        
    df_test["id"] = df_test["id"].astype(str) # Ensure ID is string
    # The HFVideoDataset expects 'time_of_alert'. For test data, it's not available.
    # We add it as NaN so the dataset class can handle it (it will use default windowing).
    if "time_of_alert" not in df_test.columns:
        df_test["time_of_alert"] = np.nan
    print(f"Loaded {len(df_test)} test video entries.")


    # --- Test Dataset and DataLoader ---
    print("Initializing test dataset...")
    test_dataset = HFVideoDataset(
        df=df_test,
        video_dir=TEST_VIDEO_DIR,
        hf_processor_name=HF_PROCESSOR_NAME,
        num_clip_frames=NUM_CLIP_FRAMES,
        target_processing_fps=TARGET_PROCESSING_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=DATALOADER_NUM_WORKERS, collate_fn=collate_fn_hf_videos,
                             pin_memory=True if device == 'cuda' else False)

    print(f"Loading model from: {BEST_MODEL_PATH}")
    if not os.path.exists(BEST_MODEL_PATH): # ... (error check) ...
        print(f"ERROR: Best model path not found: {BEST_MODEL_PATH}")
        print("Please update BEST_MODEL_PATH in the script.")
        return

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model_config_saved = checkpoint.get('config', {}) # Config saved during training

    # Use config from checkpoint primarily, with fallbacks to script defaults
    loaded_hf_model_name = model_config_saved.get("HF_MODEL_NAME", HF_MODEL_NAME)
    loaded_num_clip_frames = model_config_saved.get("NUM_CLIP_FRAMES", NUM_CLIP_FRAMES)
    # backbone_feature_dim is determined by model.config.hidden_size inside HFCustomTimeSformer
    loaded_dropout_rate = model_config_saved.get("DROPOUT_RATE", DROPOUT_RATE_INFERENCE)


    print(f"Instantiating model with: name={loaded_hf_model_name}, frames={loaded_num_clip_frames}, dropout={loaded_dropout_rate}")
    model = HFCustomTimeSformer(
        hf_model_name=loaded_hf_model_name,
        num_frames_input_clip=loaded_num_clip_frames,
        # backbone_feature_dim_config will be overridden by model.config.hidden_size in constructor
        pretrained=False, # Crucial: We are loading weights, not from HF Hub pretrained
        dropout_rate=loaded_dropout_rate,
        freeze_backbone=False # Does not matter for inference
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode (disables dropout etc.)
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}. Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")

    # --- Prediction Loop ---
    print("Starting predictions...")
    all_video_ids = []
    all_scores = []

    with torch.no_grad():
        progress_bar_test = tqdm(test_loader, desc="Predicting", unit="batch")
        for batch in progress_bar_test:
            if batch is None: # Handle empty batches if collate_fn returns None
                print("Warning: Skipping an empty batch during prediction.")
                continue

            pixel_values = batch["pixel_values"].to(device)
            video_ids_batch = batch["video_id"] # List of video IDs

            # Model outputs raw logits for frame and sequence
            _, seq_logits = model(pixel_values)

            # Apply sigmoid to get probabilities (0-1 score) for sequence
            seq_probs = torch.sigmoid(seq_logits)

            all_video_ids.extend(video_ids_batch)
            all_scores.extend(seq_probs.cpu().numpy())

    # --- Create Submission File ---
    if not all_video_ids:
        print("No predictions were made. Check your test data and model.")
        return
        
    print(f"Generated {len(all_scores)} predictions.")
    submission_df = pd.DataFrame({
        "id": all_video_ids,
        "score": all_scores
    })

    # Ensure IDs in submission match the format expected (e.g., if zfill(5) was used for filenames but CSV has int IDs)
    # The dataset currently does: video_id = str(row["id"]).zfill(5)
    # If test.csv IDs are simple integers like 1, 2, etc., and filenames are 00001.mp4, 00002.mp4
    # then the `video_id` from the dataset will be '00001'.
    # The problem asks for "video id (filename without extension)".
    # If your test.csv 'id' column is already the filename without extension (e.g. '00001'), then it's fine.
    # If test.csv 'id' column is integer and submission expects integer, you might need to convert back.
    # For now, assuming `video_id` from dataset is the correct format for submission.
    
    # Make sure 'id' column is first, then 'score'
    submission_df = submission_df[["id", "score"]]

    try:
        submission_df.to_csv(OUTPUT_SUBMISSION_FILE, index=False)
        print(f"Submission file saved to: {OUTPUT_SUBMISSION_FILE}")
    except Exception as e:
        print(f"Error saving submission file: {e}")

if __name__ == "__main__":
    # Reminder for user to set the BEST_MODEL_PATH
    if "YOUR_BEST_RUN_DIR/model_best.pth" in BEST_MODEL_PATH:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update the 'BEST_MODEL_PATH' variable in this script !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()