import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

from datasets.xclip_dataset import XCLIPVideoDataset, collate_fn_xclip
from models.xclip import CustomXCLIPModel
from transformers.models.x_clip import XCLIPProcessor

# AI prompts:
# Improve formatting
# And sensible outputs when runing the script
# Add control statements for errors

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "nexar-collision-prediction")
BEST_RUN_DIR = "xclip_run_20250511_055735"

TEST_CSV_PATH = os.path.join(BASE_DATA_DIR, "test.csv")
TEST_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "test")
BEST_MODEL_PATH = f"{BASE_PROJECT_DIR}/checkpoints_xclip/{BEST_RUN_DIR}/model_best.pth"
OUTPUT_SUBMISSION_FILE = f"{BASE_PROJECT_DIR}/submissions/submission_xclip_{TIMESTAMP}.csv"

XCLIP_MODEL_NAME = "microsoft/xclip-base-patch32"
NUM_FRAMES = 8
TARGET_FPS = 3
SEQUENCE_WINDOW_SECONDS = 10.0
BATCH_SIZE = 8
DATALOADER_NUM_WORKERS = 16
PIN_MEMORY = True

USE_TEST_TIME_AUGMENTATION = True
TTA_SAMPLES = 3  

def main():
    os.makedirs(os.path.dirname(OUTPUT_SUBMISSION_FILE), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- Prediction Configuration ---")
    print(f"Using device: {device}")
    print(f"Test CSV: {TEST_CSV_PATH}")
    print(f"Test Video Directory: {TEST_VIDEO_DIR}")
    print(f"Loading model from: {BEST_MODEL_PATH}")
    print(f"Output submission file: {OUTPUT_SUBMISSION_FILE}")
    print(f"Model: {XCLIP_MODEL_NAME}")
    print(f"Number of frames: {NUM_FRAMES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Test Time Augmentation: {'Enabled' if USE_TEST_TIME_AUGMENTATION else 'Disabled'}")
    
    df_test = pd.read_csv(TEST_CSV_PATH)
    
    df_test["id"] = df_test["id"].astype(str)
    
    if "time_of_alert" not in df_test.columns:
        df_test["time_of_alert"] = np.nan
    
    print(f"Loaded {len(df_test)} test video entries.")
    
    processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_NAME)
    
    print("Initializing test dataset...")
    test_dataset = XCLIPVideoDataset(
        df=df_test,
        video_dir=TEST_VIDEO_DIR,
        processor_name=XCLIP_MODEL_NAME,
        num_frames=NUM_FRAMES,
        target_fps=TARGET_FPS,
        sequence_window_seconds=SEQUENCE_WINDOW_SECONDS,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=collate_fn_xclip,
        pin_memory=PIN_MEMORY
    )
    
    print(f"Loading model from: {BEST_MODEL_PATH}")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model_config = checkpoint.get('config', {})
    
    loaded_model_name = model_config.get("XCLIP_MODEL_NAME", XCLIP_MODEL_NAME)
    loaded_num_frames = model_config.get("NUM_FRAMES", NUM_FRAMES)
    loaded_dropout_rate = model_config.get("DROPOUT_RATE", 0.3)
    
    print(f"Instantiating model with: name={loaded_model_name}, frames={loaded_num_frames}, dropout={loaded_dropout_rate}")
    
    model = CustomXCLIPModel(
        model_name=loaded_model_name,
        num_frames=loaded_num_frames,
        dropout_rate=0.0,  
        pretrained=False,  
        freeze_backbone=False,  
        freeze_text_model=True
    ).to(device)
    
    model.initialize_text_features(processor.tokenizer)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    def run_tta_inference():
        print("Running Test Time Augmentation...")
        all_video_ids = []
        all_scores = []
        
        tta_dataset = XCLIPVideoDataset(
            df=df_test,
            video_dir=TEST_VIDEO_DIR,
            processor_name=XCLIP_MODEL_NAME,
            num_frames=NUM_FRAMES,
            target_fps=TARGET_FPS,
            sequence_window_seconds=SEQUENCE_WINDOW_SECONDS,
            is_train=True,  
            augmentation_params={
                'strength': 0.3,  
                'color_jitter_prob': 0.7,
                'gray_scale_prob': 0.0,  
                'random_crop_scale': (0.8, 1.0),  
                'random_crop_ratio': (0.85, 1.15)  
            }
        )
        
        video_scores = {}
        
        for tta_idx in range(TTA_SAMPLES):
            print(f"TTA pass {tta_idx+1}/{TTA_SAMPLES}")
            tta_loader = DataLoader(
                tta_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=DATALOADER_NUM_WORKERS,
                collate_fn=collate_fn_xclip,
                pin_memory=PIN_MEMORY
            )
            
            with torch.no_grad():
                progress_bar = tqdm(tta_loader, desc=f"TTA {tta_idx+1}", unit="batch")
                for batch in progress_bar:
                    if batch is None:
                        continue
                    
                    pixel_values = batch["pixel_values"].to(device)
                    video_ids = batch["video_id"]
                    
                    _, seq_logits = model(pixel_values)
                    seq_probs = torch.sigmoid(seq_logits)
                    
                    for i, vid_id in enumerate(video_ids):
                        if vid_id not in video_scores:
                            video_scores[vid_id] = []
                        video_scores[vid_id].append(seq_probs[i].item())
        
        for vid_id, scores in video_scores.items():
            all_video_ids.append(vid_id)
            all_scores.append(np.mean(scores))
            
        return all_video_ids, all_scores
            
    def run_regular_inference():
        print("Running standard inference...")
        all_video_ids = []
        all_scores = []
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Predicting", unit="batch")
            for batch in progress_bar:
                if batch is None:
                    continue
                
                pixel_values = batch["pixel_values"].to(device)
                video_ids = batch["video_id"]
                
                _, seq_logits = model(pixel_values)
                seq_probs = torch.sigmoid(seq_logits)
                
                all_video_ids.extend(video_ids)
                all_scores.extend(seq_probs.cpu().numpy())
                
        return all_video_ids, all_scores
    
    if USE_TEST_TIME_AUGMENTATION:
        all_video_ids, all_scores = run_tta_inference()
    else:
        all_video_ids, all_scores = run_regular_inference()
    
    print(f"Generated {len(all_scores)} predictions.")
    
    submission_df = pd.DataFrame({
        "id": all_video_ids,
        "score": all_scores
    })
    
    submission_df = submission_df[["id", "score"]]
    
    submission_df.to_csv(OUTPUT_SUBMISSION_FILE, index=False)

if __name__ == "__main__":
    main()