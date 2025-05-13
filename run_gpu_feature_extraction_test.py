from datetime import datetime
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import json
import torch
from torch.utils.data import DataLoader
from transformers.models.clip import CLIPProcessor

from datasets.gpu_video_dataset import TestVideoFrameDataset, test_collate_fn
from models.ViT_model import get_clip_vision_model
from models.Vit_feature_extract import extract_features_single_video_optimized

# AI Prompts:
# Improve formatting
# And sensible outputs when runing the script
# Add control statements for errors

if __name__ == "__main__":
    # config 
    TEST_VIDEO_DIR = "nexar-collision-prediction/test" 
    CSV_PATH = "nexar-collision-prediction/test.csv"   
    
    VIT_MODEL_NAME = "openai/clip-vit-large-patch14"
    OUTPUT_DIR_ROOT = f"CLIP_ViT_Features_Test_{VIT_MODEL_NAME.split('/')[-1]}"
    
    FPS_TARGET = 3 
    TIME_WINDOW = 10.0 
    SEQUENCE_LENGTH = int(FPS_TARGET * TIME_WINDOW)

    FRAME_RESOLUTION_H_W_TUPLE = (720, 1280)

    VIDEO_LOADER_BATCH_SIZE = 1 
    DATALOADER_NUM_WORKERS = max(0, os.cpu_count() // 2 if os.cpu_count() else 2)
    INTERNAL_VIT_FRAME_BATCH_SIZE = 32 

    SAVE_INTERVAL_VIDEOS = 128 

    TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using target device: {TARGET_DEVICE}")

    if DATALOADER_NUM_WORKERS > 0 and TARGET_DEVICE == 'cuda':
        current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
                print("Set PyTorch multiprocessing start method to 'spawn'.")
            except RuntimeError as e:
                print(f"Warning: Could not set start method to 'spawn': {e}. Using default: {current_start_method}")

    # output directory setup
    timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
    feature_save_dir = os.path.join(OUTPUT_DIR_ROOT, f"run_{timestamp_str}")
    os.makedirs(feature_save_dir, exist_ok=True)
    print(f"Saving test features to: {feature_save_dir}")

    # load test video list
    df = pd.read_csv(CSV_PATH)
    if "id" not in df.columns:
        raise ValueError("CSV file must contain an 'id' column for video identifiers.")
    df["id"] = df["id"].apply(lambda x: str(x).zfill(5))

    num_total_videos = len(df)

    # save run metadata
    run_metadata = {
        "run_timestamp": timestamp_str, "vit_model_name": VIT_MODEL_NAME,
        "sequence_length_frames": SEQUENCE_LENGTH, "video_loader_batch_size": VIDEO_LOADER_BATCH_SIZE,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "internal_vit_frame_batch_size": INTERNAL_VIT_FRAME_BATCH_SIZE,
        "output_save_interval_videos": SAVE_INTERVAL_VIDEOS, "target_device": TARGET_DEVICE,
        "source_csv_file": CSV_PATH, "video_directory": TEST_VIDEO_DIR,
        "num_total_videos_in_run": num_total_videos,
    }
    with open(os.path.join(feature_save_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=4)
    print(f"Saved run metadata. Processing {num_total_videos} test videos.")

    # initialize model and processor once
    print(f"Initializing model {VIT_MODEL_NAME} and processor...")
    feature_extraction_model = get_clip_vision_model(model_name=VIT_MODEL_NAME).to(TARGET_DEVICE).eval()
    # ViT can be compiled for L4
    feature_extraction_model = torch.compile(feature_extraction_model)
    
    clip_processor = CLIPProcessor.from_pretrained(VIT_MODEL_NAME, use_fast=False)
    print("Model and processor initialized.")

    test_video_dataset = TestVideoFrameDataset(
        df=df,
        video_dir=TEST_VIDEO_DIR,
        sequence_length=SEQUENCE_LENGTH,
        frame_resolution_h_w_tuple=FRAME_RESOLUTION_H_W_TUPLE
    )
    
    # had batch related trouble without collate_fn, AI suggested this.
    # I understand its used for batching
    actual_collate_fn = test_collate_fn if VIDEO_LOADER_BATCH_SIZE > 1 else None
    test_video_dataloader = DataLoader(
        test_video_dataset,
        batch_size=VIDEO_LOADER_BATCH_SIZE,
        shuffle=False, 
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=actual_collate_fn,
        pin_memory=True if TARGET_DEVICE == 'cuda' and DATALOADER_NUM_WORKERS > 0 else False,
        persistent_workers=True if DATALOADER_NUM_WORKERS > 0 else False,
    )

    # main processing loop
    processed_video_count = 0
    current_saving_batch_features_list = []
    current_saving_batch_ids_list = []

    # Help of AI was taken to create debug statements
    for batch_data in tqdm(test_video_dataloader, desc="Processing Test Videos"):
        
        video_ids_list_from_batch = batch_data[0]
        frames_batch_tensor_from_loader = batch_data[1]  

        frames_batch_tensor_on_device = frames_batch_tensor_from_loader.to(TARGET_DEVICE)

        for i_in_batch in range(len(video_ids_list_from_batch)):
            video_id = video_ids_list_from_batch[i_in_batch] # string '00204'
            
            single_video_frames_tensor = frames_batch_tensor_on_device[i_in_batch] 
            # cur shape (SEQUENCE_LENGTH, C, H, W)

            # --- DEBUG PRINTS (use these new variable names for clarity) ---
            # print(f"\n--- MainScript DEBUG for Video ID: {video_id} ---") # video_id is now a string
            # print(f"  frames_batch_tensor_from_loader.shape: {frames_batch_tensor_from_loader.shape}")
            # print(f"  single_video_frames_tensor.shape (after indexing batch): {single_video_frames_tensor.shape}")
            # print(f"  single_video_frames_tensor.dtype: {single_video_frames_tensor.dtype}")
            # print(f"  Expected SEQUENCE_LENGTH for T dim: {SEQUENCE_LENGTH}")
            # --- END DEBUG PRINTS ---

            if single_video_frames_tensor.nelement() == 0 or \
               single_video_frames_tensor.shape[0] != SEQUENCE_LENGTH: # Now this check should be correct
                print(f"  WARNING: Insufficient/empty frames for test video {video_id} (shape: {single_video_frames_tensor.shape}). Appending empty features.")
                video_features_np = np.array([])
            else:
                # --- BEGIN DEBUG PRINTS ---
                # print(f"  Calling extract_features_single_video_optimized for {video_id} with frame shape {single_video_frames_tensor.shape}")
                video_features_np = extract_features_single_video_optimized(
                    video_frames_tensor_tchw=single_video_frames_tensor,
                    model=feature_extraction_model,
                    processor=clip_processor,
                    target_device=TARGET_DEVICE,
                    internal_model_batch_size=INTERNAL_VIT_FRAME_BATCH_SIZE
                )
            
            # --- BEGIN DEBUG PRINTS ---
            # print(f"  Returned video_features_np.shape for {video_id}: {video_features_np.shape if isinstance(video_features_np, np.ndarray) else 'Not a numpy array'}")

            current_saving_batch_features_list.append(video_features_np)
            current_saving_batch_ids_list.append(video_id)
            processed_video_count += 1

            if processed_video_count % SAVE_INTERVAL_VIDEOS == 0 or processed_video_count == num_total_videos:
                if not current_saving_batch_ids_list: continue # Nothing to save

                current_output_batch_num = (processed_video_count + SAVE_INTERVAL_VIDEOS -1) // SAVE_INTERVAL_VIDEOS
                print(f"\nSaving Output Batch {current_output_batch_num} ({len(current_saving_batch_ids_list)} videos)...")
                batch_file_suffix = f"saving_batch_{current_output_batch_num}"
                
                save_path_features = os.path.join(feature_save_dir, f"test_features_{batch_file_suffix}.npy")
                save_path_ids = os.path.join(feature_save_dir, f"test_ids_{batch_file_suffix}.npy") 

                try:
                    np.save(save_path_features, np.array(current_saving_batch_features_list, dtype=object))
                    np.save(save_path_ids, np.array(current_saving_batch_ids_list, dtype=str)) # Save IDs as strings
                    print(f"  Successfully saved batch {current_output_batch_num}.")
                except Exception as e:
                    print(f"  Error saving data for output batch {current_output_batch_num}: {e}")
                
                current_saving_batch_features_list, current_saving_batch_ids_list = [], []
    
    if current_saving_batch_ids_list:
        final_batch_num = (num_total_videos + SAVE_INTERVAL_VIDEOS - 1) // SAVE_INTERVAL_VIDEOS
        print(f"\nSaving Final Output Batch {final_batch_num} ({len(current_saving_batch_ids_list)} videos)...")
        batch_file_suffix = f"saving_batch_{final_batch_num}"
        save_path_features = os.path.join(feature_save_dir, f"test_features_{batch_file_suffix}.npy")
        save_path_ids = os.path.join(feature_save_dir, f"test_ids_{batch_file_suffix}.npy")
        try:
            np.save(save_path_features, np.array(current_saving_batch_features_list, dtype=object))
            np.save(save_path_ids, np.array(current_saving_batch_ids_list, dtype=str))
            print(f"  Successfully saved final batch {final_batch_num}.")
        except Exception as e:
            print(f"  Error saving data for final output batch: {e}")


    print(f"\nAll test video processing and feature saving completed in: {feature_save_dir}")